"""Phase 0: 诊断 ControlNet 分支梯度是否真的在流动。
复制 train_controlnet_dit.py 的关键搭建逻辑，跑 20 步，
打印 control_layers / zero_linears / melody_encoder 的梯度 norm 和参数变化。
"""


from __future__ import annotations

import sys
from pathlib import Path
#for p in Path(__file__).resolve().parents:
#    print(str(p))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import ipdb
bp = ipdb.set_trace
import importlib.util
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from stable_audio_control.audio_io import install_torchaudio_load_fallback
from stable_audio_control.melody.extractors import (
    MelodyExtractor,
    build_melody_extractor,
    melody_control_channels,
)
from stable_audio_control.models import (
    ControlConditionedDiffusionWrapper,
    ControlNetContinuousTransformer,
    build_control_wrapper,
)
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.data.dataset import SampleDataset, LocalDatasetConfig
from stable_audio_tools.training.diffusion import DiffusionCondTrainingWrapper

def load_json(path):
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    

install_torchaudio_load_fallback()

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATASET_CONFIG = str(_PROJECT_ROOT / "stable_audio_control/data/mtg_jamendo/dataset_config_train_30.json")



def main():

    
    configs = []
    for adc in cfg.get("datasets", []):
        fn = None
        mp = adc.get("custom_metadata_module")
        if mp:
            spec = importlib.util.spec_from_file_location("mm", mp)
            mm = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mm)
            fn = mm.get_custom_metadata
        configs.append(LocalDatasetConfig(id=adc["id"], path=adc["path"], custom_metadata_fn=fn))

    ds = SampleDataset(configs, sample_rate=sr, sample_size=ss, random_crop=cfg.get("random_crop", True), force_channels=fc)
    sb = ds[0]

    class RD(IterableDataset):
        def __init__(self, b): self.b = b
        def __iter__(self):
            while True: yield self.b

    dl = DataLoader(RD(sb), batch_size=None, num_workers=0)

    # --- 2. Build ControlNet (CQT, top_k=4 -> 8 channels) ---
    control_channels = melody_control_channels("cqt", top_k=4, chroma_bins=12)

    control_model = build_control_wrapper(
        base_wrapper=base_model,
        num_control_layers=12,
        control_id="melody_control",
        default_control_scale=1.0,
        freeze_base=True,
        melody_channels=control_channels,
        melody_num_pitch_bins=128,
        melody_embedding_dim=64,
        melody_hidden_dim=256,
        melody_conv_layers=2,
        use_melody_encoder=True,
    )
    model_dtype = next(control_model.parameters()).dtype

    extractor = build_melody_extractor(
        feature="cqt",
        sample_rate=sr,
        fmin_hz=8.175798915643707,
        highpass_cutoff_hz=261.2,
        n_bins=128,
        bins_per_octave=12,
        hop_length=512,
        top_k=4,
        cqt_backend="auto",
        chroma_bins=12,
        chroma_n_fft=2048,
    )

    # --- 3. Create MelodyControlAugmenter (same as train_controlnet_dit.py) ---
    # We embed the extractor inline rather than using the full class
    original_conditioner = control_model.base_wrapper.conditioner

    class MelodyControlAugmenter(nn.Module):
        def __init__(self, base_conditioner, control_id, extractor, control_channels):
            super().__init__()
            self.base_conditioner = base_conditioner
            self.control_id = control_id
            self.extractor = extractor
            self.control_channels = control_channels
            self._batch_audio = None

        def set_batch_audio(self, audio):
            self._batch_audio = audio.detach()

        def forward(self, metadata, device):
            conditioning = self.base_conditioner(metadata, device)
            if not metadata:
                conditioning[self.control_id] = [
                    torch.zeros((0, self.control_channels, 1), device=device, dtype=torch.long),
                    None,
                ]
                return conditioning
            if self._batch_audio is None:
                raise RuntimeError("missing batch audio")
            waveform = self._batch_audio.to(device=device, dtype=torch.float32)
            if waveform.ndim == 2:
                waveform = waveform.unsqueeze(0)
            if waveform.shape[1] == 1:
                waveform = waveform.repeat(1, 2, 1)
            melody = self.extractor.extract(waveform).to(device=device)
            conditioning[self.control_id] = [melody, None]
            self._batch_audio = None
            return conditioning

    control_model.base_wrapper.conditioner = MelodyControlAugmenter(
        base_conditioner=original_conditioner,
        control_id="melody_control",
        extractor=extractor,
        control_channels=control_channels,
    )

    # --- 4. Initialize lazy parameters & freeze policy ---
    # First, materialize lazy params by doing a dummy control_input extraction
    def _init_lazy():
        for dtype_tensor in [
            torch.zeros((1, control_channels, 8), device="cpu", dtype=torch.long),
            torch.zeros((1, control_channels, 8), device="cpu", dtype=model_dtype),
        ]:
            cond = {"melody_control": [dtype_tensor, None]}
            control_model._extract_control_input(cond=cond, target_len=8, dtype=model_dtype, device="cpu")

    _init_lazy()

    # Freeze everything, then unfreeze control branch
    from torch.nn.parameter import UninitializedParameter
    for param in control_model.parameters():
        if isinstance(param, UninitializedParameter):
            continue
        param.requires_grad_(False)

    transformer = control_model.model.model.transformer
    assert isinstance(transformer, ControlNetContinuousTransformer)

    target_modules = {
        "control_layers": transformer.control_layers,
        "zero_linears": transformer.zero_linears,
        "melody_encoder": control_model.melody_encoder,
        "control_projector": control_model.control_projector,
    }
    trainable_names = []
    for prefix, module in target_modules.items():
        if module is None:
            continue
        for name, param in module.named_parameters():
            if isinstance(param, UninitializedParameter):
                continue
            param.requires_grad_(True)
            trainable_names.append(f"{prefix}.{name}")

    if not trainable_names:
        raise RuntimeError("No trainable parameters!")
    total_trainable = sum(p.numel() for p in control_model.parameters() if p.requires_grad)
    print(f"ControlNet trainable params: {total_trainable:,}")
    print(f"Trainable name samples: {trainable_names[:6]}")

    # --- 5. Training wrapper ---
    tc = model_config.get("training", {})
    control_model = control_model.to(device).train()

    tw = DiffusionCondTrainingWrapper(
        model=control_model,
        lr=lr,
        mask_padding=tc.get("mask_padding", False),
        mask_padding_dropout=tc.get("mask_padding_dropout", 0.0),
        use_ema=False,
        log_loss_info=tc.get("log_loss_info", False),
        optimizer_configs=None,
        pre_encoded=tc.get("pre_encoded", False),
        cfg_dropout_prob=0.0,
        timestep_sampler=tc.get("timestep_sampler", "uniform"),
        timestep_sampler_options=tc.get("timestep_sampler_options", {}),
    )
    tw = tw.to(device).train()

    opt = torch.optim.Adam(tw.parameters(), lr=lr)

    class TS:
        def __init__(self, o): self.optimizers = [o]; self.barebones = False
    tw._trainer = TS(opt)
    tw.log_dict = lambda *a, **kw: None

    # --- 6. Pick probes from control branch ---
    probes = {}
    probes_before = {}
    for prefix in ["control_layers", "zero_linears", "melody_encoder", "control_projector"]:
        for n, p in control_model.named_parameters():
        # 只要名称中包含该前缀，且满足其他条件
            if prefix in n and p.requires_grad and p.ndim >= 2:
                probes[prefix] = n
                probes_before[prefix] = p.detach().clone()
                break
    #bp()
    for prefix, name in probes.items():
        p = dict(control_model.named_parameters())[name]
        print(f"probe {prefix}: {name}  shape={list(p.shape)}")


    
    # --- 7. Training loop ---
    it = iter(dl)
    conditioner = control_model.base_wrapper.conditioner  # MelodyControlAugmenter
    for step in range(20):
        b = next(it)
        r, m = b[0], b[1]
        if r.ndim == 2:
            r = r.unsqueeze(0)
        # Normalize metadata
        if isinstance(m, dict):
            pm = m.get("padding_mask")
            if isinstance(pm, torch.Tensor):
                m = dict(m)
                m["padding_mask"] = [pm]
            m = [m]
        r = r.to(device)
        for mi in m:
    
            for k, v in list(mi.items()):
                if isinstance(v, torch.Tensor):
                    mi[k] = v.to(device)
       
                elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                    mi[k] = [t.to(device) for t in v]
                 
        # Inject batch audio into conditioner before training step
        conditioner.set_batch_audio(r)

        opt.zero_grad()
        loss = tw.training_step((r, m), 0)
        loss.backward()

        # Compute per-branch grad norms
        grad_info = {}
        for prefix in ["control_layers", "zero_linears", "melody_encoder", "control_projector"]:
            gn = 0.0
            for n, p in control_model.named_parameters():
                if prefix in n and p.grad is not None:
                    gn += p.grad.norm().item() ** 2
            grad_info[prefix] = gn ** 0.5

        opt.step()

        # Compute per-branch parameter deltas
        delta_info = {}
        for prefix, name in probes.items():
            p_after = dict(control_model.named_parameters())[name].detach()
            delta = (p_after - probes_before[prefix]).abs().max().item()
            probes_before[prefix] = p_after.clone()
            delta_info[prefix] = delta
        #bp()
        print(f"step {step:2d}  loss={loss.item():.4f}  "
              f"gl_ctl={grad_info.get('control_layers',0):.3f}  gl_z={grad_info.get('zero_linears',0):.3f}  "
              f"gl_mel={grad_info.get('melody_encoder',0):.3f}  gl_proj={grad_info.get('control_projector',0):.3f}  "
              f"d_ctl={delta_info.get('control_layers',0):.2e}  d_z={delta_info.get('zero_linears',0):.2e}  "
              f"d_mel={delta_info.get('melody_encoder',0):.2e}")
    print("DONE")

model_name = "stabilityai/stable-audio-open-1.0"
lr = 1e-4
device = torch.device("cuda")
    # --- 1. Load data (same as baseline) ---
cfg = load_json(_DATASET_CONFIG)
base_model, model_config = get_pretrained_model(model_name)
ss = int(model_config["sample_size"])
sr = int(model_config["sample_rate"])
ac = int(model_config.get("audio_channels", 2))
fc = "mono" if ac == 1 else "stereo"
print("sample_rate:", sr, "sample_size:", ss, "force_channels:", fc)
main()