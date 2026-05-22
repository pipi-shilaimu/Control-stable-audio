"""诊断：验证 optimizer 是否真的在更新参数。跑 20 步，打印梯度 norm 和参数变化量。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnose_overfit_baseline import load_json
from stable_audio_control.audio_io import install_torchaudio_load_fallback
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.data.dataset import SampleDataset, LocalDatasetConfig
from stable_audio_tools.training.diffusion import DiffusionCondTrainingWrapper
import importlib.util
import torch
from torch.utils.data import DataLoader, IterableDataset

install_torchaudio_load_fallback()

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATASET_CONFIG = str(_PROJECT_ROOT / "stable_audio_control/data/mtg_jamendo/dataset_config_train_30.json")

cfg = load_json(_DATASET_CONFIG)
model, mcfg = get_pretrained_model("stabilityai/stable-audio-open-1.0")
ss, sr = int(mcfg["sample_size"]), int(mcfg["sample_rate"])
ac = int(mcfg.get("audio_channels", 2))
fc = "mono" if ac == 1 else "stereo"

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
tc = mcfg.get("training", {})
device = torch.device("cuda")

tw = DiffusionCondTrainingWrapper(
    model=model, lr=1e-4, mask_padding=False, mask_padding_dropout=0.0,
    use_ema=False, log_loss_info=False, optimizer_configs=None,
    pre_encoded=False, cfg_dropout_prob=0.0,
    timestep_sampler=tc.get("timestep_sampler", "uniform"),
    timestep_sampler_options=tc.get("timestep_sampler_options", {}),
)
tw = tw.to(device).train()
opt = torch.optim.Adam(tw.parameters(), lr=1e-4)

class TS:
    def __init__(self, o): self.optimizers = [o]; self.barebones = False
tw._trainer = TS(opt)
tw.log_dict = lambda *a, **kw: None

it = iter(dl)

# 找一个可训练参数做 probe
probe_name = None
probe_before = None
for n, p in tw.named_parameters():
    if p.requires_grad and p.ndim >= 2:
        probe_name = n
        probe_before = p.detach().clone()
        break
print(f"probe: {probe_name}  shape={list(probe_before.shape)}")

for step in range(20):
    b = next(it)
    r, m = b[0], b[1]
    if r.ndim == 2:
        r = r.unsqueeze(0)

    # Normalize metadata: padding_mask must be wrapped in a list
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

    opt.zero_grad()
    loss = tw.training_step((r, m), 0)
    loss.backward()

    total_norm = 0.0
    for p in tw.parameters():
        if p.grad is not None:
            total_norm += p.grad.norm().item() ** 2
    total_norm = total_norm ** 0.5

    opt.step()
    probe_after = dict(tw.named_parameters())[probe_name].detach()
    delta = (probe_after - probe_before).abs().max().item()
    probe_before = probe_after.clone()
    print(f"step {step:2d}  loss={loss.item():.4f}  grad_norm={total_norm:.2f}  probe_delta={delta:.6f}")

print("DONE")