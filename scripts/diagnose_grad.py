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

dataset_config = load_json(_DATASET_CONFIG)
base_model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
sample_size = int(model_config["sample_size"])
sample_rate = int(model_config["sample_rate"])
audio_channels = int(model_config.get("audio_channels", 2))
force_channels = "mono" if audio_channels == 1 else "stereo"

dataset_configs = []
for audio_dir_config in dataset_config.get("datasets", []):
    custom_metadata_fn = None
    metadata_module_path = audio_dir_config.get("custom_metadata_module")
    if metadata_module_path:
        spec = importlib.util.spec_from_file_location("metadata_module", metadata_module_path)
        metadata_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metadata_module)
        custom_metadata_fn = metadata_module.get_custom_metadata
    dataset_configs.append(LocalDatasetConfig(
        id=audio_dir_config["id"],
        path=audio_dir_config["path"],
        custom_metadata_fn=custom_metadata_fn,
    ))

dataset = SampleDataset(
    dataset_configs,
    sample_rate=sample_rate,
    sample_size=sample_size,
    random_crop=dataset_config.get("random_crop", True),
    force_channels=force_channels,
)
single_batch = dataset[0]


class SingleBatchDataset(IterableDataset):
    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        while True:
            yield self.batch


dataloader = DataLoader(SingleBatchDataset(single_batch), batch_size=None, num_workers=0)
training_config = model_config.get("training", {})
device = torch.device("cuda")

training_wrapper = DiffusionCondTrainingWrapper(
    model=base_model,
    lr=1e-4,
    mask_padding=False,
    mask_padding_dropout=0.0,
    use_ema=False,
    log_loss_info=False,
    optimizer_configs=None,
    pre_encoded=False,
    cfg_dropout_prob=0.0,
    timestep_sampler=training_config.get("timestep_sampler", "uniform"),
    timestep_sampler_options=training_config.get("timestep_sampler_options", {}),
)
training_wrapper = training_wrapper.to(device).train()
optimizer = torch.optim.Adam(training_wrapper.parameters(), lr=1e-4)


class TrainerStub:
    def __init__(self, optimizer):
        self.optimizers = [optimizer]
        self.barebones = False


training_wrapper._trainer = TrainerStub(optimizer)
training_wrapper.log_dict = lambda *args, **kwargs: None

dataloader_iter = iter(dataloader)

# 找一个可训练参数做 probe
probe_name = None
probe_before = None
for param_name, param in training_wrapper.named_parameters():
    if param.requires_grad and param.ndim >= 2:
        probe_name = param_name
        probe_before = param.detach().clone()
        break
print(f"probe: {probe_name}  shape={list(probe_before.shape)}")

for step in range(20):
    batch = next(dataloader_iter)
    reals, metadata = batch[0], batch[1]
    if reals.ndim == 2:
        reals = reals.unsqueeze(0)

    # Normalize metadata: padding_mask must be wrapped in a list
    if isinstance(metadata, dict):
        padding_mask = metadata.get("padding_mask")
        if isinstance(padding_mask, torch.Tensor):
            metadata = dict(metadata)
            metadata["padding_mask"] = [padding_mask]
        metadata = [metadata]

    reals = reals.to(device)
    for metadata_item in metadata:
        for key, value in list(metadata_item.items()):
            if isinstance(value, torch.Tensor):
                metadata_item[key] = value.to(device)
            elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                metadata_item[key] = [tensor.to(device) for tensor in value]

    optimizer.zero_grad()
    loss = training_wrapper.training_step((reals, metadata), 0)
    loss.backward()

    # Compute total gradient norm across all parameters
    total_grad_norm = 0.0
    for param in training_wrapper.parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.norm().item() ** 2
    total_grad_norm = total_grad_norm ** 0.5

    optimizer.step()

    # Measure how much the probe parameter changed in this step
    probe_after = dict(training_wrapper.named_parameters())[probe_name].detach()
    probe_delta = (probe_after - probe_before).abs().max().item()
    probe_before = probe_after.clone()

    print(f"step {step:2d}  loss={loss.item():.4f}  grad_norm={total_grad_norm:.2f}  probe_delta={probe_delta:.6f}")

print("DONE")