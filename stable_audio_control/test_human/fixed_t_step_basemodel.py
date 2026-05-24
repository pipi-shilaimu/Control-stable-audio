from __future__ import annotations
import sys
from pathlib import Path


#for p in Path(__file__).resolve().parents:
#    print(str(p))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stable_audio_control import data
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

# === Codex: 手动构造固定 t 的训练需要这两个 import ===
from stable_audio_tools.training.diffusion import get_alphas_sigmas  # t → alpha, sigma
from torch.nn.functional import mse_loss  # 直接算 MSE，不走 MultiLoss 包装


def load_json(path):
    import json
    with open(path, "r") as f:
        return json.load(f)
install_torchaudio_load_fallback()

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
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

trainning_wrapper = DiffusionCondTrainingWrapper(
    model=base_model,
    lr = 1e-4,
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

training_wrapper = trainning_wrapper.to(device).train()
optimizer = torch.optim.Adam(trainning_wrapper.parameters(), lr=1e-4)


class TD:
    def __init__(self, optimizer):
        self.optimizers = [optimizer]
        self.barebones = False

training_wrapper._trainer = TD(optimizer)
training_wrapper.log_dict = lambda *args, **kwargs: None

# === Codex: 循环前准备 — 固定 t、固定噪声 seed、提取常用子模块 ===
FIXED_T = 0.9                          # 固定 timestep，t≈1 噪声最大，目标可学
noise_seed = None                       # 第一步初始化，后面每步复用同一个噪声
pretransform = training_wrapper.diffusion.pretransform  # waveform → latent
conditioner = training_wrapper.diffusion.conditioner    # 文本条件 (T5)

dataloader_iter = iter(dataloader)
_batch = next(dataloader_iter)           # 取一次
_metadata = _batch[1]
if isinstance(_metadata, dict):
    _metadata = [_metadata]
for mi in _metadata:
    for k, v in list(mi.items()):
        if isinstance(v, torch.Tensor):
            mi[k] = v.to(device)
with torch.no_grad():
    fixed_cond = conditioner(_metadata, device)   # 只算一次

for step in range(1000):
    batch = next(dataloader_iter)
    reals, metadata = batch[0], batch[1]
    if reals.ndim == 2:
        reals = reals.unsqueeze(0)
    reals = reals.to(device)
    if isinstance(metadata, dict):
        metadata = [metadata]
    for metadata_item in metadata:
        for key, value in list(metadata_item.items()):
            if isinstance(value, torch.Tensor):
                metadata_item[key] = value.to(device)
            elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                metadata_item[key] = [tensor.to(device) for tensor in value]

    # === Codex: 手动构造训练步骤 — 绕过 training_step 的随机 t 采样 ===
    # 1. pretransform: waveform → latent（冻结，不需要梯度）
    with torch.no_grad():
        raw = pretransform.model.encoder(reals)
        latent = raw[:, :64, :] / pretransform.scale    # [1, C_latent, L]
    # 2. 固定 timestep + 固定噪声（第一步生成，后面复用）
    if noise_seed is None:
        noise_seed = torch.randn_like(latent)          # 只生成一次
    t = torch.full((1,), FIXED_T, device=device)       # 每一步都用同一个 t

    # 3. noise schedule + noising
    alphas, sigmas = get_alphas_sigmas(t)
    alphas = alphas[:, None, None]                     # [1] → [1,1,1]
    sigmas = sigmas[:, None, None]
    noised_latent = latent * alphas + noise_seed * sigmas

    # 4. v-objective target: v = alpha*noise - sigma*x0
    target = noise_seed * alphas - latent * sigmas

    # 5. 文本条件（冻结 T5，不需要梯度）
    #with torch.no_grad():
    #    cond = conditioner(metadata, device)

    # 6. diffusion forward（需要梯度，backbone 可训练）
    training_wrapper.diffusion.eval()
    output = training_wrapper.diffusion(
        noised_latent, t,
        cond=fixed_cond,
        cfg_dropout_prob=0.0,
    )
    training_wrapper.diffusion.train()

    # 7. MSE loss（比 MultiLoss 更直接，同时 output/target 的 padding_mask 一致）
    loss = mse_loss(output, target)

    # 8. backward + step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    print(f"step {step:4d}  t={FIXED_T}  loss={loss.item():.6f}")

print("DONE")

"""
fixed_t_step_basemodel.py

目的：验证"消除 VAE bottleneck 随机采样后，纯 backbone 全参数训练能否收敛"。

方法：
  - 固定 timestep t=0.9（避开 t≈0 时 target=random_noise 的不可学区域）
  - 固定噪声 seed（第一步 torch.randn_like 后复用）
  - 固定文本条件（T5 conditioner 只调用一次）
  - 绕过 VAE bottleneck，直接用 encoder 的 mean 输出作为 latent：
        raw = pretransform.model.encoder(reals)       # [B, 128, L]
        latent = raw[:, :64, :] / pretransform.scale  # 前 64 通道 = mean
  - 冻结 transformer dropout（diffusion.eval()）
  - 全参数训练（1.05B 参数），lr=1e-4

验证结果（2026-05-23）：
  step  0: loss=0.164139
  step  5: loss=0.047535
  step 10: loss=0.019868
  step 15: loss=0.009974
  step 16: loss=0.008965  ✽ 16 步收敛

结论：VAE bottleneck 的 vae_sample() → torch.randn_like() 是此前所有训练
      （backbone 5000 步、ControlNet 25000 步）loss 不降的唯一根因。
      pretransform.encode() 不应在冻结的扩散训练中使用——应改用 encoder mean。
"""
