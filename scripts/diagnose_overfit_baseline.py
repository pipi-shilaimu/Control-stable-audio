"""
Phase 0 Step 0: 纯 backbone 单 batch 过拟合 baseline。

回答核心问题：不加 ControlNet 的原始 StableAudio backbone，
全参数训练，同一条数据，v-objective MSE 能降到多低？

用法：
    python scripts/diagnose_overfit_baseline.py [--steps 500] [--lr 1e-4]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_audio_control.audio_io import install_torchaudio_load_fallback
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.data.dataset import SampleDataset, LocalDatasetConfig
from stable_audio_tools.training.diffusion import DiffusionCondTrainingWrapper

import importlib.util
import torch
from torch.utils.data import DataLoader, IterableDataset

install_torchaudio_load_fallback()

# Hardcoded dataset config path relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATASET_CONFIG = str(_PROJECT_ROOT / "stable_audio_control/data/mtg_jamendo/dataset_config_train_30.json")


def load_json(path):
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    model_name = "stabilityai/stable-audio-open-1.0"
    lr = 1e-4
    max_steps = 500
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_config = load_json(_DATASET_CONFIG)
    base_model, model_config = get_pretrained_model(model_name)

    sample_size = int(model_config["sample_size"])
    sample_rate = int(model_config["sample_rate"])
    audio_channels = int(model_config.get("audio_channels", 2))
    force_channels = "mono" if audio_channels == 1 else "stereo"

    # Build SampleDataset directly (avoids Windows multiprocessing pickle issues)
    configs = []
    for audio_dir_config in dataset_config.get("datasets", []):
        custom_metadata_fn = None
        custom_metadata_module_path = audio_dir_config.get("custom_metadata_module", None)
        if custom_metadata_module_path is not None:
            spec = importlib.util.spec_from_file_location("metadata_module", custom_metadata_module_path)
            metadata_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(metadata_module)
            custom_metadata_fn = metadata_module.get_custom_metadata

        configs.append(
            LocalDatasetConfig(
                id=audio_dir_config["id"],
                path=audio_dir_config["path"],
                custom_metadata_fn=custom_metadata_fn,
            )
        )

    train_dataset = SampleDataset(
        configs,
        sample_rate=sample_rate,
        sample_size=sample_size,
        random_crop=dataset_config.get("random_crop", True),
        force_channels=force_channels,
    )

    single_batch = train_dataset[0]

    class _RepeatDataset(IterableDataset):
        def __init__(self, batch):
            self.batch = batch
        def __iter__(self):
            while True:
                yield self.batch

    train_dl = DataLoader(_RepeatDataset(single_batch), batch_size=None, num_workers=0)

    print(f"=== PURE BACKBONE OVERFIT BASELINE ===")
    print(f"model:      {model_name}")
    print(f"dataset:    {_DATASET_CONFIG}")
    print(f"device:     {device}")
    print(f"sample_rate: {sample_rate}, sample_size: {sample_size}")
    print(f"max_steps:  {max_steps}, lr: {lr}")
    print(f"batch shape: {single_batch[0].shape}")
    prompt = single_batch[1].get("prompt", "N/A") if isinstance(single_batch[1], dict) else str(single_batch[1])
    print(f"prompt:     '{prompt[:80]}'")

    # Full training, no freezing, no ControlNet. cfg_dropout=0.
    training_config = model_config.get("training", {})

    training_wrapper = DiffusionCondTrainingWrapper(
        model=base_model,
        lr=lr,
        mask_padding=training_config.get("mask_padding", False),
        mask_padding_dropout=training_config.get("mask_padding_dropout", 0.0),
        use_ema=False,
        log_loss_info=training_config.get("log_loss_info", False),
        optimizer_configs=None,
        pre_encoded=training_config.get("pre_encoded", False),
        cfg_dropout_prob=0.0,
        timestep_sampler=training_config.get("timestep_sampler", "uniform"),
        timestep_sampler_options=training_config.get("timestep_sampler_options", {}),
    )
    training_wrapper = training_wrapper.to(device)
    training_wrapper.train()

    total_params = sum(p.numel() for p in training_wrapper.parameters() if p.requires_grad)
    print(f"trainable params: {total_params:,}")

    optimizer = torch.optim.Adam(training_wrapper.parameters(), lr=lr)

    # Mock trainer + log_dict to satisfy Lightning requirements outside pl.Trainer
    class _TrainerStub:
        def __init__(self, opt):
            self.optimizers = [opt]
            self.barebones = False
    training_wrapper._trainer = _TrainerStub(optimizer)
    training_wrapper.log_dict = lambda *args, **kwargs: None

    dataloader_iter = iter(train_dl)
    losses = []

    for step in range(max_steps):
        batch = next(dataloader_iter)
        reals, metadata = batch[0], batch[1]

        # SampleDataset returns [C, T]; pretransform.encode needs [B, C, T]
        if reals.ndim == 2:
            reals = reals.unsqueeze(0)
        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        # Normalize metadata for stable-audio-tools conditioner
        if isinstance(metadata, dict):
            pm = metadata.get("padding_mask")
            if isinstance(pm, torch.Tensor):
                metadata = dict(metadata)
                metadata["padding_mask"] = [pm]
            metadata = [metadata]
        if isinstance(metadata, list):
            for item in metadata:
                if isinstance(item, dict):
                    pm = item.get("padding_mask")
                    if isinstance(pm, torch.Tensor):
                        item = dict(item)
                        item["padding_mask"] = [pm]

        # Move to GPU
        reals = reals.to(device=device)
        for m in metadata:
            if not isinstance(m, dict):
                continue
            for k, v in list(m.items()):
                if isinstance(v, torch.Tensor):
                    m[k] = v.to(device=device)
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                    m[k] = [t.to(device=device) for t in v]

        optimizer.zero_grad()
        loss = training_wrapper.training_step((reals, metadata), 0)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.detach().cpu())
        losses.append(loss_val)

        if step % 50 == 0 or step < 10:
            print(f"  step {step:4d}  loss={loss_val:.6f}")

    min_loss = min(losses)
    avg_last_50 = sum(losses[-50:]) / 50

    print()
    print(f"=== RESULTS ===")
    print(f"Initial loss (step 0):      {losses[0]:.6f}")
    print(f"Final loss (step {max_steps - 1}):   {losses[-1]:.6f}")
    print(f"Minimum loss:               {min_loss:.6f}")
    print(f"Average loss (last 50 steps): {avg_last_50:.6f}")

    print()
    print("=== INTERPRETATION ===")
    if min_loss < 0.1:
        print("Backbone CAN overfit single batch -> ControlNet injection is the problem.")
    elif min_loss < 0.5:
        print(f"Backbone partially converges (min={min_loss:.4f}) -> ControlNet may be amplifying the issue.")
    else:
        print(f"Backbone CANNOT overfit single batch (min={min_loss:.4f}).")
        print("Problem is NOT in ControlNet -> likely in text conditioning or model config path.")


if __name__ == "__main__":
    main()