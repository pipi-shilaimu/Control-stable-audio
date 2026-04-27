from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.parameter import UninitializedParameter

# Allow script execution without installing local package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_audio_control.melody.cqt_topk import CQTTopKConfig, CQTTopKExtractor  # noqa: E402
from stable_audio_control.models import (  # noqa: E402
    ControlConditionedDiffusionWrapper,
    ControlNetContinuousTransformer,
    build_control_wrapper,
)
from stable_audio_tools import get_pretrained_model  # noqa: E402
from stable_audio_tools.data.dataset import create_dataloader_from_config  # noqa: E402
from stable_audio_tools.models.diffusion import ConditionedDiffusionModelWrapper  # noqa: E402
from stable_audio_tools.training.diffusion import DiffusionCondTrainingWrapper  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def _str_to_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train ControlNet-DiT on StableAudio Open with top-k CQT melody control."
    )
    parser.add_argument("--dataset-config", type=str, required=True, help="Path to dataset config JSON.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="stabilityai/stable-audio-open-1.0",
        help="HuggingFace pretrained model id for base StableAudio Open.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--gradient-clip-val", type=float, default=0.0)
    parser.add_argument("--log-every-n-steps", type=int, default=10)
    parser.add_argument(
        "--limit-train-batches",
        type=float,
        default=1.0,
        help="Lightning limit_train_batches. 1.0 means full epoch.",
    )
    parser.add_argument(
        "--default-root-dir",
        type=str,
        default="outputs/train_controlnet_dit",
        help="Lightning output root dir.",
    )
    parser.add_argument("--ckpt-path", type=str, default=None, help="Optional checkpoint path for resume.")
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        help="Lightning precision (e.g. 16-mixed, bf16-mixed, 32-true).",
    )
    parser.add_argument("--accelerator", type=str, default="auto", help="Lightning accelerator.")
    parser.add_argument("--devices", type=int, default=1, help="Lightning devices count.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override model_config training lr.")
    parser.add_argument("--use-ema", type=_str_to_bool, default=True, help="Enable EMA in training wrapper.")

    # ControlNet args
    parser.add_argument("--num-control-layers", type=int, default=2)
    parser.add_argument("--control-id", type=str, default="melody_control")
    parser.add_argument("--default-control-scale", type=float, default=1.0)
    parser.add_argument("--freeze-base", type=_str_to_bool, default=True)

    # CQT args
    parser.add_argument("--top-k", type=int, default=4, help="Top-k bins per stereo channel.")
    parser.add_argument("--n-bins", type=int, default=128)
    parser.add_argument("--bins-per-octave", type=int, default=12)
    parser.add_argument("--fmin-hz", type=float, default=8.175798915643707)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--highpass-cutoff-hz", type=float, default=261.2)
    parser.add_argument("--cqt-backend", type=str, choices=["auto", "nnaudio", "librosa"], default="auto")

    return parser


def initialize_lazy_parameters(
    model: ControlConditionedDiffusionWrapper,
    *,
    device: torch.device,
    dtype: torch.dtype,
    control_channels: int,
) -> None:
    """Materialize lazy parameters (e.g. control_projector) before trainer starts."""
    dummy_cond: Dict[str, Any] = {
        "melody_control": [torch.zeros((1, control_channels, 8), device=device, dtype=dtype), None],
    }
    _ = model._extract_control_input(  # type: ignore[attr-defined]
        cond=dummy_cond,
        target_len=8,
        dtype=dtype,
        device=device,
    )


def apply_control_only_freeze_policy(model: ControlConditionedDiffusionWrapper) -> List[str]:
    """Freeze everything except control branch and control projector."""
    for param in model.parameters():
        if isinstance(param, UninitializedParameter):
            continue
        param.requires_grad_(False)

    transformer = model.model.model.transformer
    if not isinstance(transformer, ControlNetContinuousTransformer):
        raise TypeError("Expected ControlNetContinuousTransformer after build_control_wrapper.")

    target_modules: Dict[str, nn.Module] = {
        "control_layers": transformer.control_layers,
        "zero_linears": transformer.zero_linears,
        "control_projector": model.control_projector,
    }

    trainable_names: List[str] = []
    for prefix, module in target_modules.items():
        for name, param in module.named_parameters():
            if isinstance(param, UninitializedParameter):
                continue
            param.requires_grad_(True)
            trainable_names.append(f"{prefix}.{name}")

    return trainable_names


class MelodyControlAugmenter(nn.Module):
    """
    Inject `melody_control` into conditioner output.

    The training wrapper sets the current batch waveform via `set_batch_audio(...)`
    right before calling the parent training step.
    """

    def __init__(
        self,
        base_conditioner: nn.Module,
        control_id: str,
        extractor: CQTTopKExtractor,
    ) -> None:
        super().__init__()
        self.base_conditioner = base_conditioner
        self.control_id = control_id
        self.extractor = extractor
        self._batch_audio: Optional[torch.Tensor] = None

    def set_batch_audio(self, audio: torch.Tensor) -> None:
        self._batch_audio = audio.detach()

    @staticmethod
    def _ensure_stereo(audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 2:
            audio = audio.unsqueeze(0)
        if audio.ndim != 3:
            raise ValueError(f"Expected [B, C, T] audio tensor, got shape={tuple(audio.shape)}")

        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)
        if audio.shape[1] != 2:
            raise ValueError(
                f"Expected mono/stereo waveform with channel dim 1 or 2, got channels={audio.shape[1]}"
            )
        return audio

    def forward(self, metadata: List[Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
        conditioning = self.base_conditioner(metadata, device)

        if not metadata:
            channels = self.extractor.config.top_k * 2
            conditioning[self.control_id] = [torch.zeros((0, channels, 1), device=device, dtype=torch.long), None]
            return conditioning

        if self._batch_audio is None:
            raise RuntimeError(
                "MelodyControlAugmenter missing batch audio. "
                "Use MelodyAwareDiffusionCondTrainingWrapper to set batch waveform before conditioning."
            )

        waveform = self._ensure_stereo(self._batch_audio.to(device=device, dtype=torch.float32))
        melody_control = self.extractor.extract(waveform).to(device=device)
        conditioning[self.control_id] = [melody_control, None]

        # Clear stale reference to avoid accidental reuse.
        self._batch_audio = None
        return conditioning


class MelodyAwareDiffusionCondTrainingWrapper(DiffusionCondTrainingWrapper):
    """Training wrapper that injects batch waveform into melody augmenter each step."""

    def __init__(self, *args, melody_augmenter: MelodyControlAugmenter, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.melody_augmenter = melody_augmenter

    def _set_melody_batch(self, batch) -> None:
        reals = batch[0]
        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        if self.pre_encoded:
            raise RuntimeError(
                "pre_encoded=True is not supported in this script because melody extraction requires waveform audio."
            )

        self.melody_augmenter.set_batch_audio(reals)

    def training_step(self, batch, batch_idx):
        self._set_melody_batch(batch)
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        self._set_melody_batch(batch)
        return super().validation_step(batch, batch_idx)


def create_training_wrapper(
    *,
    model_config: Dict[str, Any],
    model: ControlConditionedDiffusionWrapper,
    melody_augmenter: MelodyControlAugmenter,
    learning_rate_override: Optional[float],
    use_ema_override: bool,
) -> MelodyAwareDiffusionCondTrainingWrapper:
    training_config = model_config.get("training", {})
    optimizer_configs = training_config.get("optimizer_configs", None)
    learning_rate = learning_rate_override if learning_rate_override is not None else training_config.get("learning_rate")

    if learning_rate is None and optimizer_configs is None:
        learning_rate = 5e-5

    return MelodyAwareDiffusionCondTrainingWrapper(
        model=model,
        melody_augmenter=melody_augmenter,
        lr=learning_rate,
        mask_padding=training_config.get("mask_padding", False),
        mask_padding_dropout=training_config.get("mask_padding_dropout", 0.0),
        use_ema=use_ema_override,
        log_loss_info=training_config.get("log_loss_info", False),
        optimizer_configs=optimizer_configs,
        pre_encoded=training_config.get("pre_encoded", False),
        cfg_dropout_prob=training_config.get("cfg_dropout_prob", 0.1),
        timestep_sampler=training_config.get("timestep_sampler", "uniform"),
        timestep_sampler_options=training_config.get("timestep_sampler_options", {}),
        p_one_shot=training_config.get("p_one_shot", 0.0),
        inpainting_config=training_config.get("inpainting", None),
    )


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    dataset_config_path = Path(args.dataset_config)
    if not dataset_config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_config_path}")

    dataset_config = load_json(str(dataset_config_path))

    base_model, model_config = get_pretrained_model(args.model_name)
    base_model = cast(ConditionedDiffusionModelWrapper, base_model)

    control_model = build_control_wrapper(
        base_wrapper=base_model,
        num_control_layers=args.num_control_layers,
        control_id=args.control_id,
        default_control_scale=args.default_control_scale,
        freeze_base=args.freeze_base,
    )
    control_model = cast(ControlConditionedDiffusionWrapper, control_model)

    sample_rate = int(model_config["sample_rate"])
    model_dtype = next(control_model.parameters()).dtype

    extractor = CQTTopKExtractor(
        CQTTopKConfig(
            sample_rate=sample_rate,
            fmin_hz=args.fmin_hz,
            highpass_cutoff_hz=args.highpass_cutoff_hz,
            n_bins=args.n_bins,
            bins_per_octave=args.bins_per_octave,
            hop_length=args.hop_length,
            top_k=args.top_k,
            backend=args.cqt_backend,
        )
    )

    melody_augmenter = MelodyControlAugmenter(
        base_conditioner=cast(nn.Module, control_model.base_wrapper.conditioner),
        control_id=args.control_id,
        extractor=extractor,
    )
    control_model.base_wrapper.conditioner = melody_augmenter

    initialize_lazy_parameters(
        control_model,
        device=torch.device("cpu"),
        dtype=model_dtype,
        control_channels=args.top_k * 2,
    )
    trainable_names = apply_control_only_freeze_policy(control_model)

    if not trainable_names:
        raise RuntimeError("No trainable parameters left after applying control-only freeze policy.")

    # `create_dataloader_from_config` uses persistent_workers=True, so keep num_workers >= 1.
    effective_num_workers = max(1, int(args.num_workers))
    if args.num_workers < 1:
        print("num_workers < 1 is not supported by stable_audio_tools dataloader; using num_workers=1.")

    train_dl = create_dataloader_from_config(
        dataset_config=dataset_config,
        batch_size=int(args.batch_size),
        sample_size=int(model_config["sample_size"]),
        sample_rate=sample_rate,
        audio_channels=int(model_config.get("audio_channels", 2)),
        num_workers=effective_num_workers,
        shuffle=True,
    )

    training_wrapper = create_training_wrapper(
        model_config=model_config,
        model=control_model,
        melody_augmenter=melody_augmenter,
        learning_rate_override=args.learning_rate,
        use_ema_override=bool(args.use_ema),
    )

    accelerator = args.accelerator
    precision = args.precision
    if not torch.cuda.is_available() and precision != "32-true":
        print(f"CUDA unavailable; precision '{precision}' may be unsupported on CPU. Switching to '32-true'.")
        precision = "32-true"

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=int(args.devices),
        precision=precision,
        max_steps=int(args.max_steps),
        accumulate_grad_batches=int(args.accumulate_grad_batches),
        gradient_clip_val=float(args.gradient_clip_val),
        log_every_n_steps=int(args.log_every_n_steps),
        limit_train_batches=float(args.limit_train_batches),
        default_root_dir=args.default_root_dir,
    )

    print(f"model_name={args.model_name}")
    print(f"dataset_config={dataset_config_path.resolve()}")
    print(f"sample_rate={sample_rate}, sample_size={model_config['sample_size']}, batch_size={args.batch_size}")
    print(f"trainable_modules={len(trainable_names)} tensors")
    print(f"trainable_name_samples={trainable_names[:6]}")

    trainer.fit(training_wrapper, train_dataloaders=train_dl, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
