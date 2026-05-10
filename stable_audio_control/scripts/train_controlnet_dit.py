from __future__ import annotations

import argparse
import copy
import importlib
import inspect
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

from stable_audio_control.audio_io import install_torchaudio_load_fallback  # noqa: E402
from stable_audio_control.melody.extractors import (  # noqa: E402
    MelodyExtractor,
    build_melody_extractor,
    melody_control_channels,
)
from stable_audio_control.models import (  # noqa: E402
    ControlConditionedDiffusionWrapper,
    ControlNetContinuousTransformer,
    build_control_wrapper,
)
from stable_audio_tools import get_pretrained_model  # noqa: E402
from stable_audio_tools.data.dataset import create_dataloader_from_config  # noqa: E402
from stable_audio_tools.models.diffusion import ConditionedDiffusionModelWrapper  # noqa: E402
from stable_audio_tools.training.diffusion import DiffusionCondTrainingWrapper  # noqa: E402


install_torchaudio_load_fallback()


def patch_stable_audio_tools_inverse_lr_for_torch() -> None:
    """Patch stable-audio-tools' InverseLR for PyTorch versions without `verbose`."""
    from stable_audio_tools.training import utils as training_utils

    base_scheduler = getattr(torch.optim.lr_scheduler, "LRScheduler", torch.optim.lr_scheduler._LRScheduler)
    if "verbose" in inspect.signature(base_scheduler.__init__).parameters:
        return

    original_inverse_lr = training_utils.InverseLR
    if getattr(original_inverse_lr, "_stable_audio_control_torch_compat", False):
        return

    class CompatibleInverseLR(original_inverse_lr):  # type: ignore[misc, valid-type]
        _stable_audio_control_torch_compat = True

        def __init__(
            self,
            optimizer,
            inv_gamma=1.0,
            power=1.0,
            warmup=0.0,
            final_lr=0.0,
            last_epoch=-1,
            verbose=False,
        ):
            self.inv_gamma = inv_gamma
            self.power = power
            if not 0.0 <= warmup < 1:
                raise ValueError("Invalid value for warmup")
            self.warmup = warmup
            self.final_lr = final_lr
            self.verbose = verbose
            torch.optim.lr_scheduler._LRScheduler.__init__(self, optimizer, last_epoch=last_epoch)

    CompatibleInverseLR.__name__ = original_inverse_lr.__name__
    CompatibleInverseLR.__qualname__ = original_inverse_lr.__qualname__
    CompatibleInverseLR.__doc__ = original_inverse_lr.__doc__
    training_utils.InverseLR = CompatibleInverseLR


patch_stable_audio_tools_inverse_lr_for_torch()


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
        description="Train ControlNet-DiT on StableAudio Open with selectable melody control features."
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
    parser.add_argument("--num-control-layers", type=int, default=12)
    parser.add_argument("--control-id", type=str, default="melody_control")
    parser.add_argument("--default-control-scale", type=float, default=1.0)
    parser.add_argument("--freeze-base", type=_str_to_bool, default=True)
    parser.add_argument("--melody-embedding-dim", type=int, default=64)
    parser.add_argument("--melody-hidden-dim", type=int, default=256)
    parser.add_argument("--melody-conv-layers", type=int, default=2)

    # Melody feature args
    parser.add_argument(
        "--melody-feature",
        type=str,
        choices=["cqt", "chromagram"],
        default="cqt",
        help="Melody control feature extractor. Defaults to the existing top-k CQT path.",
    )

    # CQT args
    parser.add_argument("--top-k", type=int, default=4, help="Top-k bins per stereo channel.")
    parser.add_argument("--n-bins", type=int, default=128)
    parser.add_argument("--bins-per-octave", type=int, default=12)
    parser.add_argument("--fmin-hz", type=float, default=8.175798915643707)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--highpass-cutoff-hz", type=float, default=261.2)
    parser.add_argument("--cqt-backend", type=str, choices=["auto", "nnaudio", "librosa"], default="auto")

    # Chromagram args
    parser.add_argument("--chroma-bins", type=int, default=12, help="Chromagram pitch-class bins.")
    parser.add_argument("--chroma-n-fft", type=int, default=2048, help="Chromagram STFT size.")

    return parser


def initialize_lazy_parameters(
    model: ControlConditionedDiffusionWrapper,
    *,
    device: torch.device,
    dtype: torch.dtype,
    control_channels: int,
) -> None:
    """Materialize/check control conditioning modules before trainer starts."""
    index_cond: Dict[str, Any] = {
        model.control_id: [torch.zeros((1, control_channels, 8), device=device, dtype=torch.long), None],
    }
    _ = model._extract_control_input(  # type: ignore[attr-defined]
        cond=index_cond,
        target_len=8,
        dtype=dtype,
        device=device,
    )

    dense_cond: Dict[str, Any] = {
        model.control_id: [torch.zeros((1, control_channels, 8), device=device, dtype=dtype), None],
    }
    _ = model._extract_control_input(  # type: ignore[attr-defined]
        cond=dense_cond,
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

    target_modules: Dict[str, Optional[nn.Module]] = {
        "control_layers": transformer.control_layers,
        "zero_linears": transformer.zero_linears,
        "melody_encoder": model.melody_encoder,
        "control_projector": model.control_projector,
    }

    trainable_names: List[str] = []
    for prefix, module in target_modules.items():
        if module is None:
            continue
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
        extractor: MelodyExtractor,
        control_channels: int,
        empty_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.base_conditioner = base_conditioner
        self.control_id = control_id
        self.extractor = extractor
        self.control_channels = int(control_channels)
        self.empty_dtype = empty_dtype
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
            conditioning[self.control_id] = [
                torch.zeros((0, self.control_channels, 1), device=device, dtype=self.empty_dtype),
                None,
            ]
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

    def _prepare_batch(self, batch):
        reals = batch[0]
        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        if self.pre_encoded:
            raise RuntimeError(
                "pre_encoded=True is not supported in this script because melody extraction requires waveform audio."
            )

        self.melody_augmenter.set_batch_audio(reals)
        return reals, normalize_metadata_padding_masks(batch[1])

    def training_step(self, batch, batch_idx):
        batch = self._prepare_batch(batch)
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        batch = self._prepare_batch(batch)
        return super().validation_step(batch, batch_idx)


def normalize_metadata_padding_masks(metadata: Any) -> List[Dict[str, Any]]:
    """Wrap bare audio_dir padding-mask tensors in the container expected upstream."""
    normalized: List[Dict[str, Any]] = []
    for item in metadata:
        if not isinstance(item, dict):
            normalized.append(item)
            continue

        padding_mask = item.get("padding_mask")
        if isinstance(padding_mask, torch.Tensor):
            item = dict(item)
            item["padding_mask"] = [padding_mask]
        elif isinstance(padding_mask, tuple):
            item = dict(item)
            item["padding_mask"] = list(padding_mask)
        normalized.append(item)
    return normalized


def _override_optimizer_config_learning_rate(
    optimizer_configs: Dict[str, Any],
    learning_rate: float,
) -> None:
    updated = 0
    for optimizer_config in optimizer_configs.values():
        if not isinstance(optimizer_config, dict):
            continue
        optimizer = optimizer_config.get("optimizer")
        if not isinstance(optimizer, dict):
            continue
        config = optimizer.setdefault("config", {})
        if not isinstance(config, dict):
            raise TypeError("optimizer config must be a mapping when overriding --learning-rate.")
        config["lr"] = float(learning_rate)
        updated += 1

    if updated == 0:
        raise ValueError("Could not find optimizer config entries to override with --learning-rate.")


def resolve_training_optimizer_settings(
    *,
    model_config: Dict[str, Any],
    learning_rate_override: Optional[float],
) -> tuple[Optional[float], Optional[Dict[str, Any]]]:
    training_config = model_config.get("training", {})
    optimizer_configs = training_config.get("optimizer_configs", None)
    optimizer_configs = copy.deepcopy(optimizer_configs) if optimizer_configs is not None else None

    if learning_rate_override is not None:
        learning_rate = float(learning_rate_override)
        if optimizer_configs is not None:
            _override_optimizer_config_learning_rate(optimizer_configs, learning_rate)
            return None, optimizer_configs
        return learning_rate, None

    learning_rate = training_config.get("learning_rate")
    if learning_rate is None and optimizer_configs is None:
        learning_rate = 5e-5
    return learning_rate, optimizer_configs


def create_training_wrapper(
    *,
    model_config: Dict[str, Any],
    model: ControlConditionedDiffusionWrapper,
    melody_augmenter: MelodyControlAugmenter,
    learning_rate_override: Optional[float],
    use_ema_override: bool,
) -> MelodyAwareDiffusionCondTrainingWrapper:
    training_config = model_config.get("training", {})
    learning_rate, optimizer_configs = resolve_training_optimizer_settings(
        model_config=model_config,
        learning_rate_override=learning_rate_override,
    )

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


def _importable_module_name_from_path(module_path: Path) -> Optional[str]:
    repo_root = Path(__file__).resolve().parents[2]
    resolved = module_path.resolve()
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError:
        return None

    if relative.suffix != ".py" or relative.name == "__init__.py":
        return None

    package_parts = relative.with_suffix("").parts
    package_dir = repo_root
    for part in package_parts[:-1]:
        package_dir = package_dir / part
        if not (package_dir / "__init__.py").exists():
            return None

    return ".".join(package_parts)


def make_dataloader_custom_metadata_picklable(dataloader: Any, dataset_config: Dict[str, Any]) -> None:
    """
    Replace stable-audio-tools' temporary `metadata_module` functions with package imports.

    On Windows, DataLoader workers use spawn and must pickle the Dataset.  The upstream
    loader imports custom metadata files as a temporary module named `metadata_module`,
    which cannot be imported again in the worker process.  When the same file is part of
    this repo's package tree, the package-level function is picklable and equivalent.
    """
    dataset = getattr(dataloader, "dataset", None)
    custom_metadata_fns = getattr(dataset, "custom_metadata_fns", None)
    if not isinstance(custom_metadata_fns, dict):
        return

    for audio_dir_config in dataset_config.get("datasets", []):
        audio_dir_path = audio_dir_config.get("path")
        custom_metadata_module = audio_dir_config.get("custom_metadata_module")
        if audio_dir_path is None or custom_metadata_module is None:
            continue

        module_name = _importable_module_name_from_path(Path(custom_metadata_module))
        if module_name is None:
            continue

        metadata_module = importlib.import_module(module_name)
        custom_metadata_fn = metadata_module.get_custom_metadata
        configured_path = Path(audio_dir_path).resolve()

        for existing_path in list(custom_metadata_fns.keys()):
            if Path(existing_path).resolve() == configured_path:
                custom_metadata_fns[existing_path] = custom_metadata_fn


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    dataset_config_path = Path(args.dataset_config)
    if not dataset_config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_config_path}")

    dataset_config = load_json(str(dataset_config_path))

    base_model, model_config = get_pretrained_model(args.model_name)
    base_model = cast(ConditionedDiffusionModelWrapper, base_model)

    control_channels = melody_control_channels(
        args.melody_feature,
        top_k=args.top_k,
        chroma_bins=args.chroma_bins,
    )
    uses_discrete_melody = args.melody_feature == "cqt"

    control_model = build_control_wrapper(
        base_wrapper=base_model,
        num_control_layers=args.num_control_layers,
        control_id=args.control_id,
        default_control_scale=args.default_control_scale,
        freeze_base=args.freeze_base,
        melody_channels=control_channels,
        melody_num_pitch_bins=args.n_bins,
        melody_embedding_dim=args.melody_embedding_dim,
        melody_hidden_dim=args.melody_hidden_dim,
        melody_conv_layers=args.melody_conv_layers,
        use_melody_encoder=uses_discrete_melody,
    )
    control_model = cast(ControlConditionedDiffusionWrapper, control_model)

    sample_rate = int(model_config["sample_rate"])
    model_dtype = next(control_model.parameters()).dtype

    extractor = build_melody_extractor(
        feature=args.melody_feature,
        sample_rate=sample_rate,
        fmin_hz=args.fmin_hz,
        highpass_cutoff_hz=args.highpass_cutoff_hz,
        n_bins=args.n_bins,
        bins_per_octave=args.bins_per_octave,
        hop_length=args.hop_length,
        top_k=args.top_k,
        cqt_backend=args.cqt_backend,
        chroma_bins=args.chroma_bins,
        chroma_n_fft=args.chroma_n_fft,
    )

    melody_augmenter = MelodyControlAugmenter(
        base_conditioner=cast(nn.Module, control_model.base_wrapper.conditioner),
        control_id=args.control_id,
        extractor=extractor,
        control_channels=control_channels,
        empty_dtype=torch.long if uses_discrete_melody else torch.float32,
    )
    control_model.base_wrapper.conditioner = melody_augmenter

    initialize_lazy_parameters(
        control_model,
        device=torch.device("cpu"),
        dtype=model_dtype,
        control_channels=control_channels,
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
    make_dataloader_custom_metadata_picklable(train_dl, dataset_config)

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
    print(f"melody_feature={args.melody_feature}, control_channels={control_channels}")
    print(f"sample_rate={sample_rate}, sample_size={model_config['sample_size']}, batch_size={args.batch_size}")
    print(f"trainable_modules={len(trainable_names)} tensors")
    print(f"trainable_name_samples={trainable_names[:6]}")

    trainer.fit(training_wrapper, train_dataloaders=train_dl, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
