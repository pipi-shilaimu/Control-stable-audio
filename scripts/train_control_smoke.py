import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, cast

import torch
from torch import nn
from torch.nn.parameter import UninitializedParameter

# Allow script execution without installing local package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_audio_control.models import (  # noqa: E402
    ControlConditionedDiffusionWrapper,
    ControlNetContinuousTransformer,
    build_control_wrapper,
)
from stable_audio_tools import get_pretrained_model  # noqa: E402
from stable_audio_tools.models.diffusion import ConditionedDiffusionModelWrapper  # noqa: E402
from stable_audio_tools.training.diffusion import DiffusionCondTrainingWrapper  # noqa: E402


class MelodyControlAugmenter(nn.Module):
    """Inject synthetic `melody_control` into conditioner output for smoke training."""

    def __init__(
        self,
        base_conditioner: nn.Module,
        control_id: str,
        melody_channels: int,
        control_length: int,
        fallback_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.base_conditioner = base_conditioner
        self.control_id = control_id
        self.melody_channels = melody_channels
        self.control_length = control_length
        self.fallback_dtype = fallback_dtype

    def forward(self, metadata: List[Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
        conditioning = self.base_conditioner(metadata, device)
        batch_size = len(metadata)

        melody_control = torch.randn(
            (batch_size, self.melody_channels, self.control_length),
            device=device,
            dtype=self.fallback_dtype,
        )
        conditioning[self.control_id] = [melody_control, None]
        return conditioning


class DummyTrainer:
    """Minimal trainer stub for LightningModule access in `training_step`."""

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizers = [optimizer]


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(parameters: Iterable[torch.nn.Parameter]) -> int:
    return sum(p.numel() for p in parameters)


def build_minimal_batch(
    *,
    batch_size: int,
    audio_length: int,
    sample_rate: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    reals = torch.randn((batch_size, 2, audio_length), dtype=dtype, device=device)
    seconds_total = float(audio_length) / float(sample_rate)

    metadata: List[Dict[str, Any]] = []
    for _ in range(batch_size):
        metadata.append(
            {
                "prompt": "control smoke prompt",
                "seconds_start": 0.0,
                "seconds_total": seconds_total,
                "padding_mask": [torch.ones(audio_length, dtype=torch.bool)],
            }
        )

    return reals, metadata


def apply_freeze_policy(model: ControlConditionedDiffusionWrapper) -> List[str]:
    """Freeze everything except control branch modules and control projector."""
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


def pick_probe_parameter(
    model: nn.Module,
    *,
    require_grad: bool,
    max_numel: int = 262_144,
) -> Tuple[str, torch.nn.Parameter]:
    for name, param in model.named_parameters():
        if param.requires_grad != require_grad:
            continue
        if param.numel() <= max_numel:
            return name, param
    for name, param in model.named_parameters():
        if param.requires_grad == require_grad:
            return name, param
    raise RuntimeError(f"No parameter found with requires_grad={require_grad}.")


def pick_named_probe_parameter(
    model: nn.Module,
    *,
    include: str,
    require_grad: bool,
) -> Tuple[str, torch.nn.Parameter]:
    for name, param in model.named_parameters():
        if include in name and param.requires_grad == require_grad:
            return name, param
    raise RuntimeError(f"No parameter found with include='{include}' and requires_grad={require_grad}.")


def initialize_lazy_parameters(
    model: ControlConditionedDiffusionWrapper,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Materialize lazy parameters (e.g. control projector) before optimizer creation."""
    dummy_cond: Dict[str, Any] = {
        "melody_control": [torch.zeros((1, 8, 8), device=device, dtype=dtype), None],
    }
    _ = model._extract_control_input(  # type: ignore[attr-defined]
        cond=dummy_cond,
        target_len=8,
        dtype=dtype,
        device=device,
    )


def main() -> None:
    set_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "stabilityai/stable-audio-open-1.0"
    learning_rate = 1e-4
    batch_size = 1
    audio_length = 65_536 if device.type == "cuda" else 32_768
    melody_channels = 8
    control_length = 32

    base_model, model_config = get_pretrained_model(model_name)
    base_model = cast(ConditionedDiffusionModelWrapper, base_model)

    control_model = build_control_wrapper(
        base_wrapper=base_model,
        num_control_layers=2,
        control_id="melody_control",
        default_control_scale=1.0,
        freeze_base=True,
    )
    control_model = cast(ControlConditionedDiffusionWrapper, control_model)

    model_dtype = next(control_model.parameters()).dtype
    control_model.base_wrapper.conditioner = MelodyControlAugmenter(
        base_conditioner=control_model.base_wrapper.conditioner,
        control_id="melody_control",
        melody_channels=melody_channels,
        control_length=control_length,
        fallback_dtype=model_dtype,
    )

    control_model = control_model.to(device=device, dtype=model_dtype).train()
    initialize_lazy_parameters(control_model, device=device, dtype=model_dtype)

    trainable_names = apply_freeze_policy(control_model)
    trainable_params = [param for param in control_model.parameters() if param.requires_grad]

    if not trainable_params:
        raise RuntimeError("No trainable parameters left after applying freeze policy.")

    training_wrapper = DiffusionCondTrainingWrapper(
        model=control_model,
        lr=learning_rate,
        use_ema=False,
        mask_padding=False,
        cfg_dropout_prob=0.0,
        pre_encoded=False,
    ).to(device)
    training_wrapper.train()
    training_wrapper.log_dict = lambda *args, **kwargs: None  # type: ignore[method-assign]

    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
    training_wrapper._trainer = DummyTrainer(optimizer)  # type: ignore[attr-defined]

    reals, metadata = build_minimal_batch(
        batch_size=batch_size,
        audio_length=audio_length,
        sample_rate=int(model_config["sample_rate"]),
        dtype=model_dtype,
        device=device,
    )

    try:
        trainable_probe_name, trainable_probe_param = pick_named_probe_parameter(
            control_model, include="zero_linears", require_grad=True
        )
    except RuntimeError:
        trainable_probe_name, trainable_probe_param = pick_probe_parameter(control_model, require_grad=True)
    frozen_probe_name, frozen_probe_param = pick_probe_parameter(control_model, require_grad=False)
    trainable_probe_before = trainable_probe_param.detach().clone()
    frozen_probe_before = frozen_probe_param.detach().clone()

    optimizer.zero_grad(set_to_none=True)

    try:
        loss = training_wrapper.training_step((reals, metadata), batch_idx=0)
    except Exception as exc:  # noqa: BLE001
        print("training_step failed:", type(exc).__name__, str(exc))
        raise

    if not torch.isfinite(loss.detach()):
        raise RuntimeError(f"Loss is not finite before backward: {float(loss.detach().cpu())}")

    loss.backward()
    optimizer.step()

    frozen_with_grad = [
        name for name, param in control_model.named_parameters() if (not param.requires_grad) and (param.grad is not None)
    ]
    trainable_with_grad = [
        name for name, param in control_model.named_parameters() if param.requires_grad and (param.grad is not None)
    ]

    trainable_delta = float((trainable_probe_param.detach() - trainable_probe_before).abs().sum().item())
    frozen_delta = float((frozen_probe_param.detach() - frozen_probe_before).abs().sum().item())

    print("model:", model_name)
    print("device:", device.type, "dtype:", str(model_dtype))
    print("batch:", tuple(reals.shape), "padding_mask:", tuple(metadata[0]["padding_mask"][0].shape))
    print("trainable_param_count:", count_parameters(trainable_params))
    print("trainable_tensor_count:", len(trainable_params))
    print("trainable_name_samples:", trainable_names[:6])
    print("loss:", float(loss.detach().cpu()))
    print("loss_is_finite:", bool(torch.isfinite(loss.detach()).item()))
    print("trainable_with_grad_count:", len(trainable_with_grad))
    print("frozen_with_grad_count:", len(frozen_with_grad))
    print("trainable_probe:", trainable_probe_name, "delta:", trainable_delta)
    print("frozen_probe:", frozen_probe_name, "delta:", frozen_delta)

    if frozen_with_grad:
        raise RuntimeError(f"Freeze policy violated: frozen params with gradients: {frozen_with_grad[:5]}")
    if trainable_delta <= 0.0:
        raise RuntimeError("Optimizer step did not update the selected trainable probe parameter.")
    if frozen_delta != 0.0:
        raise RuntimeError("Frozen probe parameter changed after optimizer.step().")

    print("Smoke PASS: one training step completed with finite loss and expected freeze behavior.")


if __name__ == "__main__":
    main()
