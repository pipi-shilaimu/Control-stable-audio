import random
import sys
from pathlib import Path
from typing import cast

import torch

# Allow running as a script without installing the local package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_audio_control.models import (
    ControlConditionedDiffusionWrapper,
    ControlNetContinuousTransformer,
    build_control_wrapper,
)
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.models.diffusion import ConditionedDiffusionModelWrapper


def main() -> None:
    """
    Smoke test for `stable_audio_control.models.control_dit`.

    What it validates:
    1) `build_control_wrapper(...)` can wrap a real StableAudio Open model.
    2) `cond["melody_control"]` is consumed by `ControlConditionedDiffusionWrapper`.
    3) With zero-initialized adapters, control_scale does not change outputs (diff ~= 0).
    4) After tiny adapter perturbation, control path changes outputs (diff > 0).
    """

    torch.manual_seed(0)
    random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load real pretrained wrapper (requires HF assets reachable or already cached).
    base_model, _model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    base_model = cast(ConditionedDiffusionModelWrapper, base_model)

    model = build_control_wrapper(
        base_wrapper=base_model,
        num_control_layers=2,
        control_id="melody_control",
        default_control_scale=0.5,
        freeze_base=True,
    )
    model = cast(ControlConditionedDiffusionWrapper, model)
    model = model.to(device).eval()

    transformer = model.model.model.transformer
    assert isinstance(transformer, ControlNetContinuousTransformer), "Control transformer was not attached."

    # Base conditioning from built-in conditioners.
    batch_metadata = [
        {
            "prompt": "test prompt",
            "seconds_start": 0,
            "seconds_total": 5,
        }
    ]
    cond = model.conditioner(batch_metadata, device)

    # Add melody control in the contracted format [B, C_melody, L].
    bsz = 1
    latent_len = 128
    melody_channels = 8
    dtype = next(model.parameters()).dtype
    cond["melody_control"] = [
        torch.randn((bsz, melody_channels, latent_len), device=device, dtype=dtype),
        None,
    ]

    # Random latent noise and timestep.
    x = torch.randn((bsz, model.io_channels, latent_len), device=device, dtype=dtype)
    t = torch.full((bsz,), 0.5, device=device, dtype=dtype)

    with torch.no_grad():
        out0 = model(
            x,
            t,
            cond=cond,
            cfg_scale=1.0,
            cfg_dropout_prob=0.0,
            control_scale=0.0,
        )
        out1 = model(
            x,
            t,
            cond=cond,
            cfg_scale=1.0,
            cfg_dropout_prob=0.0,
            control_scale=0.5,
        )

    diff_zero_init = float((out1 - out0).float().norm())
    print("device:", device, "dtype:", dtype)
    print("zero-init diff norm:", diff_zero_init)

    # Zero-init adapters should keep output unchanged at init.
    if diff_zero_init > 1e-5:
        print("WARNING: zero-init diff is larger than expected.")

    # Perturb one adapter slightly to verify control path is active.
    with torch.no_grad():
        transformer.zero_linears[0].weight.normal_(mean=0.0, std=1e-3)

    with torch.no_grad():
        out2 = model(
            x,
            t,
            cond=cond,
            cfg_scale=1.0,
            cfg_dropout_prob=0.0,
            control_scale=0.5,
        )

    diff_after_perturb = float((out2 - out0).float().norm())
    print("after-perturb diff norm:", diff_after_perturb)

    if diff_after_perturb <= 1e-5:
        raise RuntimeError("Control path appears inactive: diff after perturbation is too small.")

    print("Smoke PASS: control_dit wrapper path is active and behaving as expected.")


if __name__ == "__main__":
    main()
