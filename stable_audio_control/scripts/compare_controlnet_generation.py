from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any, cast

import torch

# Allow script execution without installing local package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_audio_control.audio_io import install_torchaudio_load_fallback  # noqa: E402
from stable_audio_control.inference.control_compare import (  # noqa: E402
    audio_sample_size_from_seconds,
    build_compare_metadata,
    initialize_lazy_control_modules,
    load_control_checkpoint,
    load_reference_audio,
    save_audio_tensor,
    write_compare_metadata,
)
from stable_audio_control.melody.cqt_topk import CQTTopKConfig, CQTTopKExtractor  # noqa: E402
from stable_audio_control.models import ControlConditionedDiffusionWrapper, build_control_wrapper  # noqa: E402
from stable_audio_tools import get_pretrained_model  # noqa: E402
from stable_audio_tools.inference.generation import generate_diffusion_cond  # noqa: E402
from stable_audio_tools.models.diffusion import ConditionedDiffusionModelWrapper  # noqa: E402


install_torchaudio_load_fallback()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a minimal base-vs-ControlNet comparison pair from one prompt and one reference melody."
    )
    parser.add_argument("--ckpt-path", type=str, required=True, help="Lightning checkpoint from train_controlnet_dit.py.")
    parser.add_argument("--reference-audio", type=str, required=True, help="Audio file used to extract melody_control.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for both generations.")
    parser.add_argument("--negative-prompt", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/controlnet_generation_compare")
    parser.add_argument("--base-output-name", type=str, default="base.wav")
    parser.add_argument("--control-output-name", type=str, default="control.wav")
    parser.add_argument("--metadata-name", type=str, default="compare.json")

    parser.add_argument("--model-name", type=str, default="stabilityai/stable-audio-open-1.0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds-start", type=float, default=0.0)
    parser.add_argument("--seconds-total", type=float, default=5.0)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Audio sample count. Defaults to seconds_total * sample_rate rounded to min_input_length.",
    )
    parser.add_argument("--steps", type=int, default=8, help="Low default keeps the comparison smoke test quick.")
    parser.add_argument("--cfg-scale", type=float, default=7.0)
    parser.add_argument("--sampler-type", type=str, default="dpmpp-3m-sde")
    parser.add_argument("--sigma-min", type=float, default=0.3)
    parser.add_argument("--sigma-max", type=float, default=500.0)
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cuda', or 'cpu'.")
    parser.add_argument("--model-half", action="store_true", help="Move models to float16 after loading.")

    # ControlNet args must match the training run.
    parser.add_argument("--num-control-layers", type=int, default=2)
    parser.add_argument("--control-id", type=str, default="melody_control")
    parser.add_argument("--control-scale", type=float, default=1.0)
    parser.add_argument("--prefer-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--melody-embedding-dim", type=int, default=64)
    parser.add_argument("--melody-hidden-dim", type=int, default=256)
    parser.add_argument("--melody-conv-layers", type=int, default=2)

    # CQT args must match the training run.
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--n-bins", type=int, default=128)
    parser.add_argument("--bins-per-octave", type=int, default=12)
    parser.add_argument("--fmin-hz", type=float, default=8.175798915643707)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--highpass-cutoff-hz", type=float, default=261.2)
    parser.add_argument("--cqt-backend", type=str, choices=["auto", "nnaudio", "librosa"], default="auto")

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.sampler_type == "dpmpp-3m-sde" and int(args.steps) < 2:
        raise ValueError("dpmpp-3m-sde requires --steps >= 2; use a different sampler for single-step smoke tests.")


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _conditioning(prompt: str, seconds_start: float, seconds_total: float) -> list[dict[str, Any]]:
    return [
        {
            "prompt": prompt,
            "seconds_start": float(seconds_start),
            "seconds_total": float(seconds_total),
        }
    ]


def _negative_conditioning(args: argparse.Namespace) -> list[dict[str, Any]] | None:
    if not args.negative_prompt:
        return None
    return _conditioning(args.negative_prompt, args.seconds_start, args.seconds_total)


def _prepare_model(model: torch.nn.Module, *, device: torch.device, model_half: bool) -> torch.nn.Module:
    model = model.to(device).eval().requires_grad_(False)
    if model_half:
        if device.type == "cpu":
            print("model_half requested on CPU; keeping float32.")
        else:
            model = model.to(torch.float16)
    return model


@torch.no_grad()
def _generate(
    model: torch.nn.Module,
    args: argparse.Namespace,
    *,
    sample_size: int,
    device: torch.device,
    conditioning_tensors: dict[str, Any] | None = None,
    extra_sampler_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    kwargs = dict(extra_sampler_kwargs or {})
    return generate_diffusion_cond(
        model,
        steps=int(args.steps),
        cfg_scale=float(args.cfg_scale),
        conditioning=None
        if conditioning_tensors is not None
        else _conditioning(args.prompt, args.seconds_start, args.seconds_total),
        conditioning_tensors=conditioning_tensors,
        negative_conditioning=_negative_conditioning(args),
        batch_size=1,
        sample_size=int(sample_size),
        seed=int(args.seed),
        device=device,
        sampler_type=args.sampler_type,
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
        **kwargs,
    )


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)
    device = _resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_output_path = output_dir / args.base_output_name
    control_output_path = output_dir / args.control_output_name
    metadata_path = output_dir / args.metadata_name

    print(f"device={device}")
    print(f"loading base model: {args.model_name}")
    base_model, model_config = get_pretrained_model(args.model_name)
    base_model = cast(ConditionedDiffusionModelWrapper, base_model)

    sample_rate = int(model_config["sample_rate"])
    sample_size = (
        int(args.sample_size)
        if args.sample_size is not None
        else audio_sample_size_from_seconds(
            seconds_total=float(args.seconds_total),
            sample_rate=sample_rate,
            min_input_length=int(base_model.min_input_length),
        )
    )
    print(f"sample_rate={sample_rate}, sample_size={sample_size}, seconds_total={args.seconds_total}")

    base_model = cast(ConditionedDiffusionModelWrapper, _prepare_model(base_model, device=device, model_half=args.model_half))
    print(f"generating base output -> {base_output_path}")
    base_audio = _generate(base_model, args, sample_size=sample_size, device=device)
    save_audio_tensor(base_output_path, base_audio, sample_rate)
    del base_audio
    del base_model
    _empty_cuda_cache()

    print(f"loading control model: {args.model_name}")
    control_base_model, _control_model_config = get_pretrained_model(args.model_name)
    control_base_model = cast(ConditionedDiffusionModelWrapper, control_base_model)
    control_model = build_control_wrapper(
        base_wrapper=control_base_model,
        num_control_layers=int(args.num_control_layers),
        control_id=args.control_id,
        default_control_scale=float(args.control_scale),
        freeze_base=True,
        melody_channels=int(args.top_k) * 2,
        melody_num_pitch_bins=int(args.n_bins),
        melody_embedding_dim=int(args.melody_embedding_dim),
        melody_hidden_dim=int(args.melody_hidden_dim),
        melody_conv_layers=int(args.melody_conv_layers),
    )
    control_model = cast(ControlConditionedDiffusionWrapper, control_model)

    initialize_lazy_control_modules(
        control_model,
        device=torch.device("cpu"),
        dtype=next(control_model.parameters()).dtype,
        control_channels=int(args.top_k) * 2,
    )
    checkpoint_load = load_control_checkpoint(control_model, args.ckpt_path, prefer_ema=bool(args.prefer_ema))
    print(f"checkpoint loaded: use_ema={checkpoint_load['use_ema']}")

    control_model = cast(
        ControlConditionedDiffusionWrapper,
        _prepare_model(control_model, device=device, model_half=args.model_half),
    )

    reference_audio = load_reference_audio(
        args.reference_audio,
        target_sample_rate=sample_rate,
        target_sample_size=sample_size,
        device=device,
    )
    extractor = CQTTopKExtractor(
        CQTTopKConfig(
            sample_rate=sample_rate,
            fmin_hz=float(args.fmin_hz),
            highpass_cutoff_hz=float(args.highpass_cutoff_hz),
            n_bins=int(args.n_bins),
            bins_per_octave=int(args.bins_per_octave),
            hop_length=int(args.hop_length),
            top_k=int(args.top_k),
            backend=args.cqt_backend,
        )
    )
    melody_control = extractor.extract(reference_audio).to(device=device)

    latent_sample_size = sample_size
    if control_model.pretransform is not None:
        latent_sample_size = sample_size // int(control_model.pretransform.downsampling_ratio)

    control_input = control_model._extract_control_input(  # type: ignore[attr-defined]
        cond={args.control_id: [melody_control, None]},
        target_len=int(latent_sample_size),
        dtype=next(control_model.model.parameters()).dtype,
        device=device,
    )
    if control_input is None:
        raise RuntimeError("Failed to build control_input from reference audio.")

    conditioning_tensors = control_model.conditioner(
        _conditioning(args.prompt, args.seconds_start, args.seconds_total),
        device,
    )

    print(f"melody_control_shape={tuple(melody_control.shape)}, control_input_shape={tuple(control_input.shape)}")
    print(f"generating control output -> {control_output_path}")
    control_audio = _generate(
        control_model,
        args,
        sample_size=sample_size,
        device=device,
        conditioning_tensors=conditioning_tensors,
        extra_sampler_kwargs={
            "control_input": control_input,
            "control_scale": float(args.control_scale),
        },
    )
    save_audio_tensor(control_output_path, control_audio, sample_rate)

    control_config = {
        "num_control_layers": int(args.num_control_layers),
        "control_id": args.control_id,
        "top_k": int(args.top_k),
        "n_bins": int(args.n_bins),
        "bins_per_octave": int(args.bins_per_octave),
        "fmin_hz": float(args.fmin_hz),
        "hop_length": int(args.hop_length),
        "highpass_cutoff_hz": float(args.highpass_cutoff_hz),
        "cqt_backend": args.cqt_backend,
        "melody_embedding_dim": int(args.melody_embedding_dim),
        "melody_hidden_dim": int(args.melody_hidden_dim),
        "melody_conv_layers": int(args.melody_conv_layers),
    }
    metadata = build_compare_metadata(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        reference_audio_path=args.reference_audio,
        ckpt_path=args.ckpt_path,
        base_output_path=base_output_path,
        control_output_path=control_output_path,
        seed=int(args.seed),
        model_name=args.model_name,
        sample_rate=sample_rate,
        sample_size=sample_size,
        seconds_start=float(args.seconds_start),
        seconds_total=float(args.seconds_total),
        steps=int(args.steps),
        cfg_scale=float(args.cfg_scale),
        sampler_type=args.sampler_type,
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
        control_scale=float(args.control_scale),
        use_ema=bool(checkpoint_load["use_ema"]),
        control_config=control_config,
    )
    metadata["checkpoint_load"] = checkpoint_load
    metadata["reference"] = {
        "melody_control_shape": list(melody_control.shape),
        "control_input_shape": list(control_input.shape),
    }
    write_compare_metadata(metadata_path, metadata)
    print(f"metadata -> {metadata_path}")


if __name__ == "__main__":
    main()
