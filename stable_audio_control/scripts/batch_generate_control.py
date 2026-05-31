"""
Batch generate ControlNet outputs with the same reference audio + prompt,
using different seeds in a single model load (no redundant model reloading).

Usage:
    python3 batch_generate_control.py \
        --ckpt-path outputs/formal10s/checkpoints/interrupted-step-24099.ckpt \
        --reference-audio audio/Then.mp3 \
        --prompt "piano instrumental" \
        --output-dir outputs/formal10s/samples/ \
        --num-samples 10 \
        --model-half \
        --num-control-layers 12 \
        --seconds-total 10 \
        --steps 100
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any, cast

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_audio_control.audio_io import install_torchaudio_load_fallback  # noqa: E402
from stable_audio_control.inference.control_compare import (  # noqa: E402
    audio_sample_size_from_seconds,
    initialize_lazy_control_modules,
    load_control_checkpoint,
    load_reference_audio,
    save_audio_tensor,
)
from stable_audio_control.melody.extractors import build_melody_extractor, melody_control_channels  # noqa: E402
from stable_audio_control.models import ControlConditionedDiffusionWrapper, build_control_wrapper  # noqa: E402
from stable_audio_tools import get_pretrained_model  # noqa: E402
from stable_audio_tools.inference.generation import generate_diffusion_cond  # noqa: E402
from stable_audio_tools.models.diffusion import ConditionedDiffusionModelWrapper  # noqa: E402

install_torchaudio_load_fallback()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch generate ControlNet outputs with different seeds.")
    p.add_argument("--ckpt-path", type=str, required=True)
    p.add_argument("--reference-audio", type=str, required=True)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="outputs/batch_generate")
    p.add_argument("--num-samples", type=int, default=10, help="Number of samples to generate with different seeds.")
    p.add_argument("--seed-start", type=int, default=0, help="First seed value, increments by 1 per sample.")

    p.add_argument("--model-name", type=str, default="stabilityai/stable-audio-open-1.0")
    p.add_argument("--seconds-total", type=float, default=10.0)
    p.add_argument("--seconds-start", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--cfg-scale", type=float, default=7.0)
    p.add_argument("--sampler-type", type=str, default="dpmpp-3m-sde")
    p.add_argument("--sigma-min", type=float, default=0.3)
    p.add_argument("--sigma-max", type=float, default=500.0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--model-half", action="store_true")

    p.add_argument("--num-control-layers", type=int, default=12)
    p.add_argument("--control-id", type=str, default="melody_control")
    p.add_argument("--control-scale", type=float, default=1.0)
    p.add_argument("--prefer-ema", type=bool, default=True, action=argparse.BooleanOptionalAction)

    p.add_argument("--melody-feature", type=str, choices=["cqt", "chromagram"], default="cqt")
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--n-bins", type=int, default=128)
    p.add_argument("--bins-per-octave", type=int, default=12)
    p.add_argument("--fmin-hz", type=float, default=8.175798915643707)
    p.add_argument("--hop-length", type=int, default=512)
    p.add_argument("--highpass-cutoff-hz", type=float, default=261.2)
    p.add_argument("--cqt-backend", type=str, choices=["auto", "nnaudio", "librosa"], default="auto")
    p.add_argument("--melody-embedding-dim", type=int, default=64)
    p.add_argument("--melody-hidden-dim", type=int, default=256)
    p.add_argument("--melody-conv-layers", type=int, default=2)
    p.add_argument("--chroma-bins", type=int, default=12)
    p.add_argument("--chroma-n-fft", type=int, default=2048)
    p.add_argument("--output-name", type=str, default="control_{i:03d}_seed-{seed}.wav",
                   help="Template: {i} = sample index, {seed} = seed value. Example: control_{i:03d}_seed-{seed}.wav")
    return p


def main():
    args = build_parser().parse_args()
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model once ---
    print(f"Loading model: {args.model_name}")
    base_model, model_config = get_pretrained_model(args.model_name)
    base_model = cast(ConditionedDiffusionModelWrapper, base_model)

    sample_rate = int(model_config["sample_rate"])
    sample_size = audio_sample_size_from_seconds(
        seconds_total=float(args.seconds_total),
        sample_rate=sample_rate,
        min_input_length=int(base_model.min_input_length),
    )
    print(f"sample_rate={sample_rate}, sample_size={sample_size}")

    control_channels = melody_control_channels(args.melody_feature, top_k=int(args.top_k), chroma_bins=int(args.chroma_bins))
    uses_discrete_melody = args.melody_feature == "cqt"

    control_model = build_control_wrapper(
        base_wrapper=base_model,
        num_control_layers=int(args.num_control_layers),
        control_id=args.control_id,
        default_control_scale=float(args.control_scale),
        freeze_base=True,
        melody_channels=control_channels,
        melody_num_pitch_bins=int(args.n_bins),
        melody_embedding_dim=int(args.melody_embedding_dim),
        melody_hidden_dim=int(args.melody_hidden_dim),
        melody_conv_layers=int(args.melody_conv_layers),
        use_melody_encoder=uses_discrete_melody,
    )
    control_model = cast(ControlConditionedDiffusionWrapper, control_model)

    initialize_lazy_control_modules(control_model, device=torch.device("cpu"),
                                     dtype=next(control_model.parameters()).dtype, control_channels=control_channels)

    checkpoint_load = load_control_checkpoint(control_model, args.ckpt_path, prefer_ema=bool(args.prefer_ema))
    print(f"Checkpoint loaded: use_ema={checkpoint_load['use_ema']}")

    control_model = control_model.to(device).eval().requires_grad_(False)
    if args.model_half and device.type != "cpu":
        control_model = control_model.to(torch.float16)

    # --- Extract melody once ---
    reference_audio = load_reference_audio(args.reference_audio, target_sample_rate=sample_rate,
                                            target_sample_size=sample_size, device=device)
    extractor = build_melody_extractor(
        feature=args.melody_feature, sample_rate=sample_rate,
        fmin_hz=float(args.fmin_hz), highpass_cutoff_hz=float(args.highpass_cutoff_hz),
        n_bins=int(args.n_bins), bins_per_octave=int(args.bins_per_octave),
        hop_length=int(args.hop_length), top_k=int(args.top_k),
        cqt_backend=args.cqt_backend, chroma_bins=int(args.chroma_bins), chroma_n_fft=int(args.chroma_n_fft),
    )
    melody_control = extractor.extract(reference_audio).to(device=device)

    latent_sample_size = sample_size
    if control_model.pretransform is not None:
        latent_sample_size = sample_size // int(control_model.pretransform.downsampling_ratio)

    control_input = control_model._extract_control_input(
        cond={args.control_id: [melody_control, None]},
        target_len=int(latent_sample_size),
        dtype=next(control_model.model.parameters()).dtype,
        device=device,
    )
    if control_input is None:
        raise RuntimeError("Failed to build control_input.")

    conditioning = control_model.conditioner(
        [{"prompt": args.prompt, "seconds_start": float(args.seconds_start), "seconds_total": float(args.seconds_total)}],
        device,
    )

    # --- Batch generate ---
    extra_sampler_kwargs = {
        "control_input": control_input,
        "control_scale": float(args.control_scale),
    }

    for i in range(args.num_samples):
        seed = int(args.seed_start) + i
        filename = args.output_name.format(i=i, seed=seed)
        output_path = output_dir / filename
        print(f"[{i + 1}/{args.num_samples}] seed={seed} -> {output_path}")

        audio = generate_diffusion_cond(
            control_model,
            steps=int(args.steps),
            cfg_scale=float(args.cfg_scale),
            conditioning_tensors=conditioning,
            batch_size=1,
            sample_size=int(sample_size),
            seed=seed,
            device=device,
            sampler_type=args.sampler_type,
            sigma_min=float(args.sigma_min),
            sigma_max=float(args.sigma_max),
            **extra_sampler_kwargs,
        )
        save_audio_tensor(output_path, audio, sample_rate)

        # Free GPU memory between samples
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Done: {args.num_samples} samples -> {output_dir}")


if __name__ == "__main__":
    main()
