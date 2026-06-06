from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, cast

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


DEFAULT_DEMO_CONTROL_VARIANTS = ["correct", "shuffled", "zero"]
_CONTROL_VARIANT_ALIASES = {
    "correct": "correct",
    "zero": "zero",
    "shuffle": "shuffled",
    "shuffled": "shuffled",
}


@dataclass(frozen=True)
class TensorActivityStats:
    variant: str
    tensor_name: str
    shape: str
    dtype: str
    value_count: int
    unique_value_count: int
    zero_value_ratio: float
    nonzero_value_ratio: float
    mean_abs: float
    max_abs: float
    rms: float
    l2: float
    frame_count: int
    frame_rms_mean: float
    frame_rms_min: float
    frame_rms_max: float
    frame_active_ratio: float


@dataclass(frozen=True)
class PairwiseTensorStats:
    tensor_name: str
    variant_a: str
    variant_b: str
    cosine_similarity: float
    mean_abs_difference: float
    rms_difference: float
    relative_l2_difference: float


def _normalize_control_variant(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in _CONTROL_VARIANT_ALIASES:
        expected = ", ".join(sorted(_CONTROL_VARIANT_ALIASES))
        raise ValueError(f"Unknown control variant: {value}. Expected one of: {expected}.")
    return _CONTROL_VARIANT_ALIASES[normalized]


def parse_demo_control_variants(value: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"[,/]+", value) if part.strip()]
    if not parts:
        raise ValueError("--demo-control-variants must contain at least one variant.")
    variants: list[str] = []
    for part in parts:
        variant = _normalize_control_variant(part)
        if variant not in variants:
            variants.append(variant)
    return variants


def make_control_variant_audio(reference_audio: torch.Tensor, variant: str) -> torch.Tensor:
    variant = _normalize_control_variant(variant)
    if variant == "correct":
        return reference_audio
    if variant == "zero":
        return torch.zeros_like(reference_audio)
    if variant == "shuffled":
        return torch.flip(reference_audio, dims=[-1])
    raise AssertionError(f"Unhandled normalized control variant: {variant}")


def _shape_token(tensor: torch.Tensor) -> str:
    return "x".join(str(dim) for dim in tensor.shape)


def _safe_ratio(numerator: torch.Tensor, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator.to(torch.float32).sum().item()) / float(denominator)


def _round(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return round(float(value), 8)


def _frame_rms(tensor: torch.Tensor, frame_dim: int) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor.detach().to(torch.float32).reshape(1).abs()
    frame_dim = frame_dim if frame_dim >= 0 else tensor.ndim + frame_dim
    if frame_dim < 0 or frame_dim >= tensor.ndim:
        raise ValueError(f"frame_dim={frame_dim} is out of range for shape={tuple(tensor.shape)}")
    data = tensor.detach().to(torch.float32)
    reduce_dims = tuple(dim for dim in range(data.ndim) if dim != frame_dim)
    if not reduce_dims:
        return data.abs()
    return data.square().mean(dim=reduce_dims).sqrt()


def compute_tensor_activity_stats(
    tensor: torch.Tensor,
    *,
    variant: str,
    tensor_name: str,
    frame_dim: int,
    zero_epsilon: float = 1e-12,
) -> TensorActivityStats:
    detached = tensor.detach().cpu()
    data = detached.to(torch.float32)
    flat = data.reshape(-1)
    value_count = int(flat.numel())
    if value_count == 0:
        raise ValueError("Cannot compute activity stats for an empty tensor.")

    zero_mask = flat.abs() <= float(zero_epsilon)
    frame_values = _frame_rms(detached, frame_dim=frame_dim)
    frame_flat = frame_values.reshape(-1)
    frame_active = frame_flat > float(zero_epsilon)

    return TensorActivityStats(
        variant=variant,
        tensor_name=tensor_name,
        shape=_shape_token(tensor),
        dtype=str(tensor.dtype),
        value_count=value_count,
        unique_value_count=int(torch.unique(detached).numel()),
        zero_value_ratio=_round(_safe_ratio(zero_mask, value_count)),
        nonzero_value_ratio=_round(_safe_ratio(~zero_mask, value_count)),
        mean_abs=_round(float(flat.abs().mean().item())),
        max_abs=_round(float(flat.abs().max().item())),
        rms=_round(float(flat.square().mean().sqrt().item())),
        l2=_round(float(flat.square().sum().sqrt().item())),
        frame_count=int(frame_flat.numel()),
        frame_rms_mean=_round(float(frame_flat.mean().item())),
        frame_rms_min=_round(float(frame_flat.min().item())),
        frame_rms_max=_round(float(frame_flat.max().item())),
        frame_active_ratio=_round(_safe_ratio(frame_active, int(frame_flat.numel()))),
    )


def compute_forward_delta_stats(
    *,
    base_output: torch.Tensor,
    controlled_output: torch.Tensor,
    variant: str,
) -> TensorActivityStats:
    if tuple(base_output.shape) != tuple(controlled_output.shape):
        raise ValueError(
            "base_output and controlled_output must have identical shapes, got "
            f"{tuple(base_output.shape)} and {tuple(controlled_output.shape)}"
        )
    delta = controlled_output.detach() - base_output.detach()
    return compute_tensor_activity_stats(
        delta,
        variant=variant,
        tensor_name="forward_delta",
        frame_dim=-1,
    )


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.detach().to(torch.float32).reshape(-1)
    b_flat = b.detach().to(torch.float32).reshape(-1)
    denominator = a_flat.square().sum().sqrt() * b_flat.square().sum().sqrt()
    if not torch.isfinite(denominator) or float(denominator.item()) <= 1e-12:
        return 0.0
    return _round(float((a_flat * b_flat).sum().item() / denominator.item()))


def compute_pairwise_tensor_stats(
    tensors: dict[str, torch.Tensor],
    *,
    tensor_name: str,
) -> list[PairwiseTensorStats]:
    variants = list(tensors.keys())
    rows: list[PairwiseTensorStats] = []
    for i, variant_a in enumerate(variants):
        for variant_b in variants[i + 1 :]:
            a = tensors[variant_a].detach().cpu().to(torch.float32)
            b = tensors[variant_b].detach().cpu().to(torch.float32)
            if tuple(a.shape) != tuple(b.shape):
                raise ValueError(
                    f"Cannot compare {tensor_name} for {variant_a}/{variant_b}: "
                    f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}"
                )
            diff = a - b
            norm_a = float(a.reshape(-1).square().sum().sqrt().item())
            norm_b = float(b.reshape(-1).square().sum().sqrt().item())
            rows.append(
                PairwiseTensorStats(
                    tensor_name=tensor_name,
                    variant_a=variant_a,
                    variant_b=variant_b,
                    cosine_similarity=_cosine_similarity(a, b),
                    mean_abs_difference=_round(float(diff.abs().mean().item())),
                    rms_difference=_round(float(diff.square().mean().sqrt().item())),
                    relative_l2_difference=_round(
                        float(diff.reshape(-1).square().sum().sqrt().item()) / max(norm_a, norm_b, 1e-12)
                    ),
                )
            )
    return rows


def _stats_by_variant(rows: Iterable[TensorActivityStats], tensor_name: str) -> dict[str, TensorActivityStats]:
    return {row.variant: row for row in rows if row.tensor_name == tensor_name}


def _append_forward_delta_report(
    report: list[str],
    rows: list[TensorActivityStats],
    *,
    ratio_threshold: float,
) -> None:
    forward_delta = _stats_by_variant(rows, "forward_delta")
    correct = forward_delta.get("correct")
    shuffled = forward_delta.get("shuffled")
    zero = forward_delta.get("zero")
    if correct is not None and zero is not None:
        ratio = correct.rms / max(zero.rms, 1e-12)
        report.append(f"forward_delta correct/zero rms ratio={ratio:.4f}")
        if ratio >= float(ratio_threshold):
            report.append(
                "denoiser-level activity-shortcut candidate: correct forward_delta RMS is much larger than zero."
            )
    if shuffled is not None and zero is not None:
        ratio = shuffled.rms / max(zero.rms, 1e-12)
        report.append(f"forward_delta shuffled/zero rms ratio={ratio:.4f}")
        if ratio >= float(ratio_threshold):
            report.append(
                "denoiser-level activity-shortcut candidate: shuffled forward_delta RMS is much larger than zero."
            )
    if correct is not None and shuffled is not None:
        ratio = correct.rms / max(shuffled.rms, 1e-12)
        report.append(f"forward_delta correct/shuffled rms ratio={ratio:.4f}")
        if 0.8 <= ratio <= 1.25:
            report.append(
                "denoiser-level content-insensitive candidate: correct and shuffled forward_delta RMS are similar."
            )


def _append_pairwise_report(report: list[str], pairwise_rows: list[PairwiseTensorStats]) -> None:
    for row in pairwise_rows:
        if row.tensor_name != "forward_delta":
            continue
        report.append(
            f"forward_delta {row.variant_a}/{row.variant_b} cosine={row.cosine_similarity:.4f}, "
            f"relative_l2_difference={row.relative_l2_difference:.4f}"
        )
        variants = {row.variant_a, row.variant_b}
        if variants == {"correct", "shuffled"} and row.cosine_similarity >= 0.8 and row.relative_l2_difference <= 0.35:
            report.append(
                "denoiser-level activity-shortcut candidate: correct and shuffled forward deltas are very similar."
            )


def build_activity_report(
    rows: list[TensorActivityStats],
    *,
    pairwise_rows: list[PairwiseTensorStats] | None = None,
    ratio_threshold: float = 5.0,
) -> list[str]:
    report: list[str] = []
    control = _stats_by_variant(rows, "control_input")
    melody = _stats_by_variant(rows, "melody_control")

    correct = control.get("correct")
    shuffled = control.get("shuffled")
    zero = control.get("zero")
    if correct is not None and zero is not None:
        ratio = correct.rms / max(zero.rms, 1e-12)
        report.append(f"correct/zero rms ratio={ratio:.4f}")
        if ratio >= float(ratio_threshold):
            report.append(
                "activity-shortcut candidate: correct control_input RMS is much larger than zero control_input RMS."
            )
    if shuffled is not None and zero is not None:
        ratio = shuffled.rms / max(zero.rms, 1e-12)
        report.append(f"shuffled/zero rms ratio={ratio:.4f}")
        if ratio >= float(ratio_threshold):
            report.append(
                "activity-shortcut candidate: shuffled control_input RMS is much larger than zero control_input RMS."
            )
    if correct is not None and shuffled is not None:
        ratio = correct.rms / max(shuffled.rms, 1e-12)
        report.append(f"correct/shuffled rms ratio={ratio:.4f}")
        if 0.8 <= ratio <= 1.25:
            report.append(
                "content-insensitive candidate: correct and shuffled control_input RMS are similar; check pitch/content metrics."
            )
    zero_melody = melody.get("zero")
    if zero_melody is not None:
        report.append(
            "zero melody_control zero_value_ratio="
            f"{zero_melody.zero_value_ratio:.4f}, unique_value_count={zero_melody.unique_value_count}"
        )
        if zero_melody.zero_value_ratio < 0.95:
            report.append(
                "zero-audio CQT is not padding-zero: silent audio still produced nonzero pitch indices."
            )
    _append_forward_delta_report(report, rows, ratio_threshold=ratio_threshold)
    if pairwise_rows is not None:
        _append_pairwise_report(report, pairwise_rows)
    if not report:
        report.append("No activity report could be built; expected control_input rows were missing.")
    return report


def write_stats_csv(path: str | Path, rows: list[TensorActivityStats]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(TensorActivityStats.__dataclass_fields__.keys())
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_pairwise_csv(path: str | Path, rows: list[PairwiseTensorStats]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(PairwiseTensorStats.__dataclass_fields__.keys())
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_report(path: str | Path, report: list[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report) + "\n", encoding="utf-8")


def _default_report_path(output_csv: str | Path) -> Path:
    csv_path = Path(output_csv)
    return csv_path.with_suffix(".report.txt")


def _default_pairwise_path(output_csv: str | Path) -> Path:
    csv_path = Path(output_csv)
    return csv_path.with_suffix(".pairwise.csv")


def run_diagnostics(args: argparse.Namespace) -> tuple[list[TensorActivityStats], list[PairwiseTensorStats], list[str]]:
    from stable_audio_control.inference.control_compare import (  # noqa: WPS433
        audio_sample_size_from_seconds,
        initialize_lazy_control_modules,
        load_control_checkpoint,
        load_reference_audio,
    )
    from stable_audio_control.melody.extractors import build_melody_extractor, melody_control_channels  # noqa: WPS433
    from stable_audio_control.models import ControlConditionedDiffusionWrapper, build_control_wrapper  # noqa: WPS433
    from stable_audio_tools import get_pretrained_model  # noqa: WPS433
    from stable_audio_tools.models.diffusion import ConditionedDiffusionModelWrapper  # noqa: WPS433

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu"
    )
    print(f"[diagnose] device={device}", flush=True)
    print(f"[diagnose] loading model: {args.model_name}", flush=True)
    base_model, model_config = get_pretrained_model(args.model_name)
    base_model = cast(ConditionedDiffusionModelWrapper, base_model)

    sample_rate = int(model_config["sample_rate"])
    sample_size = audio_sample_size_from_seconds(
        seconds_total=float(args.seconds_total),
        sample_rate=sample_rate,
        min_input_length=int(base_model.min_input_length),
    )
    print(f"[diagnose] sample_rate={sample_rate}, sample_size={sample_size}", flush=True)

    control_channels = melody_control_channels(
        args.melody_feature,
        top_k=int(args.top_k),
        chroma_bins=int(args.chroma_bins),
    )
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
    initialize_lazy_control_modules(
        control_model,
        device=torch.device("cpu"),
        dtype=next(control_model.parameters()).dtype,
        control_channels=control_channels,
    )
    checkpoint_load = load_control_checkpoint(control_model, args.ckpt_path, prefer_ema=bool(args.prefer_ema))
    print(f"[diagnose] checkpoint loaded: use_ema={checkpoint_load['use_ema']}", flush=True)

    control_model = control_model.to(device).eval().requires_grad_(False)
    if args.model_half and device.type != "cpu":
        control_model = control_model.to(torch.float16)

    reference_audio = load_reference_audio(
        args.reference_audio,
        target_sample_rate=sample_rate,
        target_sample_size=sample_size,
        device=device,
    )
    extractor = build_melody_extractor(
        feature=args.melody_feature,
        sample_rate=sample_rate,
        fmin_hz=float(args.fmin_hz),
        highpass_cutoff_hz=float(args.highpass_cutoff_hz),
        n_bins=int(args.n_bins),
        bins_per_octave=int(args.bins_per_octave),
        hop_length=int(args.hop_length),
        top_k=int(args.top_k),
        cqt_backend=args.cqt_backend,
        chroma_bins=int(args.chroma_bins),
        chroma_n_fft=int(args.chroma_n_fft),
    )

    latent_sample_size = sample_size
    if control_model.pretransform is not None:
        latent_sample_size = sample_size // int(control_model.pretransform.downsampling_ratio)
    print(f"[diagnose] latent_sample_size={latent_sample_size}", flush=True)

    rows: list[TensorActivityStats] = []
    pairwise_rows: list[PairwiseTensorStats] = []
    control_inputs: dict[str, torch.Tensor] = {}
    for variant in parse_demo_control_variants(args.demo_control_variants):
        print(f"[diagnose] variant={variant}: extracting melody_control", flush=True)
        variant_audio = make_control_variant_audio(reference_audio, variant)
        melody_control = extractor.extract(variant_audio).to(device=device)
        rows.append(
            compute_tensor_activity_stats(
                melody_control,
                variant=variant,
                tensor_name="melody_control",
                frame_dim=-1,
            )
        )
        print(f"[diagnose] variant={variant}: building control_input", flush=True)
        control_input = control_model._extract_control_input(
            cond={args.control_id: [melody_control, None]},
            target_len=int(latent_sample_size),
            dtype=next(control_model.model.parameters()).dtype,
            device=device,
        )
        if control_input is None:
            raise RuntimeError(f"Failed to build control_input for variant={variant}.")
        rows.append(
            compute_tensor_activity_stats(
                control_input,
                variant=variant,
                tensor_name="control_input",
                frame_dim=1,
            )
        )
        control_inputs[variant] = control_input.detach()

    if bool(args.forward_delta):
        if not control_inputs:
            raise RuntimeError("Cannot run forward_delta diagnostics without at least one control_input.")
        print("[diagnose] running one denoiser forward pass per variant for forward_delta stats", flush=True)
        torch.manual_seed(int(args.diagnostic_seed))
        model_dtype = next(control_model.model.parameters()).dtype
        x = torch.randn((1, control_model.io_channels, int(latent_sample_size)), device=device, dtype=model_dtype)
        t = torch.full((1,), float(args.diagnostic_timestep), device=device, dtype=model_dtype)
        first_control = next(iter(control_inputs.values())).to(device=device, dtype=model_dtype)
        with torch.no_grad():
            base_output = control_model.model(
                x,
                t,
                cfg_scale=1.0,
                cfg_dropout_prob=0.0,
                control_input=first_control,
                control_scale=0.0,
                use_checkpointing=False,
            )

        forward_deltas: dict[str, torch.Tensor] = {}
        for variant, control_input in control_inputs.items():
            print(f"[diagnose] variant={variant}: measuring forward_delta", flush=True)
            with torch.no_grad():
                controlled_output = control_model.model(
                    x,
                    t,
                    cfg_scale=1.0,
                    cfg_dropout_prob=0.0,
                    control_input=control_input.to(device=device, dtype=model_dtype),
                    control_scale=float(args.control_scale),
                    use_checkpointing=False,
                )
            rows.append(
                compute_forward_delta_stats(
                    base_output=base_output,
                    controlled_output=controlled_output,
                    variant=variant,
                )
            )
            forward_deltas[variant] = (controlled_output.detach() - base_output.detach()).cpu()
            del controlled_output
        pairwise_rows.extend(compute_pairwise_tensor_stats(forward_deltas, tensor_name="forward_delta"))
        del base_output, x, t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = build_activity_report(rows, pairwise_rows=pairwise_rows)
    return rows, pairwise_rows, report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose correct/shuffled/zero melody-control activity without running diffusion generation."
    )
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--reference-audio", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--output-report", type=str, default=None)
    parser.add_argument("--output-pairwise-csv", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="stabilityai/stable-audio-open-1.0")
    parser.add_argument("--seconds-total", type=float, default=10.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model-half", action="store_true")
    parser.add_argument("--num-control-layers", type=int, default=12)
    parser.add_argument("--control-id", type=str, default="melody_control")
    parser.add_argument("--control-scale", type=float, default=1.0)
    parser.add_argument("--demo-control-variants", type=str, default=",".join(DEFAULT_DEMO_CONTROL_VARIANTS))
    parser.add_argument("--prefer-ema", type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--forward-delta", type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--diagnostic-seed", type=int, default=0)
    parser.add_argument("--diagnostic-timestep", type=float, default=0.5)
    parser.add_argument("--melody-feature", type=str, choices=["cqt", "chromagram"], default="cqt")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--n-bins", type=int, default=128)
    parser.add_argument("--bins-per-octave", type=int, default=12)
    parser.add_argument("--fmin-hz", type=float, default=8.175798915643707)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--highpass-cutoff-hz", type=float, default=261.2)
    parser.add_argument("--cqt-backend", type=str, choices=["auto", "nnaudio", "librosa"], default="auto")
    parser.add_argument("--melody-embedding-dim", type=int, default=64)
    parser.add_argument("--melody-hidden-dim", type=int, default=256)
    parser.add_argument("--melody-conv-layers", type=int, default=2)
    parser.add_argument("--chroma-bins", type=int, default=12)
    parser.add_argument("--chroma-n-fft", type=int, default=2048)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows, pairwise_rows, report = run_diagnostics(args)
    output_report = Path(args.output_report) if args.output_report else _default_report_path(args.output_csv)
    output_pairwise_csv = (
        Path(args.output_pairwise_csv) if args.output_pairwise_csv else _default_pairwise_path(args.output_csv)
    )
    write_stats_csv(args.output_csv, rows)
    write_pairwise_csv(output_pairwise_csv, pairwise_rows)
    write_report(output_report, report)
    print(f"[diagnose] stats_csv={args.output_csv}", flush=True)
    print(f"[diagnose] pairwise_csv={output_pairwise_csv}", flush=True)
    print(f"[diagnose] report={output_report}", flush=True)
    for line in report:
        print(f"[diagnose] {line}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
