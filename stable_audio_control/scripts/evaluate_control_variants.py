from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_audio_control.inference.melody_similarity import (  # noqa: E402
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SECONDS_TOTAL,
    compare_audio_melody_similarity,
)


def _parse_float_token(value: str) -> float:
    return float(value.replace("p", "."))


def _infer_control_scale_from_name(name: str) -> float | None:
    """Infer explicit ``control_<scale>`` tokens without treating sample IDs as scale."""

    tokens = [token for token in re.split(r"[_-]+", Path(name).stem.lower()) if token]
    variant_tokens = {"correct", "zero", "shuffle", "shuffled"}
    for index, token in enumerate(tokens[:-1]):
        if token != "control":
            continue

        candidate = tokens[index + 1]
        next_token = tokens[index + 2] if index + 2 < len(tokens) else ""
        if next_token == "seed":
            continue

        if re.fullmatch(r"\d+(?:p\d+)?|\d+\.\d+", candidate) is None:
            continue
        if candidate.isdigit() and len(candidate) > 1 and candidate.startswith("0"):
            continue
        if ("p" not in candidate and "." not in candidate) and next_token not in variant_tokens | {"step", ""}:
            continue
        return _parse_float_token(candidate)

    return None


def infer_control_metadata_from_name(path: str | Path) -> dict[str, Any]:
    """Infer diagnostic dimensions from generated demo/batch filenames."""

    name = Path(path).name
    lower = name.lower()

    variant = "unknown"
    for candidate in ("shuffled", "shuffle", "correct", "zero"):
        if re.search(rf"(^|[_-]){candidate}([_.-]|$)", lower):
            variant = "shuffled" if candidate == "shuffle" else candidate
            break

    control_scale = _infer_control_scale_from_name(lower)

    seed = None
    seed_match = re.search(r"(^|[_-])seed[-_]?(\d+)", lower)
    if seed_match:
        seed = int(seed_match.group(2))

    step = None
    step_match = re.search(r"(^|[_-])step[-_]?(\d+)", lower)
    if step_match:
        step = int(step_match.group(2))

    return {
        "variant": variant,
        "control_scale": control_scale,
        "seed": seed,
        "step": step,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate melody-control diagnostic outputs such as correct/shuffled/zero sweeps."
    )
    parser.add_argument("--reference-audio", type=str, required=True)
    parser.add_argument("--generated-dir", type=str, required=True)
    parser.add_argument("--glob", type=str, default="*.wav")
    parser.add_argument("--output-csv", type=str, default=None)

    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--seconds-total", type=float, default=DEFAULT_SECONDS_TOTAL)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--melody-feature", type=str, choices=["cqt", "chromagram"], default="cqt")

    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--n-bins", type=int, default=128)
    parser.add_argument("--bins-per-octave", type=int, default=12)
    parser.add_argument("--fmin-hz", type=float, default=8.175798915643707)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--highpass-cutoff-hz", type=float, default=261.2)
    parser.add_argument("--cqt-backend", type=str, choices=["auto", "nnaudio", "librosa"], default="auto")
    parser.add_argument("--chroma-bins", type=int, default=12)
    parser.add_argument("--chroma-n-fft", type=int, default=2048)
    return parser


def _similarity_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "feature": args.melody_feature,
        "sample_rate": int(args.sample_rate),
        "sample_size": args.sample_size,
        "seconds_total": float(args.seconds_total),
        "top_k": int(args.top_k),
        "n_bins": int(args.n_bins),
        "bins_per_octave": int(args.bins_per_octave),
        "fmin_hz": float(args.fmin_hz),
        "hop_length": int(args.hop_length),
        "highpass_cutoff_hz": float(args.highpass_cutoff_hz),
        "cqt_backend": args.cqt_backend,
        "chroma_bins": int(args.chroma_bins),
        "chroma_n_fft": int(args.chroma_n_fft),
    }


def build_report_row(
    *,
    generated_audio: str | Path,
    metadata: dict[str, Any],
    similarity_metadata: dict[str, Any],
) -> dict[str, Any]:
    similarity = similarity_metadata["similarity"]
    metric = similarity
    if "additional_metrics" in similarity and "cqt_top1_accuracy" in similarity["additional_metrics"]:
        metric = similarity["additional_metrics"]["cqt_top1_accuracy"]

    return {
        "generated_audio": str(Path(generated_audio)),
        "variant": metadata.get("variant"),
        "control_scale": metadata.get("control_scale"),
        "seed": metadata.get("seed"),
        "step": metadata.get("step"),
        "metric_name": metric["metric_name"],
        "score": metric["score"],
        "matched_tokens": metric.get("matched_tokens"),
        "total_tokens": metric.get("total_tokens"),
        "compared_frames": metric.get("compared_frames"),
    }


def evaluate_generated_files(args: argparse.Namespace) -> list[dict[str, Any]]:
    generated_dir = Path(args.generated_dir)
    files = sorted(path for path in generated_dir.glob(args.glob) if path.is_file())
    if not files:
        raise FileNotFoundError(f"No generated audio files matched {generated_dir / args.glob}")

    rows: list[dict[str, Any]] = []
    kwargs = _similarity_kwargs_from_args(args)
    for path in files:
        metadata = infer_control_metadata_from_name(path)
        similarity = compare_audio_melody_similarity(
            args.reference_audio,
            path,
            **kwargs,
        )
        rows.append(
            build_report_row(
                generated_audio=path,
                metadata=metadata,
                similarity_metadata=similarity,
            )
        )
    return rows


def write_report_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty report.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows = evaluate_generated_files(args)

    output_csv = args.output_csv
    if output_csv is None:
        output_csv = str(Path(args.generated_dir) / "melody_control_report.csv")
    write_report_csv(output_csv, rows)

    print(f"wrote {len(rows)} rows -> {output_csv}")
    for row in rows[:10]:
        print(
            f"{row['variant']} control={row['control_scale']} seed={row['seed']} "
            f"score={float(row['score']):.6f} file={row['generated_audio']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
