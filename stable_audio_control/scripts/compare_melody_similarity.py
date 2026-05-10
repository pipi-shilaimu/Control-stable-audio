from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow script execution without installing local package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_audio_control.inference.melody_similarity import (  # noqa: E402
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SECONDS_TOTAL,
    compare_audio_melody_similarity,
    write_similarity_metadata,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare how closely generated audio follows a reference melody feature."
    )
    parser.add_argument("--reference-audio", type=str, required=True, help="Reference melody audio.")
    parser.add_argument("--generated-audio", type=str, required=True, help="Generated control audio to score.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path for similarity JSON output.")

    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--seconds-total", type=float, default=DEFAULT_SECONDS_TOTAL)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Audio sample count to compare. Overrides --seconds-total when provided.",
    )

    parser.add_argument(
        "--melody-feature",
        type=str,
        choices=["cqt", "chromagram"],
        default="cqt",
        help="Melody feature used for the similarity score.",
    )

    # CQT args should match the control experiment.
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--n-bins", type=int, default=128)
    parser.add_argument("--bins-per-octave", type=int, default=12)
    parser.add_argument("--fmin-hz", type=float, default=8.175798915643707)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--highpass-cutoff-hz", type=float, default=261.2)
    parser.add_argument("--cqt-backend", type=str, choices=["auto", "nnaudio", "librosa"], default="auto")

    # Chromagram args should match the control experiment.
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


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    metadata = compare_audio_melody_similarity(
        args.reference_audio,
        args.generated_audio,
        **_similarity_kwargs_from_args(args),
    )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_similarity_metadata(output_path, metadata)
        similarity = metadata["similarity"]
        print(f"similarity={similarity['score']:.6f} metric={similarity['metric_name']}")
        print(f"metadata -> {output_path}")
    else:
        print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
