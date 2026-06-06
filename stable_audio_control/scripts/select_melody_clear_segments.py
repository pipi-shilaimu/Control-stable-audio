from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_audio_control.audio_io import install_torchaudio_load_fallback  # noqa: E402
from stable_audio_control.melody.cqt_topk import CQTTopKConfig, CQTTopKExtractor  # noqa: E402


DEFAULT_EXTENSIONS = ".wav,.mp3,.flac,.ogg,.m4a"


@dataclass(frozen=True)
class MelodyClearFilterConfig:
    sample_rate: int = 44_100
    seconds_total: float = 10.0
    n_bins: int = 128
    bins_per_octave: int = 12
    fmin_hz: float = 8.175_798_915_643_707
    hop_length: int = 512
    top_k: int = 4
    highpass_cutoff_hz: float = 261.2
    cqt_backend: str = "auto"
    active_energy_ratio: float = 0.05
    low_pitch_bin: int = 48
    jump_semitones: int = 7
    min_score: float = 0.55
    maybe_score: float = 0.42
    min_voiced_ratio: float = 0.55
    max_silence_ratio: float = 0.15
    min_top1_dominance: float = 0.32
    min_unique_pitches: int = 6
    max_unique_pitches: int = 45
    min_pitch_motion_rate: float = 0.03
    max_pitch_jump_rate: float = 0.35
    max_low_pitch_ratio: float = 0.45


@dataclass(frozen=True)
class MelodyClearMetrics:
    decision: str
    melody_clear_score: float
    silence_ratio: float
    voiced_ratio: float
    top1_dominance: float
    unique_pitch_count: int
    pitch_motion_rate: float
    pitch_jump_rate: float
    low_pitch_ratio: float
    pitch_range_semitones: int


@dataclass(frozen=True)
class CandidateRow:
    rank: int
    audio_path: str
    filename: str
    duration_sec: float
    sample_rate: int
    decision: str
    melody_clear_score: float
    silence_ratio: float
    voiced_ratio: float
    top1_dominance: float
    unique_pitch_count: int
    pitch_motion_rate: float
    pitch_jump_rate: float
    low_pitch_ratio: float
    pitch_range_semitones: int
    error: str = ""


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_mean(tensor: torch.Tensor, default: float = 0.0) -> float:
    if tensor.numel() == 0:
        return float(default)
    return float(tensor.to(torch.float32).mean().item())


def _linear_score(value: float, low: float, high: float) -> float:
    if math.isclose(high, low):
        return 1.0 if value >= high else 0.0
    return _clamp01((float(value) - low) / (high - low))


def _bounded_count_score(count: int, minimum: int, maximum: int) -> float:
    if count < minimum:
        return _clamp01(count / max(1, minimum))
    if count > maximum:
        return _clamp01(maximum / max(1, count))
    return 1.0


def _format_score_token(value: float) -> str:
    return f"{float(value):.3f}".replace("-", "neg").replace(".", "p")


def compute_melody_clear_metrics(
    magnitude: torch.Tensor,
    *,
    config: MelodyClearFilterConfig,
) -> MelodyClearMetrics:
    """Compute melody-clear heuristics from CQT magnitude [B, 2, bins, frames]."""

    if magnitude.ndim != 4:
        raise ValueError(f"Expected CQT magnitude [B,2,bins,frames], got shape={tuple(magnitude.shape)}")
    if magnitude.shape[0] != 1:
        raise ValueError("select_melody_clear_segments evaluates one file at a time.")
    if magnitude.shape[1] != 2:
        raise ValueError(f"Expected stereo CQT magnitude with 2 channels, got {magnitude.shape[1]}")
    if magnitude.shape[2] < 1 or magnitude.shape[3] < 1:
        raise ValueError("CQT magnitude must contain at least one bin and one frame.")

    combined = magnitude[0].to(torch.float32).mean(dim=0)  # [bins, frames]
    frame_energy = combined.sum(dim=0)
    max_energy = float(frame_energy.max().item()) if frame_energy.numel() else 0.0
    if max_energy <= 0.0:
        return MelodyClearMetrics(
            decision="reject",
            melody_clear_score=0.0,
            silence_ratio=1.0,
            voiced_ratio=0.0,
            top1_dominance=0.0,
            unique_pitch_count=0,
            pitch_motion_rate=0.0,
            pitch_jump_rate=0.0,
            low_pitch_ratio=0.0,
            pitch_range_semitones=0,
        )

    active = frame_energy >= (max_energy * float(config.active_energy_ratio))
    voiced_ratio = _safe_mean(active)
    silence_ratio = 1.0 - voiced_ratio

    k = min(int(config.top_k), int(combined.shape[0]))
    top_values, top_indices = torch.topk(combined, k=k, dim=0, largest=True, sorted=True)
    top1_values = top_values[0]
    topk_sum = top_values.sum(dim=0).clamp_min(1e-8)
    top1_dominance = _safe_mean((top1_values / topk_sum)[active])

    active_top1 = top_indices[0, active].to(torch.long) + 1
    if active_top1.numel() == 0:
        unique_pitch_count = 0
        pitch_motion_rate = 0.0
        pitch_jump_rate = 0.0
        low_pitch_ratio = 0.0
        pitch_range_semitones = 0
    else:
        unique_pitch_count = int(torch.unique(active_top1).numel())
        low_pitch_ratio = _safe_mean(active_top1 <= int(config.low_pitch_bin))
        pitch_range_semitones = int(active_top1.max().item() - active_top1.min().item())
        if active_top1.numel() < 2:
            pitch_motion_rate = 0.0
            pitch_jump_rate = 0.0
        else:
            deltas = torch.abs(active_top1[1:] - active_top1[:-1])
            pitch_motion_rate = _safe_mean(deltas >= 1)
            pitch_jump_rate = _safe_mean(deltas > int(config.jump_semitones))

    voiced_score = _linear_score(voiced_ratio, config.min_voiced_ratio, 0.90)
    dominance_score = _linear_score(top1_dominance, config.min_top1_dominance, 0.78)
    unique_score = _bounded_count_score(unique_pitch_count, config.min_unique_pitches, config.max_unique_pitches)
    motion_score = _linear_score(pitch_motion_rate, config.min_pitch_motion_rate, 0.18)
    range_score = _linear_score(pitch_range_semitones, 6.0, 24.0)
    silence_score = 1.0 - _clamp01(silence_ratio / max(1e-8, config.max_silence_ratio * 2.0))
    jump_score = 1.0 - _clamp01(pitch_jump_rate / max(1e-8, config.max_pitch_jump_rate))
    low_pitch_score = 1.0 - _clamp01(low_pitch_ratio / max(1e-8, config.max_low_pitch_ratio))

    melody_clear_score = (
        0.22 * voiced_score
        + 0.24 * dominance_score
        + 0.15 * unique_score
        + 0.14 * motion_score
        + 0.10 * range_score
        + 0.08 * silence_score
        + 0.04 * jump_score
        + 0.03 * low_pitch_score
    )
    melody_clear_score = _clamp01(melody_clear_score)

    hard_accept = (
        melody_clear_score >= float(config.min_score)
        and silence_ratio <= float(config.max_silence_ratio)
        and voiced_ratio >= float(config.min_voiced_ratio)
        and top1_dominance >= float(config.min_top1_dominance)
        and unique_pitch_count >= int(config.min_unique_pitches)
        and unique_pitch_count <= int(config.max_unique_pitches)
        and pitch_motion_rate >= float(config.min_pitch_motion_rate)
        and pitch_jump_rate <= float(config.max_pitch_jump_rate)
        and low_pitch_ratio <= float(config.max_low_pitch_ratio)
    )
    if hard_accept:
        decision = "accept"
    elif melody_clear_score >= float(config.maybe_score) and silence_ratio < 0.35 and pitch_jump_rate <= 0.60:
        decision = "maybe"
    else:
        decision = "reject"

    return MelodyClearMetrics(
        decision=decision,
        melody_clear_score=melody_clear_score,
        silence_ratio=silence_ratio,
        voiced_ratio=voiced_ratio,
        top1_dominance=top1_dominance,
        unique_pitch_count=unique_pitch_count,
        pitch_motion_rate=pitch_motion_rate,
        pitch_jump_rate=pitch_jump_rate,
        low_pitch_ratio=low_pitch_ratio,
        pitch_range_semitones=pitch_range_semitones,
    )


def find_audio_files(
    audio_dir: str | Path,
    *,
    extensions: Iterable[str],
    recursive: bool = True,
    max_files: int | None = None,
) -> list[Path]:
    root = Path(audio_dir)
    if not root.exists():
        raise FileNotFoundError(f"Audio directory not found: {root}")
    normalized_exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    iterator = root.rglob("*") if recursive else root.glob("*")
    files = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in normalized_exts)
    if max_files is not None:
        files = files[: max(0, int(max_files))]
    return files


def _ensure_stereo(audio: torch.Tensor) -> torch.Tensor:
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.ndim != 2:
        raise ValueError(f"Expected audio [C,T] or [T], got shape={tuple(audio.shape)}")
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    if audio.shape[0] != 2:
        raise ValueError(f"Expected mono/stereo audio, got channels={audio.shape[0]}")
    return audio.to(torch.float32)


def load_audio_for_scoring(path: str | Path, *, config: MelodyClearFilterConfig) -> tuple[torch.Tensor, int]:
    install_torchaudio_load_fallback()
    audio, source_sample_rate = torchaudio.load(str(path))
    audio = _ensure_stereo(audio)
    if int(source_sample_rate) != int(config.sample_rate):
        audio = torchaudio.functional.resample(audio, int(source_sample_rate), int(config.sample_rate))
    target_samples = max(1, int(round(float(config.seconds_total) * int(config.sample_rate))))
    if audio.shape[-1] < target_samples:
        audio = torch.nn.functional.pad(audio, (0, target_samples - audio.shape[-1]))
    elif audio.shape[-1] > target_samples:
        audio = audio[..., :target_samples]
    return audio, int(source_sample_rate)


def extract_cqt_magnitude(audio: torch.Tensor, *, config: MelodyClearFilterConfig) -> torch.Tensor:
    extractor = CQTTopKExtractor(
        CQTTopKConfig(
            sample_rate=int(config.sample_rate),
            fmin_hz=float(config.fmin_hz),
            highpass_cutoff_hz=float(config.highpass_cutoff_hz),
            n_bins=int(config.n_bins),
            bins_per_octave=int(config.bins_per_octave),
            hop_length=int(config.hop_length),
            top_k=int(config.top_k),
            backend=config.cqt_backend,  # type: ignore[arg-type]
        )
    )
    normalized = extractor._normalize_audio(audio).to(torch.float32)  # noqa: SLF001
    with torch.amp.autocast(device_type=normalized.device.type, enabled=False):
        filtered = extractor._highpass(normalized)  # noqa: SLF001
        if extractor._backend == "nnaudio":  # noqa: SLF001
            return extractor._cqt_with_nnaudio(filtered)  # noqa: SLF001
        return extractor._cqt_with_librosa(filtered)  # noqa: SLF001


def evaluate_audio_file(path: str | Path, *, config: MelodyClearFilterConfig, rank: int = 0) -> CandidateRow:
    try:
        audio, source_sample_rate = load_audio_for_scoring(path, config=config)
        magnitude = extract_cqt_magnitude(audio.unsqueeze(0), config=config)
        metrics = compute_melody_clear_metrics(magnitude, config=config)
        return CandidateRow(
            rank=rank,
            audio_path=str(Path(path)),
            filename=Path(path).name,
            duration_sec=float(audio.shape[-1]) / float(config.sample_rate),
            sample_rate=int(source_sample_rate),
            decision=metrics.decision,
            melody_clear_score=round(metrics.melody_clear_score, 6),
            silence_ratio=round(metrics.silence_ratio, 6),
            voiced_ratio=round(metrics.voiced_ratio, 6),
            top1_dominance=round(metrics.top1_dominance, 6),
            unique_pitch_count=int(metrics.unique_pitch_count),
            pitch_motion_rate=round(metrics.pitch_motion_rate, 6),
            pitch_jump_rate=round(metrics.pitch_jump_rate, 6),
            low_pitch_ratio=round(metrics.low_pitch_ratio, 6),
            pitch_range_semitones=int(metrics.pitch_range_semitones),
            error="",
        )
    except Exception as exc:  # noqa: BLE001
        return CandidateRow(
            rank=rank,
            audio_path=str(Path(path)),
            filename=Path(path).name,
            duration_sec=0.0,
            sample_rate=0,
            decision="error",
            melody_clear_score=0.0,
            silence_ratio=0.0,
            voiced_ratio=0.0,
            top1_dominance=0.0,
            unique_pitch_count=0,
            pitch_motion_rate=0.0,
            pitch_jump_rate=0.0,
            low_pitch_ratio=0.0,
            pitch_range_semitones=0,
            error=f"{type(exc).__name__}: {exc}",
        )


def _evaluate_audio_file_task(task: tuple[str, MelodyClearFilterConfig, int]) -> CandidateRow:
    path, config, rank = task
    return evaluate_audio_file(path, config=config, rank=rank)


def write_candidates_csv(path: str | Path, rows: list[CandidateRow]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(CandidateRow.__dataclass_fields__.keys())
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _safe_audition_name(row: CandidateRow) -> str:
    stem = Path(row.filename).stem
    suffix = Path(row.filename).suffix or ".wav"
    score = _format_score_token(row.melody_clear_score)
    voiced = _format_score_token(row.voiced_ratio)
    jump = _format_score_token(row.pitch_jump_rate)
    return f"rank_{row.rank:06d}_score_{score}_voiced_{voiced}_jump_{jump}_{row.decision}_{stem}{suffix}"


def copy_audition_files(rows: list[CandidateRow], audition_dir: str | Path, *, top_n: int) -> int:
    if top_n <= 0:
        return 0
    output_dir = Path(audition_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for row in rows:
        if copied >= top_n:
            break
        if row.decision == "error":
            continue
        source = Path(row.audio_path)
        if not source.exists():
            continue
        shutil.copy2(source, output_dir / _safe_audition_name(row))
        copied += 1
    return copied


def evaluate_audio_dir(
    audio_dir: str | Path,
    *,
    config: MelodyClearFilterConfig,
    extensions: Iterable[str],
    recursive: bool,
    max_files: int | None = None,
    workers: int = 1,
) -> list[CandidateRow]:
    files = find_audio_files(audio_dir, extensions=extensions, recursive=recursive, max_files=max_files)
    worker_count = max(1, int(workers))
    if worker_count == 1 or len(files) <= 1:
        rows = [evaluate_audio_file(path, config=config, rank=index + 1) for index, path in enumerate(files)]
    else:
        tasks = [(str(path), config, index + 1) for index, path in enumerate(files)]
        rows = []
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_evaluate_audio_file_task, task) for task in tasks]
            for future in as_completed(futures):
                rows.append(future.result())
    rows.sort(key=lambda row: (row.decision != "accept", -row.melody_clear_score, row.filename))
    return [
        CandidateRow(
            rank=index + 1,
            audio_path=row.audio_path,
            filename=row.filename,
            duration_sec=row.duration_sec,
            sample_rate=row.sample_rate,
            decision=row.decision,
            melody_clear_score=row.melody_clear_score,
            silence_ratio=row.silence_ratio,
            voiced_ratio=row.voiced_ratio,
            top1_dominance=row.top1_dominance,
            unique_pitch_count=row.unique_pitch_count,
            pitch_motion_rate=row.pitch_motion_rate,
            pitch_jump_rate=row.pitch_jump_rate,
            low_pitch_ratio=row.low_pitch_ratio,
            pitch_range_semitones=row.pitch_range_semitones,
            error=row.error,
        )
        for index, row in enumerate(rows)
    ]


def _parse_extensions(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank 10s audio files by melody clarity for ControlNet curriculum.")
    parser.add_argument("--audio-dir", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--audition-dir", type=str, default=None)
    parser.add_argument("--top-n", type=int, default=200, help="Copy top-N non-error files when --audition-dir is set.")
    parser.add_argument("--extensions", type=str, default=DEFAULT_EXTENSIONS)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes for CQT scoring.")

    parser.add_argument("--sample-rate", type=int, default=44_100)
    parser.add_argument("--seconds-total", type=float, default=10.0)
    parser.add_argument("--cqt-backend", type=str, choices=["auto", "nnaudio", "librosa"], default="auto")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--n-bins", type=int, default=128)
    parser.add_argument("--bins-per-octave", type=int, default=12)
    parser.add_argument("--fmin-hz", type=float, default=8.175_798_915_643_707)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--highpass-cutoff-hz", type=float, default=261.2)

    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument("--maybe-score", type=float, default=0.42)
    parser.add_argument("--min-voiced-ratio", type=float, default=0.55)
    parser.add_argument("--max-silence-ratio", type=float, default=0.15)
    parser.add_argument("--min-top1-dominance", type=float, default=0.32)
    parser.add_argument("--min-unique-pitches", type=int, default=6)
    parser.add_argument("--max-unique-pitches", type=int, default=45)
    parser.add_argument("--min-pitch-motion-rate", type=float, default=0.03)
    parser.add_argument("--max-pitch-jump-rate", type=float, default=0.35)
    parser.add_argument("--max-low-pitch-ratio", type=float, default=0.45)
    parser.add_argument("--low-pitch-bin", type=int, default=48)
    parser.add_argument("--jump-semitones", type=int, default=7)
    parser.add_argument("--active-energy-ratio", type=float, default=0.05)
    return parser


def config_from_args(args: argparse.Namespace) -> MelodyClearFilterConfig:
    return MelodyClearFilterConfig(
        sample_rate=int(args.sample_rate),
        seconds_total=float(args.seconds_total),
        n_bins=int(args.n_bins),
        bins_per_octave=int(args.bins_per_octave),
        fmin_hz=float(args.fmin_hz),
        hop_length=int(args.hop_length),
        top_k=int(args.top_k),
        highpass_cutoff_hz=float(args.highpass_cutoff_hz),
        cqt_backend=args.cqt_backend,
        active_energy_ratio=float(args.active_energy_ratio),
        low_pitch_bin=int(args.low_pitch_bin),
        jump_semitones=int(args.jump_semitones),
        min_score=float(args.min_score),
        maybe_score=float(args.maybe_score),
        min_voiced_ratio=float(args.min_voiced_ratio),
        max_silence_ratio=float(args.max_silence_ratio),
        min_top1_dominance=float(args.min_top1_dominance),
        min_unique_pitches=int(args.min_unique_pitches),
        max_unique_pitches=int(args.max_unique_pitches),
        min_pitch_motion_rate=float(args.min_pitch_motion_rate),
        max_pitch_jump_rate=float(args.max_pitch_jump_rate),
        max_low_pitch_ratio=float(args.max_low_pitch_ratio),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = config_from_args(args)
    rows = evaluate_audio_dir(
        args.audio_dir,
        config=config,
        extensions=_parse_extensions(args.extensions),
        recursive=bool(args.recursive),
        max_files=args.max_files,
        workers=int(args.workers),
    )
    write_candidates_csv(args.output_csv, rows)
    copied = copy_audition_files(rows, args.audition_dir, top_n=int(args.top_n)) if args.audition_dir else 0

    counts: dict[str, int] = {}
    for row in rows:
        counts[row.decision] = counts.get(row.decision, 0) + 1
    print(f"scored {len(rows)} files -> {args.output_csv}")
    print("decision_counts=" + ", ".join(f"{key}:{counts[key]}" for key in sorted(counts)))
    if args.audition_dir:
        print(f"copied {copied} audition files -> {args.audition_dir}")
    for row in rows[:10]:
        print(
            f"rank={row.rank} decision={row.decision} score={row.melody_clear_score:.3f} "
            f"voiced={row.voiced_ratio:.3f} dom={row.top1_dominance:.3f} "
            f"unique={row.unique_pitch_count} jump={row.pitch_jump_rate:.3f} file={row.filename}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
