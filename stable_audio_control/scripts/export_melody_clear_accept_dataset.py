from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


METRIC_FLOAT_FIELDS = (
    "melody_clear_score",
    "silence_ratio",
    "voiced_ratio",
    "top1_dominance",
    "pitch_motion_rate",
    "pitch_jump_rate",
    "low_pitch_ratio",
)
METRIC_INT_FIELDS = ("rank", "unique_pitch_count", "pitch_range_semitones")


@dataclass(frozen=True)
class ExportSummary:
    total_rows: int
    selected_count: int
    skipped_count: int
    train_count: int
    val_count: int
    train_tracks: int
    val_tracks: int
    output_root: str
    train_config: str
    val_config: str


def _default_metadata_module() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "mtg_jamendo_metadata.py"


def _split_csv_tokens(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _normalized_track_id_from_digits(digits: str) -> str:
    return f"track_{int(digits):07d}"


def extract_track_id_from_name(name: str | Path) -> str | None:
    normalized = str(name).replace("\\", "/")
    filename = normalized.rsplit("/", 1)[-1]
    stem = Path(filename).stem

    track_match = re.search(r"track[_-](\d{1,9})", stem, flags=re.IGNORECASE)
    if track_match:
        return _normalized_track_id_from_digits(track_match.group(1))

    segment_match = re.search(r"^(\d{1,9})(?:[_-]seg\d+)?$", stem, flags=re.IGNORECASE)
    if segment_match:
        return _normalized_track_id_from_digits(segment_match.group(1))

    digit_match = re.search(r"(\d{1,9})", stem)
    if digit_match:
        return _normalized_track_id_from_digits(digit_match.group(1))
    return None


def _load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        return [dict(row) for row in csv.DictReader(fp)]


def _load_source_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Source manifest not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Source manifest must be a JSON object keyed by filename: {manifest_path}")
    return data


def _manifest_candidates(filename: str, audio_path: str) -> tuple[str, ...]:
    normalized_audio_path = str(audio_path).replace("\\", "/").lstrip("./")
    return (
        filename,
        Path(filename).name,
        normalized_audio_path,
        Path(normalized_audio_path).name,
    )


def _source_manifest_entry(source_manifest: Mapping[str, Any], *, filename: str, audio_path: str) -> dict[str, Any]:
    for key in _manifest_candidates(filename, audio_path):
        if key in source_manifest:
            value = source_manifest[key]
            if isinstance(value, Mapping):
                return dict(value)
            return {"prompt": str(value)}
    return {"prompt": "instrumental piano melody"}


def _track_id_from_entry_or_filename(entry: Mapping[str, Any], filename: str) -> str:
    for key in ("track_id", "mtg_track_id", "source_track_id"):
        value = entry.get(key)
        if value:
            value_str = str(value)
            if value_str.startswith("track_"):
                return value_str
            extracted = extract_track_id_from_name(value_str)
            if extracted:
                return extracted
    return extract_track_id_from_name(filename) or Path(filename).stem


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return round(float(value), 6)


def _to_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def _build_manifest_entry(row: Mapping[str, str], source_entry: Mapping[str, Any], *, track_id: str) -> dict[str, Any]:
    entry = dict(source_entry)
    entry.setdefault("prompt", "instrumental piano melody")
    entry["track_id"] = track_id
    entry["melody_clear_decision"] = row.get("decision", "")
    entry["melody_clear_source_audio_path"] = row.get("audio_path", "")
    for field in METRIC_FLOAT_FIELDS:
        value = _to_float(row.get(field))
        if value is not None:
            entry[field] = value
    for field in METRIC_INT_FIELDS:
        value = _to_int(row.get(field))
        if value is not None:
            entry[field] = value
    return entry


def _split_track_ids(track_ids: Iterable[str], *, val_ratio: float, seed: int) -> set[str]:
    unique_ids = sorted(set(track_ids))
    if len(unique_ids) <= 1 or val_ratio <= 0.0:
        return set()
    rng = random.Random(int(seed))
    rng.shuffle(unique_ids)
    val_count = int(round(len(unique_ids) * float(val_ratio)))
    val_count = max(1, min(len(unique_ids) - 1, val_count))
    return set(unique_ids[:val_count])


def _copy_audio(source: str | Path, destination: str | Path) -> None:
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Accepted audio file not found: {source_path}")
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _dataset_config(*, dataset_id: str, split: str, audio_dir: Path, metadata_module: Path) -> dict[str, Any]:
    return {
        "dataset_type": "audio_dir",
        "datasets": [
            {
                "id": f"{dataset_id}_{split}",
                "path": str(audio_dir.resolve()),
                "custom_metadata_module": str(metadata_module.resolve()),
            }
        ],
        "random_crop": False,
    }


def export_accept_dataset(
    *,
    csv_path: str | Path,
    source_manifest_path: str | Path,
    output_root: str | Path,
    dataset_id: str,
    val_ratio: float = 0.1,
    seed: int = 1337,
    decisions: Iterable[str] = ("accept",),
    metadata_module: str | Path | None = None,
) -> ExportSummary:
    rows = _load_csv_rows(csv_path)
    source_manifest = _load_source_manifest(source_manifest_path)
    accepted_decisions = set(decisions)

    selected: list[tuple[dict[str, str], dict[str, Any], str]] = []
    skipped_count = 0
    for row in rows:
        if row.get("decision", "") not in accepted_decisions:
            skipped_count += 1
            continue
        filename = row.get("filename") or Path(row.get("audio_path", "")).name
        source_entry = _source_manifest_entry(source_manifest, filename=filename, audio_path=row.get("audio_path", ""))
        track_id = _track_id_from_entry_or_filename(source_entry, filename)
        selected.append((row, source_entry, track_id))

    val_track_ids = _split_track_ids((track_id for _, _, track_id in selected), val_ratio=val_ratio, seed=seed)
    root = Path(output_root)
    train_dir = root / "train"
    val_dir = root / "val"
    manifest_dir = root / "manifests"
    metadata_module_path = Path(metadata_module) if metadata_module else _default_metadata_module()

    train_manifest: dict[str, dict[str, Any]] = {}
    val_manifest: dict[str, dict[str, Any]] = {}
    train_track_ids: set[str] = set()
    val_track_ids_used: set[str] = set()

    for row, source_entry, track_id in selected:
        filename = row.get("filename") or Path(row.get("audio_path", "")).name
        split = "val" if track_id in val_track_ids else "train"
        split_dir = val_dir if split == "val" else train_dir
        destination = split_dir / filename
        _copy_audio(row.get("audio_path", ""), destination)
        manifest_entry = _build_manifest_entry(row, source_entry, track_id=track_id)
        if split == "val":
            val_manifest[filename] = manifest_entry
            val_track_ids_used.add(track_id)
        else:
            train_manifest[filename] = manifest_entry
            train_track_ids.add(track_id)

    _write_json(manifest_dir / "train.json", train_manifest)
    _write_json(manifest_dir / "val.json", val_manifest)
    train_config_path = root / "dataset_config_train.json"
    val_config_path = root / "dataset_config_val.json"
    _write_json(
        train_config_path,
        _dataset_config(dataset_id=dataset_id, split="train", audio_dir=train_dir, metadata_module=metadata_module_path),
    )
    _write_json(
        val_config_path,
        _dataset_config(dataset_id=dataset_id, split="val", audio_dir=val_dir, metadata_module=metadata_module_path),
    )

    return ExportSummary(
        total_rows=len(rows),
        selected_count=len(selected),
        skipped_count=skipped_count,
        train_count=len(train_manifest),
        val_count=len(val_manifest),
        train_tracks=len(train_track_ids),
        val_tracks=len(val_track_ids_used),
        output_root=str(root),
        train_config=str(train_config_path),
        val_config=str(val_config_path),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export decision=accept melody-clear CSV rows to train/val audio_dir datasets."
    )
    parser.add_argument("--csv", type=str, required=True, help="CSV produced by select_melody_clear_segments.py.")
    parser.add_argument("--source-manifest", type=str, required=True, help="Manifest from the first-stage candidate export.")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--dataset-id", type=str, default="piano_clean_v2")
    parser.add_argument("--decisions", type=str, default="accept")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--metadata-module", type=str, default=str(_default_metadata_module()))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = export_accept_dataset(
        csv_path=args.csv,
        source_manifest_path=args.source_manifest,
        output_root=args.output_root,
        dataset_id=args.dataset_id,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        decisions=_split_csv_tokens(args.decisions),
        metadata_module=args.metadata_module,
    )
    print(f"total_rows={summary.total_rows}")
    print(f"selected_count={summary.selected_count}")
    print(f"skipped_count={summary.skipped_count}")
    print(f"train_count={summary.train_count} train_tracks={summary.train_tracks}")
    print(f"val_count={summary.val_count} val_tracks={summary.val_tracks}")
    print(f"output_root={summary.output_root}")
    print(f"train_config={summary.train_config}")
    print(f"val_config={summary.val_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
