from __future__ import annotations

import argparse
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf


DEFAULT_DATASET_NAME = "renumics/song-describer-dataset"
DEFAULT_OUTPUT_DIR = Path("./stable_audio_control/data/song_describer")
DEFAULT_METADATA_MODULE = Path(__file__).resolve().with_name("song_describer_metadata.py")
DEFAULT_PROMPT_FIELDS = ("caption", "prompt", "text", "description")
DEFAULT_AUDIO_FIELDS = ("audio", "path")


@dataclass(frozen=True)
class ExportResult:
    split: str
    exported_count: int
    audio_dir: Path
    manifest_path: Path
    dataset_config_path: Path


def _as_posix_relpath(path: Path) -> str:
    return path.as_posix()


def _safe_token(value: Any) -> str:
    token = str(value).strip().lower()
    token = re.sub(r"[^a-z0-9._-]+", "-", token)
    token = token.strip("-._")
    return token or "unknown"


def _sample_id(sample: Mapping[str, Any], index: int) -> str:
    parts = [f"{index:06d}"]
    for key in ("track_id", "caption_id", "id"):
        value = sample.get(key)
        if value is not None:
            parts.append(f"{key}-{_safe_token(value)}")
    return "_".join(parts)


def get_prompt(sample: Mapping[str, Any], prompt_field: str | None = None) -> str:
    candidates = (prompt_field,) if prompt_field is not None else DEFAULT_PROMPT_FIELDS
    for field in candidates:
        if field is None or field not in sample:
            continue
        value = sample[field]
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = " ".join(str(item) for item in value if item is not None)
        prompt = str(value).strip()
        if prompt:
            return prompt
    raise KeyError(
        "Could not find a non-empty prompt field. "
        f"Tried: {', '.join(field for field in candidates if field is not None)}"
    )


def get_audio_value(sample: Mapping[str, Any], audio_field: str | None = None) -> Any:
    candidates = (audio_field,) if audio_field is not None else DEFAULT_AUDIO_FIELDS
    for field in candidates:
        if field is not None and field in sample and sample[field] is not None:
            return sample[field]
    raise KeyError(
        "Could not find an audio field. "
        f"Tried: {', '.join(field for field in candidates if field is not None)}"
    )


def _audio_array_for_soundfile(audio_array: Any) -> np.ndarray:
    audio = np.asarray(audio_array, dtype=np.float32)
    if audio.ndim == 0:
        raise ValueError("Audio array must contain at least one sample.")
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        # Hugging Face audio is usually [frames] or [frames, channels], but a
        # few sources expose [channels, frames]. SoundFile expects frames first.
        if audio.shape[0] <= 8 and audio.shape[0] < audio.shape[1]:
            return audio.T
        return audio
    raise ValueError(f"Audio array must be 1D or 2D, got shape={audio.shape}.")


def decode_audio(audio_value: Any) -> tuple[np.ndarray, int]:
    if isinstance(audio_value, Mapping):
        if audio_value.get("array") is not None:
            if "sampling_rate" not in audio_value or audio_value["sampling_rate"] is None:
                raise KeyError("Audio mapping with an 'array' value must contain a 'sampling_rate' key.")
            return _audio_array_for_soundfile(audio_value["array"]), int(audio_value["sampling_rate"])

        if audio_value.get("bytes") is not None:
            audio, sample_rate = sf.read(io.BytesIO(audio_value["bytes"]), dtype="float32", always_2d=False)
            return _audio_array_for_soundfile(audio), int(sample_rate)

        if audio_value.get("path") is not None:
            audio, sample_rate = sf.read(str(audio_value["path"]), dtype="float32", always_2d=False)
            return _audio_array_for_soundfile(audio), int(sample_rate)

        keys = ", ".join(str(key) for key in audio_value.keys()) or "<none>"
        raise KeyError(
            "Audio mapping must contain either decoded 'array'/'sampling_rate' values "
            f"or readable 'bytes'/'path'. Available keys: {keys}."
        )

    if isinstance(audio_value, (str, Path)):
        audio, sample_rate = sf.read(str(audio_value), dtype="float32", always_2d=False)
        return _audio_array_for_soundfile(audio), int(sample_rate)

    raise TypeError(f"Unsupported audio value type: {type(audio_value)!r}")


def _should_export(sample: Mapping[str, Any], valid_only: bool) -> bool:
    if not valid_only:
        return True
    if "is_valid_subset" not in sample:
        return True
    return bool(sample["is_valid_subset"])


def _manifest_entry(sample: Mapping[str, Any], prompt: str, split: str, source_index: int) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "prompt": prompt,
        "split": split,
        "source_index": source_index,
    }
    for key in ("track_id", "caption_id", "is_valid_subset", "is_audioset_eval", "caption"):
        if key in sample:
            value = sample[key]
            if isinstance(value, np.generic):
                value = value.item()
            entry[key] = value
    return entry


def write_dataset_config(
    *,
    output_dir: Path,
    split: str,
    audio_dir: Path,
    metadata_module_path: Path,
    dataset_id: str | None = None,
    random_crop: bool = True,
) -> Path:
    dataset_config = {
        "dataset_type": "audio_dir",
        "datasets": [
            {
                "id": dataset_id or f"song_describer_{split}",
                "path": str(audio_dir.resolve()),
                "custom_metadata_module": str(metadata_module_path.resolve()),
            }
        ],
        "random_crop": bool(random_crop),
    }
    config_path = output_dir / f"dataset_config_{split}.json"
    config_path.write_text(json.dumps(dataset_config, indent=2, ensure_ascii=False), encoding="utf-8")
    return config_path


def export_samples_to_audio_dir(
    samples: Iterable[Mapping[str, Any]],
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    split: str = "train",
    max_items: int | None = None,
    valid_only: bool = False,
    prompt_field: str | None = None,
    audio_field: str | None = None,
    metadata_module_path: Path | str = DEFAULT_METADATA_MODULE,
    random_crop: bool = True,
) -> ExportResult:
    output_root = Path(output_dir)
    audio_dir = output_root / split
    manifest_dir = output_root / "manifests"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict[str, Any]] = {}
    exported_count = 0

    for source_index, sample in enumerate(samples):
        if max_items is not None and exported_count >= max_items:
            break
        if not _should_export(sample, valid_only=valid_only):
            continue

        prompt = get_prompt(sample, prompt_field=prompt_field)
        audio, sample_rate = decode_audio(get_audio_value(sample, audio_field=audio_field))

        filename = f"{_sample_id(sample, source_index)}.wav"
        audio_path = audio_dir / filename
        sf.write(str(audio_path), audio, sample_rate, format="WAV", subtype="PCM_16")

        relpath = _as_posix_relpath(audio_path.relative_to(audio_dir))
        manifest[relpath] = _manifest_entry(sample, prompt=prompt, split=split, source_index=source_index)
        exported_count += 1

    manifest_path = manifest_dir / f"{split}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    dataset_config_path = write_dataset_config(
        output_dir=output_root,
        split=split,
        audio_dir=audio_dir,
        metadata_module_path=Path(metadata_module_path),
        random_crop=random_crop,
    )

    return ExportResult(
        split=split,
        exported_count=exported_count,
        audio_dir=audio_dir,
        manifest_path=manifest_path,
        dataset_config_path=dataset_config_path,
    )


def load_song_describer_dataset(dataset_name: str = DEFAULT_DATASET_NAME):
    from datasets import load_dataset

    return load_dataset(dataset_name)


def export_huggingface_dataset(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    splits: Sequence[str] = ("train",),
    max_items: int | None = None,
    valid_only: bool = False,
    prompt_field: str | None = None,
    audio_field: str | None = None,
    metadata_module_path: Path | str = DEFAULT_METADATA_MODULE,
    random_crop: bool = True,
) -> list[ExportResult]:
    dataset = load_song_describer_dataset(dataset_name)
    results: list[ExportResult] = []
    for split in splits:
        if split not in dataset:
            available = ", ".join(dataset.keys())
            raise KeyError(f"Split '{split}' not found in dataset. Available splits: {available}")
        results.append(
            export_samples_to_audio_dir(
                dataset[split],
                output_dir=output_dir,
                split=split,
                max_items=max_items,
                valid_only=valid_only,
                prompt_field=prompt_field,
                audio_field=audio_field,
                metadata_module_path=metadata_module_path,
                random_crop=random_crop,
            )
        )
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Song Describer from Hugging Face to audio_dir + custom_metadata_module format."
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--valid-only", action="store_true", help="Keep rows with is_valid_subset=True when present.")
    parser.add_argument("--prompt-field", default=None)
    parser.add_argument("--audio-field", default=None)
    parser.add_argument("--metadata-module", type=Path, default=DEFAULT_METADATA_MODULE)
    parser.add_argument("--no-random-crop", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    results = export_huggingface_dataset(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        splits=args.splits,
        max_items=args.max_items,
        valid_only=args.valid_only,
        prompt_field=args.prompt_field,
        audio_field=args.audio_field,
        metadata_module_path=args.metadata_module,
        random_crop=not args.no_random_crop,
    )

    for result in results:
        print(
            f"{result.split}: exported {result.exported_count} files\n"
            f"  audio_dir: {result.audio_dir.resolve()}\n"
            f"  manifest: {result.manifest_path.resolve()}\n"
            f"  dataset_config: {result.dataset_config_path.resolve()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
