from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


_MANIFEST_CACHE: dict[Path, dict[str, Any]] = {}


def _normalize_relpath(value: str) -> str:
    return value.replace("\\", "/").lstrip("./")


def _candidate_manifest_paths(info: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []

    env_path = os.environ.get("STABLE_AUDIO_CONTROL_METADATA_MANIFEST")
    if env_path:
        candidates.append(Path(env_path))

    audio_path_value = info.get("path")
    if audio_path_value:
        audio_path = Path(audio_path_value)
        split_dir = audio_path.parent
        dataset_root = split_dir.parent
        candidates.append(dataset_root / "manifests" / f"{split_dir.name}.json")

    return candidates


def _load_manifest(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved not in _MANIFEST_CACHE:
        _MANIFEST_CACHE[resolved] = json.loads(resolved.read_text(encoding="utf-8"))
    return _MANIFEST_CACHE[resolved]


def _find_manifest(info: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    for candidate in _candidate_manifest_paths(info):
        if candidate.exists():
            return candidate, _load_manifest(candidate)
    raise FileNotFoundError(
        "Could not find Song Describer manifest. Expected an environment override "
        "`STABLE_AUDIO_CONTROL_METADATA_MANIFEST` or a sibling `manifests/<split>.json` "
        "next to the audio split directory."
    )


def get_custom_metadata(info, audio):
    _manifest_path, manifest = _find_manifest(info)
    relpath = _normalize_relpath(str(info["relpath"]))

    entry = manifest.get(relpath)
    if entry is None:
        entry = manifest.get(Path(relpath).name)
    if entry is None:
        raise KeyError(f"Could not find relpath '{relpath}' in Song Describer manifest.")

    prompt = entry["prompt"] if isinstance(entry, dict) else str(entry)
    metadata = {"prompt": prompt}

    if isinstance(entry, dict):
        for key, value in entry.items():
            if key == "prompt":
                continue
            metadata[f"song_describer_{key}"] = value

    return metadata
