from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import soundfile as sf
import torch
import torchaudio

from stable_audio_control.audio_io import install_torchaudio_load_fallback
from stable_audio_control.models import ControlConditionedDiffusionWrapper


install_torchaudio_load_fallback()


@dataclass(frozen=True)
class ControlCheckpointStates:
    online_wrapper_state: dict[str, torch.Tensor]
    ema_model_state: dict[str, torch.Tensor]
    use_ema: bool


def _path_string(path: str | Path) -> str:
    return str(Path(path).resolve())


def strip_state_dict_prefix(
    state_dict: Mapping[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    """Return state-dict entries below a prefix with that prefix removed."""

    return {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}


def extract_control_checkpoint_state_dicts(
    checkpoint: Mapping[str, Any],
    *,
    prefer_ema: bool = True,
) -> ControlCheckpointStates:
    """Split a Lightning training checkpoint into online wrapper and EMA model states."""

    raw_state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(raw_state, Mapping):
        raise TypeError("Checkpoint must be a state dict or contain a mapping under 'state_dict'.")

    state_dict = dict(raw_state)
    online_wrapper_state = strip_state_dict_prefix(state_dict, "diffusion.")
    if not online_wrapper_state:
        raise ValueError("Checkpoint does not contain any 'diffusion.' weights.")

    ema_model_state = strip_state_dict_prefix(state_dict, "diffusion_ema.ema_model.")
    use_ema = bool(prefer_ema and ema_model_state)
    if not use_ema:
        ema_model_state = {}

    return ControlCheckpointStates(
        online_wrapper_state=online_wrapper_state,
        ema_model_state=ema_model_state,
        use_ema=use_ema,
    )


def load_control_checkpoint(
    model: ControlConditionedDiffusionWrapper,
    ckpt_path: str | Path,
    *,
    prefer_ema: bool = True,
) -> dict[str, Any]:
    """Load online ControlNet wrapper weights, then optionally overlay EMA DiT weights."""

    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    states = extract_control_checkpoint_state_dicts(checkpoint, prefer_ema=prefer_ema)

    online_result = model.load_state_dict(states.online_wrapper_state, strict=False)
    ema_result = None
    if states.use_ema:
        ema_result = model.model.load_state_dict(states.ema_model_state, strict=False)

    return {
        "ckpt_path": _path_string(ckpt_path),
        "use_ema": states.use_ema,
        "online_missing_keys": list(online_result.missing_keys),
        "online_unexpected_keys": list(online_result.unexpected_keys),
        "ema_missing_keys": [] if ema_result is None else list(ema_result.missing_keys),
        "ema_unexpected_keys": [] if ema_result is None else list(ema_result.unexpected_keys),
    }


def initialize_lazy_control_modules(
    model: ControlConditionedDiffusionWrapper,
    *,
    device: torch.device,
    dtype: torch.dtype,
    control_channels: int,
) -> None:
    """Materialize lazy control projection modules before loading checkpoint weights."""

    index_cond: dict[str, Any] = {
        model.control_id: [torch.zeros((1, control_channels, 8), device=device, dtype=torch.long), None],
    }
    model._extract_control_input(  # type: ignore[attr-defined]
        cond=index_cond,
        target_len=8,
        dtype=dtype,
        device=device,
    )

    dense_cond: dict[str, Any] = {
        model.control_id: [torch.zeros((1, control_channels, 8), device=device, dtype=dtype), None],
    }
    model._extract_control_input(  # type: ignore[attr-defined]
        cond=dense_cond,
        target_len=8,
        dtype=dtype,
        device=device,
    )


def ensure_stereo(audio: torch.Tensor) -> torch.Tensor:
    """Normalize a waveform to [2, T], duplicating mono input."""

    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.ndim != 2:
        raise ValueError(f"Expected audio tensor [C,T] or [T], got shape={tuple(audio.shape)}")
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    if audio.shape[0] != 2:
        raise ValueError(f"Expected mono/stereo audio, got channels={audio.shape[0]}")
    return audio


def load_reference_audio(
    audio_path: str | Path,
    *,
    target_sample_rate: int,
    target_sample_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Load, resample, crop/pad, and return a stereo reference waveform [1,2,T]."""

    audio, sample_rate = torchaudio.load(str(audio_path))
    audio = ensure_stereo(audio.to(torch.float32))

    if int(sample_rate) != int(target_sample_rate):
        resampler = torchaudio.transforms.Resample(int(sample_rate), int(target_sample_rate))
        audio = resampler(audio)

    if audio.shape[-1] < target_sample_size:
        audio = torch.nn.functional.pad(audio, (0, target_sample_size - audio.shape[-1]))
    elif audio.shape[-1] > target_sample_size:
        audio = audio[..., :target_sample_size]

    return audio.unsqueeze(0).to(device=device, dtype=torch.float32)


def audio_sample_size_from_seconds(
    *,
    seconds_total: float,
    sample_rate: int,
    min_input_length: int = 1,
) -> int:
    """Convert seconds to samples and round up to the model's minimum input multiple."""

    sample_size = max(1, int(round(float(seconds_total) * int(sample_rate))))
    min_input_length = max(1, int(min_input_length))
    remainder = sample_size % min_input_length
    if remainder:
        sample_size += min_input_length - remainder
    return sample_size


def save_audio_tensor(path: str | Path, audio: torch.Tensor, sample_rate: int) -> None:
    """Peak-normalize and save a generated tensor as WAV using soundfile."""

    audio = audio.detach().to(torch.float32).cpu()
    if audio.ndim == 3:
        if audio.shape[0] != 1:
            raise ValueError(f"Expected one generated batch item, got batch={audio.shape[0]}")
        audio = audio[0]
    audio = ensure_stereo(audio)

    peak = audio.abs().max()
    if torch.isfinite(peak) and float(peak) > 0.0:
        audio = audio / peak
    audio = audio.clamp(-1.0, 1.0)

    data = audio.transpose(0, 1).contiguous().numpy().astype(np.float32)
    sf.write(str(path), data, int(sample_rate), format="WAV", subtype="PCM_16")


def build_compare_metadata(
    *,
    prompt: str,
    negative_prompt: str | None,
    reference_audio_path: str | Path,
    ckpt_path: str | Path,
    base_output_path: str | Path,
    control_output_path: str | Path,
    seed: int,
    model_name: str,
    sample_rate: int,
    sample_size: int,
    seconds_start: float,
    seconds_total: float,
    steps: int,
    cfg_scale: float,
    sampler_type: str,
    sigma_min: float,
    sigma_max: float,
    control_scale: float,
    use_ema: bool,
    control_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the reproducibility sidecar for a base-vs-control generation run."""

    return {
        "schema_version": 1,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "model_name": model_name,
        "paths": {
            "reference_audio": _path_string(reference_audio_path),
            "checkpoint": _path_string(ckpt_path),
            "base_output": _path_string(base_output_path),
            "control_output": _path_string(control_output_path),
        },
        "generation": {
            "seed": int(seed),
            "sample_rate": int(sample_rate),
            "sample_size": int(sample_size),
            "seconds_start": float(seconds_start),
            "seconds_total": float(seconds_total),
            "steps": int(steps),
            "cfg_scale": float(cfg_scale),
            "sampler_type": sampler_type,
            "sigma_min": float(sigma_min),
            "sigma_max": float(sigma_max),
        },
        "control": {
            "scale": float(control_scale),
            "use_ema": bool(use_ema),
            "config": dict(control_config),
        },
    }


def write_compare_metadata(path: str | Path, metadata: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
