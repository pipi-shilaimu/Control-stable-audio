from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
import torchaudio

from stable_audio_control.audio_io import install_torchaudio_load_fallback
from stable_audio_control.inference.control_compare import ensure_stereo
from stable_audio_control.melody.extractors import build_melody_extractor


install_torchaudio_load_fallback()

MelodyFeature = Literal["cqt", "chromagram"]

DEFAULT_SAMPLE_RATE = 44_100
DEFAULT_SECONDS_TOTAL = 5.0

SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aif", ".aiff")


def _path_string(path: str | Path) -> str:
    return str(Path(path).resolve())


def _prepare_target_sample_size(
    *,
    sample_rate: int,
    sample_size: int | None,
    seconds_total: float | None,
) -> int | None:
    if sample_size is not None:
        return int(sample_size)
    if seconds_total is None:
        return None
    return max(1, int(round(float(seconds_total) * int(sample_rate))))


def _load_audio_tensor(
    audio_path: str | Path,
    *,
    target_sample_rate: int,
) -> torch.Tensor:
    audio, sample_rate = torchaudio.load(str(audio_path))
    audio = ensure_stereo(audio.to(torch.float32))

    if int(sample_rate) != int(target_sample_rate):
        resampler = torchaudio.transforms.Resample(int(sample_rate), int(target_sample_rate))
        audio = resampler(audio)

    return audio.unsqueeze(0)


def _align_audio_pair(
    reference_audio_path: str | Path,
    generated_audio_path: str | Path,
    *,
    target_sample_rate: int,
    target_sample_size: int | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    reference_audio = _load_audio_tensor(reference_audio_path, target_sample_rate=target_sample_rate)
    generated_audio = _load_audio_tensor(generated_audio_path, target_sample_rate=target_sample_rate)

    if target_sample_size is None:
        common_sample_size = min(int(reference_audio.shape[-1]), int(generated_audio.shape[-1]))
        if common_sample_size < 1:
            raise ValueError("Cannot compare empty audio clips.")
        reference_audio = reference_audio[..., :common_sample_size]
        generated_audio = generated_audio[..., :common_sample_size]
        alignment_mode = "common_prefix"
    else:
        common_sample_size = int(target_sample_size)
        if reference_audio.shape[-1] < common_sample_size:
            reference_audio = torch.nn.functional.pad(reference_audio, (0, common_sample_size - reference_audio.shape[-1]))
        else:
            reference_audio = reference_audio[..., :common_sample_size]
        if generated_audio.shape[-1] < common_sample_size:
            generated_audio = torch.nn.functional.pad(generated_audio, (0, common_sample_size - generated_audio.shape[-1]))
        else:
            generated_audio = generated_audio[..., :common_sample_size]
        alignment_mode = "fixed_length"

    alignment = {
        "sample_rate": int(target_sample_rate),
        "sample_size": int(common_sample_size),
        "seconds_total": float(common_sample_size) / float(target_sample_rate),
        "mode": alignment_mode,
    }
    return reference_audio, generated_audio, alignment


def _build_feature_config(
    *,
    feature: MelodyFeature,
    sample_rate: int,
    top_k: int,
    n_bins: int,
    bins_per_octave: int,
    fmin_hz: float,
    hop_length: int,
    highpass_cutoff_hz: float,
    cqt_backend: str,
    chroma_bins: int,
    chroma_n_fft: int,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "feature": feature,
        "sample_rate": int(sample_rate),
        "hop_length": int(hop_length),
    }
    if feature == "cqt":
        config.update(
            {
                "top_k": int(top_k),
                "n_bins": int(n_bins),
                "bins_per_octave": int(bins_per_octave),
                "fmin_hz": float(fmin_hz),
                "highpass_cutoff_hz": float(highpass_cutoff_hz),
                "cqt_backend": cqt_backend,
            }
        )
    else:
        config.update(
            {
                "chroma_bins": int(chroma_bins),
                "chroma_n_fft": int(chroma_n_fft),
            }
        )
    return config


def compare_cqt_topk_features(
    reference: torch.Tensor,
    generated: torch.Tensor,
    *,
    top_k: int,
) -> dict[str, Any]:
    if reference.ndim != 3 or generated.ndim != 3:
        raise ValueError(
            f"Expected reference and generated CQT tensors with shape [B, C, F]; "
            f"got {tuple(reference.shape)} and {tuple(generated.shape)}"
        )

    expected_channels = int(top_k) * 2
    if reference.shape[1] != expected_channels or generated.shape[1] != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} CQT channels for top_k={top_k}; "
            f"got {reference.shape[1]} and {generated.shape[1]}"
        )

    compared_frames = min(int(reference.shape[-1]), int(generated.shape[-1]))
    if compared_frames < 1:
        raise ValueError("Cannot compare empty CQT features.")

    reference = reference[..., :compared_frames].to(torch.long)
    generated = generated[..., :compared_frames].to(torch.long)

    reference = reference.reshape(reference.shape[0], int(top_k), 2, compared_frames).permute(0, 2, 3, 1)
    generated = generated.reshape(generated.shape[0], int(top_k), 2, compared_frames).permute(0, 2, 3, 1)

    # [B, 2, F, K] -> [B, 2, F, K, K]
    overlap = generated.unsqueeze(-1).eq(reference.unsqueeze(-2))
    matched_tokens = int(overlap.any(dim=-1).sum().item())
    total_tokens = int(generated.numel())
    score = float(matched_tokens / total_tokens)

    return {
        "metric_name": "cqt_topk_pitch_overlap_rate",
        "score": score,
        "matched_tokens": matched_tokens,
        "total_tokens": total_tokens,
        "compared_frames": compared_frames,
    }


def compare_chromagram_features(reference: torch.Tensor, generated: torch.Tensor) -> dict[str, Any]:
    if reference.ndim != 3 or generated.ndim != 3:
        raise ValueError(
            f"Expected reference and generated chromagram tensors with shape [B, C, F]; "
            f"got {tuple(reference.shape)} and {tuple(generated.shape)}"
        )

    if reference.shape[1] != generated.shape[1]:
        raise ValueError(
            f"Chromagram channel count must match; got {reference.shape[1]} and {generated.shape[1]}"
        )

    compared_frames = min(int(reference.shape[-1]), int(generated.shape[-1]))
    if compared_frames < 1:
        raise ValueError("Cannot compare empty chromagram features.")

    reference = reference[..., :compared_frames].to(torch.float32).transpose(1, 2).contiguous()
    generated = generated[..., :compared_frames].to(torch.float32).transpose(1, 2).contiguous()

    frame_similarity = F.cosine_similarity(reference, generated, dim=-1, eps=1e-8)
    score = float(frame_similarity.mean().item())

    return {
        "metric_name": "chromagram_frame_cosine_mean",
        "score": score,
        "compared_frames": compared_frames,
    }


def compare_melody_features(
    reference: torch.Tensor,
    generated: torch.Tensor,
    *,
    feature: MelodyFeature,
    top_k: int,
) -> dict[str, Any]:
    if feature == "cqt":
        return compare_cqt_topk_features(reference, generated, top_k=top_k)
    if feature == "chromagram":
        return compare_chromagram_features(reference, generated)
    raise ValueError(f"Unsupported melody feature: {feature}")


def compare_audio_melody_similarity(
    reference_audio_path: str | Path,
    generated_audio_path: str | Path,
    *,
    feature: MelodyFeature,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    sample_size: int | None = None,
    seconds_total: float = DEFAULT_SECONDS_TOTAL,
    top_k: int = 4,
    n_bins: int = 128,
    bins_per_octave: int = 12,
    fmin_hz: float = 8.175_798_915_643_707,
    hop_length: int = 512,
    highpass_cutoff_hz: float = 261.2,
    cqt_backend: str = "auto",
    chroma_bins: int = 12,
    chroma_n_fft: int = 2048,
) -> dict[str, Any]:
    target_sample_size = _prepare_target_sample_size(
        sample_rate=sample_rate,
        sample_size=sample_size,
        seconds_total=seconds_total,
    )
    reference_audio, generated_audio, alignment = _align_audio_pair(
        reference_audio_path,
        generated_audio_path,
        target_sample_rate=sample_rate,
        target_sample_size=target_sample_size,
    )

    extractor = build_melody_extractor(
        feature=feature,
        sample_rate=sample_rate,
        top_k=top_k,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        fmin_hz=fmin_hz,
        hop_length=hop_length,
        highpass_cutoff_hz=highpass_cutoff_hz,
        cqt_backend=cqt_backend,  # type: ignore[arg-type]
        chroma_bins=chroma_bins,
        chroma_n_fft=chroma_n_fft,
    )

    reference_feature = extractor.extract(reference_audio)
    generated_feature = extractor.extract(generated_audio)
    similarity = compare_melody_features(
        reference_feature,
        generated_feature,
        feature=feature,
        top_k=top_k,
    )

    feature_config = _build_feature_config(
        feature=feature,
        sample_rate=sample_rate,
        top_k=top_k,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        fmin_hz=fmin_hz,
        hop_length=hop_length,
        highpass_cutoff_hz=highpass_cutoff_hz,
        cqt_backend=cqt_backend,
        chroma_bins=chroma_bins,
        chroma_n_fft=chroma_n_fft,
    )

    return {
        "schema_version": 1,
        "paths": {
            "reference_audio": _path_string(reference_audio_path),
            "generated_audio": _path_string(generated_audio_path),
        },
        "alignment": alignment,
        "feature": {
            "name": feature,
            "config": feature_config,
            "reference_shape": list(reference_feature.shape),
            "generated_shape": list(generated_feature.shape),
        },
        "similarity": similarity,
    }


def write_similarity_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
