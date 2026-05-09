from __future__ import annotations

from typing import Literal, Protocol

import torch

from stable_audio_control.melody.chromagram import ChromagramConfig, ChromagramExtractor
from stable_audio_control.melody.cqt_topk import BackendLiteral, CQTTopKConfig, CQTTopKExtractor


MelodyFeature = Literal["cqt", "chromagram"]


class MelodyExtractor(Protocol):
    def extract(self, audio: torch.Tensor) -> torch.Tensor: ...


def melody_control_channels(
    feature: MelodyFeature,
    *,
    top_k: int,
    chroma_bins: int,
) -> int:
    if feature == "cqt":
        return int(top_k) * 2
    if feature == "chromagram":
        return int(chroma_bins)
    raise ValueError(f"Unsupported melody feature: {feature}")


def build_melody_extractor(
    *,
    feature: MelodyFeature,
    sample_rate: int,
    top_k: int,
    n_bins: int,
    bins_per_octave: int,
    fmin_hz: float,
    hop_length: int,
    highpass_cutoff_hz: float,
    cqt_backend: BackendLiteral,
    chroma_bins: int,
    chroma_n_fft: int,
) -> MelodyExtractor:
    if feature == "cqt":
        return CQTTopKExtractor(
            CQTTopKConfig(
                sample_rate=sample_rate,
                fmin_hz=fmin_hz,
                highpass_cutoff_hz=highpass_cutoff_hz,
                n_bins=n_bins,
                bins_per_octave=bins_per_octave,
                hop_length=hop_length,
                top_k=top_k,
                backend=cqt_backend,
            )
        )
    if feature == "chromagram":
        return ChromagramExtractor(
            ChromagramConfig(
                sample_rate=sample_rate,
                hop_length=hop_length,
                n_fft=chroma_n_fft,
                n_chroma=chroma_bins,
            )
        )
    raise ValueError(f"Unsupported melody feature: {feature}")
