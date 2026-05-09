from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ChromagramConfig:
    sample_rate: int
    hop_length: int = 512
    n_fft: int = 2048
    n_chroma: int = 12


class ChromagramExtractor:
    """Extract a mono 12-bin chromagram control tensor from stereo audio."""

    def __init__(self, config: ChromagramConfig) -> None:
        if config.n_chroma < 1:
            raise ValueError("n_chroma must be greater than 0.")
        if config.hop_length < 1:
            raise ValueError("hop_length must be greater than 0.")
        if config.n_fft < 1:
            raise ValueError("n_fft must be greater than 0.")
        self.config = config

    def _normalize_audio(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 2:
            if audio.shape[0] != 2:
                raise ValueError(f"Expected [2, T] for 2D input, got shape {tuple(audio.shape)}")
            audio = audio.unsqueeze(0)
        elif audio.ndim == 3:
            if audio.shape[1] != 2:
                raise ValueError(f"Expected [B, 2, T] for 3D input, got shape {tuple(audio.shape)}")
        else:
            raise ValueError(f"Expected [2, T] or [B, 2, T], got shape {tuple(audio.shape)}")

        if audio.shape[-1] < self.config.hop_length:
            raise ValueError(
                f"Audio length ({audio.shape[-1]}) must be >= hop_length ({self.config.hop_length})"
            )

        return audio

    def extract(self, audio: torch.Tensor) -> torch.Tensor:
        import librosa
        import numpy as np

        normalized = self._normalize_audio(audio)
        device = normalized.device
        mono = normalized.detach().to(torch.float32).cpu().mean(dim=1)

        chroma_frames: list[torch.Tensor] = []
        for batch_ix in range(mono.shape[0]):
            chroma = librosa.feature.chroma_stft(
                y=mono[batch_ix].numpy(),
                sr=self.config.sample_rate,
                hop_length=self.config.hop_length,
                n_fft=self.config.n_fft,
                n_chroma=self.config.n_chroma,
            )
            chroma = np.nan_to_num(chroma, nan=0.0, posinf=0.0, neginf=0.0)
            chroma = np.clip(chroma, 0.0, 1.0)
            chroma_frames.append(torch.from_numpy(chroma.astype("float32", copy=False)))

        return torch.stack(chroma_frames, dim=0).to(device=device, dtype=torch.float32)
