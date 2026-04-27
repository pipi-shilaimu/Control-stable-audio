from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torchaudio

BackendLiteral = Literal["auto", "nnaudio", "librosa"]


@dataclass(frozen=True)
class CQTTopKConfig:
    sample_rate: int
    fmin_hz: float = 8.175798915643707  # MIDI note 0
    highpass_cutoff_hz: float = 261.2
    n_bins: int = 128
    bins_per_octave: int = 12
    hop_length: int = 512
    top_k: int = 4
    backend: BackendLiteral = "auto"


class CQTTopKExtractor:
    def __init__(self, config: CQTTopKConfig) -> None:
        self.config = config
        self._backend = self._resolve_backend(config.backend)
        self._nnaudio_cqt: Optional[torch.nn.Module] = None

    @staticmethod
    def _resolve_backend(backend: BackendLiteral) -> Literal["nnaudio", "librosa"]:
        if backend not in {"auto", "nnaudio", "librosa"}:
            raise ValueError(f"Unsupported backend '{backend}'.")

        if backend in {"auto", "nnaudio"}:
            try:
                import nnAudio  # noqa: F401

                return "nnaudio"
            except Exception:  # noqa: BLE001
                if backend == "nnaudio":
                    raise RuntimeError("Backend 'nnaudio' requested but nnAudio is not available.")

        if backend in {"auto", "librosa"}:
            try:
                import librosa  # noqa: F401

                return "librosa"
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError("No available CQT backend. Install nnAudio or librosa.") from exc

        raise RuntimeError("Failed to resolve CQT backend.")

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

        if self.config.top_k > self.config.n_bins:
            raise ValueError(f"top_k ({self.config.top_k}) must be <= n_bins ({self.config.n_bins})")

        return audio

    def _highpass(self, audio: torch.Tensor) -> torch.Tensor:
        b, c, t = audio.shape
        flattened = audio.reshape(b * c, t)
        filtered = torchaudio.functional.highpass_biquad(
            waveform=flattened,
            sample_rate=self.config.sample_rate,
            cutoff_freq=self.config.highpass_cutoff_hz,
        )
        return filtered.reshape(b, c, t)

    def _build_nnaudio(self, device: torch.device) -> torch.nn.Module:
        if self._nnaudio_cqt is None:
            from nnAudio.features import CQT

            self._nnaudio_cqt = CQT(
                sr=self.config.sample_rate,
                hop_length=self.config.hop_length,
                fmin=self.config.fmin_hz,
                n_bins=self.config.n_bins,
                bins_per_octave=self.config.bins_per_octave,
                output_format="Magnitude",
                trainable=False,
                center=True,
            )
        return self._nnaudio_cqt.to(device)

    def _cqt_with_nnaudio(self, audio: torch.Tensor) -> torch.Tensor:
        b, c, t = audio.shape
        flattened = audio.reshape(b * c, t)
        cqt_module = self._build_nnaudio(audio.device)
        magnitude = cqt_module(flattened)
        if magnitude.ndim != 3:
            raise RuntimeError(f"Unexpected nnAudio CQT output shape: {tuple(magnitude.shape)}")
        _, n_bins, frames = magnitude.shape
        return magnitude.reshape(b, c, n_bins, frames)

    def _cqt_with_librosa(self, audio: torch.Tensor) -> torch.Tensor:
        import librosa

        b, c, _ = audio.shape
        cpu_audio = audio.detach().to(torch.float32).cpu()

        per_channel = []
        for batch_ix in range(b):
            channel_mags = []
            for ch_ix in range(c):
                waveform_np = cpu_audio[batch_ix, ch_ix].numpy()
                cqt_complex = librosa.cqt(
                    y=waveform_np,
                    sr=self.config.sample_rate,
                    hop_length=self.config.hop_length,
                    fmin=self.config.fmin_hz,
                    n_bins=self.config.n_bins,
                    bins_per_octave=self.config.bins_per_octave,
                )
                channel_mags.append(torch.from_numpy(abs(cqt_complex)).to(torch.float32))
            per_channel.append(torch.stack(channel_mags, dim=0))

        magnitude = torch.stack(per_channel, dim=0)
        return magnitude.to(audio.device)

    def extract(self, audio: torch.Tensor) -> torch.LongTensor:
        normalized = self._normalize_audio(audio)
        filtered = self._highpass(normalized)

        if self._backend == "nnaudio":
            magnitude = self._cqt_with_nnaudio(filtered)
        else:
            magnitude = self._cqt_with_librosa(filtered)

        # [B, 2, n_bins, F] -> top-k over n_bins -> [B, 2, K, F]
        _, topk_idx = torch.topk(magnitude, k=self.config.top_k, dim=2, largest=True, sorted=True)

        # 1-based index in 1..n_bins; reserve 0 for mask.
        topk_idx = topk_idx + 1

        # Interleave as [L0, R0, L1, R1, ...] and return [B, 2K, F].
        left = topk_idx[:, 0, :, :]
        right = topk_idx[:, 1, :, :]
        interleaved = torch.stack((left, right), dim=3).reshape(
            topk_idx.shape[0], topk_idx.shape[2] * 2, topk_idx.shape[3]
        )

        return interleaved.to(torch.long)
