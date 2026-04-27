"""
Runtime audio I/O fallbacks for environments where TorchCodec is unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torchaudio


_FALLBACK_INSTALLED = False


def _iter_exception_chain(exc: BaseException):
    """Yield an exception and its chained causes/contexts."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_torchcodec_failure(exc: BaseException) -> bool:
    for chained in _iter_exception_chain(exc):
        message = str(chained).lower()
        if "torchcodec" in message or "libtorchcodec" in message or "save_with_torchcodec" in message:
            return True
    return False


def _save_wav_with_soundfile(uri: str | Path, src: torch.Tensor, sample_rate: int) -> None:
    if src.ndim == 1:
        src = src.unsqueeze(0)
    if src.ndim != 2:
        raise ValueError(f"Expected a 1D/2D audio tensor, got shape {tuple(src.shape)}")

    # torchaudio.save expects [channels, frames]; soundfile expects [frames, channels].
    audio = src.detach().cpu().transpose(0, 1).contiguous().numpy()

    subtype = "PCM_16" if src.dtype == torch.int16 else None
    sf.write(file=str(uri), data=audio, samplerate=sample_rate, format="WAV", subtype=subtype)


def install_torchaudio_save_fallback() -> None:
    """
    Patch ``torchaudio.save`` to gracefully fall back to ``soundfile`` for WAV.

    This is a narrow workaround for Windows environments where TorchAudio routes
    save() through TorchCodec and TorchCodec cannot load its native dependencies.
    """

    global _FALLBACK_INSTALLED
    if _FALLBACK_INSTALLED:
        return

    original_save = torchaudio.save

    def save_with_fallback(
        uri: Any,
        src: torch.Tensor,
        sample_rate: int,
        *args: Any,
        **kwargs: Any,
    ):
        try:
            return original_save(uri, src, sample_rate, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if not _is_torchcodec_failure(exc):
                raise
            _save_wav_with_soundfile(uri=uri, src=src, sample_rate=sample_rate)
            return None

    torchaudio.save = save_with_fallback
    _FALLBACK_INSTALLED = True
