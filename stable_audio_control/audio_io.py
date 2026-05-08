"""
Runtime audio I/O fallbacks for environments where TorchCodec is unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torchaudio


_SAVE_FALLBACK_INSTALLED = False
_LOAD_FALLBACK_INSTALLED = False


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


def _load_audio_with_soundfile(
    uri: Any,
    *,
    frame_offset: int = 0,
    num_frames: int = -1,
    channels_first: bool = True,
) -> tuple[torch.Tensor, int]:
    frames = -1 if num_frames is None or num_frames < 0 else int(num_frames)
    file = str(uri) if isinstance(uri, (str, Path)) else uri
    audio, sample_rate = sf.read(
        file=file,
        start=int(frame_offset),
        frames=frames,
        dtype="float32",
        always_2d=True,
    )
    tensor = torch.from_numpy(audio.copy())
    if channels_first:
        tensor = tensor.transpose(0, 1).contiguous()
    return tensor, int(sample_rate)


def install_torchaudio_load_fallback() -> None:
    """
    Patch ``torchaudio.load`` to fall back to ``soundfile`` when TorchCodec fails.

    This keeps local WAV datasets usable on Windows setups where TorchCodec cannot
    load its FFmpeg/PyTorch-compatible native DLLs.
    """

    global _LOAD_FALLBACK_INSTALLED
    if _LOAD_FALLBACK_INSTALLED:
        return

    original_load = torchaudio.load

    def load_with_fallback(
        uri: Any,
        frame_offset: int = 0,
        num_frames: int = -1,
        normalize: bool = True,
        channels_first: bool = True,
        format: str | None = None,
        buffer_size: int = 4096,
        backend: str | None = None,
    ):
        try:
            return original_load(
                uri,
                frame_offset=frame_offset,
                num_frames=num_frames,
                normalize=normalize,
                channels_first=channels_first,
                format=format,
                buffer_size=buffer_size,
                backend=backend,
            )
        except Exception as exc:  # noqa: BLE001
            if not _is_torchcodec_failure(exc):
                raise
            return _load_audio_with_soundfile(
                uri,
                frame_offset=frame_offset,
                num_frames=num_frames,
                channels_first=channels_first,
            )

    torchaudio.load = load_with_fallback
    _LOAD_FALLBACK_INSTALLED = True


def install_torchaudio_save_fallback() -> None:
    """
    Patch ``torchaudio.save`` to gracefully fall back to ``soundfile`` for WAV.

    This is a narrow workaround for Windows environments where TorchAudio routes
    save() through TorchCodec and TorchCodec cannot load its native dependencies.
    """

    global _SAVE_FALLBACK_INSTALLED
    if _SAVE_FALLBACK_INSTALLED:
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
    _SAVE_FALLBACK_INSTALLED = True
