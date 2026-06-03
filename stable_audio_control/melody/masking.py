from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class MelodyMaskingConfig:
    """Training-only curriculum masking for top-k CQT melody tokens."""

    enabled: bool = False
    full_mask_steps: int = 0
    schedule_steps: int = 10_000
    frame_mask_ratio_start: float = 0.75
    frame_mask_ratio_end: float = 0.10
    secondary_mask_prob: float = 0.15
    secondary_shuffle_prob: float = 0.15
    preserve_top1: bool = True

    def __post_init__(self) -> None:
        if self.full_mask_steps < 0:
            raise ValueError("full_mask_steps must be non-negative.")
        if self.schedule_steps < 0:
            raise ValueError("schedule_steps must be non-negative.")
        for name in (
            "frame_mask_ratio_start",
            "frame_mask_ratio_end",
            "secondary_mask_prob",
            "secondary_shuffle_prob",
        ):
            value = float(getattr(self, name))
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be in 0..1; got {value}.")


def _normalize_melody(melody: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if melody.ndim == 2:
        return melody.unsqueeze(0), True
    if melody.ndim != 3:
        raise ValueError(f"melody must be [B,C,F] or [C,F]; got shape={tuple(melody.shape)}")
    return melody, False


def _restore_melody_shape(melody: torch.Tensor, squeezed: bool) -> torch.Tensor:
    return melody.squeeze(0) if squeezed else melody


def _curriculum_progress(global_step: int, config: MelodyMaskingConfig) -> float:
    if config.schedule_steps <= 0:
        return 1.0
    step_after_full_mask = max(0, int(global_step) - int(config.full_mask_steps))
    return min(1.0, step_after_full_mask / float(config.schedule_steps))


def melody_frame_mask_ratio(global_step: int, config: MelodyMaskingConfig) -> float:
    """Return the current frame-mask ratio after the full-mask phase."""

    progress = _curriculum_progress(global_step, config)
    return float(
        config.frame_mask_ratio_start
        + (config.frame_mask_ratio_end - config.frame_mask_ratio_start) * progress
    )


def apply_progressive_melody_mask(
    melody: torch.Tensor,
    *,
    global_step: int,
    config: MelodyMaskingConfig,
    training: bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply paper-style training curriculum masking to top-k CQT tokens.

    Channel layout is expected to be [L0, R0, L1, R1, L2, R2, L3, R3].
    Channels 0 and 1 are top-1 left/right and are preserved after the
    early full-mask phase when `preserve_top1=True`.
    """

    melody_bcf, squeezed = _normalize_melody(melody)
    masked = melody_bcf.clone()
    if not training or not config.enabled:
        return _restore_melody_shape(masked, squeezed)

    if int(global_step) < int(config.full_mask_steps):
        return _restore_melody_shape(torch.zeros_like(masked), squeezed)

    batch, channels, frames = masked.shape
    if batch == 0 or channels == 0 or frames == 0:
        return _restore_melody_shape(masked, squeezed)

    secondary_start = 2 if config.preserve_top1 and channels >= 2 else 0
    frame_ratio = melody_frame_mask_ratio(global_step, config)

    if config.secondary_shuffle_prob > 0.0 and secondary_start < channels and frames > 1:
        secondary = masked[:, secondary_start:, :]
        shuffle_scores = torch.rand(
            secondary.shape,
            generator=generator,
            device=masked.device,
        )
        order = shuffle_scores.argsort(dim=-1)
        shuffled = secondary.gather(dim=-1, index=order)
        shuffle_choice = torch.rand(
            secondary.shape[:2],
            generator=generator,
            device=masked.device,
        ) < float(config.secondary_shuffle_prob)
        masked[:, secondary_start:, :] = torch.where(
            shuffle_choice.unsqueeze(-1),
            shuffled,
            secondary,
        )

    if config.secondary_mask_prob > 0.0 and secondary_start < channels:
        secondary_shape = (batch, channels - secondary_start, frames)
        secondary_mask = torch.rand(
            secondary_shape,
            generator=generator,
            device=masked.device,
        ) < float(config.secondary_mask_prob)
        masked[:, secondary_start:, :] = masked[:, secondary_start:, :].masked_fill(
            secondary_mask,
            0,
        )

    if frame_ratio > 0.0:
        frame_mask = torch.rand(
            (batch, frames),
            generator=generator,
            device=masked.device,
        ) < frame_ratio
        masked = masked.masked_fill(frame_mask[:, None, :], 0)

    return _restore_melody_shape(masked, squeezed)


def _as_padding_tensor(padding_mask: Any) -> torch.Tensor | None:
    if padding_mask is None:
        return None
    if torch.is_tensor(padding_mask):
        return padding_mask
    if isinstance(padding_mask, (list, tuple)):
        tensors = [_as_padding_tensor(item) for item in padding_mask]
        tensors = [item for item in tensors if item is not None]
        if not tensors:
            return None
        if len(tensors) == 1:
            return tensors[0]
        return torch.stack(tensors, dim=0)
    raise TypeError(f"padding_mask must be a Tensor, list, tuple, or None; got {type(padding_mask)}")


def _normalize_padding_mask(
    padding_mask: torch.Tensor,
    *,
    batch_size: int,
    audio_num_samples: int,
    device: torch.device,
) -> torch.Tensor:
    mask = padding_mask.to(device=device, dtype=torch.float32)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.ndim == 3 and mask.shape[1] == 1:
        mask = mask[:, 0, :]
    if mask.ndim != 2:
        raise ValueError(f"padding_mask must be [T], [B,T], or [B,1,T]; got shape={tuple(mask.shape)}")

    if mask.shape[0] == 1 and batch_size > 1:
        mask = mask.expand(batch_size, -1)
    if mask.shape[0] != batch_size:
        raise ValueError(
            f"padding_mask batch size must be 1 or {batch_size}; got {mask.shape[0]}."
        )

    if mask.shape[-1] != int(audio_num_samples):
        mask = F.interpolate(
            mask[:, None, :],
            size=max(1, int(audio_num_samples)),
            mode="nearest",
        )[:, 0, :]
    return mask


def zero_melody_frames_from_padding_mask(
    melody: torch.Tensor,
    *,
    padding_mask: torch.Tensor | list[Any] | tuple[Any, ...] | None,
    audio_num_samples: int,
) -> torch.Tensor:
    """Set CQT melody frames that correspond to padded waveform samples to token 0."""

    melody_bcf, squeezed = _normalize_melody(melody)
    masked = melody_bcf.clone()
    padding_tensor = _as_padding_tensor(padding_mask)
    if padding_tensor is None:
        return _restore_melody_shape(masked, squeezed)

    valid_audio = _normalize_padding_mask(
        padding_tensor,
        batch_size=masked.shape[0],
        audio_num_samples=audio_num_samples,
        device=masked.device,
    )
    valid_frames = F.interpolate(
        valid_audio[:, None, :],
        size=masked.shape[-1],
        mode="nearest",
    )[:, 0, :]
    padded_frames = valid_frames < 0.5
    masked = masked.masked_fill(padded_frames[:, None, :], 0)
    return _restore_melody_shape(masked, squeezed)


def padding_masks_from_metadata(metadata: Iterable[dict[str, Any]]) -> torch.Tensor | None:
    masks: list[torch.Tensor] = []
    for item in metadata:
        if not isinstance(item, dict):
            return None
        mask = _as_padding_tensor(item.get("padding_mask"))
        if mask is None:
            return None
        if mask.ndim > 1:
            mask = mask.reshape(-1, mask.shape[-1])[0]
        masks.append(mask)
    if not masks:
        return None
    return torch.stack(masks, dim=0)
