from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MelodyControlEncoder(nn.Module):
    """
    Convert top-k CQT pitch indices into ControlNet-ready token features.

    Input indices use the paper-aligned contract:
    - shape: [B, C_melody, F] or [B, F, C_melody]
    - values: 0..num_pitch_bins, where 0 is mask/padding

    Output:
    - shape: [B, target_len, output_dim]
    """

    def __init__(
        self,
        *,
        num_pitch_bins: int = 128,
        melody_channels: int = 8,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        output_dim: int,
        conv_layers: int = 2,
        conv_kernel_size: int = 3,
        interp_mode: str = "nearest",
    ) -> None:
        super().__init__()

        if num_pitch_bins < 1:
            raise ValueError("num_pitch_bins must be greater than 0.")
        if melody_channels < 1:
            raise ValueError("melody_channels must be greater than 0.")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be greater than 0.")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be greater than 0.")
        if output_dim < 1:
            raise ValueError("output_dim must be greater than 0.")
        if conv_layers < 1:
            raise ValueError("conv_layers must be greater than 0.")
        if conv_kernel_size < 1 or conv_kernel_size % 2 == 0:
            raise ValueError("conv_kernel_size must be a positive odd integer.")

        self.num_pitch_bins = int(num_pitch_bins)
        self.melody_channels = int(melody_channels)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.conv_layers = int(conv_layers)
        self.conv_kernel_size = int(conv_kernel_size)
        self.interp_mode = interp_mode

        self.pitch_embedding = nn.Embedding(
            num_embeddings=self.num_pitch_bins + 1,
            embedding_dim=self.embedding_dim,
            padding_idx=0,
        )

        conv_modules: list[nn.Module] = []
        in_channels = self.melody_channels * self.embedding_dim
        padding = self.conv_kernel_size // 2
        for layer_ix in range(self.conv_layers):
            conv_modules.append(
                nn.Conv1d(
                    in_channels=in_channels if layer_ix == 0 else self.hidden_dim,
                    out_channels=self.hidden_dim,
                    kernel_size=self.conv_kernel_size,
                    padding=padding,
                )
            )
            conv_modules.append(nn.SiLU())
        conv_modules.append(nn.Conv1d(self.hidden_dim, self.output_dim, kernel_size=1))
        self.conv = nn.Sequential(*conv_modules)

    def _normalize_indices(self, melody: torch.Tensor) -> torch.LongTensor:
        if melody.ndim == 2:
            melody = melody.unsqueeze(0)
        if melody.ndim != 3:
            raise ValueError(
                f"melody must be 3D ([B,C,F] or [B,F,C]); got shape={tuple(melody.shape)}"
            )

        if melody.shape[1] == self.melody_channels:
            melody_bcf = melody
        elif melody.shape[2] == self.melody_channels:
            melody_bcf = melody.transpose(1, 2).contiguous()
        else:
            raise ValueError(
                "melody must have a melody channel dimension equal to "
                f"{self.melody_channels}; got shape={tuple(melody.shape)}"
            )

        if torch.is_floating_point(melody_bcf):
            rounded = melody_bcf.round()
            if not torch.equal(rounded, melody_bcf):
                raise TypeError("floating melody indices must contain integer values.")
            melody_bcf = rounded

        melody_long = melody_bcf.to(dtype=torch.long)
        if melody_long.numel() > 0:
            min_value = int(melody_long.min().item())
            max_value = int(melody_long.max().item())
            if min_value < 0 or max_value > self.num_pitch_bins:
                raise ValueError(
                    f"melody pitch indices must be in 0..{self.num_pitch_bins}; "
                    f"got min={min_value}, max={max_value}"
                )
        return melody_long

    def forward(
        self,
        melody: torch.Tensor,
        *,
        target_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if target_len <= 0:
            raise ValueError("target_len must be greater than 0.")

        melody_long = self._normalize_indices(melody).to(device=device)
        self.to(device=device)

        # [B, C, F] -> [B, C, F, E] -> [B, C*E, F]
        embedded = self.pitch_embedding(melody_long)
        bsz, channels, frames, embed_dim = embedded.shape
        embedded = embedded.permute(0, 1, 3, 2).reshape(bsz, channels * embed_dim, frames)

        features = self.conv(embedded.to(dtype=next(self.parameters()).dtype))
        if features.shape[-1] != target_len:
            features = F.interpolate(features, size=target_len, mode=self.interp_mode)

        return features.transpose(1, 2).contiguous().to(dtype=dtype, device=device)
