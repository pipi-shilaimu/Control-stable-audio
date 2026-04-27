import typing as tp

import torch
from torch import nn
from torch.nn import functional as F

from stable_audio_control.models.control_transformer import ControlNetContinuousTransformer


class ControlConditionedDiffusionWrapper(nn.Module):
    """
    Task-1 wrapper:
    - Keeps `stable_audio_tools` public calling style unchanged.
    - Extracts `melody_control` from `cond`.
    - Converts it into `control_input` and forwards it to ControlNet transformer.
    """

    def __init__(
        self,
        base_wrapper: nn.Module,
        control_id: str = "melody_control",
        default_control_scale: float = 1.0,
        control_interp_mode: str = "nearest",
    ) -> None:
        super().__init__()

        self.base_wrapper = base_wrapper
        self.control_id = control_id
        self.default_control_scale = float(default_control_scale)
        self.control_interp_mode = control_interp_mode

        base_transformer = self.model.model.transformer
        if not isinstance(base_transformer, ControlNetContinuousTransformer):
            raise TypeError(
                "Expected base_wrapper.model.model.transformer to be ControlNetContinuousTransformer. "
                "Call `attach_controlnet_transformer(...)` first."
            )

        self.control_dim_in = base_transformer.base.project_in.in_features
        self.control_projector = nn.LazyLinear(self.control_dim_in)

    # Properties are exposed to remain compatible with existing training code that
    # expects a ConditionedDiffusionModelWrapper-like object.
    @property
    def model(self):
        return self.base_wrapper.model

    @property
    def conditioner(self):
        return self.base_wrapper.conditioner

    @property
    def io_channels(self):
        return self.base_wrapper.io_channels

    @property
    def sample_rate(self):
        return self.base_wrapper.sample_rate

    @property
    def diffusion_objective(self):
        return self.base_wrapper.diffusion_objective

    @property
    def pretransform(self):
        return self.base_wrapper.pretransform

    @property
    def cross_attn_cond_ids(self):
        return self.base_wrapper.cross_attn_cond_ids

    @property
    def global_cond_ids(self):
        return self.base_wrapper.global_cond_ids

    @property
    def input_concat_ids(self):
        return self.base_wrapper.input_concat_ids

    @property
    def prepend_cond_ids(self):
        return self.base_wrapper.prepend_cond_ids

    @property
    def min_input_length(self):
        return self.base_wrapper.min_input_length

    @property
    def dist_shift(self):
        return self.base_wrapper.dist_shift

    def get_conditioning_inputs(self, conditioning_tensors: tp.Dict[str, tp.Any], negative: bool = False):
        return self.base_wrapper.get_conditioning_inputs(conditioning_tensors, negative=negative)

    def generate(self, *args, **kwargs):
        return self.base_wrapper.generate(*args, **kwargs)

    def _extract_control_input(
        self,
        cond: tp.Dict[str, tp.Any],
        target_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tp.Optional[torch.Tensor]:
        if cond is None or self.control_id not in cond:
            return None

        raw = cond[self.control_id]
        control_tensor = raw[0] if isinstance(raw, (list, tuple)) else raw
        if control_tensor is None:
            return None
        if not torch.is_tensor(control_tensor):
            raise TypeError(f"`cond[{self.control_id}]` must be a Tensor or [Tensor, mask], got {type(control_tensor)}")

        control_tensor = control_tensor.to(device=device)

        if control_tensor.ndim == 2:
            control_tensor = control_tensor.unsqueeze(0)
        if control_tensor.ndim != 3:
            raise ValueError(
                f"`cond[{self.control_id}]` must be 3D ([B,C,L] or [B,L,C]); got shape={tuple(control_tensor.shape)}"
            )

        # Accept both [B, C, L] and [B, L, C].
        if control_tensor.shape[1] == target_len and control_tensor.shape[2] != target_len:
            control_blc = control_tensor
        else:
            control_bcl = control_tensor
            if not torch.is_floating_point(control_bcl):
                control_bcl = control_bcl.to(dtype=dtype)
            if control_bcl.shape[-1] != target_len:
                control_bcl = F.interpolate(
                    control_bcl,
                    size=target_len,
                    mode=self.control_interp_mode,
                )
            control_blc = control_bcl.transpose(1, 2).contiguous()

        control_blc = control_blc.to(dtype=dtype)
        self.control_projector = self.control_projector.to(device=device, dtype=dtype)
        control_input = self.control_projector(control_blc)  # [B, L, dim_in]
        return control_input

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: tp.Dict[str, tp.Any],
        cfg_dropout_prob: float = 0.0,
        control_scale: tp.Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        if control_scale is None:
            control_scale = self.default_control_scale

        control_input = self._extract_control_input(
            cond=cond,
            target_len=x.shape[-1],
            dtype=x.dtype,
            device=x.device,
        )

        # Keep melody control private to the ControlNet path; do not route it through
        # base cross/prepend/input-concat conditioning ids.
        cond_for_base = dict(cond) if cond is not None else {}
        cond_for_base.pop(self.control_id, None)

        return self.base_wrapper(
            x,
            t,
            cond=cond_for_base,
            cfg_dropout_prob=cfg_dropout_prob,
            control_input=control_input,
            control_scale=float(control_scale),
            **kwargs,
        )


def attach_controlnet_transformer(
    base_wrapper: nn.Module,
    num_control_layers: int = 12,
    default_control_scale: float = 1.0,
    freeze_base: bool = True,
) -> nn.Module:
    """
    Replace pretrained StableAudio Open transformer in-place with a ControlNet variant.
    Returns the same wrapper object for chaining.
    """

    base_transformer = base_wrapper.model.model.transformer
    control_transformer = ControlNetContinuousTransformer(
        base_transformer=base_transformer,
        num_control_layers=num_control_layers,
        default_control_scale=default_control_scale,
    )
    if freeze_base:
        control_transformer.freeze_base()
    base_wrapper.model.model.transformer = control_transformer
    return base_wrapper


def build_control_wrapper(
    base_wrapper: nn.Module,
    num_control_layers: int = 12,
    control_id: str = "melody_control",
    default_control_scale: float = 1.0,
    freeze_base: bool = True,
) -> ControlConditionedDiffusionWrapper:
    """
    One-liner helper:
    1) attach ControlNet transformer
    2) return training-compatible ControlConditionedDiffusionWrapper
    """

    wrapped = attach_controlnet_transformer(
        base_wrapper=base_wrapper,
        num_control_layers=num_control_layers,
        default_control_scale=default_control_scale,
        freeze_base=freeze_base,
    )
    return ControlConditionedDiffusionWrapper(
        base_wrapper=wrapped,
        control_id=control_id,
        default_control_scale=default_control_scale,
    )
