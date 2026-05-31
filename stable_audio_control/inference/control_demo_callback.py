"""
ControlNet-aware demo callback for stable_audio_tools diffusion training.

Extends DiffusionCondDemoCallback to inject melody control into the
generation path, so demo audio reflects what the ControlNet has learned.

核心思路：在 conditioning 里保留 melody_control，直接传 `cond=conditioning` 给
sample()，让 ControlConditionedDiffusionWrapper.forward 内部自动处理
_extract_control_input → control_input → base_wrapper forwarding。

不需要改 stable_audio_tools 源码——所有逻辑在这个子类里。
"""

from __future__ import annotations

import typing as tp
import torch
import soundfile as sf
import pytorch_lightning as pl
from einops import rearrange

from stable_audio_tools.inference.sampling import sample
from stable_audio_tools.training.diffusion import DiffusionCondTrainingWrapper
from stable_audio_tools.training.utils import log_audio, log_image
from stable_audio_tools.interface.aeiou import audio_spectrogram_image
from pytorch_lightning.utilities.rank_zero import rank_zero_only


DEFAULT_CONTROL_SCALES = [0.0, 0.1, 0.3, 0.6, 1.0]
DEFAULT_CONTROL_VARIANTS = ["correct", "shuffled", "zero"]
_CONTROL_SCALES_UNSET = object()


def _format_scale(value: float | int) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "neg").replace(".", "p")


class ControlNetDemoCallback(pl.Callback):
    """每 demo_every 步生成 demo 音频，自动注入 ControlNet 旋律控制条件。"""

    def __init__(
        self,
        demo_every: int = 2000,
        num_demos: int = 4,
        sample_size: int = 65536,
        demo_steps: int = 250,
        sample_rate: int = 48000,
        demo_cfg_scales: tp.List[int] = [3, 6, 9],
        control_scale: tp.Optional[float] = None,
        control_scales: tp.Union[tp.Sequence[float], None, object] = _CONTROL_SCALES_UNSET,
        control_variants: tp.Optional[tp.Sequence[str]] = None,
        control_id: str = "melody_control",
    ):
        super().__init__()
        self.demo_every = demo_every
        self.num_demos = num_demos
        self.demo_samples = sample_size
        self.demo_steps = demo_steps
        self.sample_rate = sample_rate
        self.last_demo_step = -1
        self.demo_cfg_scales = demo_cfg_scales
        self.control_scales = self._resolve_control_scales(control_scale, control_scales)
        self.control_scale = float(self.control_scales[-1])
        self.control_variants = list(control_variants or DEFAULT_CONTROL_VARIANTS)
        self.control_id = control_id

    def _resolve_control_scales(
        self,
        control_scale: tp.Optional[float],
        control_scales: tp.Union[tp.Sequence[float], None, object],
    ) -> tp.List[float]:
        if control_scales is _CONTROL_SCALES_UNSET:
            if control_scale is None:
                return list(DEFAULT_CONTROL_SCALES)
            return [float(control_scale)]
        if control_scales is None:
            if control_scale is None:
                return list(DEFAULT_CONTROL_SCALES)
            return [float(control_scale)]
        return [float(scale) for scale in control_scales]

    def iter_control_demo_combinations(self) -> tp.List[tp.Tuple[int, float, str]]:
        return [
            (cfg_scale, control_scale, variant)
            for cfg_scale in self.demo_cfg_scales
            for control_scale in self.control_scales
            for variant in self.control_variants
        ]

    def format_demo_stem(self, cfg_scale: int, control_scale: float, variant: str, step: int) -> str:
        return (
            f"demo_cfg_{_format_scale(cfg_scale)}"
            f"_control_{_format_scale(control_scale)}"
            f"_{variant}_step_{step:08d}"
        )

    def format_audio_tag(self, cfg_scale: int, control_scale: float, variant: str, step: int) -> str:
        return (
            f"demo_step_{step:08d}"
            f"_cfg_{_format_scale(cfg_scale)}"
            f"_control_{_format_scale(control_scale)}"
            f"_{variant}"
        )

    def format_melspec_tag(self, cfg_scale: int, control_scale: float, variant: str, step: int) -> str:
        return (
            f"demo_melspec_step_{step:08d}"
            f"_cfg_{_format_scale(cfg_scale)}"
            f"_control_{_format_scale(control_scale)}"
            f"_{variant}"
        )

    def make_variant_reals(self, reals: torch.Tensor, variant: str) -> torch.Tensor:
        if variant == "correct":
            return reals
        if variant == "shuffled":
            if reals.shape[0] > 1:
                return torch.roll(reals, shifts=1, dims=0)
            return torch.flip(reals, dims=[-1])
        if variant == "zero":
            return torch.zeros_like(reals)
        raise ValueError(f"Unknown control variant: {variant}")

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module: DiffusionCondTrainingWrapper, outputs, batch, batch_idx):
        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return

        module.eval()
        print(f"[ControlNetDemo] Generating demo at step {trainer.global_step}")
        self.last_demo_step = trainer.global_step

        # --- 1. 从当前 batch 取音频和 metadata ---
        reals = batch[0]
        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]
        metadata = batch[1]
        if isinstance(metadata, tuple):
            metadata = list(metadata)
        if isinstance(metadata, list):
            demo_count = min(self.num_demos, reals.shape[0], len(metadata))
            demo_cond = metadata[:demo_count]
        else:
            demo_count = min(self.num_demos, reals.shape[0])
            demo_cond = [metadata] * demo_count

        if demo_count <= 0:
            print("[ControlNetDemo] Skipping demo: empty batch.")
            module.train()
            return

        demo_reals = reals[:demo_count]

        # Normalize metadata: unconditionally set prompt + wrap padding_mask in list

        for item in demo_cond:
            try:
                item["prompt"] = str(item.get("prompt", "music"))
                pm = item.get("padding_mask")
                if isinstance(pm, torch.Tensor):
                    item["padding_mask"] = [pm]
            except (TypeError, AttributeError):
                pass

        # --- 2. 准备 demo noise。所有对照共用同一份 noise，便于比较控制差异 ---
        melody_augmenter = getattr(module, "melody_augmenter", None)

        diffusion = module.diffusion

        demo_samples = self.demo_samples
        if diffusion.pretransform is not None:
            demo_samples = demo_samples // diffusion.pretransform.downsampling_ratio

        noise = torch.randn([demo_count, diffusion.io_channels, demo_samples]).to(module.device)

        try:
            for cfg_scale, control_scale, variant in self.iter_control_demo_combinations():
                print(f"[ControlNetDemo] cfg_scale={cfg_scale} control_scale={control_scale} variant={variant}")

                if melody_augmenter is not None:
                    melody_augmenter.set_batch_audio(self.make_variant_reals(demo_reals, variant))

                with torch.cuda.amp.autocast():
                    conditioning = diffusion.conditioner(demo_cond, module.device)

                # --- 3. 直接传 cond=conditioning 给 wrapper ---
                # ControlConditionedDiffusionWrapper.forward 内部会：
                #   _extract_control_input(cond) → control_input
                #   pop melody_control from cond → cond_for_base
                #   base_wrapper(x, t, cond_for_base, control_input=..., control_scale=..., ...)
                # EMA model is the raw DiT — doesn't know control_input routing.
                # Always use full ControlConditionedDiffusionWrapper for demo.
                fakes = sample(
                    diffusion,
                    noise,
                    self.demo_steps,
                    0,
                    cond=conditioning,
                    cfg_scale=cfg_scale,
                    control_scale=control_scale,
                    batch_cfg=True,
                )

                if diffusion.pretransform is not None:
                    fakes = diffusion.pretransform.decode(fakes)

                # --- 4. 保存音频 ---
                fakes = rearrange(fakes, "b d n -> d (b n)")
                stem = self.format_demo_stem(cfg_scale, control_scale, variant, trainer.global_step)
                filename = f"{trainer.default_root_dir}/{stem}.wav"
                fakes_out = (
                    fakes.to(torch.float32)
                    .div(torch.max(torch.abs(fakes)))
                    .mul(32767)
                    .to(torch.int16)
                    .cpu()
                )
                sf.write(filename, fakes_out.cpu().numpy().T, self.sample_rate)
                log_audio(
                    trainer.logger,
                    self.format_audio_tag(cfg_scale, control_scale, variant, trainer.global_step),
                    filename,
                    self.sample_rate,
                )
                log_image(
                    trainer.logger,
                    self.format_melspec_tag(cfg_scale, control_scale, variant, trainer.global_step),
                    audio_spectrogram_image(fakes_out),
                )

        finally:
            module.train()
