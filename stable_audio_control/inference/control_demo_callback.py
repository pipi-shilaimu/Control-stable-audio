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
import torchaudio
import pytorch_lightning as pl
from einops import rearrange
from torch import nn

from stable_audio_tools.inference.sampling import sample
from stable_audio_tools.training.diffusion import DiffusionCondTrainingWrapper
from stable_audio_tools.training.utils import log_audio, log_image, log_point_cloud
from stable_audio_tools.interface.aeiou import audio_spectrogram_image
from pytorch_lightning.utilities.rank_zero import rank_zero_only


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
        control_scale: float = 1.0,
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
        self.control_scale = float(control_scale)
        self.control_id = control_id

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
        if isinstance(metadata, list):
            demo_cond = metadata[: self.num_demos]
        else:
            demo_cond = [metadata] * self.num_demos

        # --- 2. 注入音频到 MelodyControlAugmenter，然后走 conditioner ---
        melody_augmenter = getattr(module, "melody_augmenter", None)
        if melody_augmenter is not None:
            # 用当前 batch 的真实波形提取旋律（demo 和训练用同一批数据）
            melody_augmenter.set_batch_audio(reals)

        diffusion = module.diffusion

        demo_samples = self.demo_samples
        if diffusion.pretransform is not None:
            demo_samples = demo_samples // diffusion.pretransform.downsampling_ratio

        noise = torch.randn([self.num_demos, diffusion.io_channels, demo_samples]).to(module.device)

        try:
            with torch.cuda.amp.autocast():
                conditioning = diffusion.conditioner(demo_cond, module.device)

            # --- 3. 直接传 cond=conditioning 给 wrapper ---
            # ControlConditionedDiffusionWrapper.forward 内部会：
            #   _extract_control_input(cond) → control_input
            #   pop melody_control from cond → cond_for_base
            #   base_wrapper(x, t, cond_for_base, control_input=..., control_scale=..., ...)
            # EMA model is the raw DiT — doesn't know control_input routing.
            # Always use full ControlConditionedDiffusionWrapper for demo.
            model = diffusion

            for cfg_scale in self.demo_cfg_scales:
                print(f"[ControlNetDemo] cfg_scale={cfg_scale}")

                fakes = sample(
                    model,
                    noise,
                    self.demo_steps,
                    0,
                    cond=conditioning,
                    cfg_scale=cfg_scale,
                    control_scale=self.control_scale,
                    batch_cfg=True,
                )

                if diffusion.pretransform is not None:
                    fakes = diffusion.pretransform.decode(fakes)

                # --- 4. 保存音频 ---
                fakes = rearrange(fakes, "b d n -> d (b n)")
                filename = f"demo_cfg_{cfg_scale}_{trainer.global_step:08d}.wav"
                fakes_out = (
                    fakes.to(torch.float32)
                    .div(torch.max(torch.abs(fakes)))
                    .mul(32767)
                    .to(torch.int16)
                    .cpu()
                )
                torchaudio.save(filename, fakes_out, self.sample_rate)
                log_audio(trainer.logger, f"demo_cfg_{cfg_scale}", filename, self.sample_rate)
                log_image(
                    trainer.logger,
                    f"demo_melspec_cfg_{cfg_scale}",
                    audio_spectrogram_image(fakes_out),
                )

        finally:
            module.train()