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

import csv
import typing as tp
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
import pytorch_lightning as pl

from stable_audio_tools.inference.sampling import sample
from stable_audio_tools.training.diffusion import DiffusionCondTrainingWrapper
from stable_audio_tools.training.utils import log_audio, log_image
from stable_audio_tools.interface.aeiou import audio_spectrogram_image
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from stable_audio_control.inference.melody_similarity import compare_audio_tensors_melody_similarity


DEFAULT_CONTROL_SCALES = [0.0, 0.1, 0.3, 0.6, 1.0]
DEFAULT_CONTROL_VARIANTS = ["correct", "shuffled", "zero"]
_CONTROL_SCALES_UNSET = object()
DEMO_MELODY_SIMILARITY_FIELDS = [
    "step",
    "demo_index",
    "cfg_scale",
    "control_scale",
    "variant",
    "audio_path",
    "metric_name",
    "score",
    "cqt_top1_score",
    "cqt_topk_score",
    "matched_tokens",
    "total_tokens",
    "compared_frames",
    "skipped_reason",
]


def prepare_demo_control_audio(
    audio: torch.Tensor,
    *,
    source_sample_rate: int,
    target_sample_rate: int,
    target_sample_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Normalize a user-provided control audio file to [1, 2, target_sample_size]."""

    audio = audio.to(torch.float32)
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.ndim != 2:
        raise ValueError(f"Expected control audio [C,T] or [T], got shape={tuple(audio.shape)}")

    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    if audio.shape[0] != 2:
        raise ValueError(f"Expected mono/stereo control audio, got channels={audio.shape[0]}")

    if int(source_sample_rate) != int(target_sample_rate):
        resampler = torchaudio.transforms.Resample(int(source_sample_rate), int(target_sample_rate))
        audio = resampler(audio)

    target_sample_size = max(1, int(target_sample_size))
    if audio.shape[-1] < target_sample_size:
        audio = torch.nn.functional.pad(audio, (0, target_sample_size - audio.shape[-1]))
    elif audio.shape[-1] > target_sample_size:
        audio = audio[..., :target_sample_size]

    return audio.unsqueeze(0).to(device=device, dtype=torch.float32)


def _format_scale(value: float | int) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "neg").replace(".", "p")


class ControlNetDemoCallback(pl.Callback):
    """每 demo_every 步生成 demo 音频，自动注入 ControlNet 旋律控制条件。"""

    def __init__(
        self,
        demo_every: int = 2000,
        num_demos: int = 1,
        sample_size: int = 65536,
        demo_steps: int = 250,
        sample_rate: int = 48000,
        demo_cfg_scales: tp.List[int] = [3, 6, 9],
        control_scale: tp.Optional[float] = None,
        control_scales: tp.Union[tp.Sequence[float], None, object] = _CONTROL_SCALES_UNSET,
        control_variants: tp.Optional[tp.Sequence[str]] = None,
        demo_control_audio_path: tp.Optional[str] = None,
        demo_prompt: tp.Optional[str] = None,
        control_id: str = "melody_control",
        demo_melody_similarity: bool = True,
        demo_melody_similarity_csv: str = "demo_melody_similarity.csv",
        demo_melody_similarity_feature: str = "cqt",
        demo_melody_similarity_top_k: int = 4,
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
        self.demo_control_audio_path = demo_control_audio_path
        self.demo_prompt = demo_prompt
        self.control_id = control_id
        self.demo_melody_similarity = bool(demo_melody_similarity)
        self.demo_melody_similarity_csv = demo_melody_similarity_csv
        self.demo_melody_similarity_feature = demo_melody_similarity_feature
        self.demo_melody_similarity_top_k = int(demo_melody_similarity_top_k)

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

    def make_variant_reals(self, reals: torch.Tensor, variant: str, *, shuffle_by_time: bool = False) -> torch.Tensor:
        if variant == "correct":
            return reals
        if variant == "shuffled":
            if reals.shape[0] > 1 and not shuffle_by_time:
                return torch.roll(reals, shifts=1, dims=0)
            return torch.flip(reals, dims=[-1])
        if variant == "zero":
            return torch.zeros_like(reals)
        raise ValueError(f"Unknown control variant: {variant}")

    def load_demo_control_audio(self, device: torch.device) -> torch.Tensor:
        if self.demo_control_audio_path is None:
            raise RuntimeError("load_demo_control_audio called without demo_control_audio_path.")
        audio, source_sample_rate = torchaudio.load(self.demo_control_audio_path)
        return prepare_demo_control_audio(
            audio,
            source_sample_rate=int(source_sample_rate),
            target_sample_rate=int(self.sample_rate),
            target_sample_size=int(self.demo_samples),
            device=device,
        )

    def _similarity_csv_path(self, default_root_dir: str | Path) -> Path:
        path = Path(self.demo_melody_similarity_csv)
        if path.is_absolute():
            return path
        return Path(default_root_dir) / path

    def _write_similarity_row(self, path: Path, row: dict[str, tp.Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=DEMO_MELODY_SIMILARITY_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow({field: row.get(field, "") for field in DEMO_MELODY_SIMILARITY_FIELDS})

    def _log_similarity_metrics(
        self,
        trainer,
        *,
        cfg_scale: float,
        control_scale: float,
        variant: str,
        step: int,
        score: float | None,
        cqt_top1_score: float | None,
        cqt_topk_score: float | None,
    ) -> None:
        logger = getattr(trainer, "logger", None)
        log_metrics = getattr(logger, "log_metrics", None)
        if log_metrics is None:
            return

        prefix = (
            "demo_melody_similarity/"
            f"cfg_{_format_scale(cfg_scale)}"
            f"_control_{_format_scale(control_scale)}"
            f"_{variant}"
        )
        metrics: dict[str, float] = {}
        if score is not None:
            metrics[f"{prefix}/score"] = float(score)
        if cqt_top1_score is not None:
            metrics[f"{prefix}/top1"] = float(cqt_top1_score)
        if cqt_topk_score is not None:
            metrics[f"{prefix}/topk"] = float(cqt_topk_score)
        if metrics:
            log_metrics(metrics, step=step)

    def score_demo_melody_similarity(
        self,
        *,
        trainer,
        melody_augmenter,
        reference_reals: torch.Tensor | None,
        fake: torch.Tensor,
        cfg_scale: float,
        control_scale: float,
        variant: str,
        step: int,
        demo_index: int,
        audio_path: str,
    ) -> None:
        if not self.demo_melody_similarity:
            return

        row: dict[str, tp.Any] = {
            "step": int(step),
            "demo_index": int(demo_index),
            "cfg_scale": float(cfg_scale),
            "control_scale": float(control_scale),
            "variant": variant,
            "audio_path": audio_path,
        }

        if variant == "zero":
            row["skipped_reason"] = "zero_control_has_no_reference_melody"
            self._write_similarity_row(self._similarity_csv_path(trainer.default_root_dir), row)
            print(
                f"[ControlNetDemoSimilarity] step={step} cfg={cfg_scale} "
                f"control={control_scale} variant={variant} skipped={row['skipped_reason']}"
            )
            return

        extractor = getattr(melody_augmenter, "extractor", None)
        if reference_reals is None or extractor is None:
            row["skipped_reason"] = "missing_reference_or_extractor"
            self._write_similarity_row(self._similarity_csv_path(trainer.default_root_dir), row)
            return

        try:
            reference_audio = reference_reals[demo_index : demo_index + 1]
            generated_audio = fake.unsqueeze(0) if fake.ndim == 2 else fake
            metadata = compare_audio_tensors_melody_similarity(
                reference_audio,
                generated_audio,
                extractor=extractor,
                feature=tp.cast(tp.Any, self.demo_melody_similarity_feature),
                sample_rate=int(self.sample_rate),
                sample_size=int(self.demo_samples),
                top_k=int(self.demo_melody_similarity_top_k),
            )
            similarity = metadata["similarity"]
            score = float(similarity["score"])
            row.update(
                {
                    "metric_name": similarity["metric_name"],
                    "score": score,
                    "matched_tokens": similarity.get("matched_tokens", ""),
                    "total_tokens": similarity.get("total_tokens", ""),
                    "compared_frames": similarity.get("compared_frames", ""),
                }
            )

            cqt_top1_score = None
            cqt_topk_score = score if similarity["metric_name"] == "cqt_topk_pitch_overlap_rate" else None
            row["cqt_topk_score"] = "" if cqt_topk_score is None else cqt_topk_score
            additional = similarity.get("additional_metrics", {})
            cqt_top1 = additional.get("cqt_top1_accuracy") if isinstance(additional, dict) else None
            if isinstance(cqt_top1, dict):
                cqt_top1_score = float(cqt_top1["score"])
                row["cqt_top1_score"] = cqt_top1_score

            self._write_similarity_row(self._similarity_csv_path(trainer.default_root_dir), row)
            self._log_similarity_metrics(
                trainer,
                cfg_scale=cfg_scale,
                control_scale=control_scale,
                variant=variant,
                step=step,
                score=score,
                cqt_top1_score=cqt_top1_score,
                cqt_topk_score=cqt_topk_score,
            )
            print(
                f"[ControlNetDemoSimilarity] step={step} cfg={cfg_scale} "
                f"control={control_scale} variant={variant} "
                f"top1={cqt_top1_score if cqt_top1_score is not None else 'n/a'} "
                f"topk={cqt_topk_score if cqt_topk_score is not None else 'n/a'}"
            )
        except Exception as exc:  # noqa: BLE001
            row["skipped_reason"] = f"{type(exc).__name__}: {exc}"
            self._write_similarity_row(self._similarity_csv_path(trainer.default_root_dir), row)
            print(
                f"[ControlNetDemoSimilarity] step={step} cfg={cfg_scale} "
                f"control={control_scale} variant={variant} skipped={row['skipped_reason']}"
            )

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
            demo_cond = [
                dict(item) if isinstance(item, dict) else item
                for item in metadata[:demo_count]
            ]
        else:
            demo_count = min(self.num_demos, reals.shape[0])
            demo_cond = [
                dict(metadata) if isinstance(metadata, dict) else metadata
                for _ in range(demo_count)
            ]

        if demo_count <= 0:
            print("[ControlNetDemo] Skipping demo: empty batch.")
            module.train()
            return

        demo_reals = reals[:demo_count]
        external_control_reals = None
        if self.demo_control_audio_path is not None:
            external_control_reals = self.load_demo_control_audio(module.device).repeat(demo_count, 1, 1)

        # Normalize metadata: unconditionally set prompt + wrap padding_mask in list

        for item in demo_cond:
            try:
                item["prompt"] = str(self.demo_prompt if self.demo_prompt is not None else item.get("prompt", "music"))
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

                variant_reals = None
                if melody_augmenter is not None:
                    control_reals = external_control_reals if external_control_reals is not None else demo_reals
                    variant_reals = self.make_variant_reals(
                        control_reals,
                        variant,
                        shuffle_by_time=external_control_reals is not None,
                    )
                    melody_augmenter.set_batch_audio(variant_reals)

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
                stem = self.format_demo_stem(cfg_scale, control_scale, variant, trainer.global_step)
                audio_tag = self.format_audio_tag(cfg_scale, control_scale, variant, trainer.global_step)
                melspec_tag = self.format_melspec_tag(cfg_scale, control_scale, variant, trainer.global_step)

                for demo_index, fake in enumerate(fakes):
                    demo_suffix = f"_demo_{demo_index:02d}" if demo_count > 1 else ""
                    filename = f"{trainer.default_root_dir}/{stem}{demo_suffix}.wav"
                    fakes_out = (
                        fake.to(torch.float32)
                        .div(torch.max(torch.abs(fake)))
                        .mul(32767)
                        .to(torch.int16)
                        .cpu()
                    )
                    sf.write(filename, fakes_out.cpu().numpy().T, self.sample_rate)
                    log_audio(
                        trainer.logger,
                        f"{audio_tag}{demo_suffix}",
                        filename,
                        self.sample_rate,
                    )
                    log_image(
                        trainer.logger,
                        f"{melspec_tag}{demo_suffix}",
                        audio_spectrogram_image(fakes_out),
                    )
                    self.score_demo_melody_similarity(
                        trainer=trainer,
                        melody_augmenter=melody_augmenter,
                        reference_reals=variant_reals,
                        fake=fake,
                        cfg_scale=cfg_scale,
                        control_scale=control_scale,
                        variant=variant,
                        step=trainer.global_step,
                        demo_index=demo_index,
                        audio_path=filename,
                    )

        finally:
            module.train()
