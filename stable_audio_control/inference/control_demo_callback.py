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
import math
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
DEFAULT_CONTROL_VARIANTS = ["correct", "shuffled", "zero", "null"]
_CONTROL_SCALES_UNSET = object()
CONTROL_VARIANT_ALIASES = {
    "correct": "correct",
    "shuffled": "shuffled",
    "shuffle": "shuffled",
    "zero": "zero",
    "zero_audio": "zero",
    "null": "null",
    "disabled": "null",
    "none": "null",
}
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
DEMO_CONTROL_DIAGNOSTIC_FIELDS = [
    "step",
    "demo_index",
    "cfg_scale",
    "control_scale",
    "variant",
    "row_type",
    "variant_a",
    "variant_b",
    "control_input_rms",
    "control_input_mean_abs",
    "control_input_zero_ratio",
    "forward_delta_rms",
    "forward_delta_mean_abs",
    "forward_delta_zero_ratio",
    "forward_delta_frame_active_ratio",
    "forward_delta_cosine",
    "forward_delta_relative_l2_difference",
    "audio_rms",
    "audio_peak",
    "audio_silence_ratio",
    "low_band_ratio",
    "mid_band_ratio",
    "high_band_ratio",
    "low_mid_band_ratio",
    "spectral_centroid_hz",
    "warning",
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


def normalize_control_variant(value: str) -> str:
    variant = value.strip().lower()
    if variant not in CONTROL_VARIANT_ALIASES:
        expected = ", ".join(sorted(CONTROL_VARIANT_ALIASES))
        raise ValueError(f"Unknown control variant: {value}. Expected one of: {expected}.")
    return CONTROL_VARIANT_ALIASES[variant]


def _safe_float(value: torch.Tensor | float | int) -> float:
    result = float(value.item() if torch.is_tensor(value) else value)
    if not math.isfinite(result):
        return 0.0
    return result


def _tensor_zero_ratio(tensor: torch.Tensor, *, epsilon: float = 1e-12) -> float:
    flat = tensor.detach().to(torch.float32).reshape(-1)
    if flat.numel() == 0:
        return 0.0
    return _safe_float((flat.abs() <= float(epsilon)).to(torch.float32).mean())


def _tensor_rms(tensor: torch.Tensor) -> float:
    flat = tensor.detach().to(torch.float32).reshape(-1)
    if flat.numel() == 0:
        return 0.0
    return _safe_float(flat.square().mean().sqrt())


def _tensor_mean_abs(tensor: torch.Tensor) -> float:
    flat = tensor.detach().to(torch.float32).reshape(-1)
    if flat.numel() == 0:
        return 0.0
    return _safe_float(flat.abs().mean())


def _frame_active_ratio(delta: torch.Tensor, *, epsilon: float = 1e-8) -> float:
    data = delta.detach().to(torch.float32)
    if data.ndim < 2:
        return 0.0
    frame_dim = data.ndim - 1
    reduce_dims = tuple(dim for dim in range(data.ndim) if dim != frame_dim)
    frame_rms = data.square().mean(dim=reduce_dims).sqrt()
    return _safe_float((frame_rms > float(epsilon)).to(torch.float32).mean())


def compute_pairwise_delta_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a_flat = a.detach().to(torch.float32).reshape(-1)
    b_flat = b.detach().to(torch.float32).reshape(-1)
    if a_flat.numel() != b_flat.numel():
        raise ValueError(f"Delta tensors must have the same number of values: {a_flat.numel()} vs {b_flat.numel()}")
    diff = a_flat - b_flat
    norm_a = _safe_float(a_flat.square().sum().sqrt())
    norm_b = _safe_float(b_flat.square().sum().sqrt())
    denom = max(norm_a * norm_b, 1e-12)
    cosine = _safe_float((a_flat * b_flat).sum()) / denom
    return {
        "cosine": cosine,
        "relative_l2_difference": _safe_float(diff.square().sum().sqrt()) / max(norm_a, norm_b, 1e-12),
    }


def _normalize_audio_for_stats(audio: torch.Tensor) -> torch.Tensor:
    data = audio.detach().to(torch.float32).cpu()
    if data.ndim == 3:
        if data.shape[0] != 1:
            data = data[:1]
        data = data[0]
    if data.ndim == 1:
        data = data.unsqueeze(0)
    if data.ndim != 2:
        raise ValueError(f"audio must be [T], [C,T], or [1,C,T]; got shape={tuple(data.shape)}")
    return data.mean(dim=0)


def normalize_audio_for_wav(audio: torch.Tensor, *, epsilon: float = 1e-12) -> torch.Tensor:
    """Peak-normalize demo audio for int16 WAV writing without turning silence into NaNs."""

    data = torch.nan_to_num(audio.detach().to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    peak = _safe_float(data.abs().max()) if data.numel() > 0 else 0.0
    if peak <= float(epsilon):
        return torch.zeros_like(data, dtype=torch.int16).cpu()
    return data.div(peak).clamp(-1.0, 1.0).mul(32767).to(torch.int16).cpu()


def compute_audio_spectral_stats(audio: torch.Tensor, *, sample_rate: int) -> dict[str, float]:
    mono = _normalize_audio_for_stats(audio)
    if mono.numel() == 0:
        return {
            "audio_rms": 0.0,
            "audio_peak": 0.0,
            "audio_silence_ratio": 1.0,
            "low_band_ratio": 0.0,
            "mid_band_ratio": 0.0,
            "high_band_ratio": 0.0,
            "low_mid_band_ratio": 0.0,
            "spectral_centroid_hz": 0.0,
        }

    mono = mono - mono.mean()
    rms = _tensor_rms(mono)
    peak = _safe_float(mono.abs().max())
    frame_size = min(2048, max(16, int(mono.numel())))
    hop = max(1, frame_size // 2)
    if mono.numel() >= frame_size:
        frames = mono.unfold(0, frame_size, hop)
        frame_rms = frames.square().mean(dim=-1).sqrt()
        silence_ratio = _safe_float((frame_rms <= max(1e-5, rms * 0.02)).to(torch.float32).mean())
    else:
        silence_ratio = 1.0 if rms <= 1e-5 else 0.0

    spectrum = torch.fft.rfft(mono)
    power = spectrum.abs().square()
    freqs = torch.fft.rfftfreq(mono.numel(), d=1.0 / int(sample_rate))
    total_power = max(_safe_float(power.sum()), 1e-12)

    def band_ratio(low_hz: float, high_hz: float) -> float:
        mask = (freqs >= float(low_hz)) & (freqs < float(high_hz))
        if not bool(mask.any()):
            return 0.0
        return _safe_float(power[mask].sum()) / total_power

    low_ratio = band_ratio(0.0, 250.0)
    mid_ratio = band_ratio(250.0, 2000.0)
    high_ratio = band_ratio(2000.0, float(sample_rate) / 2.0 + 1.0)
    centroid = _safe_float((freqs * power).sum()) / total_power
    return {
        "audio_rms": rms,
        "audio_peak": peak,
        "audio_silence_ratio": silence_ratio,
        "low_band_ratio": low_ratio,
        "mid_band_ratio": mid_ratio,
        "high_band_ratio": high_ratio,
        "low_mid_band_ratio": low_ratio + mid_ratio,
        "spectral_centroid_hz": centroid,
    }


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
        demo_control_diagnostics: bool = True,
        demo_control_diagnostics_csv: str = "demo_control_diagnostics.csv",
        stop_on_collapse: bool = False,
        collapse_cosine_threshold: float = 0.98,
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
        self.control_variants = []
        for variant in list(control_variants or DEFAULT_CONTROL_VARIANTS):
            normalized = normalize_control_variant(variant)
            if normalized not in self.control_variants:
                self.control_variants.append(normalized)
        self.demo_control_audio_path = demo_control_audio_path
        self.demo_prompt = demo_prompt
        self.control_id = control_id
        self.demo_melody_similarity = bool(demo_melody_similarity)
        self.demo_melody_similarity_csv = demo_melody_similarity_csv
        self.demo_melody_similarity_feature = demo_melody_similarity_feature
        self.demo_melody_similarity_top_k = int(demo_melody_similarity_top_k)
        self.demo_control_diagnostics = bool(demo_control_diagnostics)
        self.demo_control_diagnostics_csv = demo_control_diagnostics_csv
        self.stop_on_collapse = bool(stop_on_collapse)
        self.collapse_cosine_threshold = float(collapse_cosine_threshold)

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

    def make_variant_reals(
        self,
        reals: torch.Tensor,
        variant: str,
        *,
        shuffle_by_time: bool = False,
    ) -> torch.Tensor | None:
        variant = normalize_control_variant(variant)
        if variant == "correct":
            return reals
        if variant == "shuffled":
            if reals.shape[0] > 1 and not shuffle_by_time:
                return torch.roll(reals, shifts=1, dims=0)
            return torch.flip(reals, dims=[-1])
        if variant == "zero":
            return torch.zeros_like(reals)
        if variant == "null":
            return None
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

    def _control_diagnostics_csv_path(self, default_root_dir: str | Path) -> Path:
        path = Path(self.demo_control_diagnostics_csv)
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

    def _write_control_diagnostic_row(self, path: Path, row: dict[str, tp.Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=DEMO_CONTROL_DIAGNOSTIC_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow({field: row.get(field, "") for field in DEMO_CONTROL_DIAGNOSTIC_FIELDS})

    def _build_conditioning_for_variant(
        self,
        *,
        diffusion,
        melody_augmenter,
        demo_cond,
        device: torch.device,
        variant: str,
    ) -> dict[str, tp.Any]:
        if normalize_control_variant(variant) == "null":
            base_conditioner = getattr(melody_augmenter, "base_conditioner", None)
            if base_conditioner is not None:
                conditioning = base_conditioner(demo_cond, device)
            else:
                conditioning = diffusion.conditioner(demo_cond, device)
            conditioning = dict(conditioning)
            conditioning.pop(self.control_id, None)
            return conditioning
        return diffusion.conditioner(demo_cond, device)

    def _extract_control_input_for_diagnostics(
        self,
        *,
        diffusion,
        conditioning: dict[str, tp.Any],
        target_len: int,
        device: torch.device,
        fallback_dtype: torch.dtype,
    ) -> torch.Tensor | None:
        extract = getattr(diffusion, "_extract_control_input", None)
        if extract is None:
            return None
        try:
            model_dtype = next(diffusion.model.parameters()).dtype
        except Exception:  # noqa: BLE001
            model_dtype = fallback_dtype
        return extract(
            cond=conditioning,
            target_len=int(target_len),
            dtype=model_dtype,
            device=device,
        )

    def _compute_forward_delta(
        self,
        *,
        diffusion,
        noise: torch.Tensor,
        control_input: torch.Tensor | None,
        control_scale: float,
    ) -> torch.Tensor | None:
        model = getattr(diffusion, "model", None)
        if model is None:
            return None
        x = noise.detach()
        try:
            model_dtype = next(model.parameters()).dtype
        except Exception:  # noqa: BLE001
            model_dtype = x.dtype
        x = x.to(dtype=model_dtype)
        t = torch.full((x.shape[0],), 0.5, device=x.device, dtype=model_dtype)
        kwargs: dict[str, tp.Any] = {
            "cfg_scale": 1.0,
            "cfg_dropout_prob": 0.0,
            "control_input": None if control_input is None else control_input.to(device=x.device, dtype=model_dtype),
            "use_checkpointing": False,
        }
        try:
            with torch.no_grad():
                base_output = model(x, t, control_scale=0.0, **kwargs)
                controlled_output = model(x, t, control_scale=float(control_scale), **kwargs)
        except TypeError:
            kwargs.pop("use_checkpointing", None)
            with torch.no_grad():
                base_output = model(x, t, control_scale=0.0, **kwargs)
                controlled_output = model(x, t, control_scale=float(control_scale), **kwargs)
        return (controlled_output.detach() - base_output.detach()).to(torch.float32).cpu()

    def _control_diagnostic_base_row(
        self,
        *,
        step: int,
        demo_index: int,
        cfg_scale: float,
        control_scale: float,
        variant: str,
        control_input: torch.Tensor | None,
        forward_delta: torch.Tensor | None,
        fake: torch.Tensor | None,
    ) -> dict[str, tp.Any]:
        row: dict[str, tp.Any] = {
            "step": int(step),
            "demo_index": int(demo_index),
            "cfg_scale": float(cfg_scale),
            "control_scale": float(control_scale),
            "variant": variant,
            "row_type": "variant",
        }
        if control_input is not None:
            row.update(
                {
                    "control_input_rms": _tensor_rms(control_input),
                    "control_input_mean_abs": _tensor_mean_abs(control_input),
                    "control_input_zero_ratio": _tensor_zero_ratio(control_input),
                }
            )
        if forward_delta is not None:
            row.update(
                {
                    "forward_delta_rms": _tensor_rms(forward_delta),
                    "forward_delta_mean_abs": _tensor_mean_abs(forward_delta),
                    "forward_delta_zero_ratio": _tensor_zero_ratio(forward_delta),
                    "forward_delta_frame_active_ratio": _frame_active_ratio(forward_delta),
                }
            )
        if fake is not None:
            row.update(compute_audio_spectral_stats(fake, sample_rate=int(self.sample_rate)))
        return row

    def _maybe_write_pairwise_collapse_row(
        self,
        *,
        trainer,
        cfg_scale: float,
        control_scale: float,
        step: int,
        deltas_for_combo: dict[str, torch.Tensor],
    ) -> None:
        correct = deltas_for_combo.get("correct")
        shuffled = deltas_for_combo.get("shuffled")
        if correct is None or shuffled is None:
            return
        stats = compute_pairwise_delta_stats(correct, shuffled)
        warning = ""
        if float(control_scale) > 0.0 and stats["cosine"] >= self.collapse_cosine_threshold:
            warning = (
                "correct/shuffled forward_delta collapse: "
                f"cosine={stats['cosine']:.4f} >= {self.collapse_cosine_threshold:.4f}"
            )
            print(f"[ControlNetDemoDiagnostics] WARNING step={step} {warning}")
            if self.stop_on_collapse:
                trainer.should_stop = True
        row = {
            "step": int(step),
            "demo_index": "",
            "cfg_scale": float(cfg_scale),
            "control_scale": float(control_scale),
            "variant": "correct_vs_shuffled",
            "row_type": "pairwise",
            "variant_a": "correct",
            "variant_b": "shuffled",
            "forward_delta_cosine": stats["cosine"],
            "forward_delta_relative_l2_difference": stats["relative_l2_difference"],
            "warning": warning,
        }
        self._write_control_diagnostic_row(
            self._control_diagnostics_csv_path(trainer.default_root_dir),
            row,
        )

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
        metric_prefix: str = "",
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
            metrics[f"{prefix}/{metric_prefix}score"] = float(score)
        if cqt_top1_score is not None:
            metrics[f"{prefix}/{metric_prefix}top1"] = float(cqt_top1_score)
        if cqt_topk_score is not None:
            metrics[f"{prefix}/{metric_prefix}topk"] = float(cqt_topk_score)
        if metrics:
            log_metrics(metrics, step=step)

    def _score_and_write_similarity_row(
        self,
        *,
        trainer,
        extractor,
        base_row: dict[str, tp.Any],
        reference_reals: torch.Tensor,
        fake: torch.Tensor,
        demo_index: int,
        metric_name_prefix: str = "",
        log_metric_prefix: str = "",
    ) -> tuple[float | None, float | None, float | None]:
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
        row = dict(base_row)
        row.update(
            {
                "metric_name": f"{metric_name_prefix}{similarity['metric_name']}",
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
            cfg_scale=float(base_row["cfg_scale"]),
            control_scale=float(base_row["control_scale"]),
            variant=str(base_row["variant"]),
            step=int(base_row["step"]),
            score=score,
            cqt_top1_score=cqt_top1_score,
            cqt_topk_score=cqt_topk_score,
            metric_prefix=log_metric_prefix,
        )
        return score, cqt_top1_score, cqt_topk_score

    def score_demo_melody_similarity(
        self,
        *,
        trainer,
        melody_augmenter,
        reference_reals: torch.Tensor | None,
        original_reference_reals: torch.Tensor | None,
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

        base_row: dict[str, tp.Any] = {
            "step": int(step),
            "demo_index": int(demo_index),
            "cfg_scale": float(cfg_scale),
            "control_scale": float(control_scale),
            "variant": variant,
            "audio_path": audio_path,
        }

        extractor = getattr(melody_augmenter, "extractor", None)

        if variant == "zero":
            row = dict(base_row)
            row["skipped_reason"] = "zero_control_has_no_reference_melody"
            self._write_similarity_row(self._similarity_csv_path(trainer.default_root_dir), row)
            print(
                f"[ControlNetDemoSimilarity] step={step} cfg={cfg_scale} "
                f"control={control_scale} variant={variant} skipped={row['skipped_reason']}"
            )
        else:
            if reference_reals is None or extractor is None:
                row = dict(base_row)
                row["skipped_reason"] = "missing_reference_or_extractor"
                self._write_similarity_row(self._similarity_csv_path(trainer.default_root_dir), row)
            else:
                try:
                    _, cqt_top1_score, cqt_topk_score = self._score_and_write_similarity_row(
                        trainer=trainer,
                        extractor=extractor,
                        base_row=base_row,
                        reference_reals=reference_reals,
                        fake=fake,
                        demo_index=demo_index,
                    )
                    print(
                        f"[ControlNetDemoSimilarity] step={step} cfg={cfg_scale} "
                        f"control={control_scale} variant={variant} self_ref "
                        f"top1={cqt_top1_score if cqt_top1_score is not None else 'n/a'} "
                        f"topk={cqt_topk_score if cqt_topk_score is not None else 'n/a'}"
                    )
                except Exception as exc:  # noqa: BLE001
                    row = dict(base_row)
                    row["skipped_reason"] = f"{type(exc).__name__}: {exc}"
                    self._write_similarity_row(self._similarity_csv_path(trainer.default_root_dir), row)
                    print(
                        f"[ControlNetDemoSimilarity] step={step} cfg={cfg_scale} "
                        f"control={control_scale} variant={variant} skipped={row['skipped_reason']}"
                    )

        if original_reference_reals is None or extractor is None:
            return

        try:
            _, cqt_top1_score, cqt_topk_score = self._score_and_write_similarity_row(
                trainer=trainer,
                extractor=extractor,
                base_row=base_row,
                reference_reals=original_reference_reals,
                fake=fake,
                demo_index=demo_index,
                metric_name_prefix="original_ref_",
                log_metric_prefix="original_ref_",
            )
            print(
                f"[ControlNetDemoSimilarity] step={step} cfg={cfg_scale} "
                f"control={control_scale} variant={variant} original_ref "
                f"top1={cqt_top1_score if cqt_top1_score is not None else 'n/a'} "
                f"topk={cqt_topk_score if cqt_topk_score is not None else 'n/a'}"
            )
        except Exception as exc:  # noqa: BLE001
            row = dict(base_row)
            row["metric_name"] = "original_ref"
            row["skipped_reason"] = f"{type(exc).__name__}: {exc}"
            self._write_similarity_row(self._similarity_csv_path(trainer.default_root_dir), row)
            print(
                f"[ControlNetDemoSimilarity] step={step} cfg={cfg_scale} "
                f"control={control_scale} variant={variant} original_ref skipped={row['skipped_reason']}"
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

        forward_deltas_by_combo: dict[tuple[float, float], dict[str, torch.Tensor]] = {}
        pairwise_written_combos: set[tuple[float, float]] = set()
        try:
            for cfg_scale, control_scale, variant in self.iter_control_demo_combinations():
                variant = normalize_control_variant(variant)
                print(f"[ControlNetDemo] cfg_scale={cfg_scale} control_scale={control_scale} variant={variant}")

                variant_reals = None
                control_reals = external_control_reals if external_control_reals is not None else demo_reals
                if melody_augmenter is not None:
                    variant_reals = self.make_variant_reals(
                        control_reals,
                        variant,
                        shuffle_by_time=external_control_reals is not None,
                    )
                    if variant_reals is not None:
                        melody_augmenter.set_batch_audio(variant_reals)

                with torch.cuda.amp.autocast():
                    conditioning = self._build_conditioning_for_variant(
                        diffusion=diffusion,
                        melody_augmenter=melody_augmenter,
                        demo_cond=demo_cond,
                        device=module.device,
                        variant=variant,
                    )

                control_input = None
                forward_delta = None
                if self.demo_control_diagnostics:
                    control_input = self._extract_control_input_for_diagnostics(
                        diffusion=diffusion,
                        conditioning=conditioning,
                        target_len=int(demo_samples),
                        device=module.device,
                        fallback_dtype=noise.dtype,
                    )
                    forward_delta = self._compute_forward_delta(
                        diffusion=diffusion,
                        noise=noise,
                        control_input=control_input,
                        control_scale=float(control_scale),
                    )
                    if forward_delta is not None:
                        combo_key = (float(cfg_scale), float(control_scale))
                        forward_deltas_by_combo.setdefault(combo_key, {})[variant] = forward_delta

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
                    fakes_out = normalize_audio_for_wav(fake)
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
                        original_reference_reals=control_reals,
                        fake=fake,
                        cfg_scale=cfg_scale,
                        control_scale=control_scale,
                        variant=variant,
                        step=trainer.global_step,
                        demo_index=demo_index,
                        audio_path=filename,
                    )
                    if self.demo_control_diagnostics:
                        row = self._control_diagnostic_base_row(
                            step=trainer.global_step,
                            demo_index=demo_index,
                            cfg_scale=float(cfg_scale),
                            control_scale=float(control_scale),
                            variant=variant,
                            control_input=control_input,
                            forward_delta=forward_delta,
                            fake=fake,
                        )
                        self._write_control_diagnostic_row(
                            self._control_diagnostics_csv_path(trainer.default_root_dir),
                            row,
                        )

                if self.demo_control_diagnostics:
                    combo_key = (float(cfg_scale), float(control_scale))
                    deltas_for_combo = forward_deltas_by_combo.get(combo_key, {})
                    should_write_pairwise = (
                        combo_key not in pairwise_written_combos
                        and "correct" in deltas_for_combo
                        and "shuffled" in deltas_for_combo
                    )
                else:
                    should_write_pairwise = False

                if should_write_pairwise:
                    self._maybe_write_pairwise_collapse_row(
                        trainer=trainer,
                        cfg_scale=float(cfg_scale),
                        control_scale=float(control_scale),
                        step=trainer.global_step,
                        deltas_for_combo=deltas_for_combo,
                    )
                    pairwise_written_combos.add(combo_key)

        finally:
            module.train()
