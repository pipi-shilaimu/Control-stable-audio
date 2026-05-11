from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable, NamedTuple, cast

import torch

# Allow script execution without installing local package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_audio_control.audio_io import install_torchaudio_load_fallback  # noqa: E402
from stable_audio_control.inference.control_compare import (  # noqa: E402
    audio_sample_size_from_seconds,
    compute_audio_difference_stats,
    initialize_lazy_control_modules,
    load_control_checkpoint,
    load_reference_audio,
    save_audio_tensor,
)
from stable_audio_control.inference.melody_similarity import (  # noqa: E402
    SUPPORTED_AUDIO_EXTENSIONS,
    compare_audio_melody_similarity,
    write_similarity_metadata,
)
from stable_audio_control.melody.extractors import build_melody_extractor, melody_control_channels  # noqa: E402
from stable_audio_control.models import ControlConditionedDiffusionWrapper, build_control_wrapper  # noqa: E402
from stable_audio_tools import get_pretrained_model  # noqa: E402
from stable_audio_tools.inference.generation import generate_diffusion_cond  # noqa: E402
from stable_audio_tools.models.diffusion import ConditionedDiffusionModelWrapper  # noqa: E402


install_torchaudio_load_fallback()


class RandomGenerationPlanItem(NamedTuple):
    index: int
    reference_audio_path: Path
    seed: int


_REFERENCE_PROMPT_MANIFEST_CACHE: dict[Path, dict[str, Any]] = {}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate multiple ControlNet outputs from random reference audios and score melody similarity."
    )
    parser.add_argument("--ckpt-path", type=str, required=True, help="Lightning checkpoint from train_controlnet_dit.py.")
    parser.add_argument(
        "--reference-root",
        type=str,
        default="stable_audio_control/data",
        help="Root directory scanned recursively for reference audio files.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Fallback text prompt. If omitted or blank, each reference audio must provide a manifest prompt.",
    )
    parser.add_argument(
        "--use-reference-prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use each reference audio's manifest prompt instead of the shared --prompt.",
    )
    parser.add_argument(
        "--reference-prompt-manifest",
        type=str,
        default=None,
        help="Optional JSON manifest used to resolve reference prompts. Defaults to sibling manifests/<split>.json.",
    )
    parser.add_argument("--negative-prompt", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/controlnet_generation_random")
    parser.add_argument("--summary-name", type=str, default="summary.json")
    parser.add_argument("--reference-output-name", type=str, default="reference.wav")
    parser.add_argument("--control-output-name", type=str, default="control.wav")
    parser.add_argument("--control-bypass-output-name", type=str, default="control_bypass.wav")
    parser.add_argument("--similarity-name", type=str, default="similarity.json")

    parser.add_argument("--model-name", type=str, default="stabilityai/stable-audio-open-1.0")
    parser.add_argument("--random-seed", type=int, default=0, help="Controls both reference sampling and generated seeds.")
    parser.add_argument(
        "--fixed-seed",
        type=int,
        default=None,
        help="If set, use the same generation seed for every sample while --random-seed still selects references.",
    )
    parser.add_argument(
        "--compare-base",
        action="store_true",
        help="Also generate a pure base reference for each sample and print waveform diff against the control output.",
    )
    parser.add_argument(
        "--compare-control-bypass",
        action="store_true",
        help=(
            "Also generate with the same loaded ControlNet model but without passing control_input/control_scale. "
            "When --control-scale 0 is transparent, this should match the control output closely."
        ),
    )
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed-min", type=int, default=0)
    parser.add_argument("--seed-max", type=int, default=2**31 - 1)
    parser.add_argument(
        "--allow-reference-reuse",
        action="store_true",
        help="Allow the same reference audio to appear in multiple samples.",
    )

    parser.add_argument("--seconds-start", type=float, default=0.0)
    parser.add_argument("--seconds-total", type=float, default=5.0)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Audio sample count. Defaults to seconds_total * sample_rate rounded to min_input_length.",
    )
    parser.add_argument("--steps", type=int, default=8, help="Low default keeps the batch comparison relatively quick.")
    parser.add_argument("--cfg-scale", type=float, default=7.0)
    parser.add_argument("--sampler-type", type=str, default="dpmpp-3m-sde")
    parser.add_argument("--sigma-min", type=float, default=0.3)
    parser.add_argument("--sigma-max", type=float, default=500.0)
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cuda', or 'cpu'.")
    parser.add_argument("--model-half", action="store_true", help="Move models to float16 after loading.")

    # ControlNet args must match the training run.
    parser.add_argument("--num-control-layers", type=int, default=12)
    parser.add_argument("--control-id", type=str, default="melody_control")
    parser.add_argument("--control-scale", type=float, default=1.0)
    parser.add_argument("--prefer-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--melody-embedding-dim", type=int, default=64)
    parser.add_argument("--melody-hidden-dim", type=int, default=256)
    parser.add_argument("--melody-conv-layers", type=int, default=2)

    # Melody feature args must match the training run.
    parser.add_argument(
        "--melody-feature",
        type=str,
        choices=["cqt", "chromagram"],
        default="cqt",
        help="Melody control feature extractor used by the checkpoint.",
    )

    # CQT args must match the training run.
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--n-bins", type=int, default=128)
    parser.add_argument("--bins-per-octave", type=int, default=12)
    parser.add_argument("--fmin-hz", type=float, default=8.175798915643707)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--highpass-cutoff-hz", type=float, default=261.2)
    parser.add_argument("--cqt-backend", type=str, choices=["auto", "nnaudio", "librosa"], default="auto")

    # Chromagram args must match the training run.
    parser.add_argument("--chroma-bins", type=int, default=12)
    parser.add_argument("--chroma-n-fft", type=int, default=2048)

    return parser


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _conditioning(prompt: str, seconds_start: float, seconds_total: float) -> list[dict[str, Any]]:
    return [
        {
            "prompt": prompt,
            "seconds_start": float(seconds_start),
            "seconds_total": float(seconds_total),
        }
    ]


def _normalized_optional_prompt(prompt: str | None) -> str | None:
    if prompt is None:
        return None
    prompt = prompt.strip()
    return prompt if prompt else None


def _reference_prompt_mode_requested(args: argparse.Namespace) -> bool:
    return bool(args.use_reference_prompt) or _normalized_optional_prompt(args.prompt) is None


def _negative_conditioning(args: argparse.Namespace) -> list[dict[str, Any]] | None:
    if not args.negative_prompt:
        return None
    return _conditioning(args.negative_prompt, args.seconds_start, args.seconds_total)


def _required_argument_prompt(args: argparse.Namespace) -> str:
    prompt = _normalized_optional_prompt(args.prompt)
    if prompt is None:
        raise ValueError("No shared --prompt is available for this generation call.")
    return prompt


def _prepare_model(model: torch.nn.Module, *, device: torch.device, model_half: bool) -> torch.nn.Module:
    model = model.to(device).eval().requires_grad_(False)
    if model_half:
        if device.type == "cpu":
            print("model_half requested on CPU; keeping float32.")
        else:
            model = model.to(torch.float16)
    return model


@torch.no_grad()
def _generate(
    model: torch.nn.Module,
    args: argparse.Namespace,
    *,
    seed: int,
    sample_size: int,
    device: torch.device,
    conditioning_tensors: dict[str, Any] | None = None,
    extra_sampler_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    kwargs = dict(extra_sampler_kwargs or {})
    return generate_diffusion_cond(
        model,
        steps=int(args.steps),
        cfg_scale=float(args.cfg_scale),
        conditioning=None
        if conditioning_tensors is not None
        else _conditioning(_required_argument_prompt(args), args.seconds_start, args.seconds_total),
        conditioning_tensors=conditioning_tensors,
        negative_conditioning=_negative_conditioning(args),
        batch_size=1,
        sample_size=int(sample_size),
        seed=int(seed),
        device=device,
        sampler_type=args.sampler_type,
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
        **kwargs,
    )


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def find_audio_files(root: str | Path, *, extensions: tuple[str, ...] = SUPPORTED_AUDIO_EXTENSIONS) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Reference root not found: {root_path}")

    normalized_extensions = tuple(ext.lower() for ext in extensions)
    if root_path.is_file():
        return [root_path] if root_path.suffix.lower() in normalized_extensions else []

    files = [
        path
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() in normalized_extensions
    ]
    return sorted(files, key=lambda path: path.resolve().as_posix().lower())


def _candidate_reference_prompt_manifests(
    audio_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> list[Path]:
    if manifest_path is not None:
        return [Path(manifest_path)]

    audio_path = Path(audio_path)
    split_dir = audio_path.parent
    return [split_dir.parent / "manifests" / f"{split_dir.name}.json"]


def _load_reference_prompt_manifest(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if resolved not in _REFERENCE_PROMPT_MANIFEST_CACHE:
        _REFERENCE_PROMPT_MANIFEST_CACHE[resolved] = json.loads(resolved.read_text(encoding="utf-8"))
    return _REFERENCE_PROMPT_MANIFEST_CACHE[resolved]


def _reference_prompt_manifest_keys(audio_path: str | Path, manifest_path: str | Path) -> list[str]:
    audio_path = Path(audio_path)
    manifest_path = Path(manifest_path)
    keys = [audio_path.name]

    for root in (audio_path.parent, audio_path.parent.parent, manifest_path.parent.parent):
        try:
            keys.append(audio_path.relative_to(root).as_posix())
        except ValueError:
            continue

    deduped: list[str] = []
    for key in keys:
        normalized = key.replace("\\", "/").lstrip("./")
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped


def load_reference_prompt(
    audio_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> str | None:
    for candidate in _candidate_reference_prompt_manifests(audio_path, manifest_path=manifest_path):
        if not candidate.exists():
            continue

        manifest = _load_reference_prompt_manifest(candidate)
        for key in _reference_prompt_manifest_keys(audio_path, candidate):
            entry = manifest.get(key)
            if entry is None:
                continue

            prompt = entry.get("prompt") if isinstance(entry, dict) else str(entry)
            prompt = _normalized_optional_prompt(prompt)
            if prompt is not None:
                return prompt

    return None


def resolve_generation_prompt(args: argparse.Namespace, reference_audio_path: str | Path) -> tuple[str, str]:
    if _reference_prompt_mode_requested(args):
        prompt = load_reference_prompt(
            reference_audio_path,
            manifest_path=getattr(args, "reference_prompt_manifest", None),
        )
        if prompt is not None:
            return prompt, "reference"

    argument_prompt = _normalized_optional_prompt(args.prompt)
    if argument_prompt is not None and not bool(args.use_reference_prompt):
        return argument_prompt, "argument"

    raise ValueError(
        "No prompt available for reference audio "
        f"{Path(reference_audio_path)}. Provide --prompt, or make sure the audio has a manifest entry with "
        "`prompt` in a sibling manifests/<split>.json file, or pass --reference-prompt-manifest."
    )


def build_random_generation_plan(
    audio_paths: list[Path],
    *,
    num_samples: int,
    random_seed: int,
    seed_min: int,
    seed_max: int,
    allow_reference_reuse: bool,
    fixed_seed: int | None = None,
) -> list[RandomGenerationPlanItem]:
    if num_samples < 1:
        raise ValueError("num_samples must be greater than 0.")
    if not audio_paths:
        raise ValueError("No audio files were found under the reference root.")
    if not allow_reference_reuse and num_samples > len(audio_paths):
        raise ValueError(
            f"Requested {num_samples} samples but only {len(audio_paths)} unique audio files are available."
        )

    rng = random.Random(int(random_seed))
    selected_audio_paths = (
        rng.choices(audio_paths, k=num_samples)
        if allow_reference_reuse
        else rng.sample(audio_paths, k=num_samples)
    )
    if fixed_seed is not None:
        selected_seeds = [int(fixed_seed)] * num_samples
    else:
        if seed_max < seed_min:
            raise ValueError("seed_max must be greater than or equal to seed_min.")

        seed_space_size = int(seed_max) - int(seed_min) + 1
        if seed_space_size < num_samples:
            raise ValueError(
                f"Seed range [{seed_min}, {seed_max}] is too small for {num_samples} unique seeds."
            )

        selected_seeds = rng.sample(range(int(seed_min), int(seed_max) + 1), k=num_samples)
    return [
        RandomGenerationPlanItem(index=index, reference_audio_path=reference_path, seed=seed)
        for index, (reference_path, seed) in enumerate(zip(selected_audio_paths, selected_seeds))
    ]


def load_pretrained_model_with_context(
    model_name: str,
    *,
    loader: Callable[[str], tuple[torch.nn.Module, dict[str, Any]]] = get_pretrained_model,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    try:
        return loader(model_name)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to load the pretrained StableAudio model via "
            f"get_pretrained_model({model_name!r}). The failure happened while "
            "stable-audio-tools was building the text conditioner, which loads a "
            "T5 tokenizer/encoder through Hugging Face. This usually means the "
            "container cannot download Hugging Face files, or the HF cache is incomplete/corrupt.\n\n"
            "Try pre-caching the required assets inside the same container/environment before rerunning:\n"
            "  export HF_HOME=/outputs/hf_cache\n"
            "  python3 -c \"from transformers import AutoTokenizer, T5EncoderModel; "
            "AutoTokenizer.from_pretrained('t5-base'); "
            "T5EncoderModel.from_pretrained('t5-base'); "
            "from stable_audio_tools import get_pretrained_model; "
            f"get_pretrained_model('{model_name}')\"\n\n"
            "If the container is offline, populate HF_HOME from a connected environment first, then rerun "
            "with HF_HOME pointing at that cache."
        ) from exc


def _build_control_model(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[ControlConditionedDiffusionWrapper, int, bool, int]:
    print(f"loading base model: {args.model_name}")
    base_model, model_config = load_pretrained_model_with_context(args.model_name)
    base_model = cast(ConditionedDiffusionModelWrapper, base_model)

    sample_rate = int(model_config["sample_rate"])
    control_channels = melody_control_channels(
        args.melody_feature,
        top_k=int(args.top_k),
        chroma_bins=int(args.chroma_bins),
    )
    uses_discrete_melody = args.melody_feature == "cqt"
    control_model = build_control_wrapper(
        base_wrapper=base_model,
        num_control_layers=int(args.num_control_layers),
        control_id=args.control_id,
        default_control_scale=float(args.control_scale),
        freeze_base=True,
        melody_channels=control_channels,
        melody_num_pitch_bins=int(args.n_bins),
        melody_embedding_dim=int(args.melody_embedding_dim),
        melody_hidden_dim=int(args.melody_hidden_dim),
        melody_conv_layers=int(args.melody_conv_layers),
        use_melody_encoder=uses_discrete_melody,
    )
    control_model = cast(ControlConditionedDiffusionWrapper, control_model)

    initialize_lazy_control_modules(
        control_model,
        device=torch.device("cpu"),
        dtype=next(control_model.parameters()).dtype,
        control_channels=control_channels,
    )
    checkpoint_load = load_control_checkpoint(control_model, args.ckpt_path, prefer_ema=bool(args.prefer_ema))
    print(f"checkpoint loaded: use_ema={checkpoint_load['use_ema']}")

    control_model = cast(
        ControlConditionedDiffusionWrapper,
        _prepare_model(control_model, device=device, model_half=args.model_half),
    )
    return control_model, sample_rate, bool(checkpoint_load["use_ema"]), control_channels


def _build_base_model(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, int]:
    print(f"loading base model for comparison: {args.model_name}")
    base_model, model_config = load_pretrained_model_with_context(args.model_name)
    sample_rate = int(model_config["sample_rate"])
    base_model = cast(ConditionedDiffusionModelWrapper, _prepare_model(base_model, device=device, model_half=args.model_half))
    return base_model, sample_rate


def _build_similarity_kwargs(args: argparse.Namespace, *, sample_rate: int, sample_size: int) -> dict[str, Any]:
    return {
        "feature": args.melody_feature,
        "sample_rate": int(sample_rate),
        "sample_size": int(sample_size),
        "seconds_total": float(args.seconds_total),
        "top_k": int(args.top_k),
        "n_bins": int(args.n_bins),
        "bins_per_octave": int(args.bins_per_octave),
        "fmin_hz": float(args.fmin_hz),
        "hop_length": int(args.hop_length),
        "highpass_cutoff_hz": float(args.highpass_cutoff_hz),
        "cqt_backend": args.cqt_backend,
        "chroma_bins": int(args.chroma_bins),
        "chroma_n_fft": int(args.chroma_n_fft),
    }


def _build_control_config(args: argparse.Namespace, *, control_channels: int) -> dict[str, Any]:
    return {
        "melody_feature": args.melody_feature,
        "num_control_layers": int(args.num_control_layers),
        "control_id": args.control_id,
        "control_channels": int(control_channels),
        "top_k": int(args.top_k),
        "n_bins": int(args.n_bins),
        "bins_per_octave": int(args.bins_per_octave),
        "fmin_hz": float(args.fmin_hz),
        "hop_length": int(args.hop_length),
        "highpass_cutoff_hz": float(args.highpass_cutoff_hz),
        "cqt_backend": args.cqt_backend,
        "chroma_bins": int(args.chroma_bins),
        "chroma_n_fft": int(args.chroma_n_fft),
        "melody_embedding_dim": int(args.melody_embedding_dim),
        "melody_hidden_dim": int(args.melody_hidden_dim),
        "melody_conv_layers": int(args.melody_conv_layers),
    }


def validate_args(args: argparse.Namespace) -> None:
    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. "
            "Check that this path exists inside the current container/session. "
            "If the file is on the host or in another container, mount/copy it so the script can see it."
        )
    if args.sampler_type == "dpmpp-3m-sde" and int(args.steps) < 2:
        raise ValueError("dpmpp-3m-sde requires --steps >= 2; use a different sampler for single-step smoke tests.")


def _output_item_dir(output_dir: Path, *, template: str, item: RandomGenerationPlanItem, reference_name: str) -> Path:
    return output_dir / template.format(index=item.index, seed=item.seed, reference_name=reference_name)


def _summarize_scores(items: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(item["similarity"]["score"]) for item in items]
    metric_name = items[0]["similarity"]["metric_name"] if items else None
    return {
        "metric_name": metric_name,
        "mean_score": float(sum(scores) / len(scores)) if scores else None,
        "min_score": float(min(scores)) if scores else None,
        "max_score": float(max(scores)) if scores else None,
    }


def format_similarity_progress_line(
    *,
    index: int,
    total: int,
    similarity: dict[str, Any],
    output_path: Path,
) -> str:
    metric_name = similarity["metric_name"]
    score = float(similarity["score"])
    return (
        f"[{index + 1}/{total}] similarity_score={score:.6f} "
        f"metric={metric_name} metadata={output_path.as_posix()}"
    )


def format_similarity_summary_line(aggregate: dict[str, Any]) -> str:
    metric_name = aggregate["metric_name"]
    mean_score = aggregate["mean_score"]
    min_score = aggregate["min_score"]
    max_score = aggregate["max_score"]
    if mean_score is None or min_score is None or max_score is None:
        return f"similarity_summary metric={metric_name} mean=None min=None max=None"
    return (
        f"similarity_summary metric={metric_name} "
        f"mean={float(mean_score):.6f} min={float(min_score):.6f} max={float(max_score):.6f}"
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)
    device = _resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_files = find_audio_files(args.reference_root)
    plan = build_random_generation_plan(
        reference_files,
        num_samples=int(args.num_samples),
        random_seed=int(args.random_seed),
        seed_min=int(args.seed_min),
        seed_max=int(args.seed_max),
        allow_reference_reuse=bool(args.allow_reference_reuse),
        fixed_seed=args.fixed_seed,
    )
    prompt_plan = {
        item.index: resolve_generation_prompt(args, item.reference_audio_path)
        for item in plan
    }

    control_model, sample_rate, use_ema, control_channels = _build_control_model(args, device=device)
    sample_size = (
        int(args.sample_size)
        if args.sample_size is not None
        else audio_sample_size_from_seconds(
            seconds_total=float(args.seconds_total),
            sample_rate=sample_rate,
            min_input_length=int(control_model.min_input_length),
        )
    )
    print(f"device={device}")
    print(f"sample_rate={sample_rate}, sample_size={sample_size}, seconds_total={args.seconds_total}")
    print(f"reference_root={Path(args.reference_root).resolve()}")
    print(f"selected_references={len(plan)}")
    if args.fixed_seed is not None:
        print(f"fixed_seed={int(args.fixed_seed)}")
    if args.compare_base:
        print("compare_base=True")
    print(f"melody_feature={args.melody_feature}, control_channels={control_channels}")
    print(
        "prompt_mode="
        f"{'reference' if _reference_prompt_mode_requested(args) else 'argument'}"
    )

    extractor = build_melody_extractor(
        feature=args.melody_feature,
        sample_rate=sample_rate,
        fmin_hz=float(args.fmin_hz),
        highpass_cutoff_hz=float(args.highpass_cutoff_hz),
        n_bins=int(args.n_bins),
        bins_per_octave=int(args.bins_per_octave),
        hop_length=int(args.hop_length),
        top_k=int(args.top_k),
        cqt_backend=args.cqt_backend,
        chroma_bins=int(args.chroma_bins),
        chroma_n_fft=int(args.chroma_n_fft),
    )

    latent_sample_size = sample_size
    if control_model.pretransform is not None:
        latent_sample_size = sample_size // int(control_model.pretransform.downsampling_ratio)

    base_model = None
    if args.compare_base:
        base_model, _base_sample_rate = _build_base_model(args, device=device)
        if int(_base_sample_rate) != int(sample_rate):
            raise RuntimeError(
                f"Base model sample_rate {_base_sample_rate} does not match control model sample_rate {sample_rate}."
            )

    items: list[dict[str, Any]] = []
    for item in plan:
        reference_name = item.reference_audio_path.stem
        item_dir = _output_item_dir(output_dir, template="{index:02d}_seed-{seed}_{reference_name}", item=item, reference_name=reference_name)
        item_dir.mkdir(parents=True, exist_ok=True)

        reference_output_path = item_dir / args.reference_output_name
        control_output_path = item_dir / args.control_output_name
        control_bypass_output_path = item_dir / args.control_bypass_output_name
        similarity_output_path = item_dir / args.similarity_name
        item_prompt, item_prompt_source = prompt_plan[item.index]

        print(
            f"[{item.index + 1}/{len(plan)}] reference={item.reference_audio_path.name} "
            f"seed={item.seed} prompt_source={item_prompt_source} -> {item_dir}"
        )

        reference_audio = load_reference_audio(
            item.reference_audio_path,
            target_sample_rate=sample_rate,
            target_sample_size=sample_size,
            device=device,
        )
        save_audio_tensor(reference_output_path, reference_audio, sample_rate)

        melody_control = extractor.extract(reference_audio).to(device=device)
        control_input = control_model._extract_control_input(  # type: ignore[attr-defined]
            cond={args.control_id: [melody_control, None]},
            target_len=int(latent_sample_size),
            dtype=next(control_model.model.parameters()).dtype,
            device=device,
        )
        if control_input is None:
            raise RuntimeError("Failed to build control_input from reference audio.")

        conditioning_tensors = control_model.conditioner(
            _conditioning(item_prompt, args.seconds_start, args.seconds_total),
            device,
        )

        control_audio = _generate(
            control_model,
            args,
            seed=int(item.seed),
            sample_size=sample_size,
            device=device,
            conditioning_tensors=conditioning_tensors,
            extra_sampler_kwargs={
                "control_input": control_input,
                "control_scale": float(args.control_scale),
            },
        )
        save_audio_tensor(control_output_path, control_audio, sample_rate)

        control_bypass_audio_difference = None
        control_bypass_output_resolved = None
        if args.compare_control_bypass:
            control_bypass_audio = _generate(
                control_model,
                args,
                seed=int(item.seed),
                sample_size=sample_size,
                device=device,
                conditioning_tensors=conditioning_tensors,
            )
            save_audio_tensor(control_bypass_output_path, control_bypass_audio, sample_rate)
            control_bypass_output_resolved = str(control_bypass_output_path.resolve())
            control_bypass_audio_difference = compute_audio_difference_stats(control_bypass_audio, control_audio)
            print(
                "control_bypass_vs_control_audio_difference "
                f"max_abs={control_bypass_audio_difference['max_abs_diff']:.6f} "
                f"mean_abs={control_bypass_audio_difference['mean_abs_diff']:.6f} "
                f"rms={control_bypass_audio_difference['rms_diff']:.6f} "
                f"relative_rms={control_bypass_audio_difference['relative_rms_diff']:.6f} "
                f"corr={control_bypass_audio_difference['channel_correlation']}"
            )

        audio_difference = None
        if base_model is not None:
            base_conditioning_tensors = base_model.conditioner(
                _conditioning(item_prompt, args.seconds_start, args.seconds_total),
                device,
            )
            base_audio = _generate(
                base_model,
                args,
                seed=int(item.seed),
                sample_size=sample_size,
                device=device,
                conditioning_tensors=base_conditioning_tensors,
            )
            audio_difference = compute_audio_difference_stats(base_audio, control_audio)
            print(
                "base_vs_control_audio_difference "
                f"max_abs={audio_difference['max_abs_diff']:.6f} "
                f"mean_abs={audio_difference['mean_abs_diff']:.6f} "
                f"rms={audio_difference['rms_diff']:.6f} "
                f"relative_rms={audio_difference['relative_rms_diff']:.6f} "
                f"corr={audio_difference['channel_correlation']}"
            )

        similarity = compare_audio_melody_similarity(
            reference_output_path,
            control_output_path,
            **_build_similarity_kwargs(args, sample_rate=sample_rate, sample_size=sample_size),
        )
        write_similarity_metadata(similarity_output_path, similarity)
        print(
            format_similarity_progress_line(
                index=item.index,
                total=len(plan),
                similarity=similarity["similarity"],
                output_path=similarity_output_path,
            )
        )

        items.append(
            {
                "index": int(item.index),
                "seed": int(item.seed),
                "reference_audio_source": str(item.reference_audio_path.resolve()),
                "reference_audio_output": str(reference_output_path.resolve()),
                "control_output": str(control_output_path.resolve()),
                "control_bypass_output": control_bypass_output_resolved,
                "similarity_output": str(similarity_output_path.resolve()),
                "prompt": item_prompt,
                "prompt_source": item_prompt_source,
                "similarity": similarity["similarity"],
                "feature": similarity["feature"],
                "alignment": similarity["alignment"],
                "reference_melody_control_shape": list(melody_control.shape),
                "control_input_shape": list(control_input.shape),
                "audio_difference": audio_difference,
                "base_vs_control_audio_difference": audio_difference,
                "control_bypass_vs_control_audio_difference": control_bypass_audio_difference,
            }
        )

        del reference_audio
        del melody_control
        del control_input
        del conditioning_tensors
        del control_audio
        if args.compare_control_bypass:
            del control_bypass_audio
        if base_model is not None:
            del base_conditioning_tensors
            del base_audio
        _empty_cuda_cache()

    aggregate = _summarize_scores(items)
    summary = {
        "schema_version": 1,
        "model_name": args.model_name,
        "prompt": {
            "argument": _normalized_optional_prompt(args.prompt),
            "mode": "reference" if _reference_prompt_mode_requested(args) else "argument",
            "reference_prompt_manifest": args.reference_prompt_manifest,
            "use_reference_prompt": bool(args.use_reference_prompt),
        },
        "paths": {
            "checkpoint": str(Path(args.ckpt_path).resolve()),
            "reference_root": str(Path(args.reference_root).resolve()),
            "output_dir": str(output_dir.resolve()),
        },
        "selection": {
            "random_seed": int(args.random_seed),
            "num_samples": int(args.num_samples),
            "seed_min": int(args.seed_min),
            "seed_max": int(args.seed_max),
            "allow_reference_reuse": bool(args.allow_reference_reuse),
        },
        "generation": {
            "seconds_start": float(args.seconds_start),
            "seconds_total": float(args.seconds_total),
            "sample_size": int(sample_size),
            "steps": int(args.steps),
            "cfg_scale": float(args.cfg_scale),
            "sampler_type": args.sampler_type,
            "sigma_min": float(args.sigma_min),
            "sigma_max": float(args.sigma_max),
            "seed_policy": "fixed" if args.fixed_seed is not None else "random_unique",
            "fixed_seed": None if args.fixed_seed is None else int(args.fixed_seed),
            "compare_base": bool(args.compare_base),
            "compare_control_bypass": bool(args.compare_control_bypass),
        },
        "control": {
            "scale": float(args.control_scale),
            "use_ema": bool(use_ema),
            "config": _build_control_config(args, control_channels=control_channels),
        },
        "similarity": {
            "config": _build_similarity_kwargs(args, sample_rate=sample_rate, sample_size=sample_size),
            "aggregate": aggregate,
        },
        "items": items,
    }

    summary_path = output_dir / args.summary_name
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(format_similarity_summary_line(aggregate))
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    raise SystemExit(main())
