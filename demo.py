from __future__ import annotations

from pathlib import Path

import torch

from stable_audio_control.inference.control_compare import audio_sample_size_from_seconds, save_audio_tensor
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond


MODEL_NAME = "stabilityai/stable-audio-open-1.0"
PROMPT = "Warm arpeggios on an analog synthesizer with a gradually rising filter cutoff and a reverb tail"
SECONDS_START = 0.0
SECONDS_TOTAL = 47.0
STEPS = 100
CFG_SCALE = 5.0
SAMPLER_TYPE = "dpmpp-3m-sde"
SIGMA_MIN = 0.3
SIGMA_MAX = 500.0
SEED = 12345
MODEL_HALF = True
OUTPUT_PATH = Path("output.wav")


def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepare_model(model: torch.nn.Module, *, device: torch.device, model_half: bool) -> torch.nn.Module:
    model = model.to(device).eval().requires_grad_(False)
    if model_half:
        if device.type == "cpu":
            print("model_half requested on CPU; keeping float32.")
        else:
            model = model.to(torch.float16)
    return model


def _conditioning(prompt: str, seconds_start: float, seconds_total: float) -> list[dict[str, float | str]]:
    return [
        {
            "prompt": prompt,
            "seconds_start": float(seconds_start),
            "seconds_total": float(seconds_total),
        }
    ]


def _resolve_sample_size(*, sample_rate: int, min_input_length: int) -> int:
    return audio_sample_size_from_seconds(
        seconds_total=float(SECONDS_TOTAL),
        sample_rate=int(sample_rate),
        min_input_length=int(min_input_length),
    )


def main() -> int:
    device = _resolve_device()
    print(f"device={device}")

    model, model_config = get_pretrained_model(MODEL_NAME)
    sample_rate = int(model_config["sample_rate"])
    sample_size = _resolve_sample_size(sample_rate=sample_rate, min_input_length=int(model.min_input_length))
    print(f"sample_rate={sample_rate}, sample_size={sample_size}, seconds_total={SECONDS_TOTAL}")

    model = _prepare_model(model, device=device, model_half=MODEL_HALF)

    conditioning = _conditioning(PROMPT, SECONDS_START, SECONDS_TOTAL)
    output = generate_diffusion_cond(
        model,
        steps=STEPS,
        cfg_scale=CFG_SCALE,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
        sampler_type=SAMPLER_TYPE,
        device=device,
        seed=SEED,
    )

    save_audio_tensor(OUTPUT_PATH, output, sample_rate)
    print(f"音频已成功保存为 {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
