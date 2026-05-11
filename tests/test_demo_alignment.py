from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_demo_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "demo.py"
    spec = importlib.util.spec_from_file_location("demo", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DemoAlignmentTests(unittest.TestCase):
    def test_builds_conditioning_dict_like_generation_scripts(self) -> None:
        module = _load_demo_module()

        conditioning = module._conditioning(
            "Warm arpeggios on an analog synthesizer with a gradually rising filter cutoff and a reverb tail",
            0.0,
            47.0,
        )

        self.assertEqual(
            conditioning,
            [
                {
                    "prompt": "Warm arpeggios on an analog synthesizer with a gradually rising filter cutoff and a reverb tail",
                    "seconds_start": 0.0,
                    "seconds_total": 47.0,
                }
            ],
        )

    def test_resolves_sample_size_from_seconds_total_and_min_input_length(self) -> None:
        module = _load_demo_module()

        sample_size = module._resolve_sample_size(sample_rate=44_100, min_input_length=2_048)

        self.assertEqual(sample_size, 2_074_624)

    def test_prepares_model_in_eval_mode_and_keeps_cpu_float32_for_model_half(self) -> None:
        module = _load_demo_module()
        model = torch.nn.Linear(4, 4)

        prepared = module._prepare_model(model, device=torch.device("cpu"), model_half=True)

        self.assertIs(prepared, model)
        self.assertFalse(prepared.training)
        self.assertFalse(any(parameter.requires_grad for parameter in prepared.parameters()))
        self.assertEqual(next(prepared.parameters()).dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
