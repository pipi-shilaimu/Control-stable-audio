from __future__ import annotations

import json
import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_audio_control.inference.control_compare import (
    build_compare_metadata,
    compute_audio_difference_stats,
    extract_control_checkpoint_state_dicts,
    write_compare_metadata,
)


def _load_compare_script_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "stable_audio_control" / "scripts" / "compare_controlnet_generation.py"
    spec = importlib.util.spec_from_file_location("compare_controlnet_generation", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ControlNetGenerationCompareTests(unittest.TestCase):
    def test_extracts_online_wrapper_state_and_ema_model_overlay(self) -> None:
        online_value = torch.tensor([1.0])
        ema_value = torch.tensor([2.0])
        checkpoint = {
            "state_dict": {
                "diffusion.melody_encoder.weight": online_value,
                "diffusion.base_wrapper.model.model.weight": online_value,
                "diffusion_ema.ema_model.model.weight": ema_value,
                "diffusion_ema.step": torch.tensor([50]),
                "losses.loss_modules.0.weight": torch.tensor([1.0]),
            }
        }

        states = extract_control_checkpoint_state_dicts(checkpoint, prefer_ema=True)

        self.assertIn("melody_encoder.weight", states.online_wrapper_state)
        self.assertIn("base_wrapper.model.model.weight", states.online_wrapper_state)
        self.assertEqual(set(states.ema_model_state), {"model.weight"})
        torch.testing.assert_close(states.ema_model_state["model.weight"], ema_value)
        self.assertTrue(states.use_ema)

    def test_falls_back_to_online_only_when_ema_is_not_available(self) -> None:
        checkpoint = {
            "state_dict": {
                "diffusion.melody_encoder.weight": torch.tensor([1.0]),
            }
        }

        states = extract_control_checkpoint_state_dicts(checkpoint, prefer_ema=True)

        self.assertIn("melody_encoder.weight", states.online_wrapper_state)
        self.assertEqual(states.ema_model_state, {})
        self.assertFalse(states.use_ema)

    def test_extract_remaps_training_melody_augmenter_base_conditioner_keys_for_inference(self) -> None:
        augmented_conditioner_value = torch.tensor([3.0])
        canonical_conditioner_value = torch.tensor([4.0])
        checkpoint = {
            "state_dict": {
                "diffusion.base_wrapper.conditioner.base_conditioner.conditioners.seconds_total.embedder.embedding.1.weight": augmented_conditioner_value,
                "diffusion.base_wrapper.conditioner.conditioners.prompt.embedder.weight": canonical_conditioner_value,
                "diffusion.control_projector.weight": torch.tensor([5.0]),
            }
        }

        states = extract_control_checkpoint_state_dicts(checkpoint, prefer_ema=True)

        self.assertIn(
            "base_wrapper.conditioner.conditioners.seconds_total.embedder.embedding.1.weight",
            states.online_wrapper_state,
        )
        self.assertNotIn(
            "base_wrapper.conditioner.base_conditioner.conditioners.seconds_total.embedder.embedding.1.weight",
            states.online_wrapper_state,
        )
        torch.testing.assert_close(
            states.online_wrapper_state[
                "base_wrapper.conditioner.conditioners.seconds_total.embedder.embedding.1.weight"
            ],
            augmented_conditioner_value,
        )
        torch.testing.assert_close(
            states.online_wrapper_state["base_wrapper.conditioner.conditioners.prompt.embedder.weight"],
            canonical_conditioner_value,
        )

    def test_writes_compare_metadata_sidecar(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = build_compare_metadata(
                prompt="bright piano melody",
                negative_prompt="noise",
                reference_audio_path=root / "ref.wav",
                ckpt_path=root / "epoch=0-step=50.ckpt",
                base_output_path=root / "base.wav",
                control_output_path=root / "control.wav",
                seed=1234,
                model_name="stabilityai/stable-audio-open-1.0",
                sample_rate=44_100,
                sample_size=44_100,
                seconds_start=0.0,
                seconds_total=1.0,
                steps=8,
                cfg_scale=7.0,
                sampler_type="dpmpp-3m-sde",
                sigma_min=0.3,
                sigma_max=500.0,
                control_scale=1.0,
                use_ema=True,
                control_config={"top_k": 4, "n_bins": 128},
            )
            sidecar_path = root / "compare.json"

            write_compare_metadata(sidecar_path, metadata)
            loaded = json.loads(sidecar_path.read_text(encoding="utf-8"))

        self.assertEqual(loaded["schema_version"], 1)
        self.assertEqual(loaded["prompt"], "bright piano melody")
        self.assertEqual(loaded["negative_prompt"], "noise")
        self.assertEqual(loaded["generation"]["seed"], 1234)
        self.assertEqual(loaded["generation"]["steps"], 8)
        self.assertEqual(loaded["control"]["scale"], 1.0)
        self.assertTrue(loaded["control"]["use_ema"])
        self.assertTrue(loaded["paths"]["base_output"].endswith("base.wav"))
        self.assertTrue(loaded["paths"]["control_output"].endswith("control.wav"))

    def test_computes_audio_difference_stats_from_generated_tensors_after_output_normalization(self) -> None:
        base = torch.tensor([[[0.0, 0.5, -0.5], [0.25, -0.25, 0.0]]])
        control = torch.tensor([[[0.0, 0.25, -0.5], [0.25, 0.0, 0.0]]])

        stats = compute_audio_difference_stats(base, control)

        self.assertEqual(stats["base_shape"], [2, 3])
        self.assertEqual(stats["control_shape"], [2, 3])
        self.assertEqual(stats["compared_samples"], 3)
        self.assertEqual(stats["channels"], 2)
        self.assertAlmostEqual(stats["max_abs_diff"], 0.5)
        self.assertAlmostEqual(stats["mean_abs_diff"], 0.16666666666666666)
        self.assertGreater(stats["base_rms"], 0.0)
        self.assertGreater(stats["control_rms"], 0.0)
        self.assertLess(stats["rms_diff"], stats["base_rms"])
        self.assertEqual(len(stats["channel_correlation"]), 2)

    def test_compare_script_parser_defaults_to_minimal_reproducible_outputs(self) -> None:
        module = _load_compare_script_module()
        parser = module.build_arg_parser()

        args = parser.parse_args(
            [
                "--ckpt-path",
                "model.ckpt",
                "--reference-audio",
                "reference.wav",
                "--prompt",
                "a melody",
            ]
        )

        self.assertEqual(args.model_name, "stabilityai/stable-audio-open-1.0")
        self.assertEqual(args.output_dir, "outputs/controlnet_generation_compare")
        self.assertEqual(args.base_output_name, "base.wav")
        self.assertEqual(args.control_output_name, "control.wav")
        self.assertEqual(args.metadata_name, "compare.json")
        self.assertEqual(args.seed, 0)
        self.assertEqual(args.steps, 8)
        self.assertEqual(args.melody_feature, "cqt")
        self.assertEqual(args.chroma_bins, 12)
        self.assertEqual(args.chroma_n_fft, 2048)
        self.assertTrue(args.prefer_ema)

    def test_compare_script_rejects_dpmpp_3m_sde_single_step(self) -> None:
        module = _load_compare_script_module()
        parser = module.build_arg_parser()
        args = parser.parse_args(
            [
                "--ckpt-path",
                "model.ckpt",
                "--reference-audio",
                "reference.wav",
                "--prompt",
                "a melody",
                "--sampler-type",
                "dpmpp-3m-sde",
                "--steps",
                "1",
            ]
        )

        with self.assertRaisesRegex(ValueError, "dpmpp-3m-sde requires --steps >= 2"):
            module.validate_args(args)


if __name__ == "__main__":
    unittest.main()
