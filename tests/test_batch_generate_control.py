from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch


def _load_batch_generate_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "stable_audio_control" / "scripts" / "batch_generate_control.py"
    spec = importlib.util.spec_from_file_location("batch_generate_control", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BatchGenerateControlTests(unittest.TestCase):
    def test_parser_defaults_to_correct_control_variant(self) -> None:
        module = _load_batch_generate_module()
        parser = module.build_parser()

        args = parser.parse_args(
            [
                "--ckpt-path",
                "model.ckpt",
                "--reference-audio",
                "reference.wav",
                "--prompt",
                "piano melody",
            ]
        )

        self.assertEqual(module.parse_demo_control_variants(args.demo_control_variants), ["correct"])
        self.assertTrue(args.prefer_ema)

    def test_parser_accepts_no_prefer_ema_diagnostic_mode(self) -> None:
        module = _load_batch_generate_module()
        parser = module.build_parser()

        args = parser.parse_args(
            [
                "--ckpt-path",
                "model.ckpt",
                "--reference-audio",
                "reference.wav",
                "--prompt",
                "piano melody",
                "--no-prefer-ema",
            ]
        )

        self.assertFalse(args.prefer_ema)

    def test_ema_policy_notes_recommend_no_ema_diagnostic_when_ema_is_used(self) -> None:
        module = _load_batch_generate_module()

        notes = module.describe_inference_policy(
            {
                "use_ema": True,
                "ema_missing_keys": ["control_layers.0.weight", "control_layers.1.weight"],
                "ema_unexpected_keys": ["legacy.weight"],
            }
        )

        joined = "\n".join(notes)
        self.assertIn("--no-prefer-ema", joined)
        self.assertIn("hybrid", joined.lower())
        self.assertIn("melody control", joined)
        self.assertIn("ema_missing_keys=2", joined)
        self.assertIn("ema_unexpected_keys=1", joined)

    def test_inference_policy_notes_document_cfg_keeps_melody_control_on(self) -> None:
        module = _load_batch_generate_module()

        notes = module.describe_inference_policy({"use_ema": False})

        joined = "\n".join(notes)
        self.assertIn("CFG", joined)
        self.assertIn("melody control stays enabled", joined)

    def test_parses_slash_separated_control_variants_and_aliases(self) -> None:
        module = _load_batch_generate_module()

        variants = module.parse_demo_control_variants("correct/zero/shuffle/shuffled/null/disabled/none")

        self.assertEqual(variants, ["correct", "zero", "shuffled", "null"])

    def test_rejects_unknown_control_variant(self) -> None:
        module = _load_batch_generate_module()

        with self.assertRaisesRegex(ValueError, "Unknown control variant"):
            module.parse_demo_control_variants("correct/random")

    def test_makes_variant_reference_audio(self) -> None:
        module = _load_batch_generate_module()
        reference_audio = torch.arange(1 * 2 * 4, dtype=torch.float32).reshape(1, 2, 4)

        correct = module.make_control_variant_audio(reference_audio, "correct")
        zero = module.make_control_variant_audio(reference_audio, "zero")
        shuffled = module.make_control_variant_audio(reference_audio, "shuffled")
        null = module.make_control_variant_audio(reference_audio, "null")

        torch.testing.assert_close(correct, reference_audio)
        torch.testing.assert_close(zero, torch.zeros_like(reference_audio))
        torch.testing.assert_close(shuffled, torch.flip(reference_audio, dims=[-1]))
        self.assertIsNone(null)

    def test_formats_output_name_without_variant_for_default_single_correct(self) -> None:
        module = _load_batch_generate_module()

        filename = module.format_output_name(
            "control_{i:03d}_seed-{seed}.wav",
            i=2,
            seed=42,
            variant="correct",
            include_variant=False,
        )

        self.assertEqual(filename, "control_002_seed-42.wav")

    def test_formats_output_name_with_variant_suffix_when_needed(self) -> None:
        module = _load_batch_generate_module()

        filename = module.format_output_name(
            "control_{i:03d}_seed-{seed}.wav",
            i=2,
            seed=42,
            variant="zero",
            include_variant=True,
        )

        self.assertEqual(filename, "control_002_seed-42_zero.wav")

    def test_formats_output_name_with_explicit_variant_placeholder(self) -> None:
        module = _load_batch_generate_module()

        filename = module.format_output_name(
            "control_{variant}_{i:03d}_seed-{seed}.wav",
            i=2,
            seed=42,
            variant="shuffled",
            include_variant=True,
        )

        self.assertEqual(filename, "control_shuffled_002_seed-42.wav")


if __name__ == "__main__":
    unittest.main()
