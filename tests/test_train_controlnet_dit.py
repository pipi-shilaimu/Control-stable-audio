from __future__ import annotations

import importlib.util
import pickle
import types
import unittest
from pathlib import Path

import torch


def _load_script_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "stable_audio_control" / "scripts" / "train_controlnet_dit.py"
    spec = importlib.util.spec_from_file_location("train_controlnet_dit", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TrainControlNetDiTScriptTests(unittest.TestCase):
    def test_arg_parser_defaults(self) -> None:
        module = _load_script_module()
        parser = module.build_arg_parser()

        args = parser.parse_args(["--dataset-config", "dummy_dataset.json"])

        self.assertEqual(args.model_name, "stabilityai/stable-audio-open-1.0")
        self.assertEqual(args.melody_feature, "cqt")
        self.assertEqual(args.num_control_layers, 12)
        self.assertEqual(args.control_id, "melody_control")
        self.assertEqual(args.cqt_backend, "auto")
        self.assertEqual(args.chroma_bins, 12)
        self.assertEqual(args.chroma_n_fft, 2048)
        self.assertEqual(args.top_k, 4)
        self.assertEqual(args.melody_embedding_dim, 64)
        self.assertEqual(args.melody_hidden_dim, 256)
        self.assertEqual(args.melody_conv_layers, 2)
        self.assertEqual(args.cqt_silence_threshold_ratio, 0.01)
        self.assertEqual(args.cqt_silence_threshold_abs, 1e-8)
        self.assertIsNone(args.seconds_total)
        self.assertIsNone(args.sample_size)
        self.assertEqual(module.parse_demo_cfg_scales(args.demo_cfg_scales), [3.0, 6.0, 9.0])
        self.assertEqual(args.demo_steps, 100)
        self.assertEqual(module.parse_demo_control_scales(args.demo_control_scales), [0.0, 0.1, 0.3, 0.6, 1.0])
        self.assertEqual(args.demo_control_variants, "correct,shuffled,zero,null")
        self.assertIsNone(args.demo_control_audio)
        self.assertIsNone(args.demo_prompt)
        self.assertTrue(args.demo_melody_similarity)
        self.assertEqual(args.demo_melody_similarity_csv, "demo_melody_similarity.csv")
        self.assertTrue(args.demo_control_diagnostics)
        self.assertEqual(args.demo_control_diagnostics_csv, "demo_control_diagnostics.csv")
        self.assertFalse(args.demo_stop_on_collapse)
        self.assertEqual(args.demo_collapse_cosine_threshold, 0.98)
        self.assertFalse(args.melody_mask)
        self.assertEqual(args.melody_full_mask_steps, 0)
        self.assertEqual(args.melody_mask_schedule_steps, 10000)
        self.assertEqual(args.melody_frame_mask_ratio_start, 0.75)
        self.assertEqual(args.melody_frame_mask_ratio_end, 0.10)
        self.assertEqual(args.melody_secondary_mask_prob, 0.15)
        self.assertEqual(args.melody_secondary_shuffle_prob, 0.15)

    def test_arg_parser_accepts_demo_cost_overrides(self) -> None:
        module = _load_script_module()
        parser = module.build_arg_parser()

        args = parser.parse_args(
            [
                "--dataset-config",
                "dummy_dataset.json",
                "--demo-cfg-scales",
                "6",
                "--demo-steps",
                "30",
                "--demo-control-scales",
                "0,1",
                "--demo-control-variants",
                "correct,disabled",
                "--demo-prompt",
                "instrumental piano melody",
                "--demo-melody-similarity",
                "false",
                "--demo-melody-similarity-csv",
                "demo_scores.csv",
                "--demo-control-diagnostics",
                "false",
                "--demo-control-diagnostics-csv",
                "diag.csv",
                "--demo-stop-on-collapse",
                "true",
                "--demo-collapse-cosine-threshold",
                "0.95",
            ]
        )

        self.assertEqual(module.parse_demo_cfg_scales(args.demo_cfg_scales), [6.0])
        self.assertEqual(args.demo_steps, 30)
        self.assertEqual(module.parse_demo_control_scales(args.demo_control_scales), [0.0, 1.0])
        self.assertEqual(module.parse_demo_control_variants(args.demo_control_variants), ["correct", "null"])
        self.assertEqual(args.demo_prompt, "instrumental piano melody")
        self.assertFalse(args.demo_melody_similarity)
        self.assertEqual(args.demo_melody_similarity_csv, "demo_scores.csv")
        self.assertFalse(args.demo_control_diagnostics)
        self.assertEqual(args.demo_control_diagnostics_csv, "diag.csv")
        self.assertTrue(args.demo_stop_on_collapse)
        self.assertEqual(args.demo_collapse_cosine_threshold, 0.95)

    def test_arg_parser_accepts_melody_mask_overrides(self) -> None:
        module = _load_script_module()
        parser = module.build_arg_parser()

        args = parser.parse_args(
            [
                "--dataset-config",
                "dummy_dataset.json",
                "--melody-mask",
                "true",
                "--melody-full-mask-steps",
                "500",
                "--melody-mask-schedule-steps",
                "4000",
                "--melody-frame-mask-ratio-start",
                "0.9",
                "--melody-frame-mask-ratio-end",
                "0.05",
                "--melody-secondary-mask-prob",
                "0.25",
                "--melody-secondary-shuffle-prob",
                "0.35",
            ]
        )

        config = module.build_melody_masking_config(args)

        self.assertTrue(config.enabled)
        self.assertEqual(config.full_mask_steps, 500)
        self.assertEqual(config.schedule_steps, 4000)
        self.assertEqual(config.frame_mask_ratio_start, 0.9)
        self.assertEqual(config.frame_mask_ratio_end, 0.05)
        self.assertEqual(config.secondary_mask_prob, 0.25)
        self.assertEqual(config.secondary_shuffle_prob, 0.35)

    def test_melody_mask_config_rejects_invalid_ratios(self) -> None:
        module = _load_script_module()
        parser = module.build_arg_parser()
        args = parser.parse_args(
            [
                "--dataset-config",
                "dummy_dataset.json",
                "--melody-mask",
                "true",
                "--melody-frame-mask-ratio-start",
                "1.2",
            ]
        )

        with self.assertRaisesRegex(ValueError, "frame_mask_ratio_start"):
            module.build_melody_masking_config(args)

    def test_training_demo_count_is_independent_of_batch_size(self) -> None:
        module = _load_script_module()

        self.assertEqual(module.resolve_training_demo_count(batch_size=1), 1)
        self.assertEqual(module.resolve_training_demo_count(batch_size=4), 1)

    def test_arg_parser_rejects_conflicting_training_length_overrides(self) -> None:
        module = _load_script_module()
        parser = module.build_arg_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--dataset-config",
                    "dummy_dataset.json",
                    "--seconds-total",
                    "10",
                    "--sample-size",
                    "441000",
                ]
            )

    def test_resolves_effective_sample_size_from_seconds_total_with_model_alignment(self) -> None:
        module = _load_script_module()

        resolved = module.resolve_effective_train_sample_size(
            model_config_sample_size=2_097_152,
            sample_rate=44_100,
            min_input_length=2_048,
            seconds_total=10.0,
            sample_size=None,
        )

        self.assertEqual(resolved.sample_size, 442_368)
        self.assertAlmostEqual(resolved.seconds_total, 442_368 / 44_100)
        self.assertEqual(resolved.source, "--seconds-total")
        self.assertEqual(resolved.model_config_sample_size, 2_097_152)
        self.assertEqual(resolved.min_input_length, 2_048)

    def test_resolves_effective_sample_size_from_explicit_sample_size_with_model_alignment(self) -> None:
        module = _load_script_module()

        resolved = module.resolve_effective_train_sample_size(
            model_config_sample_size=2_097_152,
            sample_rate=44_100,
            min_input_length=2_048,
            seconds_total=None,
            sample_size=441_001,
        )

        self.assertEqual(resolved.sample_size, 442_368)
        self.assertAlmostEqual(resolved.seconds_total, 442_368 / 44_100)
        self.assertEqual(resolved.source, "--sample-size")

    def test_resolves_effective_sample_size_defaults_to_model_config(self) -> None:
        module = _load_script_module()

        resolved = module.resolve_effective_train_sample_size(
            model_config_sample_size=2_097_152,
            sample_rate=44_100,
            min_input_length=2_048,
            seconds_total=None,
            sample_size=None,
        )

        self.assertEqual(resolved.sample_size, 2_097_152)
        self.assertAlmostEqual(resolved.seconds_total, 2_097_152 / 44_100)
        self.assertEqual(resolved.source, "model_config")

    def test_parses_demo_control_scales_csv(self) -> None:
        module = _load_script_module()

        self.assertEqual(module.parse_demo_control_scales("0, 0.3, 1"), [0.0, 0.3, 1.0])

    def test_parses_demo_cfg_scales_csv(self) -> None:
        module = _load_script_module()

        self.assertEqual(module.parse_demo_cfg_scales("3, 6, 9"), [3.0, 6.0, 9.0])

    def test_parses_demo_control_variants_csv(self) -> None:
        module = _load_script_module()

        self.assertEqual(
            module.parse_demo_control_variants("correct, shuffle, shuffled, zero_audio, zero, disabled, none"),
            ["correct", "shuffled", "zero", "null"],
        )

    def test_rejects_unknown_demo_control_variant(self) -> None:
        module = _load_script_module()

        with self.assertRaisesRegex(ValueError, "Unknown control variant"):
            module.parse_demo_control_variants("correct,random")

    def test_import_patches_stable_audio_tools_inverse_lr_for_current_torch(self) -> None:
        _load_script_module()
        from stable_audio_tools.training.utils import InverseLR

        param = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.SGD([param], lr=1e-3)

        scheduler = InverseLR(optimizer, inv_gamma=10.0, power=0.5)

        self.assertEqual(len(scheduler.get_last_lr()), 1)

    def test_learning_rate_override_updates_optimizer_config_without_mutating_model_config(self) -> None:
        module = _load_script_module()
        model_config = {
            "training": {
                "learning_rate": 5e-5,
                "optimizer_configs": {
                    "diffusion": {
                        "optimizer": {
                            "type": "AdamW",
                            "config": {
                                "lr": 5e-5,
                                "betas": (0.9, 0.999),
                            },
                        },
                        "scheduler": {
                            "type": "InverseLR",
                            "config": {
                                "inv_gamma": 1_000_000,
                            },
                        },
                    }
                },
            }
        }

        learning_rate, optimizer_configs = module.resolve_training_optimizer_settings(
            model_config=model_config,
            learning_rate_override=1e-4,
        )

        self.assertIsNone(learning_rate)
        self.assertEqual(optimizer_configs["diffusion"]["optimizer"]["config"]["lr"], 1e-4)
        self.assertEqual(
            model_config["training"]["optimizer_configs"]["diffusion"]["optimizer"]["config"]["lr"],
            5e-5,
        )
        self.assertEqual(
            optimizer_configs["diffusion"]["scheduler"]["config"]["inv_gamma"],
            1_000_000,
        )

    def test_replaces_temp_metadata_function_with_importable_package_function(self) -> None:
        module = _load_script_module()
        repo_root = Path(__file__).resolve().parents[1]
        metadata_path = repo_root / "stable_audio_control" / "data" / "song_describer_metadata.py"
        audio_dir = repo_root / "stable_audio_control" / "data" / "song_describer" / "train"
        dataloader = types.SimpleNamespace(
            dataset=types.SimpleNamespace(
                custom_metadata_fns={str(audio_dir): lambda info, audio: {"prompt": "temporary"}}
            )
        )
        dataset_config = {
            "dataset_type": "audio_dir",
            "datasets": [
                {
                    "id": "song_describer_train",
                    "path": str(audio_dir),
                    "custom_metadata_module": str(metadata_path),
                }
            ],
        }

        module.make_dataloader_custom_metadata_picklable(dataloader, dataset_config)

        custom_metadata_fn = dataloader.dataset.custom_metadata_fns[str(audio_dir)]
        self.assertEqual(custom_metadata_fn.__module__, "stable_audio_control.data.song_describer_metadata")
        pickle.dumps(custom_metadata_fn)

    def test_normalizes_audio_dir_padding_mask_for_training_wrapper(self) -> None:
        module = _load_script_module()
        padding_mask = torch.ones(16, dtype=torch.bool)
        metadata = ({"prompt": "test", "padding_mask": padding_mask},)

        normalized = module.normalize_metadata_padding_masks(metadata)

        self.assertIsInstance(normalized, list)
        self.assertIsInstance(normalized[0]["padding_mask"], list)
        self.assertIs(normalized[0]["padding_mask"][0], padding_mask)
        self.assertEqual(tuple(torch.stack([md["padding_mask"][0] for md in normalized], dim=0).shape), (1, 16))

    def test_melody_augmenter_applies_progressive_mask_only_in_training_context(self) -> None:
        module = _load_script_module()

        class FakeConditioner(torch.nn.Module):
            def forward(self, metadata, device):
                return {}

        class FakeExtractor:
            def extract(self, waveform):
                return torch.arange(1, 1 + 8 * 4, dtype=torch.long).reshape(1, 8, 4)

        augmenter = module.MelodyControlAugmenter(
            base_conditioner=FakeConditioner(),
            control_id="melody_control",
            extractor=FakeExtractor(),
            control_channels=8,
            empty_dtype=torch.long,
            melody_masking_config=module.MelodyMaskingConfig(
                enabled=True,
                full_mask_steps=10,
            ),
        )
        metadata = [{"prompt": "test", "padding_mask": torch.ones(8)}]
        audio = torch.zeros((1, 2, 8))

        augmenter.set_training_context(global_step=0, training=True)
        augmenter.set_batch_audio(audio)
        training_conditioning = augmenter(metadata, torch.device("cpu"))

        augmenter.set_training_context(global_step=0, training=False)
        augmenter.set_batch_audio(audio)
        validation_conditioning = augmenter(metadata, torch.device("cpu"))

        self.assertTrue(torch.equal(training_conditioning["melody_control"][0], torch.zeros((1, 8, 4), dtype=torch.long)))
        self.assertFalse(torch.equal(validation_conditioning["melody_control"][0], torch.zeros((1, 8, 4), dtype=torch.long)))

    def test_melody_augmenter_zeroes_padded_cqt_frames_from_metadata(self) -> None:
        module = _load_script_module()

        class FakeConditioner(torch.nn.Module):
            def forward(self, metadata, device):
                return {}

        class FakeExtractor:
            def extract(self, waveform):
                return torch.ones((1, 8, 4), dtype=torch.long)

        augmenter = module.MelodyControlAugmenter(
            base_conditioner=FakeConditioner(),
            control_id="melody_control",
            extractor=FakeExtractor(),
            control_channels=8,
            empty_dtype=torch.long,
            melody_masking_config=module.MelodyMaskingConfig(enabled=False),
        )
        metadata = [{"prompt": "test", "padding_mask": torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])}]

        augmenter.set_training_context(global_step=100, training=True)
        augmenter.set_batch_audio(torch.zeros((1, 2, 8)))
        conditioning = augmenter(metadata, torch.device("cpu"))

        melody_control = conditioning["melody_control"][0]
        self.assertTrue(torch.equal(melody_control[:, :, :2], torch.ones_like(melody_control[:, :, :2])))
        self.assertTrue(torch.equal(melody_control[:, :, 2:], torch.zeros_like(melody_control[:, :, 2:])))


if __name__ == "__main__":
    unittest.main()
