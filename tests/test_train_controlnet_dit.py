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
        self.assertIsNone(args.seconds_total)
        self.assertIsNone(args.sample_size)
        self.assertEqual(module.parse_demo_cfg_scales(args.demo_cfg_scales), [3.0, 6.0, 9.0])
        self.assertEqual(args.demo_steps, 100)
        self.assertEqual(module.parse_demo_control_scales(args.demo_control_scales), [0.0, 0.1, 0.3, 0.6, 1.0])
        self.assertEqual(args.demo_control_variants, "correct,shuffled,zero")
        self.assertIsNone(args.demo_control_audio)
        self.assertIsNone(args.demo_prompt)

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
                "correct,zero",
                "--demo-prompt",
                "instrumental piano melody",
            ]
        )

        self.assertEqual(module.parse_demo_cfg_scales(args.demo_cfg_scales), [6.0])
        self.assertEqual(args.demo_steps, 30)
        self.assertEqual(module.parse_demo_control_scales(args.demo_control_scales), [0.0, 1.0])
        self.assertEqual(module.parse_demo_control_variants(args.demo_control_variants), ["correct", "zero"])
        self.assertEqual(args.demo_prompt, "instrumental piano melody")

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
            module.parse_demo_control_variants("correct, shuffled, zero"),
            ["correct", "shuffled", "zero"],
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


if __name__ == "__main__":
    unittest.main()
