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
        self.assertEqual(args.num_control_layers, 2)
        self.assertEqual(args.control_id, "melody_control")
        self.assertEqual(args.cqt_backend, "auto")
        self.assertEqual(args.top_k, 4)
        self.assertEqual(args.melody_embedding_dim, 64)
        self.assertEqual(args.melody_hidden_dim, 256)
        self.assertEqual(args.melody_conv_layers, 2)

    def test_import_patches_stable_audio_tools_inverse_lr_for_current_torch(self) -> None:
        _load_script_module()
        from stable_audio_tools.training.utils import InverseLR

        param = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.SGD([param], lr=1e-3)

        scheduler = InverseLR(optimizer, inv_gamma=10.0, power=0.5)

        self.assertEqual(len(scheduler.get_last_lr()), 1)

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
