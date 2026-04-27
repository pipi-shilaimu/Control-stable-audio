from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
