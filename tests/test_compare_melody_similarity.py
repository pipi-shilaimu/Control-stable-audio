from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_audio_control.inference import melody_similarity


def _load_script_module(script_name: str):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "stable_audio_control" / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MelodySimilarityTests(unittest.TestCase):
    def test_cqt_topk_similarity_scores_pitch_overlap_instead_of_rank_order(self) -> None:
        reference = torch.tensor(
            [
                [
                    [10, 30],
                    [11, 31],
                    [20, 40],
                    [21, 41],
                ]
            ],
            dtype=torch.long,
        )
        generated = torch.tensor(
            [
                [
                    [20, 30],
                    [21, 88],
                    [10, 99],
                    [11, 41],
                ]
            ],
            dtype=torch.long,
        )

        result = melody_similarity.compare_cqt_topk_features(reference, generated, top_k=2)

        self.assertEqual(result["metric_name"], "cqt_topk_pitch_overlap_rate")
        self.assertEqual(result["compared_frames"], 2)
        self.assertEqual(result["matched_tokens"], 6)
        self.assertEqual(result["total_tokens"], 8)
        self.assertAlmostEqual(result["score"], 0.75)

    def test_chromagram_similarity_uses_mean_frame_cosine(self) -> None:
        reference = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
        generated = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]], dtype=torch.float32)

        result = melody_similarity.compare_chromagram_features(reference, generated)

        self.assertEqual(result["metric_name"], "chromagram_frame_cosine_mean")
        self.assertEqual(result["compared_frames"], 2)
        self.assertAlmostEqual(result["score"], 0.5)

    def test_similarity_writer_round_trips_json(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "similarity.json"
            payload = {
                "schema_version": 1,
                "similarity": {"metric_name": "chromagram_frame_cosine_mean", "score": 0.5},
            }

            melody_similarity.write_similarity_metadata(path, payload)
            loaded = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(loaded["schema_version"], 1)
        self.assertEqual(loaded["similarity"]["score"], 0.5)

    def test_compare_melody_script_parser_defaults_to_cqt(self) -> None:
        module = _load_script_module("compare_melody_similarity.py")
        parser = module.build_arg_parser()

        args = parser.parse_args(
            [
                "--reference-audio",
                "reference.wav",
                "--generated-audio",
                "generated.wav",
            ]
        )

        self.assertEqual(args.melody_feature, "cqt")
        self.assertEqual(args.top_k, 4)
        self.assertEqual(args.chroma_bins, 12)
        self.assertEqual(args.output_json, None)


if __name__ == "__main__":
    unittest.main()
