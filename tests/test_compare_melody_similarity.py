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

    def test_cqt_top1_accuracy_scores_primary_left_right_channels(self) -> None:
        reference = torch.tensor(
            [
                [
                    [10, 20, 30],
                    [40, 50, 60],
                    [11, 21, 31],
                    [41, 51, 61],
                ]
            ],
            dtype=torch.long,
        )
        generated = torch.tensor(
            [
                [
                    [10, 99, 31],
                    [40, 51, 88],
                    [12, 22, 32],
                    [42, 52, 62],
                ]
            ],
            dtype=torch.long,
        )

        result = melody_similarity.compare_cqt_top1_accuracy(reference, generated, tolerance_bins=1)

        self.assertEqual(result["metric_name"], "cqt_top1_pitch_accuracy")
        self.assertEqual(result["matched_tokens"], 4)
        self.assertEqual(result["total_tokens"], 6)
        self.assertAlmostEqual(result["score"], 4 / 6)

    def test_cqt_top1_accuracy_ignores_zero_reference_tokens(self) -> None:
        reference = torch.tensor([[[0, 10], [0, 20], [1, 1], [2, 2]]], dtype=torch.long)
        generated = torch.tensor([[[99, 10], [99, 21], [1, 1], [2, 2]]], dtype=torch.long)

        result = melody_similarity.compare_cqt_top1_accuracy(reference, generated, tolerance_bins=0)

        self.assertEqual(result["matched_tokens"], 1)
        self.assertEqual(result["total_tokens"], 2)
        self.assertAlmostEqual(result["score"], 0.5)

    def test_compare_audio_tensors_uses_supplied_extractor_without_writing_files(self) -> None:
        class IdentityPitchExtractor:
            def extract(self, audio: torch.Tensor) -> torch.Tensor:
                return audio.to(torch.long)

        reference_audio = torch.tensor([[[10, 20, 0], [40, 50, 0]]], dtype=torch.float32)
        generated_audio = torch.tensor([[[10, 99, 0], [40, 50, 0]]], dtype=torch.float32)

        result = melody_similarity.compare_audio_tensors_melody_similarity(
            reference_audio,
            generated_audio,
            extractor=IdentityPitchExtractor(),
            feature="cqt",
            sample_rate=44_100,
            top_k=1,
            sample_size=3,
        )

        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["alignment"]["mode"], "fixed_length")
        self.assertEqual(result["alignment"]["sample_size"], 3)
        self.assertEqual(result["feature"]["reference_shape"], [1, 2, 3])
        self.assertEqual(result["similarity"]["metric_name"], "cqt_topk_pitch_overlap_rate")
        top1 = result["similarity"]["additional_metrics"]["cqt_top1_accuracy"]
        self.assertEqual(top1["matched_tokens"], 3)
        self.assertEqual(top1["total_tokens"], 4)
        self.assertAlmostEqual(top1["score"], 0.75)

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

    def test_evaluate_control_variants_parser_and_filename_metadata(self) -> None:
        module = _load_script_module("evaluate_control_variants.py")
        parser = module.build_arg_parser()

        args = parser.parse_args(
            [
                "--reference-audio",
                "reference.wav",
                "--generated-dir",
                "outputs/demo",
            ]
        )
        metadata = module.infer_control_metadata_from_name(
            "demo_cfg_5_control_0p3_shuffled_step_00008000_seed-7.wav"
        )

        self.assertEqual(args.melody_feature, "cqt")
        self.assertEqual(args.glob, "*.wav")
        self.assertEqual(metadata["variant"], "shuffled")
        self.assertEqual(metadata["control_scale"], 0.3)
        self.assertEqual(metadata["seed"], 7)

    def test_evaluate_control_variants_does_not_parse_batch_index_as_control_scale(self) -> None:
        module = _load_script_module("evaluate_control_variants.py")

        metadata = module.infer_control_metadata_from_name("control_002_seed-42_zero.wav")

        self.assertEqual(metadata["variant"], "zero")
        self.assertIsNone(metadata["control_scale"])
        self.assertEqual(metadata["seed"], 42)


if __name__ == "__main__":
    unittest.main()
