from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


def _load_script_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "stable_audio_control" / "scripts" / "diagnose_control_activity.py"
    spec = importlib.util.spec_from_file_location("diagnose_control_activity", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DiagnoseControlActivityTests(unittest.TestCase):
    def test_compute_tensor_activity_stats_for_bld_control_input(self) -> None:
        module = _load_script_module()
        tensor = torch.tensor([[[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]]])

        stats = module.compute_tensor_activity_stats(
            tensor,
            variant="correct",
            tensor_name="control_input",
            frame_dim=1,
        )

        self.assertEqual(stats.variant, "correct")
        self.assertEqual(stats.tensor_name, "control_input")
        self.assertEqual(stats.shape, "1x3x2")
        self.assertGreater(stats.rms, 2.0)
        self.assertEqual(stats.frame_count, 3)
        self.assertAlmostEqual(stats.frame_active_ratio, 2.0 / 3.0, places=6)
        self.assertEqual(stats.unique_value_count, 5)

    def test_zero_audio_cqt_indices_are_reported_as_not_padding_zero(self) -> None:
        module = _load_script_module()
        melody_indices = torch.ones((1, 8, 16), dtype=torch.long)

        stats = module.compute_tensor_activity_stats(
            melody_indices,
            variant="zero",
            tensor_name="melody_control",
            frame_dim=2,
        )

        self.assertEqual(stats.zero_value_ratio, 0.0)
        self.assertEqual(stats.nonzero_value_ratio, 1.0)
        self.assertEqual(stats.unique_value_count, 1)

    def test_activity_report_flags_correct_and_shuffled_much_louder_than_zero(self) -> None:
        module = _load_script_module()
        rows = [
            module.TensorActivityStats(
                variant="correct",
                tensor_name="control_input",
                shape="1x4x2",
                dtype="torch.float32",
                value_count=8,
                unique_value_count=8,
                zero_value_ratio=0.0,
                nonzero_value_ratio=1.0,
                mean_abs=2.0,
                max_abs=3.0,
                rms=2.0,
                l2=5.0,
                frame_count=4,
                frame_rms_mean=2.0,
                frame_rms_min=1.5,
                frame_rms_max=2.5,
                frame_active_ratio=1.0,
            ),
            module.TensorActivityStats(
                variant="shuffled",
                tensor_name="control_input",
                shape="1x4x2",
                dtype="torch.float32",
                value_count=8,
                unique_value_count=8,
                zero_value_ratio=0.0,
                nonzero_value_ratio=1.0,
                mean_abs=1.8,
                max_abs=2.7,
                rms=1.8,
                l2=4.5,
                frame_count=4,
                frame_rms_mean=1.8,
                frame_rms_min=1.4,
                frame_rms_max=2.3,
                frame_active_ratio=1.0,
            ),
            module.TensorActivityStats(
                variant="zero",
                tensor_name="control_input",
                shape="1x4x2",
                dtype="torch.float32",
                value_count=8,
                unique_value_count=2,
                zero_value_ratio=0.5,
                nonzero_value_ratio=0.5,
                mean_abs=0.1,
                max_abs=0.2,
                rms=0.1,
                l2=0.3,
                frame_count=4,
                frame_rms_mean=0.1,
                frame_rms_min=0.0,
                frame_rms_max=0.2,
                frame_active_ratio=0.5,
            ),
        ]

        report = module.build_activity_report(rows)

        joined = "\n".join(report)
        self.assertIn("activity-shortcut", joined)
        self.assertIn("correct/zero rms ratio", joined)

    def test_compute_delta_stats_uses_control_minus_base(self) -> None:
        module = _load_script_module()
        base = torch.tensor([[[1.0, 2.0, 3.0]]])
        controlled = torch.tensor([[[1.0, 4.0, -1.0]]])

        stats = module.compute_forward_delta_stats(
            base_output=base,
            controlled_output=controlled,
            variant="correct",
        )

        self.assertEqual(stats.variant, "correct")
        self.assertEqual(stats.tensor_name, "forward_delta")
        self.assertEqual(stats.shape, "1x1x3")
        self.assertGreater(stats.rms, 2.0)
        self.assertAlmostEqual(stats.zero_value_ratio, 1.0 / 3.0, places=6)

    def test_pairwise_tensor_stats_reports_cosine_and_relative_difference(self) -> None:
        module = _load_script_module()
        tensors = {
            "correct": torch.tensor([1.0, 1.0, 0.0]),
            "shuffled": torch.tensor([1.0, 1.0, 0.0]),
            "zero": torch.tensor([0.0, 0.0, 1.0]),
        }

        pairs = module.compute_pairwise_tensor_stats(tensors, tensor_name="forward_delta")

        by_pair = {(row.variant_a, row.variant_b): row for row in pairs}
        self.assertAlmostEqual(by_pair[("correct", "shuffled")].cosine_similarity, 1.0, places=6)
        self.assertAlmostEqual(by_pair[("correct", "shuffled")].relative_l2_difference, 0.0, places=6)
        self.assertLess(by_pair[("correct", "zero")].cosine_similarity, 0.1)

    def test_activity_report_flags_similar_correct_and_shuffled_forward_delta(self) -> None:
        module = _load_script_module()
        rows = [
            module.TensorActivityStats(
                variant=variant,
                tensor_name="forward_delta",
                shape="1x4x2",
                dtype="torch.float32",
                value_count=8,
                unique_value_count=8,
                zero_value_ratio=0.0,
                nonzero_value_ratio=1.0,
                mean_abs=rms,
                max_abs=rms,
                rms=rms,
                l2=rms,
                frame_count=4,
                frame_rms_mean=rms,
                frame_rms_min=rms,
                frame_rms_max=rms,
                frame_active_ratio=1.0,
            )
            for variant, rms in [("correct", 3.0), ("shuffled", 2.8), ("zero", 0.2)]
        ]
        pairs = [
            module.PairwiseTensorStats(
                tensor_name="forward_delta",
                variant_a="correct",
                variant_b="shuffled",
                cosine_similarity=0.93,
                mean_abs_difference=0.1,
                rms_difference=0.2,
                relative_l2_difference=0.08,
            )
        ]

        report = module.build_activity_report(rows, pairwise_rows=pairs)

        joined = "\n".join(report)
        self.assertIn("forward_delta correct/zero rms ratio", joined)
        self.assertIn("denoiser-level activity-shortcut", joined)
        self.assertIn("forward_delta correct/shuffled cosine=0.9300", joined)

    def test_parser_accepts_diagnostic_paths(self) -> None:
        module = _load_script_module()
        args = module.build_arg_parser().parse_args(
            [
                "--ckpt-path",
                "model.ckpt",
                "--reference-audio",
                "reference.wav",
                "--output-csv",
                "activity.csv",
                "--num-control-layers",
                "8",
                "--demo-control-variants",
                "correct/zero/shuffle",
                "--seconds-total",
                "10",
                "--no-prefer-ema",
                "--no-forward-delta",
            ]
        )

        self.assertEqual(args.ckpt_path, "model.ckpt")
        self.assertEqual(args.reference_audio, "reference.wav")
        self.assertEqual(args.output_csv, "activity.csv")
        self.assertEqual(args.num_control_layers, 8)
        self.assertFalse(args.prefer_ema)
        self.assertFalse(args.forward_delta)


if __name__ == "__main__":
    unittest.main()
