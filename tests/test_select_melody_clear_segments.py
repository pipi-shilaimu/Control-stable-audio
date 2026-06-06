from __future__ import annotations

import csv
import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


def _load_script_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "stable_audio_control" / "scripts" / "select_melody_clear_segments.py"
    spec = importlib.util.spec_from_file_location("select_melody_clear_segments", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_cqt_magnitude(pitches: list[int], *, n_bins: int = 128, peak: float = 1.0, noise: float = 0.03):
    frames = len(pitches)
    magnitude = torch.full((1, 2, n_bins, frames), noise, dtype=torch.float32)
    for frame_ix, pitch in enumerate(pitches):
        bin_ix = max(0, min(n_bins - 1, pitch))
        magnitude[0, :, bin_ix, frame_ix] = peak
        if bin_ix + 7 < n_bins:
            magnitude[0, :, bin_ix + 7, frame_ix] = peak * 0.12
    return magnitude


class SelectMelodyClearSegmentsTests(unittest.TestCase):
    def test_clear_moving_melody_scores_as_accept(self) -> None:
        module = _load_script_module()
        config = module.MelodyClearFilterConfig()
        pitches = [60] * 8 + [62] * 8 + [64] * 8 + [67] * 8 + [69] * 8 + [72] * 8
        magnitude = _make_cqt_magnitude(pitches)

        metrics = module.compute_melody_clear_metrics(magnitude, config=config)

        self.assertGreater(metrics.melody_clear_score, 0.65)
        self.assertEqual(metrics.decision, "accept")
        self.assertGreaterEqual(metrics.unique_pitch_count, 6)
        self.assertLess(metrics.pitch_jump_rate, 0.2)

    def test_silent_segment_scores_as_reject(self) -> None:
        module = _load_script_module()
        config = module.MelodyClearFilterConfig()
        magnitude = torch.zeros((1, 2, 128, 48), dtype=torch.float32)

        metrics = module.compute_melody_clear_metrics(magnitude, config=config)

        self.assertEqual(metrics.decision, "reject")
        self.assertGreater(metrics.silence_ratio, 0.9)
        self.assertLess(metrics.melody_clear_score, 0.3)

    def test_random_jump_contour_is_penalized(self) -> None:
        module = _load_script_module()
        config = module.MelodyClearFilterConfig()
        pitches = [20, 90, 25, 100, 30, 110, 35, 120] * 6
        magnitude = _make_cqt_magnitude(pitches)

        metrics = module.compute_melody_clear_metrics(magnitude, config=config)

        self.assertEqual(metrics.decision, "reject")
        self.assertGreater(metrics.pitch_jump_rate, 0.5)

    def test_write_candidates_csv_round_trips_expected_fields(self) -> None:
        module = _load_script_module()
        row = module.CandidateRow(
            rank=1,
            audio_path="track.mp3",
            filename="track.mp3",
            duration_sec=10.0,
            sample_rate=44_100,
            decision="accept",
            melody_clear_score=0.75,
            silence_ratio=0.01,
            voiced_ratio=0.99,
            top1_dominance=0.8,
            unique_pitch_count=12,
            pitch_motion_rate=0.2,
            pitch_jump_rate=0.0,
            low_pitch_ratio=0.1,
            pitch_range_semitones=18,
            error="",
        )

        with TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "candidates.csv"
            module.write_candidates_csv(csv_path, [row])
            with csv_path.open("r", encoding="utf-8", newline="") as fp:
                rows = list(csv.DictReader(fp))

        self.assertEqual(rows[0]["decision"], "accept")
        self.assertEqual(rows[0]["filename"], "track.mp3")
        self.assertEqual(rows[0]["melody_clear_score"], "0.75")

    def test_parser_accepts_worker_count_for_large_directories(self) -> None:
        module = _load_script_module()
        parser = module.build_arg_parser()

        args = parser.parse_args(
            [
                "--audio-dir",
                "audio",
                "--output-csv",
                "candidates.csv",
                "--workers",
                "4",
            ]
        )

        self.assertEqual(args.workers, 4)


if __name__ == "__main__":
    unittest.main()
