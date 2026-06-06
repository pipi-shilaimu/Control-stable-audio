from __future__ import annotations

import csv
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


def _load_script_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "stable_audio_control" / "scripts" / "export_melody_clear_accept_dataset.py"
    spec = importlib.util.spec_from_file_location("export_melody_clear_accept_dataset", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "audio_path",
        "filename",
        "decision",
        "melody_clear_score",
        "low_pitch_ratio",
        "top1_dominance",
        "pitch_jump_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class ExportMelodyClearAcceptDatasetTests(unittest.TestCase):
    def test_export_filters_accept_and_keeps_track_segments_in_same_split(self) -> None:
        module = _load_script_module()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            source_dir.mkdir()
            for name in ["0000001_seg000.mp3", "0000001_seg001.mp3", "0000002_seg000.mp3", "0000003_seg000.mp3"]:
                (source_dir / name).write_bytes(f"audio:{name}".encode("utf-8"))

            csv_path = root / "melody_clear.csv"
            _write_csv(
                csv_path,
                [
                    {
                        "rank": "1",
                        "audio_path": str(source_dir / "0000001_seg000.mp3"),
                        "filename": "0000001_seg000.mp3",
                        "decision": "accept",
                        "melody_clear_score": "0.80",
                        "low_pitch_ratio": "0.02",
                        "top1_dominance": "0.50",
                        "pitch_jump_rate": "0.10",
                    },
                    {
                        "rank": "2",
                        "audio_path": str(source_dir / "0000001_seg001.mp3"),
                        "filename": "0000001_seg001.mp3",
                        "decision": "accept",
                        "melody_clear_score": "0.78",
                        "low_pitch_ratio": "0.03",
                        "top1_dominance": "0.48",
                        "pitch_jump_rate": "0.11",
                    },
                    {
                        "rank": "3",
                        "audio_path": str(source_dir / "0000002_seg000.mp3"),
                        "filename": "0000002_seg000.mp3",
                        "decision": "accept",
                        "melody_clear_score": "0.70",
                        "low_pitch_ratio": "0.04",
                        "top1_dominance": "0.44",
                        "pitch_jump_rate": "0.12",
                    },
                    {
                        "rank": "4",
                        "audio_path": str(source_dir / "0000003_seg000.mp3"),
                        "filename": "0000003_seg000.mp3",
                        "decision": "reject",
                        "melody_clear_score": "0.20",
                        "low_pitch_ratio": "0.90",
                        "top1_dominance": "0.20",
                        "pitch_jump_rate": "0.80",
                    },
                ],
            )
            source_manifest = root / "source_manifest.json"
            source_manifest.write_text(
                json.dumps(
                    {
                        "0000001_seg000.mp3": {"prompt": "clear piano melody A", "track_id": "track_0000001"},
                        "0000001_seg001.mp3": {"prompt": "clear piano melody A2", "track_id": "track_0000001"},
                        "0000002_seg000.mp3": {"prompt": "clear piano melody B", "track_id": "track_0000002"},
                        "0000003_seg000.mp3": {"prompt": "bad bass", "track_id": "track_0000003"},
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            output_root = root / "piano_clean_v2"

            summary = module.export_accept_dataset(
                csv_path=csv_path,
                source_manifest_path=source_manifest,
                output_root=output_root,
                dataset_id="piano_clean_v2",
                val_ratio=0.5,
                seed=123,
                decisions=("accept",),
            )

            train_manifest = json.loads((output_root / "manifests" / "train.json").read_text(encoding="utf-8"))
            val_manifest = json.loads((output_root / "manifests" / "val.json").read_text(encoding="utf-8"))
            all_exported = {**train_manifest, **val_manifest}

            self.assertEqual(summary.selected_count, 3)
            self.assertEqual(len(all_exported), 3)
            self.assertNotIn("0000003_seg000.mp3", all_exported)
            self.assertEqual(all_exported["0000001_seg000.mp3"]["prompt"], "clear piano melody A")
            self.assertEqual(all_exported["0000001_seg000.mp3"]["melody_clear_score"], 0.8)
            self.assertTrue((output_root / "train").exists())
            self.assertTrue((output_root / "val").exists())

            train_track_ids = {entry["track_id"] for entry in train_manifest.values()}
            val_track_ids = {entry["track_id"] for entry in val_manifest.values()}
            self.assertTrue(train_track_ids.isdisjoint(val_track_ids))

            train_config = json.loads((output_root / "dataset_config_train.json").read_text(encoding="utf-8"))
            val_config = json.loads((output_root / "dataset_config_val.json").read_text(encoding="utf-8"))
            self.assertEqual(train_config["datasets"][0]["id"], "piano_clean_v2_train")
            self.assertEqual(val_config["datasets"][0]["id"], "piano_clean_v2_val")
            self.assertEqual(Path(train_config["datasets"][0]["path"]), (output_root / "train").resolve())

    def test_parser_accepts_required_paths_and_split_options(self) -> None:
        module = _load_script_module()
        args = module.build_arg_parser().parse_args(
            [
                "--csv",
                "melody.csv",
                "--source-manifest",
                "manifest.json",
                "--output-root",
                "dataset",
                "--dataset-id",
                "piano_clean_v2",
                "--val-ratio",
                "0.1",
                "--seed",
                "42",
            ]
        )

        self.assertEqual(args.csv, "melody.csv")
        self.assertEqual(args.source_manifest, "manifest.json")
        self.assertEqual(args.output_root, "dataset")
        self.assertEqual(args.dataset_id, "piano_clean_v2")
        self.assertEqual(args.val_ratio, 0.1)
        self.assertEqual(args.seed, 42)


if __name__ == "__main__":
    unittest.main()
