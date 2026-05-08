from __future__ import annotations

import importlib
import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def import_dataset_load_with_fake_datasets():
    sys.modules.pop("stable_audio_control.data.dataset_load", None)

    def fail_load_dataset(*args, **kwargs):
        raise AssertionError("load_dataset must not be called during module import")

    previous = sys.modules.get("datasets")
    sys.modules["datasets"] = types.SimpleNamespace(load_dataset=fail_load_dataset)
    try:
        return importlib.import_module("stable_audio_control.data.dataset_load")
    finally:
        if previous is None:
            sys.modules.pop("datasets", None)
        else:
            sys.modules["datasets"] = previous


class DatasetLoadExportTests(unittest.TestCase):
    def test_module_import_does_not_load_huggingface_dataset(self) -> None:
        module = import_dataset_load_with_fake_datasets()

        self.assertTrue(hasattr(module, "export_samples_to_audio_dir"))

    def test_decode_audio_reads_huggingface_audio_mapping_path(self) -> None:
        module = import_dataset_load_with_fake_datasets()

        with TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "tone.wav"
            expected = np.linspace(-0.25, 0.25, num=64, dtype=np.float32)
            sf.write(audio_path, expected, 16_000, format="WAV", subtype="FLOAT")

            audio, sample_rate = module.decode_audio({"path": str(audio_path)})

            self.assertEqual(sample_rate, 16_000)
            np.testing.assert_allclose(audio, expected, atol=1e-6)

    def test_decode_audio_reads_huggingface_audio_mapping_bytes(self) -> None:
        module = import_dataset_load_with_fake_datasets()

        with TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "tone.wav"
            expected = np.linspace(-0.25, 0.25, num=64, dtype=np.float32)
            sf.write(audio_path, expected, 22_050, format="WAV", subtype="FLOAT")

            audio, sample_rate = module.decode_audio({"bytes": audio_path.read_bytes()})

            self.assertEqual(sample_rate, 22_050)
            np.testing.assert_allclose(audio, expected, atol=1e-6)

    def test_export_samples_writes_audio_manifest_and_dataset_config(self) -> None:
        module = import_dataset_load_with_fake_datasets()
        samples = [
            {
                "caption": "bright piano melody",
                "caption_id": 10,
                "track_id": 20,
                "is_valid_subset": True,
                "path": {
                    "array": np.linspace(-0.5, 0.5, num=64, dtype=np.float32),
                    "sampling_rate": 8_000,
                },
            },
            {
                "caption": "filtered vocal sample",
                "caption_id": 11,
                "track_id": 21,
                "is_valid_subset": False,
                "path": {
                    "array": np.zeros(64, dtype=np.float32),
                    "sampling_rate": 8_000,
                },
            },
        ]

        with TemporaryDirectory() as tmp:
            result = module.export_samples_to_audio_dir(
                samples,
                output_dir=Path(tmp),
                split="train",
                valid_only=True,
                metadata_module_path=REPO_ROOT / "stable_audio_control" / "data" / "song_describer_metadata.py",
            )

            self.assertEqual(result.exported_count, 1)
            self.assertTrue(result.manifest_path.exists())
            self.assertTrue(result.dataset_config_path.exists())

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest), 1)
            relpath, entry = next(iter(manifest.items()))
            self.assertEqual(entry["prompt"], "bright piano melody")
            self.assertEqual(entry["caption_id"], 10)

            audio_path = Path(tmp) / "train" / relpath
            self.assertTrue(audio_path.exists())
            self.assertEqual(sf.info(audio_path).samplerate, 8_000)

            dataset_config = json.loads(result.dataset_config_path.read_text(encoding="utf-8"))
            self.assertEqual(dataset_config["dataset_type"], "audio_dir")
            self.assertEqual(dataset_config["datasets"][0]["path"], str((Path(tmp) / "train").resolve()))
            self.assertEqual(
                dataset_config["datasets"][0]["custom_metadata_module"],
                str((REPO_ROOT / "stable_audio_control" / "data" / "song_describer_metadata.py").resolve()),
            )

    def test_custom_metadata_reads_manifest_for_audio_relpath(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_dir = root / "train"
            manifest_dir = root / "manifests"
            audio_dir.mkdir(parents=True)
            manifest_dir.mkdir(parents=True)
            audio_path = audio_dir / "tone.wav"
            sf.write(audio_path, np.zeros(32, dtype=np.float32), 8_000)
            (manifest_dir / "train.json").write_text(
                json.dumps({"tone.wav": {"prompt": "soft synth loop", "track_id": 123}}),
                encoding="utf-8",
            )

            metadata_module = importlib.import_module("stable_audio_control.data.song_describer_metadata")
            metadata = metadata_module.get_custom_metadata(
                {"path": str(audio_path), "relpath": "tone.wav"},
                audio=None,
            )

            self.assertEqual(metadata["prompt"], "soft synth loop")
            self.assertEqual(metadata["song_describer_track_id"], 123)


if __name__ == "__main__":
    unittest.main()
