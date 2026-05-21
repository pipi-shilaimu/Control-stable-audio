"""Test that dataset metadata modules can load and return correct prompts."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "stable_audio_control" / "data"


class TestSongDescriberMetadata(unittest.TestCase):
    """song_describer_metadata: existing known-good dataset."""

    def _get_metadata(self, info, audio=None):
        from stable_audio_control.data.song_describer_metadata import get_custom_metadata
        return get_custom_metadata(info, audio)

    def test_known_track_returns_prompt(self):
        info = {
            "path": str(DATA_DIR / "song_describer" / "train" / "000000_track_id-1004034_caption_id-859.wav"),
            "relpath": "000000_track_id-1004034_caption_id-859.wav",
        }
        result = self._get_metadata(info)
        self.assertIn("prompt", result)
        self.assertTrue(len(result["prompt"]) > 10)

    def test_missing_track_raises_key_error(self):
        info = {
            "path": str(DATA_DIR / "song_describer" / "train" / "nonexistent.wav"),
            "relpath": "nonexistent.wav",
        }
        with self.assertRaises(KeyError):
            self._get_metadata(info)

    def test_returns_extra_fields(self):
        info = {
            "path": str(DATA_DIR / "song_describer" / "train" / "000000_track_id-1004034_caption_id-859.wav"),
            "relpath": "000000_track_id-1004034_caption_id-859.wav",
        }
        result = self._get_metadata(info)
        self.assertIn("song_describer_track_id", result)
        self.assertIn("song_describer_caption", result)

    def test_manifest_cache_hit_does_not_raise(self):
        info = {
            "path": str(DATA_DIR / "song_describer" / "train" / "000000_track_id-1004034_caption_id-859.wav"),
            "relpath": "000000_track_id-1004034_caption_id-859.wav",
        }
        self._get_metadata(info)
        result = self._get_metadata(info)
        self.assertIn("prompt", result)


class TestMTGJamendoMetadata(unittest.TestCase):
    """mtg_jamendo_metadata: newly added dataset."""

    @classmethod
    def setUpClass(cls):
        cls.manifest_path = DATA_DIR / "mtg_jamendo" / "manifests" / "train.json"
        if not cls.manifest_path.exists():
            raise unittest.SkipTest("MTG-Jamendo manifest not found, skipping.")
        cls.manifest = json.loads(cls.manifest_path.read_text("utf-8"))

    def _get_metadata(self, info, audio=None):
        from stable_audio_control.data.mtg_jamendo_metadata import get_custom_metadata
        return get_custom_metadata(info, audio)

    def test_manifest_is_not_empty(self):
        self.assertGreater(len(self.manifest), 0)

    def test_known_track_returns_prompt(self):
        first_key = list(self.manifest.keys())[0]
        info = {
            "path": str(DATA_DIR / "mtg_jamendo" / "train" / first_key),
            "relpath": first_key,
        }
        result = self._get_metadata(info)
        self.assertIn("prompt", result)
        prompt = result["prompt"]
        self.assertIn("piano", prompt, f"Expected 'piano' in prompt, got: '{prompt}'")
        self.assertNotIn("  ", prompt, f"Prompt has double space: '{prompt}'")

    def test_every_wav_has_matching_manifest_entry(self):
        audio_dir = DATA_DIR / "mtg_jamendo" / "train"
        wav_files = sorted(f.name for f in audio_dir.glob("*.wav"))
        manifest_keys = set(self.manifest.keys())
        for wav in wav_files:
            with self.subTest(wav=wav):
                self.assertIn(wav, manifest_keys, f"{wav} missing from manifest")

    def test_every_manifest_entry_has_corresponding_wav(self):
        audio_dir = DATA_DIR / "mtg_jamendo" / "train"
        wav_files = {f.name for f in audio_dir.glob("*.wav")}
        for key in self.manifest:
            with self.subTest(key=key):
                self.assertIn(key, wav_files, f"Manifest key {key} has no WAV file")

    def test_returns_extra_fields(self):
        first_key = list(self.manifest.keys())[0]
        info = {
            "path": str(DATA_DIR / "mtg_jamendo" / "train" / first_key),
            "relpath": first_key,
        }
        result = self._get_metadata(info)
        self.assertIn("mtg_tags", result)
        self.assertIn("mtg_track_id", result)

    def test_missing_track_returns_default(self):
        info = {
            "path": str(DATA_DIR / "mtg_jamendo" / "train" / "nonexistent.wav"),
            "relpath": "nonexistent.wav",
        }
        result = self._get_metadata(info)
        self.assertEqual(result["prompt"], "music")

    def test_manifest_cache_hit_does_not_raise(self):
        info = {
            "path": str(DATA_DIR / "mtg_jamendo" / "train" / "nonexistent.wav"),
            "relpath": "nonexistent.wav",
        }
        self._get_metadata(info)
        result = self._get_metadata(info)
        self.assertIn("prompt", result)


if __name__ == "__main__":
    unittest.main()
