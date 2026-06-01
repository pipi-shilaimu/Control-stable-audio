"""TDD: 测试 organize_dataset 的音频-清单双向一致性校验。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from stable_audio_control.data.organize_dataset import validate_manifest_audio


class TestValidateManifestAudio(unittest.TestCase):
    """双向一致性校验的单元测试。"""

    # ── 正向场景 ──────────────────────────────────────────

    def test_perfect_match(self):
        """音频与 manifest 完全一致 -> is_consistent=True, 两个方向均无缺失。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            (audio_dir / "a.wav").write_text("dummy")
            (audio_dir / "b.wav").write_text("dummy")

            manifest = {"a.wav": {"prompt": "p1"}, "b.wav": {"prompt": "p2"}}

            result = validate_manifest_audio(audio_dir, manifest)

            self.assertTrue(result["is_consistent"])
            self.assertEqual(result["manifest_count"], 2)
            self.assertEqual(result["audio_count"], 2)
            self.assertEqual(result["missing_audio"], [])
            self.assertEqual(result["missing_manifest"], [])

    def test_perfect_match_single_file(self):
        """只有一个文件时也能正确校验。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            (audio_dir / "only.wav").write_text("dummy")

            manifest = {"only.wav": {"prompt": "one"}}

            result = validate_manifest_audio(audio_dir, manifest)

            self.assertTrue(result["is_consistent"])
            self.assertEqual(result["manifest_count"], 1)
            self.assertEqual(result["audio_count"], 1)

    # ── 反向场景：manifest 多出条目 ─────────────────────────

    def test_manifest_has_extra_keys(self):
        """manifest 中有的 key，磁盘上没有对应音频 -> missing_audio 非空。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            (audio_dir / "a.wav").write_text("dummy")

            manifest = {"a.wav": {"prompt": "p1"}, "ghost.wav": {"prompt": "gone"}}

            result = validate_manifest_audio(audio_dir, manifest)

            self.assertFalse(result["is_consistent"])
            self.assertEqual(result["manifest_count"], 2)
            self.assertEqual(result["audio_count"], 1)
            self.assertEqual(result["missing_audio"], ["ghost.wav"])
            self.assertEqual(result["missing_manifest"], [])

    # ── 反向场景：磁盘多出文件 ─────────────────────────────

    def test_audio_has_extra_files(self):
        """磁盘上有音频，但 manifest 里没有对应 key -> missing_manifest 非空。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            (audio_dir / "a.wav").write_text("dummy")
            (audio_dir / "orphan.wav").write_text("dummy")

            manifest = {"a.wav": {"prompt": "p1"}}

            result = validate_manifest_audio(audio_dir, manifest)

            self.assertFalse(result["is_consistent"])
            self.assertEqual(result["manifest_count"], 1)
            self.assertEqual(result["audio_count"], 2)
            self.assertEqual(result["missing_audio"], [])
            self.assertEqual(result["missing_manifest"], ["orphan.wav"])

    # ── 完全不对齐 ───────────────────────────────────────

    def test_complete_mismatch(self):
        """两边的集合完全不重叠。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            (audio_dir / "x.wav").write_text("dummy")
            (audio_dir / "y.wav").write_text("dummy")

            manifest = {"p.wav": {"prompt": "p"}, "q.wav": {"prompt": "q"}}

            result = validate_manifest_audio(audio_dir, manifest)

            self.assertFalse(result["is_consistent"])
            self.assertEqual(result["missing_audio"], ["p.wav", "q.wav"])
            self.assertEqual(result["missing_manifest"], ["x.wav", "y.wav"])

    # ── 边界：空 manifest ─────────────────────────────────

    def test_empty_manifest_with_audio(self):
        """磁盘有音频但 manifest 为空 -> missing_manifest 列出全部音频。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            (audio_dir / "a.wav").write_text("dummy")

            manifest: dict[str, object] = {}

            result = validate_manifest_audio(audio_dir, manifest)

            self.assertFalse(result["is_consistent"])
            self.assertEqual(result["manifest_count"], 0)
            self.assertEqual(result["audio_count"], 1)
            self.assertEqual(result["missing_audio"], [])
            self.assertEqual(result["missing_manifest"], ["a.wav"])

    # ── 边界：空音频目录 ─────────────────────────────────

    def test_empty_audio_dir_with_manifest(self):
        """manifest 有条目但音频目录为空 -> missing_audio 列出全部 key。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_dir = root / "audio"
            audio_dir.mkdir()

            manifest = {"gone.wav": {"prompt": "x"}}

            result = validate_manifest_audio(audio_dir, manifest)

            self.assertFalse(result["is_consistent"])
            self.assertEqual(result["manifest_count"], 1)
            self.assertEqual(result["audio_count"], 0)
            self.assertEqual(result["missing_audio"], ["gone.wav"])
            self.assertEqual(result["missing_manifest"], [])

    def test_both_empty(self):
        """两边都为空 -> 理论上一致（但实际无数据）。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_dir = root / "audio"
            audio_dir.mkdir()

            manifest: dict[str, object] = {}

            result = validate_manifest_audio(audio_dir, manifest)

            self.assertTrue(result["is_consistent"])
            self.assertEqual(result["manifest_count"], 0)
            self.assertEqual(result["audio_count"], 0)
            self.assertEqual(result["missing_audio"], [])
            self.assertEqual(result["missing_manifest"], [])

    # ── audio_extensions 参数 ─────────────────────────────

    def test_audio_extensions_filters_non_wav(self):
        """audio_extensions=('.wav',) 时排除 .txt 等非音频文件。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            (audio_dir / "a.wav").write_text("dummy")
            (audio_dir / "readme.txt").write_text("hello")
            (audio_dir / "subdir").mkdir()

            manifest = {"a.wav": {"prompt": "p1"}}

            result = validate_manifest_audio(
                audio_dir, manifest, audio_extensions=(".wav",)
            )

            self.assertTrue(result["is_consistent"])
            self.assertEqual(result["audio_count"], 1)
            self.assertEqual(result["manifest_count"], 1)

    def test_audio_extensions_none_counts_all_files(self):
        """audio_extensions=None 时统计所有文件（包括 .txt）。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            (audio_dir / "a.wav").write_text("dummy")
            (audio_dir / "readme.txt").write_text("hello")

            manifest = {"a.wav": {"prompt": "p1"}}

            result = validate_manifest_audio(
                audio_dir, manifest, audio_extensions=None
            )

            self.assertFalse(result["is_consistent"])
            self.assertEqual(result["audio_count"], 2)
            self.assertIn("readme.txt", result["missing_manifest"])

    # ── 数量相等但内容不同 ───────────────────────────────

    def test_same_count_different_keys(self):
        """两边数量相等但 key 完全不同，仍然报不一致。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            (audio_dir / "a.wav").write_text("dummy")
            (audio_dir / "b.wav").write_text("dummy")

            manifest = {"c.wav": {"prompt": "c"}, "d.wav": {"prompt": "d"}}

            result = validate_manifest_audio(audio_dir, manifest)

            self.assertFalse(result["is_consistent"])
            self.assertEqual(result["manifest_count"], 2)
            self.assertEqual(result["audio_count"], 2)
            self.assertEqual(len(result["missing_audio"]), 2)
            self.assertEqual(len(result["missing_manifest"]), 2)


if __name__ == "__main__":
    unittest.main()
