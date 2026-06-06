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
    script_path = repo_root / "stable_audio_control" / "scripts" / "build_mtg_piano_candidates.py"
    spec = importlib.util.spec_from_file_location("build_mtg_piano_candidates", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_tsv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join("\t".join(row) for row in rows) + "\n", encoding="utf-8")


class BuildMTGPianoCandidatesTests(unittest.TestCase):
    def test_extract_track_id_from_segment_filename(self) -> None:
        module = _load_script_module()

        self.assertEqual(module.extract_track_id_from_name("0095400_seg000.wav"), "track_0095400")
        self.assertEqual(module.extract_track_id_from_name("1000113_seg020.mp3"), "track_1000113")
        self.assertEqual(module.extract_track_id_from_name("track_0000382.mp3"), "track_0000382")
        self.assertEqual(module.extract_track_id_from_name("82/382.mp3"), "track_0000382")

    def test_build_rows_ranks_piano_segments_and_rejects_voice(self) -> None:
        module = _load_script_module()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            mtg_root = root / "mtg"
            audio_dir = root / "audio_10s"
            audio_dir.mkdir()
            for filename in ["0000001_seg000.mp3", "0000002_seg000.mp3", "0000003_seg000.mp3", "0000004_seg000.mp3"]:
                (audio_dir / filename).write_bytes(b"fake")

            _write_tsv(
                mtg_root / "data" / "raw_30s_cleantags.tsv",
                [
                    ["TRACK_ID", "ARTIST_ID", "ALBUM_ID", "PATH", "DURATION", "TAGS"],
                    ["track_0000001", "artist_1", "album_1", "01/1.mp3", "90.0", "genre---classical\tinstrument---piano"],
                    [
                        "track_0000002",
                        "artist_2",
                        "album_2",
                        "02/2.mp3",
                        "91.0",
                        "instrument---piano\tinstrument---voice",
                    ],
                    ["track_0000003", "artist_3", "album_3", "03/3.mp3", "92.0", "instrument---guitar"],
                    [
                        "track_0000004",
                        "artist_4",
                        "album_4",
                        "04/4.mp3",
                        "93.0",
                        "instrument---piano\tinstrument---drums\tinstrument---bass",
                    ],
                ],
            )
            _write_tsv(
                mtg_root / "data" / "raw.meta.tsv",
                [
                    ["TRACK_ID", "ARTIST_ID", "ALBUM_ID", "TRACK_NAME", "ARTIST_NAME", "ALBUM_NAME", "RELEASEDATE", "URL"],
                    ["track_0000001", "artist_1", "album_1", "Clear Piano", "Alice", "Solo", "2020", "url1"],
                    ["track_0000002", "artist_2", "album_2", "Voice Piano", "Bob", "Vocal", "2020", "url2"],
                    ["track_0000004", "artist_4", "album_4", "Busy Piano", "Duo", "Band", "2020", "url4"],
                ],
            )

            rows = module.build_piano_candidate_rows(
                mtg_root=mtg_root,
                audio_dir=audio_dir,
                config=module.PianoCandidateConfig(max_soft_excludes=1),
            )

        by_file = {row.filename: row for row in rows}
        self.assertEqual(by_file["0000001_seg000.mp3"].decision, "accept")
        self.assertEqual(by_file["0000002_seg000.mp3"].decision, "reject")
        self.assertEqual(by_file["0000003_seg000.mp3"].decision, "reject")
        self.assertEqual(by_file["0000004_seg000.mp3"].decision, "maybe")
        self.assertEqual(by_file["0000001_seg000.mp3"].artist_name, "Alice")
        self.assertGreater(by_file["0000001_seg000.mp3"].tag_score, by_file["0000004_seg000.mp3"].tag_score)
        self.assertLess(by_file["0000001_seg000.mp3"].rank, by_file["0000004_seg000.mp3"].rank)

    def test_write_outputs_include_manifest_with_original_prompt(self) -> None:
        module = _load_script_module()
        row = module.PianoCandidateRow(
            rank=1,
            decision="accept",
            tag_score=1.35,
            track_id="track_0000001",
            filename="0000001_seg000.mp3",
            audio_path="audio/0000001_seg000.mp3",
            mtg_path="01/1.mp3",
            duration=90.0,
            artist_id="artist_1",
            album_id="album_1",
            artist_name="Alice",
            track_name="Clear Piano",
            album_name="Solo",
            tags="genre---classical\tinstrument---piano",
            include_tags_hit="instrument---piano",
            hard_exclude_tags_hit="",
            soft_exclude_tags_hit="",
            boost_tags_hit="genre---classical",
            prompt="bright piano melody",
            prompt_hits="piano",
            prompt_penalties="",
            reason="piano candidate",
        )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "candidates.csv"
            jsonl_path = root / "candidates.jsonl"
            manifest_path = root / "manifest.json"
            module.write_candidate_outputs(
                rows=[row],
                output_csv=csv_path,
                output_jsonl=jsonl_path,
                output_manifest=manifest_path,
            )

            with csv_path.open("r", encoding="utf-8", newline="") as fp:
                csv_rows = list(csv.DictReader(fp))
            jsonl_rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(csv_rows[0]["decision"], "accept")
        self.assertEqual(jsonl_rows[0]["track_id"], "track_0000001")
        self.assertEqual(manifest["0000001_seg000.mp3"]["prompt"], "bright piano melody")
        self.assertEqual(manifest["0000001_seg000.mp3"]["track_id"], "track_0000001")
        self.assertEqual(manifest["0000001_seg000.mp3"]["decision"], "accept")

    def test_select_rows_for_export_limits_segments_per_track(self) -> None:
        module = _load_script_module()

        def make_row(track_id: str, filename: str, rank: int) -> object:
            return module.PianoCandidateRow(
                rank=rank,
                decision="accept",
                tag_score=1.0,
                track_id=track_id,
                filename=filename,
                audio_path=filename,
                mtg_path=f"{track_id}.mp3",
                duration=90.0,
                artist_id="artist",
                album_id="album",
                artist_name="",
                track_name="",
                album_name="",
                tags="instrument---piano",
                include_tags_hit="instrument---piano",
                hard_exclude_tags_hit="",
                soft_exclude_tags_hit="",
                boost_tags_hit="",
                prompt="piano",
                prompt_hits="piano",
                prompt_penalties="",
                reason="piano candidate",
            )

        rows = [
            make_row("track_0000001", "0000001_seg000.mp3", 1),
            make_row("track_0000001", "0000001_seg001.mp3", 2),
            make_row("track_0000002", "0000002_seg000.mp3", 3),
        ]

        selected = module.select_rows_for_export(rows, decisions=("accept",), max_per_track=1)

        self.assertEqual([row.filename for row in selected], ["0000001_seg000.mp3", "0000002_seg000.mp3"])

    def test_copy_selected_audio_files_uses_selected_rows(self) -> None:
        module = _load_script_module()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp3"
            source.write_bytes(b"audio")
            copy_dir = root / "copy"
            row = module.PianoCandidateRow(
                rank=1,
                decision="accept",
                tag_score=1.0,
                track_id="track_0000001",
                filename="0000001_seg000.mp3",
                audio_path=str(source),
                mtg_path="01/1.mp3",
                duration=90.0,
                artist_id="artist",
                album_id="album",
                artist_name="",
                track_name="",
                album_name="",
                tags="instrument---piano",
                include_tags_hit="instrument---piano",
                hard_exclude_tags_hit="",
                soft_exclude_tags_hit="",
                boost_tags_hit="",
                prompt="piano",
                prompt_hits="piano",
                prompt_penalties="",
                reason="piano candidate",
            )

            copied = module.copy_selected_audio_files([row], copy_dir)

            self.assertEqual(copied, 1)
            self.assertTrue((copy_dir / "0000001_seg000.mp3").exists())

    def test_parser_accepts_audio_dir_and_output_paths(self) -> None:
        module = _load_script_module()
        args = module.build_arg_parser().parse_args(
            [
                "--audio-dir",
                "audio_10s",
                "--output-csv",
                "candidates.csv",
                "--output-manifest",
                "manifest.json",
                "--max-soft-excludes",
                "0",
                "--copy-audio-dir",
                "piano_audio",
                "--export-max-per-track",
                "3",
            ]
        )

        self.assertEqual(args.audio_dir, "audio_10s")
        self.assertEqual(args.output_manifest, "manifest.json")
        self.assertEqual(args.max_soft_excludes, 0)
        self.assertEqual(args.copy_audio_dir, "piano_audio")
        self.assertEqual(args.export_max_per_track, 3)


if __name__ == "__main__":
    unittest.main()
