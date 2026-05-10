from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_batch_script_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "stable_audio_control" / "scripts" / "compare_controlnet_generation_random.py"
    spec = importlib.util.spec_from_file_location("compare_controlnet_generation_random", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RandomControlNetGenerationCompareTests(unittest.TestCase):
    def test_finds_supported_audio_files_recursively_in_stable_order(self) -> None:
        module = _load_batch_script_module()

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "b").mkdir()
            (root / "a").mkdir()
            wav = root / "b" / "ref.wav"
            flac = root / "a" / "ref.flac"
            text = root / "a" / "notes.txt"
            wav.write_bytes(b"placeholder")
            flac.write_bytes(b"placeholder")
            text.write_text("skip", encoding="utf-8")

            files = module.find_audio_files(root, extensions=(".wav", ".flac"))

        self.assertEqual(files, [flac, wav])

    def test_build_random_generation_plan_uses_unique_references_and_seeds(self) -> None:
        module = _load_batch_script_module()
        audio_paths = [Path(f"ref_{ix}.wav") for ix in range(5)]

        plan = module.build_random_generation_plan(
            audio_paths,
            num_samples=3,
            random_seed=123,
            seed_min=10,
            seed_max=99,
            allow_reference_reuse=False,
        )

        self.assertEqual(len(plan), 3)
        self.assertEqual(len({item.reference_audio_path for item in plan}), 3)
        self.assertEqual(len({item.seed for item in plan}), 3)
        self.assertTrue(all(10 <= item.seed <= 99 for item in plan))
        self.assertEqual([item.index for item in plan], [0, 1, 2])

    def test_batch_script_parser_defaults_to_ten_samples_from_data_root(self) -> None:
        module = _load_batch_script_module()
        parser = module.build_arg_parser()

        args = parser.parse_args(
            [
                "--ckpt-path",
                "model.ckpt",
                "--prompt",
                "a clear melody",
            ]
        )

        self.assertEqual(args.num_samples, 10)
        self.assertEqual(args.reference_root, "stable_audio_control/data")
        self.assertEqual(args.output_dir, "outputs/controlnet_generation_random")
        self.assertEqual(args.melody_feature, "cqt")

    def test_pretrained_model_loader_wraps_huggingface_t5_errors_with_actionable_context(self) -> None:
        module = _load_batch_script_module()

        def failing_loader(model_name: str):
            raise RuntimeError("Cannot send a request, as the client has been closed.")

        with self.assertRaisesRegex(RuntimeError, "T5 tokenizer/encoder"):
            module.load_pretrained_model_with_context(
                "stabilityai/stable-audio-open-1.0",
                loader=failing_loader,
            )

        try:
            module.load_pretrained_model_with_context(
                "stabilityai/stable-audio-open-1.0",
                loader=failing_loader,
            )
        except RuntimeError as exc:
            message = str(exc)
            self.assertIn("t5-base", message)
            self.assertIn("HF_HOME", message)
            self.assertIn("get_pretrained_model", message)
            self.assertIn("stabilityai/stable-audio-open-1.0", message)
        else:
            self.fail("Expected RuntimeError")

    def test_validate_args_rejects_missing_checkpoint_before_model_load(self) -> None:
        module = _load_batch_script_module()
        parser = module.build_arg_parser()
        args = parser.parse_args(
            [
                "--ckpt-path",
                "missing-model.ckpt",
                "--prompt",
                "a clear melody",
            ]
        )

        with self.assertRaisesRegex(FileNotFoundError, "Checkpoint not found"):
            module.validate_args(args)

    def test_formats_similarity_progress_and_summary_lines(self) -> None:
        module = _load_batch_script_module()

        item_line = module.format_similarity_progress_line(
            index=1,
            total=3,
            similarity={"metric_name": "cqt_topk_pitch_overlap_rate", "score": 0.8123456},
            output_path=Path("sample/similarity.json"),
        )
        self.assertIn("[2/3]", item_line)
        self.assertIn("similarity_score=0.812346", item_line)
        self.assertIn("metric=cqt_topk_pitch_overlap_rate", item_line)
        self.assertIn("sample/similarity.json", item_line)

        summary_line = module.format_similarity_summary_line(
            {
                "metric_name": "cqt_topk_pitch_overlap_rate",
                "mean_score": 0.7,
                "min_score": 0.5,
                "max_score": 0.9,
            }
        )
        self.assertEqual(
            summary_line,
            "similarity_summary metric=cqt_topk_pitch_overlap_rate mean=0.700000 min=0.500000 max=0.900000",
        )


if __name__ == "__main__":
    unittest.main()
