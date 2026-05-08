from __future__ import annotations

import importlib.util
import contextlib
import io
import sys
import tempfile
import types
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "fft_cqt_visual_compare.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fft_cqt_visual_compare", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sine_wave(sr: int = 22050, seconds: float = 0.5, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    return (0.5 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


class ProcessingTests(unittest.TestCase):
    def test_normalize_bars_handles_silence(self):
        mod = load_module()

        values = mod.normalize_bars(np.zeros((4, 8), dtype=np.float32))

        self.assertEqual(values.shape, (4, 8))
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all((0.0 <= values) & (values <= 1.0)))

    def test_make_fft_bars_returns_requested_bar_count(self):
        mod = load_module()

        bars = mod.make_fft_bars(
            sine_wave(),
            sr=22050,
            hop_length=256,
            n_fft=1024,
            bars=32,
        )

        self.assertEqual(bars.shape[1], 32)
        self.assertGreater(bars.shape[0], 0)
        self.assertTrue(np.all(np.isfinite(bars)))

    def test_make_cqt_bars_returns_requested_bin_count(self):
        mod = load_module()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            bars = mod.make_cqt_bars(
                sine_wave(),
                sr=22050,
                hop_length=256,
                cqt_bins=36,
                bins_per_octave=12,
            )

        self.assertEqual(bars.shape[1], 36)
        self.assertGreater(bars.shape[0], 0)
        self.assertTrue(np.all(np.isfinite(bars)))

    def test_make_cqt_bars_can_return_topk_highlight_mask(self):
        mod = load_module()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            bars, mask = mod.make_cqt_bars(
                sine_wave(),
                sr=22050,
                hop_length=256,
                cqt_bins=24,
                bins_per_octave=12,
                return_highlight=True,
            )

        self.assertEqual(bars.shape, mask.shape)
        self.assertEqual(mask.dtype, np.bool_)
        self.assertTrue(np.all(mask.sum(axis=1) <= 4))
        self.assertGreater(int(mask.sum()), 0)
        self.assertTrue(np.all(np.isfinite(bars)))

    def test_boost_topk_per_frame_marks_and_multiplies_loudest_bins(self):
        mod = load_module()
        values = np.array(
            [
                [1.0, 4.0, 2.0, 8.0, 3.0],
                [9.0, 7.0, 5.0, 3.0, 1.0],
            ],
            dtype=np.float32,
        )

        boosted, mask = mod.boost_topk_per_frame(values, top_k=2, multiplier=1.5)

        expected = values.copy()
        expected[0, [1, 3]] *= 1.5
        expected[1, [0, 1]] *= 1.5
        np.testing.assert_allclose(boosted, expected)
        np.testing.assert_array_equal(
            mask,
            np.array(
                [
                    [False, True, False, True, False],
                    [True, True, False, False, False],
                ],
                dtype=bool,
            ),
        )
        np.testing.assert_allclose(values[0], [1.0, 4.0, 2.0, 8.0, 3.0])

    def test_resample_bool_frames_uses_nearest_source_frames(self):
        mod = load_module()
        mask = np.array(
            [
                [True, False],
                [False, True],
                [True, True],
            ],
            dtype=bool,
        )

        resampled = mod.resample_bool_frames(mask, target_frames=5)

        np.testing.assert_array_equal(
            resampled,
            np.array(
                [
                    [True, False],
                    [False, True],
                    [False, True],
                    [True, True],
                    [True, True],
                ],
                dtype=bool,
            ),
        )

    def test_resample_bar_frames_returns_target_count_and_preserves_edges(self):
        mod = load_module()
        values = np.array(
            [
                [0.0, 0.0],
                [0.25, 0.5],
                [0.5, 0.75],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )

        resampled = mod.resample_bar_frames(values, target_frames=2)

        self.assertEqual(resampled.shape, (2, 2))
        np.testing.assert_allclose(resampled[0], values[0])
        np.testing.assert_allclose(resampled[-1], values[-1])

    def test_describe_frequency_layout_explains_ranges(self):
        mod = load_module()

        info = mod.describe_frequency_layout(
            sample_rate=44100,
            n_fft=2048,
            fft_bars=64,
            cqt_bins=84,
            bins_per_octave=12,
        )

        self.assertIn("21.5 Hz - 22050.0 Hz", info.fft_axis_label)
        self.assertIn("21.5 Hz/bin", info.fft_summary)
        self.assertIn("32.7 Hz - 3951.1 Hz", info.cqt_axis_label)
        self.assertIn("semitone", info.cqt_summary)


class ProgressTests(unittest.TestCase):
    def test_progress_reporter_prints_frame_progress(self):
        mod = load_module()
        stream = io.StringIO()
        reporter = mod.ProgressReporter(label="Rendering", total_frames=4, stream=stream, use_tqdm=False)

        reporter(0, 4)
        reporter(3, 4)
        reporter.close()

        output = stream.getvalue()
        self.assertIn("Rendering", output)
        self.assertIn("4/4", output)
        self.assertIn("100.0%", output)

    def test_progress_reporter_defaults_to_stdout(self):
        mod = load_module()
        stdout = io.StringIO()
        stderr = io.StringIO()

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            reporter = mod.ProgressReporter(label="Rendering", total_frames=1, use_tqdm=False)
            reporter(0, 1)
            reporter.close()

        self.assertIn("Rendering", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")


class CliAndAudioTests(unittest.TestCase):
    def test_parse_args_uses_higher_resolution_fft_defaults(self):
        mod = load_module()

        args = mod.parse_args([])

        self.assertEqual(args.bars, 128)
        self.assertEqual(args.n_fft, 4096)
        self.assertEqual(args.renderer, "auto")
        self.assertEqual(args.encoder, "auto")
        self.assertEqual(args.width, 1280)
        self.assertEqual(args.height, 720)
        self.assertEqual(args.ncmdump.name, "ncmdump.exe")
        self.assertFalse(args.keep_converted)
        self.assertEqual(args.output, Path("music.mp4"))

    def test_parse_args_defaults_output_to_input_stem(self):
        mod = load_module()

        self.assertEqual(mod.parse_args(["song.mp3"]).output, Path("song.mp4"))
        self.assertEqual(mod.parse_args(["song.ncm"]).output, Path("song.mp4"))
        self.assertEqual(
            mod.parse_args([str(Path("nested") / "song.name.flac")]).output,
            Path("nested") / "song.name.mp4",
        )

    def test_parse_args_keeps_explicit_output(self):
        mod = load_module()

        args = mod.parse_args(["song.ncm", "--output", "custom.mp4"])

        self.assertEqual(args.output, Path("custom.mp4"))

    def test_select_renderer_uses_fast_for_mp4_and_matplotlib_for_gif(self):
        mod = load_module()

        self.assertEqual(mod.select_renderer(Path("out.mp4"), "auto"), "fast")
        self.assertEqual(mod.select_renderer(Path("out.gif"), "auto"), "matplotlib")

        with self.assertRaises(ValueError):
            mod.select_renderer(Path("out.gif"), "fast")

    def test_select_video_encoder_prefers_nvenc_when_available(self):
        mod = load_module()

        self.assertEqual(mod.select_video_encoder("auto", {"h264_nvenc", "libx264"}), "h264_nvenc")
        self.assertEqual(mod.select_video_encoder("auto", {"libx264"}), "libx264")
        self.assertEqual(mod.select_video_encoder("libx264", {"h264_nvenc", "libx264"}), "libx264")

    def test_validate_output_rejects_unsupported_suffix(self):
        mod = load_module()

        with self.assertRaises(ValueError):
            mod.validate_output_path(Path("bad.webm"))

    def test_load_audio_respects_duration(self):
        mod = load_module()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tone.wav"
            sf.write(path, sine_wave(sr=22050, seconds=1.0), 22050)

            audio = mod.load_audio(path, sr=22050, duration=0.25)

        self.assertEqual(audio.sample_rate, 22050)
        self.assertLessEqual(len(audio.samples), 22050)
        self.assertGreater(len(audio.samples), 0)
        self.assertEqual(audio.samples.ndim, 1)

    def test_prepare_input_audio_returns_plain_audio_without_conversion(self):
        mod = load_module()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tone.mp3"
            path.write_bytes(b"fake")

            with mod.prepare_input_audio(path, Path("missing.exe"), keep_converted=False) as prepared:
                self.assertEqual(prepared.original_path, path)
                self.assertEqual(prepared.audio_path, path)

    def test_prepare_input_audio_converts_ncm_with_ncmdump(self):
        mod = load_module()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ncm_path = tmp_path / "song.ncm"
            ncm_path.write_bytes(b"ncm")
            ncmdump_path = tmp_path / "ncmdump.exe"
            ncmdump_path.write_bytes(b"exe")

            def fake_run(command, check=False, capture_output=True):
                output_dir = Path(command[command.index("-o") + 1])
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "song.mp3").write_bytes(b"mp3")
                return types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

            with mock.patch.object(mod.subprocess, "run", side_effect=fake_run) as run:
                with mod.prepare_input_audio(ncm_path, ncmdump_path, keep_converted=False) as prepared:
                    converted_path = prepared.audio_path
                    self.assertEqual(prepared.original_path, ncm_path)
                    self.assertEqual(converted_path.suffix.lower(), ".mp3")
                    self.assertTrue(converted_path.exists())

            self.assertFalse(converted_path.exists())
            command = run.call_args.args[0]
            self.assertEqual(Path(command[0]), ncmdump_path)
            self.assertIn("-o", command)
            self.assertEqual(Path(command[-1]), ncm_path)


class RenderTests(unittest.TestCase):
    def test_draw_fast_frame_returns_rgb_image(self):
        mod = load_module()

        frame = mod.draw_fast_frame(
            fft_values=np.linspace(0.0, 1.0, 8, dtype=np.float32),
            cqt_values=np.linspace(1.0, 0.0, 6, dtype=np.float32),
            width=320,
            height=180,
            title="test",
            fft_label="FFT",
            cqt_label="CQT",
            time_label="0.0s / 1.0s",
        )

        self.assertEqual(frame.shape, (180, 320, 3))
        self.assertEqual(frame.dtype, np.uint8)
        self.assertGreater(int(frame.max()), int(frame.min()))

    def test_draw_fast_frame_marks_cqt_highlight_bars_red(self):
        mod = load_module()

        frame = mod.draw_fast_frame(
            fft_values=np.zeros(4, dtype=np.float32),
            cqt_values=np.ones(4, dtype=np.float32),
            width=320,
            height=180,
            title="test",
            fft_label="FFT",
            cqt_label="CQT",
            time_label="0.0s / 1.0s",
            cqt_highlight_mask=np.array([False, False, True, False], dtype=bool),
        )

        highlighted_pixel = frame[130, 170]
        normal_pixel = frame[130, 100]
        self.assertGreaterEqual(int(highlighted_pixel[0]), 220)
        self.assertLessEqual(int(highlighted_pixel[1]), 110)
        self.assertLessEqual(int(highlighted_pixel[2]), 110)
        np.testing.assert_array_equal(normal_pixel, np.array([255, 184, 77], dtype=np.uint8))

    def test_render_fast_video_writes_nonempty_mp4(self):
        mod = load_module()
        fft = np.random.default_rng(0).random((3, 8)).astype(np.float32)
        cqt = np.random.default_rng(1).random((3, 6)).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "fast.mp4"

            mod.render_fast_video(
                fft,
                cqt,
                output=output,
                fps=3,
                width=320,
                height=180,
                title="test",
                fft_axis_label="FFT",
                cqt_axis_label="CQT",
                encoder="libx264",
                show_progress=False,
            )

            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_render_animation_writes_nonempty_gif(self):
        mod = load_module()
        fft = np.random.default_rng(0).random((4, 8)).astype(np.float32)
        cqt = np.random.default_rng(1).random((4, 12)).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "preview.gif"

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                mod.render_animation(fft, cqt, output=output, fps=4, title="test", show_progress=False)

            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)
            self.assertEqual(stderr.getvalue(), "")


class MainTests(unittest.TestCase):
    def test_main_renders_gif_from_audio_file(self):
        mod = load_module()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path = tmp_path / "tone.wav"
            output_path = tmp_path / "compare.gif"
            sf.write(audio_path, sine_wave(sr=22050, seconds=0.35), 22050)

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    exit_code = mod.main(
                        [
                            str(audio_path),
                            "--output",
                            str(output_path),
                            "--duration",
                            "0.2",
                            "--fps",
                            "4",
                            "--bars",
                            "8",
                            "--cqt-bins",
                            "24",
                            "--hop-length",
                            "1024",
                        ]
                    )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


class MuxTests(unittest.TestCase):
    def test_mux_audio_captures_ffmpeg_output_as_bytes(self):
        mod = load_module()

        with mock.patch.object(mod.shutil, "which", return_value="ffmpeg"):
            with mock.patch.object(mod.subprocess, "run") as run:
                run.return_value = types.SimpleNamespace(returncode=0, stderr=b"\x80", stdout=b"")

                mod.mux_audio_into_video(Path("silent.mp4"), Path("audio.wav"), Path("out.mp4"))

        kwargs = run.call_args.kwargs
        self.assertTrue(kwargs["capture_output"])
        self.assertIsNot(kwargs.get("text"), True)


if __name__ == "__main__":
    unittest.main()
