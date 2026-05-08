from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np


DEFAULT_FFT_BARS = 128
DEFAULT_N_FFT = 4096
DEFAULT_CQT_HIGHLIGHT_TOP_K = 4
DEFAULT_CQT_HIGHLIGHT_MULTIPLIER = 1.0
DEFAULT_NCMDUMP_PATH = Path(__file__).resolve().with_name("ncmdump.exe")
SUPPORTED_CONVERTED_AUDIO_SUFFIXES = {".mp3", ".flac", ".wav", ".m4a", ".aac", ".ogg", ".opus"}
CQT_HIGHLIGHT_COLOR = (255, 77, 77)


@dataclass(frozen=True)
class AudioData:
    samples: np.ndarray
    sample_rate: int


@dataclass
class PreparedInput:
    original_path: Path
    audio_path: Path
    temp_dir: tempfile.TemporaryDirectory | None = None

    def __enter__(self) -> "PreparedInput":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        if self.temp_dir is not None:
            self.temp_dir.cleanup()
            self.temp_dir = None


@dataclass(frozen=True)
class FrequencyLayout:
    fft_axis_label: str
    cqt_axis_label: str
    fft_summary: str
    cqt_summary: str


class ProgressReporter:
    def __init__(
        self,
        label: str,
        total_frames: int,
        stream: TextIO | None = None,
        use_tqdm: bool = True,
    ) -> None:
        self.label = label
        self.total_frames = max(1, int(total_frames))
        self.stream = stream or sys.stdout
        self._last_completed = 0
        self._fallback_started = False
        self._bar = None

        if use_tqdm:
            try:
                from tqdm import tqdm

                self._bar = tqdm(total=self.total_frames, desc=label, unit="frame", file=self.stream)
            except Exception:
                self._bar = None

    def __call__(self, current_frame: int, total_frames: int) -> None:
        total = max(1, int(total_frames or self.total_frames))
        completed = min(total, max(0, int(current_frame)) + 1)

        if self._bar is not None:
            delta = completed - self._last_completed
            if delta > 0:
                self._bar.update(delta)
        else:
            percent = completed / total * 100.0
            end = "\n" if completed >= total else ""
            print(
                f"\r{self.label}: {completed}/{total} frames ({percent:.1f}%)",
                end=end,
                file=self.stream,
                flush=True,
            )
            self._fallback_started = True

        self._last_completed = max(self._last_completed, completed)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
        elif self._fallback_started and self._last_completed < self.total_frames:
            print(file=self.stream, flush=True)


def validate_output_path(path: Path | str) -> Path:
    output = Path(path)
    if output.suffix.lower() not in {".mp4", ".gif"}:
        raise ValueError(f"Unsupported output suffix '{output.suffix}'. Use .mp4 or .gif.")
    return output


def select_renderer(output: Path | str, requested: str) -> str:
    output_path = validate_output_path(output)
    if requested == "auto":
        return "fast" if output_path.suffix.lower() == ".mp4" else "matplotlib"
    if requested == "fast" and output_path.suffix.lower() != ".mp4":
        raise ValueError("The fast renderer supports MP4 output only. Use --renderer matplotlib for GIF.")
    if requested not in {"fast", "matplotlib"}:
        raise ValueError(f"Unsupported renderer '{requested}'.")
    return requested


def list_ffmpeg_encoders() -> set[str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return set()

    completed = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], check=False, capture_output=True)
    if completed.returncode != 0:
        return set()

    text = completed.stdout.decode("utf-8", errors="replace")
    encoders: set[str] = set()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            encoders.add(parts[1])
    return encoders


def select_video_encoder(requested: str, available_encoders: set[str] | None = None) -> str:
    available = available_encoders if available_encoders is not None else list_ffmpeg_encoders()
    if requested == "auto":
        if "h264_nvenc" in available:
            return "h264_nvenc"
        if "libx264" in available:
            return "libx264"
        raise RuntimeError("No supported MP4 encoder found. Install ffmpeg with h264_nvenc or libx264.")

    if requested not in {"h264_nvenc", "libx264"}:
        raise ValueError(f"Unsupported encoder '{requested}'.")
    if available_encoders is not None and requested not in available:
        raise RuntimeError(f"Requested encoder '{requested}' is not available in ffmpeg.")
    return requested


def _decode_process_output(data: bytes) -> str:
    return data.decode("utf-8", errors="replace").strip()


def _audio_mtimes(directory: Path) -> dict[Path, int]:
    if not directory.exists():
        return {}
    return {
        path: path.stat().st_mtime_ns
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_CONVERTED_AUDIO_SUFFIXES
    }


def _find_converted_audio(output_dir: Path, before: dict[Path, int], input_stem: str) -> Path:
    after = _audio_mtimes(output_dir)
    changed = [path for path, mtime in after.items() if before.get(path) != mtime]
    stem_changed = [path for path in changed if path.stem.lower() == input_stem.lower()]
    stem_all = [path for path in after if path.stem.lower() == input_stem.lower()]
    candidates = stem_changed or changed or stem_all or list(after)
    if not candidates:
        supported = ", ".join(sorted(SUPPORTED_CONVERTED_AUDIO_SUFFIXES))
        raise RuntimeError(f"ncmdump finished but no converted audio was found in {output_dir} ({supported}).")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def prepare_input_audio(input_path: Path | str, ncmdump_path: Path | str, keep_converted: bool) -> PreparedInput:
    source = Path(input_path)
    if source.suffix.lower() != ".ncm":
        return PreparedInput(original_path=source, audio_path=source)

    converter = Path(ncmdump_path)
    if not converter.exists():
        raise FileNotFoundError(f"ncmdump executable does not exist: {converter}")

    temp_dir: tempfile.TemporaryDirectory | None = None
    if keep_converted:
        output_dir = source.parent
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="ncmdump_")
        output_dir = Path(temp_dir.name)

    before = _audio_mtimes(output_dir)
    command = [str(converter), "-o", str(output_dir), str(source)]
    completed = subprocess.run(command, check=False, capture_output=True)
    if completed.returncode != 0:
        if temp_dir is not None:
            temp_dir.cleanup()
        details = _decode_process_output(completed.stderr) or _decode_process_output(completed.stdout)
        raise RuntimeError(f"ncmdump failed for {source}: {details}")

    try:
        audio_path = _find_converted_audio(output_dir, before, source.stem)
    except Exception:
        if temp_dir is not None:
            temp_dir.cleanup()
        raise

    return PreparedInput(original_path=source, audio_path=audio_path, temp_dir=temp_dir)


def load_audio(path: Path | str, sr: int | None = None, duration: float | None = None) -> AudioData:
    import librosa

    samples, sample_rate = librosa.load(path, sr=sr, mono=True, duration=duration)
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        raise ValueError(f"No audio samples were loaded from {path}.")
    return AudioData(samples=samples, sample_rate=int(sample_rate))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an offline FFT/CQT audio visualization comparison.",
    )
    parser.add_argument("input", nargs="?", type=Path, default=Path("music.wav"), help="Input audio path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output animation path. Supports .mp4 or .gif. Defaults to <input-stem>.mp4.",
    )
    parser.add_argument("--sr", type=int, default=None, help="Optional analysis sample rate.")
    parser.add_argument("--duration", type=float, default=None, help="Optional duration limit in seconds.")
    parser.add_argument("--fps", type=int, default=30, help="Animation frames per second.")
    parser.add_argument("--bars", type=int, default=DEFAULT_FFT_BARS, help="Number of FFT visual bars.")
    parser.add_argument("--cqt-bins", type=int, default=84, help="Number of CQT pitch bins.")
    parser.add_argument("--bins-per-octave", type=int, default=12, help="CQT bins per octave.")
    parser.add_argument("--hop-length", type=int, default=512, help="Analysis hop length.")
    parser.add_argument("--n-fft", type=int, default=DEFAULT_N_FFT, help="FFT window size.")
    parser.add_argument(
        "--renderer",
        choices=("auto", "fast", "matplotlib"),
        default="auto",
        help="Renderer backend. auto uses fast for MP4 and matplotlib for GIF.",
    )
    parser.add_argument(
        "--encoder",
        choices=("auto", "h264_nvenc", "libx264"),
        default="auto",
        help="MP4 video encoder for the fast renderer.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Fast renderer video width.")
    parser.add_argument("--height", type=int, default=720, help="Fast renderer video height.")
    parser.add_argument(
        "--ncmdump",
        type=Path,
        default=DEFAULT_NCMDUMP_PATH,
        help="Path to ncmdump.exe for .ncm input files.",
    )
    parser.add_argument(
        "--keep-converted",
        action="store_true",
        help="Keep audio converted from .ncm instead of using a temporary directory.",
    )
    parser.add_argument(
        "--with-audio",
        action="store_true",
        help="Mux the source audio into MP4 outputs with ffmpeg.",
    )
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = args.input.with_suffix(".mp4")
    args.output = validate_output_path(args.output)
    if args.fps <= 0:
        parser.error("--fps must be greater than 0.")
    if args.bars <= 0:
        parser.error("--bars must be greater than 0.")
    if args.cqt_bins <= 0:
        parser.error("--cqt-bins must be greater than 0.")
    if args.hop_length <= 0:
        parser.error("--hop-length must be greater than 0.")
    if args.n_fft <= 0:
        parser.error("--n-fft must be greater than 0.")
    if args.width <= 0 or args.width % 2 != 0:
        parser.error("--width must be a positive even integer.")
    if args.height <= 0 or args.height % 2 != 0:
        parser.error("--height must be a positive even integer.")
    try:
        select_renderer(args.output, args.renderer)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def normalize_bars(values: np.ndarray, gamma: float = 0.7, floor_db: float = -80.0) -> np.ndarray:
    """Convert non-negative magnitudes to stable visual bar heights in [0, 1]."""

    magnitudes = np.asarray(values, dtype=np.float32)
    if magnitudes.size == 0:
        return magnitudes

    magnitudes = np.nan_to_num(magnitudes, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(magnitudes))
    if peak <= np.finfo(np.float32).tiny:
        return np.zeros_like(magnitudes, dtype=np.float32)

    safe = np.maximum(magnitudes, np.finfo(np.float32).tiny)
    db = 20.0 * np.log10(safe / peak)
    db = np.clip(db, floor_db, 0.0)
    normalized = (db - floor_db) / abs(floor_db)
    normalized = np.clip(normalized, 0.0, 1.0)
    return np.power(normalized, gamma).astype(np.float32)


def boost_topk_per_frame(
    values: np.ndarray,
    top_k: int = DEFAULT_CQT_HIGHLIGHT_TOP_K,
    multiplier: float = DEFAULT_CQT_HIGHLIGHT_MULTIPLIER,
) -> tuple[np.ndarray, np.ndarray]:
    """Boost the loudest bins in each frame and return the selected-bin mask."""

    magnitudes = np.asarray(values, dtype=np.float32)
    if magnitudes.ndim != 2:
        raise ValueError("values must be a 2D array shaped [frames, bins].")

    boosted = magnitudes.copy()
    mask = np.zeros_like(boosted, dtype=bool)
    if boosted.size == 0 or top_k <= 0:
        return boosted, mask

    k = min(int(top_k), boosted.shape[1])
    if k <= 0:
        return boosted, mask

    top_indices = np.argpartition(boosted, -k, axis=1)[:, -k:]
    rows = np.arange(boosted.shape[0])[:, None]
    mask[rows, top_indices] = True
    boosted[mask] *= float(multiplier)
    return boosted, mask


def describe_frequency_layout(
    sample_rate: int,
    n_fft: int,
    fft_bars: int,
    cqt_bins: int,
    bins_per_octave: int,
) -> FrequencyLayout:
    import librosa

    fft_resolution = float(sample_rate) / float(n_fft)
    fft_min = max(20.0, fft_resolution)
    fft_max = float(sample_rate) / 2.0
    fft_edges = np.geomspace(fft_min, fft_max, fft_bars + 1)
    fft_first_width = fft_edges[1] - fft_edges[0]
    fft_mid_ix = max(1, min(fft_bars - 1, fft_bars // 2))
    fft_mid_width = fft_edges[fft_mid_ix + 1] - fft_edges[fft_mid_ix]
    fft_last_width = fft_edges[-1] - fft_edges[-2]

    cqt_min = float(librosa.note_to_hz("C1"))
    cqt_max = cqt_min * (2.0 ** ((cqt_bins - 1) / float(bins_per_octave)))
    cqt_ratio = 2.0 ** (1.0 / float(bins_per_octave))

    return FrequencyLayout(
        fft_axis_label=f"FFT bar index, log-spaced {fft_min:.1f} Hz - {fft_max:.1f} Hz",
        cqt_axis_label=f"CQT bin index, semitones from C1, {cqt_min:.1f} Hz - {cqt_max:.1f} Hz",
        fft_summary=(
            f"FFT: {fft_bars} log-spaced visual bars, {fft_min:.1f} Hz - {fft_max:.1f} Hz, "
            f"FFT resolution {fft_resolution:.1f} Hz/bin, visual widths approx "
            f"{fft_first_width:.1f}/{fft_mid_width:.1f}/{fft_last_width:.1f} Hz "
            "at low/mid/high bars"
        ),
        cqt_summary=(
            f"CQT: {cqt_bins} semitone bins from C1, {cqt_min:.1f} Hz - {cqt_max:.1f} Hz, "
            f"adjacent-bin ratio {cqt_ratio:.5f}"
        ),
    )


def resample_bar_frames(values: np.ndarray, target_frames: int) -> np.ndarray:
    bars = np.asarray(values, dtype=np.float32)
    if bars.ndim != 2:
        raise ValueError("values must be a 2D array shaped [frames, bars].")
    if target_frames <= 0:
        raise ValueError("target_frames must be greater than 0.")
    if bars.shape[0] == target_frames:
        return bars.copy()
    if bars.shape[0] == 1:
        return np.repeat(bars, target_frames, axis=0)

    source_x = np.linspace(0.0, 1.0, bars.shape[0], dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, target_frames, dtype=np.float32)
    resampled = np.empty((target_frames, bars.shape[1]), dtype=np.float32)
    for bar_ix in range(bars.shape[1]):
        resampled[:, bar_ix] = np.interp(target_x, source_x, bars[:, bar_ix])
    return np.clip(resampled, 0.0, 1.0)


def resample_bool_frames(values: np.ndarray, target_frames: int) -> np.ndarray:
    mask = np.asarray(values, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("values must be a 2D array shaped [frames, bars].")
    if target_frames <= 0:
        raise ValueError("target_frames must be greater than 0.")
    if mask.shape[0] == target_frames:
        return mask.copy()
    if mask.shape[0] == 0:
        return np.zeros((target_frames, mask.shape[1]), dtype=bool)

    source_positions = np.linspace(0.0, float(mask.shape[0] - 1), target_frames, dtype=np.float32)
    indices = np.floor(source_positions + 0.5).astype(np.intp)
    indices = np.clip(indices, 0, mask.shape[0] - 1)
    return mask[indices]


def make_fft_bars(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    n_fft: int = DEFAULT_N_FFT,
    bars: int = DEFAULT_FFT_BARS,
) -> np.ndarray:
    import librosa

    samples = np.asarray(y, dtype=np.float32)
    if samples.ndim != 1:
        samples = np.mean(samples, axis=0).astype(np.float32)

    spectrum = np.abs(librosa.stft(samples, n_fft=n_fft, hop_length=hop_length, center=True))
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    min_freq = max(20.0, float(sr) / float(n_fft))
    max_freq = float(sr) / 2.0
    edges = np.geomspace(min_freq, max_freq, bars + 1)
    grouped = np.zeros((spectrum.shape[1], bars), dtype=np.float32)

    for bar_ix in range(bars):
        low = edges[bar_ix]
        high = edges[bar_ix + 1]
        mask = (frequencies >= low) & (frequencies < high)
        if not np.any(mask):
            nearest = int(np.argmin(np.abs(frequencies - ((low + high) * 0.5))))
            grouped[:, bar_ix] = spectrum[nearest, :]
        else:
            grouped[:, bar_ix] = np.mean(spectrum[mask, :], axis=0)

    return normalize_bars(grouped)


def make_cqt_bars(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    cqt_bins: int = 84,
    bins_per_octave: int = 12,
    highlight_top_k: int = DEFAULT_CQT_HIGHLIGHT_TOP_K,
    highlight_multiplier: float = DEFAULT_CQT_HIGHLIGHT_MULTIPLIER,
    return_highlight: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    import librosa

    samples = np.asarray(y, dtype=np.float32)
    if samples.ndim != 1:
        samples = np.mean(samples, axis=0).astype(np.float32)

    cqt = librosa.cqt(
        y=samples,
        sr=sr,
        hop_length=hop_length,
        fmin=librosa.note_to_hz("C1"),
        n_bins=cqt_bins,
        bins_per_octave=bins_per_octave,
    )
    magnitudes = np.abs(cqt).T
    boosted, highlight_mask = boost_topk_per_frame(
        magnitudes,
        top_k=highlight_top_k,
        multiplier=highlight_multiplier,
    )
    bars = normalize_bars(boosted)
    if return_highlight:
        return bars, highlight_mask
    return bars


def _fill_rect(frame: np.ndarray, rect: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = rect
    frame[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)] = color


def _draw_bar_panel(
    frame: np.ndarray,
    values: np.ndarray,
    rect: tuple[int, int, int, int],
    bar_color: tuple[int, int, int],
    panel_color: tuple[int, int, int],
    grid_color: tuple[int, int, int],
    highlight_mask: np.ndarray | None = None,
    highlight_color: tuple[int, int, int] = CQT_HIGHLIGHT_COLOR,
) -> None:
    x0, y0, x1, y1 = rect
    _fill_rect(frame, rect, panel_color)

    for ratio in (0.25, 0.5, 0.75):
        y = int(y1 - (y1 - y0) * ratio)
        frame[y : y + 1, x0:x1] = grid_color

    inner_x0 = x0 + 8
    inner_x1 = x1 - 8
    inner_y0 = y0 + 18
    inner_y1 = y1 - 8
    bar_area_w = max(1, inner_x1 - inner_x0)
    bar_area_h = max(1, inner_y1 - inner_y0)
    clipped = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    if clipped.size == 0:
        return

    highlights = None
    if highlight_mask is not None:
        highlights = np.asarray(highlight_mask, dtype=bool)
        if highlights.ndim != 1 or highlights.shape[0] != clipped.shape[0]:
            raise ValueError("highlight_mask must be a 1D array matching values.")

    for ix, value in enumerate(clipped):
        left = inner_x0 + int(ix * bar_area_w / len(clipped))
        right = inner_x0 + int((ix + 1) * bar_area_w / len(clipped)) - 1
        if right <= left:
            right = left + 1
        top = inner_y1 - int(float(value) * bar_area_h)
        if top < inner_y1:
            color = highlight_color if highlights is not None and highlights[ix] else bar_color
            frame[top:inner_y1, left:right] = color


def draw_fast_frame(
    fft_values: np.ndarray,
    cqt_values: np.ndarray,
    width: int,
    height: int,
    title: str,
    fft_label: str,
    cqt_label: str,
    time_label: str,
    cqt_highlight_mask: np.ndarray | None = None,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    bg = (16, 18, 23)
    panel_bg = (23, 26, 33)
    grid = (42, 47, 58)
    fft_color = (53, 194, 255)
    cqt_color = (255, 184, 77)
    text = (241, 244, 248)
    muted = (174, 182, 194)

    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:] = bg

    margin_x = max(16, width // 32)
    top_pad = max(34, height // 12)
    footer_h = max(20, height // 18)
    gap = max(12, height // 30)
    panel_h = max(24, (height - top_pad - footer_h - gap - 8) // 2)
    fft_rect = (margin_x, top_pad, width - margin_x, top_pad + panel_h)
    cqt_top = top_pad + panel_h + gap
    cqt_rect = (margin_x, cqt_top, width - margin_x, cqt_top + panel_h)

    _draw_bar_panel(frame, fft_values, fft_rect, fft_color, panel_bg, grid)
    _draw_bar_panel(
        frame,
        cqt_values,
        cqt_rect,
        cqt_color,
        panel_bg,
        grid,
        highlight_mask=cqt_highlight_mask,
    )

    image = Image.fromarray(frame, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.text((margin_x, 10), title, fill=text)
    draw.text((fft_rect[0] + 8, fft_rect[1] + 3), "FFT frequency energy", fill=text)
    draw.text((fft_rect[0] + 8, fft_rect[3] + 2), fft_label, fill=muted)
    draw.text((cqt_rect[0] + 8, cqt_rect[1] + 3), "CQT pitch energy", fill=text)
    draw.text((cqt_rect[0] + 8, cqt_rect[3] + 2), cqt_label, fill=muted)
    draw.text((margin_x, height - footer_h + 2), time_label, fill=muted)
    return np.asarray(image, dtype=np.uint8)


def _ffmpeg_rawvideo_command(
    output: Path,
    width: int,
    height: int,
    fps: int,
    encoder: str,
) -> list[str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for the fast renderer but was not found on PATH.")

    command = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        encoder,
    ]
    if encoder == "libx264":
        command.extend(["-preset", "veryfast", "-crf", "18"])
    elif encoder == "h264_nvenc":
        command.extend(["-preset", "p4", "-cq", "19"])
    command.extend(["-pix_fmt", "yuv420p", str(output)])
    return command


def render_fast_video(
    fft_bars: np.ndarray,
    cqt_bars: np.ndarray,
    output: Path | str,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    title: str = "FFT vs CQT",
    fft_axis_label: str = "FFT bar index",
    cqt_axis_label: str = "CQT bin index",
    encoder: str = "auto",
    show_progress: bool = True,
    cqt_highlight_mask: np.ndarray | None = None,
) -> Path:
    output_path = validate_output_path(output)
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("The fast renderer supports MP4 output only.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fft_values = np.asarray(fft_bars, dtype=np.float32)
    cqt_values = np.asarray(cqt_bars, dtype=np.float32)
    if fft_values.ndim != 2 or cqt_values.ndim != 2:
        raise ValueError("fft_bars and cqt_bars must be 2D arrays shaped [frames, bars].")

    frame_count = min(fft_values.shape[0], cqt_values.shape[0])
    highlight_values = None
    if cqt_highlight_mask is not None:
        highlight_values = np.asarray(cqt_highlight_mask, dtype=bool)
        if highlight_values.ndim != 2:
            raise ValueError("cqt_highlight_mask must be a 2D array shaped [frames, bars].")
        if highlight_values.shape[1] != cqt_values.shape[1]:
            raise ValueError("cqt_highlight_mask must have the same bar count as cqt_bars.")
        frame_count = min(frame_count, highlight_values.shape[0])
    if frame_count <= 0:
        raise ValueError("Cannot render video with zero frames.")
    selected_encoder = select_video_encoder(encoder)
    command = _ffmpeg_rawvideo_command(output_path, width, height, fps, selected_encoder)
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    progress = ProgressReporter(label=f"Rendering {output_path.name}", total_frames=frame_count) if show_progress else None
    try:
        assert process.stdin is not None
        for frame_ix in range(frame_count):
            current_seconds = frame_ix / float(fps)
            total_seconds = frame_count / float(fps)
            frame = draw_fast_frame(
                fft_values[frame_ix],
                cqt_values[frame_ix],
                width=width,
                height=height,
                title=title,
                fft_label=fft_axis_label,
                cqt_label=cqt_axis_label,
                time_label=f"{current_seconds:0.2f}s / {total_seconds:0.2f}s",
                cqt_highlight_mask=None if highlight_values is None else highlight_values[frame_ix],
            )
            process.stdin.write(frame.tobytes())
            if progress is not None:
                progress(frame_ix, frame_count)

        process.stdin.close()
        process.stdin = None
        stdout, stderr = process.communicate()
    finally:
        if progress is not None:
            progress.close()
        if process.stdin is not None:
            process.stdin.close()

    if process.returncode != 0:
        details = stderr.decode("utf-8", errors="replace").strip() or stdout.decode(
            "utf-8", errors="replace"
        ).strip()
        raise RuntimeError(f"ffmpeg failed while rendering fast video: {details}")

    return output_path


def render_animation(
    fft_bars: np.ndarray,
    cqt_bars: np.ndarray,
    output: Path | str,
    fps: int = 30,
    title: str = "FFT vs CQT",
    dpi: int = 120,
    fft_axis_label: str | None = None,
    cqt_axis_label: str | None = None,
    show_progress: bool = True,
    cqt_highlight_mask: np.ndarray | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import animation
    from matplotlib import pyplot as plt

    output_path = validate_output_path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fft_values = np.asarray(fft_bars, dtype=np.float32)
    cqt_values = np.asarray(cqt_bars, dtype=np.float32)
    if fft_values.ndim != 2 or cqt_values.ndim != 2:
        raise ValueError("fft_bars and cqt_bars must be 2D arrays shaped [frames, bars].")

    frame_count = min(fft_values.shape[0], cqt_values.shape[0])
    highlight_values = None
    if cqt_highlight_mask is not None:
        highlight_values = np.asarray(cqt_highlight_mask, dtype=bool)
        if highlight_values.ndim != 2:
            raise ValueError("cqt_highlight_mask must be a 2D array shaped [frames, bars].")
        if highlight_values.shape[1] != cqt_values.shape[1]:
            raise ValueError("cqt_highlight_mask must have the same bar count as cqt_bars.")
        frame_count = min(frame_count, highlight_values.shape[0])
    if frame_count <= 0:
        raise ValueError("Cannot render animation with zero frames.")
    fft_values = np.clip(fft_values[:frame_count], 0.0, 1.0)
    cqt_values = np.clip(cqt_values[:frame_count], 0.0, 1.0)
    if highlight_values is not None:
        highlight_values = highlight_values[:frame_count]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharey=False)
    fig.patch.set_facecolor("#101217")
    fig.suptitle(title, color="#f1f4f8", fontsize=15, y=0.98)

    cqt_color = "#ffb84d"
    cqt_highlight_color = "#ff4d4d"
    panels = [
        (axes[0], fft_values, "FFT frequency energy", fft_axis_label or "FFT bar index", "#35c2ff", None),
        (axes[1], cqt_values, "CQT pitch energy", cqt_axis_label or "CQT bin index", cqt_color, highlight_values),
    ]
    bar_artists = []
    for ax, values, title_label, axis_label, color, highlights in panels:
        x = np.arange(values.shape[1])
        colors = color
        if highlights is not None:
            colors = [cqt_highlight_color if active else color for active in highlights[0]]
        bars = ax.bar(x, values[0], width=0.85, color=colors, edgecolor="none")
        ax.set_facecolor("#171a21")
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(-0.5, values.shape[1] - 0.5)
        ax.set_title(title_label, color="#d8dee9", loc="left", fontsize=11)
        ax.set_xlabel(axis_label, color="#aeb6c2", fontsize=8)
        ax.tick_params(colors="#7f8794", labelsize=8)
        ax.grid(axis="y", color="#2a2f3a", linewidth=0.6, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color("#2a2f3a")
        bar_artists.append(bars)

    time_label = axes[0].text(
        0.99,
        0.9,
        "",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        color="#aeb6c2",
        fontsize=9,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    def update(frame_ix: int):
        for bars, values in zip(bar_artists, (fft_values, cqt_values)):
            for rect, height in zip(bars, values[frame_ix]):
                rect.set_height(float(height))
        if highlight_values is not None:
            for rect, active in zip(bar_artists[1], highlight_values[frame_ix]):
                rect.set_color(cqt_highlight_color if active else cqt_color)
        time_label.set_text(f"frame {frame_ix + 1}/{frame_count}")
        artists = [time_label]
        for bars in bar_artists:
            artists.extend(bars)
        return artists

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frame_count,
        interval=1000.0 / max(fps, 1),
        blit=False,
        repeat=False,
    )

    try:
        if output_path.suffix.lower() == ".gif":
            writer = animation.PillowWriter(fps=fps)
        else:
            if not animation.writers.is_available("ffmpeg"):
                raise RuntimeError("Matplotlib ffmpeg writer is not available; use .gif or install ffmpeg.")
            writer = animation.FFMpegWriter(
                fps=fps,
                codec="libx264",
                bitrate=4000,
                extra_args=["-pix_fmt", "yuv420p"],
            )
        if show_progress:
            progress = ProgressReporter(label=f"Rendering {output_path.name}", total_frames=frame_count)
            try:
                anim.save(output_path, writer=writer, dpi=dpi, progress_callback=progress)
            finally:
                progress.close()
        else:
            anim.save(output_path, writer=writer, dpi=dpi)
    finally:
        plt.close(fig)

    return output_path


def mux_audio_into_video(video_path: Path, audio_path: Path, output_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for --with-audio MP4 muxing but was not found on PATH.")

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True)
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        stdout = completed.stdout.decode("utf-8", errors="replace").strip()
        details = stderr or stdout
        raise RuntimeError(f"ffmpeg failed while muxing audio: {details}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input audio file does not exist: {input_path}")

    prepared = prepare_input_audio(input_path, args.ncmdump, args.keep_converted)
    try:
        audio_input_path = prepared.audio_path
        print(f"Input: {input_path}")
        if audio_input_path != input_path:
            print(f"Converted audio: {audio_input_path}")

        audio = load_audio(audio_input_path, sr=args.sr, duration=args.duration)
        duration_seconds = len(audio.samples) / float(audio.sample_rate)
        print(f"Loaded: {duration_seconds:.2f}s at {audio.sample_rate} Hz")
        frequency_layout = describe_frequency_layout(
            sample_rate=audio.sample_rate,
            n_fft=args.n_fft,
            fft_bars=args.bars,
            cqt_bins=args.cqt_bins,
            bins_per_octave=args.bins_per_octave,
        )
        print(frequency_layout.fft_summary)
        print(frequency_layout.cqt_summary)

        print("Computing FFT bars...")
        fft_bars = make_fft_bars(
            audio.samples,
            sr=audio.sample_rate,
            hop_length=args.hop_length,
            n_fft=args.n_fft,
            bars=args.bars,
        )
        print(
            "Computing CQT bars "
            f"(highlight top {DEFAULT_CQT_HIGHLIGHT_TOP_K} bins x{DEFAULT_CQT_HIGHLIGHT_MULTIPLIER:g})..."
        )
        cqt_bars, cqt_highlight_mask = make_cqt_bars(
            audio.samples,
            sr=audio.sample_rate,
            hop_length=args.hop_length,
            cqt_bins=args.cqt_bins,
            bins_per_octave=args.bins_per_octave,
            return_highlight=True,
        )

        target_frames = max(1, int(round(duration_seconds * args.fps)))
        fft_bars = resample_bar_frames(fft_bars, target_frames)
        cqt_bars = resample_bar_frames(cqt_bars, target_frames)
        cqt_highlight_mask = resample_bool_frames(cqt_highlight_mask, target_frames)
        title = (
            f"{input_path.name} | FFT {args.bars} bars | "
            f"CQT {args.cqt_bins} bins | {args.fps} fps"
        )

        if args.with_audio and args.output.suffix.lower() == ".gif":
            print("Warning: --with-audio is ignored for GIF output.", file=sys.stderr)

        renderer = select_renderer(args.output, args.renderer)
        selected_encoder = select_video_encoder(args.encoder) if renderer == "fast" else None
        if selected_encoder is None:
            print(f"Rendering: {args.output} ({target_frames} frames at {args.fps} fps, renderer={renderer})")
        else:
            print(
                f"Rendering: {args.output} "
                f"({target_frames} frames at {args.fps} fps, renderer={renderer}, encoder={selected_encoder})"
            )

        if renderer == "fast":
            if args.with_audio:
                with tempfile.TemporaryDirectory(prefix="fft_cqt_visual_") as tmp:
                    silent_video = Path(tmp) / "silent.mp4"
                    render_fast_video(
                        fft_bars,
                        cqt_bars,
                        output=silent_video,
                        fps=args.fps,
                        width=args.width,
                        height=args.height,
                        title=title,
                        fft_axis_label=frequency_layout.fft_axis_label,
                        cqt_axis_label=frequency_layout.cqt_axis_label,
                        encoder=selected_encoder,
                        cqt_highlight_mask=cqt_highlight_mask,
                    )
                    mux_audio_into_video(silent_video, audio_input_path, args.output)
            else:
                render_fast_video(
                    fft_bars,
                    cqt_bars,
                    output=args.output,
                    fps=args.fps,
                    width=args.width,
                    height=args.height,
                    title=title,
                    fft_axis_label=frequency_layout.fft_axis_label,
                    cqt_axis_label=frequency_layout.cqt_axis_label,
                    encoder=selected_encoder,
                    cqt_highlight_mask=cqt_highlight_mask,
                )
        elif args.with_audio and args.output.suffix.lower() == ".mp4":
            with tempfile.TemporaryDirectory(prefix="fft_cqt_visual_") as tmp:
                silent_video = Path(tmp) / "silent.mp4"
                render_animation(
                    fft_bars,
                    cqt_bars,
                    output=silent_video,
                    fps=args.fps,
                    title=title,
                    fft_axis_label=frequency_layout.fft_axis_label,
                    cqt_axis_label=frequency_layout.cqt_axis_label,
                    cqt_highlight_mask=cqt_highlight_mask,
                )
                mux_audio_into_video(silent_video, audio_input_path, args.output)
        else:
            render_animation(
                fft_bars,
                cqt_bars,
                output=args.output,
                fps=args.fps,
                title=title,
                fft_axis_label=frequency_layout.fft_axis_label,
                cqt_axis_label=frequency_layout.cqt_axis_label,
                cqt_highlight_mask=cqt_highlight_mask,
            )

        print(f"Done: {args.output}")
        return 0
    finally:
        prepared.close()


if __name__ == "__main__":
    raise SystemExit(main())
