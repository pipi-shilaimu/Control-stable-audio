# FFT/CQT Visual Compare Design

## Goal

Add an offline audio visualization script under `scripts/` that compares FFT and CQT in a music-player-style animation. The default artifact should be a smooth MP4, with GIF available only as a preview or fallback format.

## User Experience

Primary command:

```powershell
.\.venv\Scripts\python.exe scripts\fft_cqt_visual_compare.py music.wav --output visual_compare.mp4 --fps 30 --with-audio
```

Preview command:

```powershell
.\.venv\Scripts\python.exe scripts\fft_cqt_visual_compare.py music.wav --output preview.gif --fps 12 --duration 10
```

The script should:

- accept an input audio path, defaulting to `music.wav` when omitted;
- default to `visual_compare.mp4`;
- generate a two-panel animation with FFT bars and CQT bars advancing over time;
- support `--duration` to limit long files during development;
- support `--fps`, `--bars`, `--cqt-bins`, `--hop-length`, and `--with-audio`;
- use MP4 through `ffmpeg` when available;
- support GIF through Pillow for short previews.

## Existing Environment

The local `.venv` already contains the libraries needed for the first version:

- `librosa` for audio loading and CQT;
- `numpy` and `scipy` for signal processing;
- `matplotlib` for rendering and animation;
- `soundfile` for robust audio file access;
- `imageio` and Pillow-backed animation support.

The machine also has `ffmpeg`, and `matplotlib.animation.writers.list()` includes `ffmpeg`, so MP4 is viable without adding dependencies.

## Architecture

Implement `scripts/fft_cqt_visual_compare.py` with a small pure-function core and a CLI wrapper.

Core functions:

- `load_audio(path, sr, duration) -> AudioData`
  Loads audio with `librosa`, converts to mono for analysis, and preserves the effective sample rate.
- `make_fft_bars(y, sr, hop_length, n_fft, bars) -> np.ndarray`
  Computes an STFT magnitude spectrogram and groups frequency bins into visually stable bar buckets. Output shape is `[frames, bars]`.
- `make_cqt_bars(y, sr, hop_length, cqt_bins, bins_per_octave) -> np.ndarray`
  Computes CQT magnitude bars. Output shape is `[frames, cqt_bins]`.
- `normalize_bars(values, gamma) -> np.ndarray`
  Converts magnitudes to decibels, normalizes to `[0, 1]`, and applies a mild gamma curve for player-like motion.
- `render_animation(fft_bars, cqt_bars, output, fps, title, with_audio, audio_path, sr)`
  Renders a two-panel animated bar chart. For `--with-audio` and MP4, render a silent temporary video first and mux audio with `ffmpeg`.

CLI concerns:

- parse arguments with `argparse`;
- validate output suffix (`.mp4` or `.gif`);
- choose MP4 writer for `.mp4`, Pillow writer for `.gif`;
- surface clear errors when `ffmpeg` is missing and MP4 is requested;
- keep temporary files inside a local temp directory and clean them up.

## Visual Design

Use a dark, restrained player-style visualization:

- top panel: FFT frequency-energy bars, emphasizing bass and midrange movement;
- bottom panel: CQT pitch bars, showing semitone/pitch activity more clearly;
- fixed y-axis `[0, 1]` so the animation does not jump;
- stable bar count and axes to avoid layout resizing;
- title includes the input filename and render settings;
- colors should differ between FFT and CQT without making the palette one-note.

## Audio Handling

Analysis is mono for clarity and speed. `--with-audio` should not try to synchronize live playback; it should mux the original audio into the MP4 after rendering. For non-WAV input, `ffmpeg` can read the original file when available. GIF outputs ignore `--with-audio` and should warn rather than fail.

## Testing

Add focused tests for pure processing functions:

- FFT bars return `[frames, bars]` with finite normalized values;
- CQT bars return `[frames, cqt_bins]` with finite normalized values;
- normalization handles silence without NaN/Inf;
- CLI argument validation rejects unsupported output suffixes.

Add a smoke verification command that renders a very short MP4 or GIF from generated sine-wave audio, to confirm the animation path writes a non-empty file.

## Non-Goals

- Real-time playback and live visualization are out of scope for this version.
- Perfect audiovisual synchronization during preview is out of scope; muxed MP4 is the intended playback artifact.
- Integration with StableAudio model training or ControlNet conditioning is out of scope.
