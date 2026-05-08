# FFT/CQT Visual Compare Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline script that renders a smooth side-by-side FFT and CQT audio visualization, defaulting to MP4 and supporting GIF previews.

**Architecture:** Keep signal processing in small testable functions inside `scripts/fft_cqt_visual_compare.py`, with a thin `argparse` CLI at the bottom. The renderer consumes precomputed normalized bar matrices so animation logic stays separate from audio feature extraction.

**Tech Stack:** Python standard library, `numpy`, `librosa`, `matplotlib.animation`, local `ffmpeg` for MP4 and audio muxing, `unittest` for tests.

---

## File Structure

- Create `scripts/fft_cqt_visual_compare.py`
  - Owns CLI parsing, audio loading, FFT/CQT bar extraction, normalization, animation rendering, and optional audio muxing.
- Create `tests/test_fft_cqt_visual_compare.py`
  - Imports the script module by path and tests pure functions plus argument validation.
- Use existing `music.wav`
  - Manual/demo input only; tests generate synthetic signals in memory.

## Task 1: Pure Processing Functions

**Files:**
- Create: `scripts/fft_cqt_visual_compare.py`
- Test: `tests/test_fft_cqt_visual_compare.py`

- [ ] **Step 1: Write failing tests for normalization, FFT bars, and CQT bars**

```python
def test_normalize_bars_handles_silence():
    values = mod.normalize_bars(np.zeros((4, 8)))
    assert values.shape == (4, 8)
    assert np.all(np.isfinite(values))
    assert np.all((0.0 <= values) & (values <= 1.0))

def test_make_fft_bars_returns_requested_bar_count():
    y = sine_wave()
    bars = mod.make_fft_bars(y, sr=22050, hop_length=256, n_fft=1024, bars=32)
    assert bars.shape[1] == 32
    assert bars.shape[0] > 0
    assert np.all(np.isfinite(bars))

def test_make_cqt_bars_returns_requested_bin_count():
    y = sine_wave()
    bars = mod.make_cqt_bars(y, sr=22050, hop_length=256, cqt_bins=36, bins_per_octave=12)
    assert bars.shape[1] == 36
    assert bars.shape[0] > 0
    assert np.all(np.isfinite(bars))
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_fft_cqt_visual_compare -v
```

Expected: FAIL because `scripts/fft_cqt_visual_compare.py` does not exist or functions are missing.

- [ ] **Step 3: Implement minimal processing functions**

Implement:

- `normalize_bars(values, gamma=0.7, floor_db=-80.0)`
- `make_fft_bars(y, sr, hop_length=512, n_fft=2048, bars=64)`
- `make_cqt_bars(y, sr, hop_length=512, cqt_bins=84, bins_per_octave=12)`

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_fft_cqt_visual_compare -v
```

Expected: PASS.

## Task 2: CLI Validation and Audio Loading

**Files:**
- Modify: `scripts/fft_cqt_visual_compare.py`
- Modify: `tests/test_fft_cqt_visual_compare.py`

- [ ] **Step 1: Write failing tests for output suffix validation and duration handling**

```python
def test_validate_output_rejects_unsupported_suffix():
    with pytest_like_assert_raises(ValueError):
        mod.validate_output_path(Path("bad.webm"))

def test_load_audio_respects_duration(tmp_path):
    path = tmp_path / "tone.wav"
    sf.write(path, sine_wave(sr=22050, seconds=1.0), 22050)
    audio = mod.load_audio(path, sr=22050, duration=0.25)
    assert audio.sample_rate == 22050
    assert len(audio.samples) <= 22050
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_fft_cqt_visual_compare -v
```

Expected: FAIL because validation/load helpers are missing.

- [ ] **Step 3: Implement CLI helper functions**

Implement:

- `AudioData` dataclass with `samples` and `sample_rate`
- `validate_output_path(path)`
- `load_audio(path, sr=None, duration=None)`
- `parse_args(argv=None)`

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_fft_cqt_visual_compare -v
```

Expected: PASS.

## Task 3: Animation Rendering

**Files:**
- Modify: `scripts/fft_cqt_visual_compare.py`
- Modify: `tests/test_fft_cqt_visual_compare.py`

- [ ] **Step 1: Write failing smoke test for a tiny GIF render**

```python
def test_render_animation_writes_nonempty_gif(tmp_path):
    fft = np.random.default_rng(0).random((4, 8))
    cqt = np.random.default_rng(1).random((4, 12))
    output = tmp_path / "preview.gif"
    mod.render_animation(fft, cqt, output=output, fps=4, title="test")
    assert output.exists()
    assert output.stat().st_size > 0
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_fft_cqt_visual_compare -v
```

Expected: FAIL because `render_animation` is missing.

- [ ] **Step 3: Implement minimal renderer**

Implement a dark two-panel bar animation with:

- fixed y-limits `[0, 1]`;
- FFT bars in one color and CQT bars in another;
- MP4 writer when suffix is `.mp4`;
- Pillow writer when suffix is `.gif`;
- non-interactive Matplotlib backend (`Agg`) for script/test stability.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_fft_cqt_visual_compare -v
```

Expected: PASS.

## Task 4: End-to-End CLI and MP4 Audio Mux

**Files:**
- Modify: `scripts/fft_cqt_visual_compare.py`

- [ ] **Step 1: Add `main(argv=None)` orchestration**

Flow:

1. parse args;
2. load audio;
3. compute FFT bars and CQT bars;
4. trim both matrices to a shared frame count;
5. render output;
6. when `--with-audio` and output is MP4, mux original audio with `ffmpeg`.

- [ ] **Step 2: Add clear console messages**

Print:

- input path, sample rate, duration;
- output path;
- selected writer;
- warning when `--with-audio` is ignored for GIF.

- [ ] **Step 3: Manual smoke render**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\fft_cqt_visual_compare.py music.wav --output fft_cqt_preview.mp4 --duration 5 --fps 15 --with-audio
```

Expected: a non-empty MP4 file is created.

- [ ] **Step 4: Preview GIF smoke render**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\fft_cqt_visual_compare.py music.wav --output fft_cqt_preview.gif --duration 3 --fps 8
```

Expected: a non-empty GIF file is created.

## Task 5: Final Verification

**Files:**
- Modify: none unless verification finds issues.

- [ ] **Step 1: Run unit tests**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_fft_cqt_visual_compare -v
```

Expected: PASS.

- [ ] **Step 2: Run MP4 smoke render**

```powershell
.\.venv\Scripts\python.exe scripts\fft_cqt_visual_compare.py music.wav --output fft_cqt_preview.mp4 --duration 5 --fps 15 --with-audio
```

Expected: PASS and output file exists.

- [ ] **Step 3: Check working tree**

```powershell
git status --short
```

Expected: only intentional new/modified files from this task are present, alongside pre-existing unrelated user changes.
