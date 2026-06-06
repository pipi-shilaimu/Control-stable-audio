# ControlNet Melody Failure Diagnosis

Date: 2026-06-06
Status: Active diagnosis for preventing the `piano_clear` run from repeating the `formal10s_v3` failure.

## Scope

This note records the observed failure mode, the current hypothesis, the evidence from the server-side diagnostic run, and the next validation plan.

The immediate goal is not general audio quality. The goal is narrower:

```text
ControlNet must reconstruct/follow melody, and correct control must be measurably different from shuffled control.
```

## Observed Symptoms

The failed training run discussed here is `formal10s_v3`, especially the `step=90000` checkpoint.

Reported listening behavior:

- `control_scale=0` sounds normal, which indicates the frozen/base generation path is still usable.
- `control_scale=0.1`, `0.3`, `0.5`, and `1.0` all degrade when ControlNet is enabled.
- With `correct` control, output tends toward continuous, smooth, meaningless mid/low-frequency sound, with little or no audible melody following.
- With `zero` control at low control scales, the output differs from `correct`: it contains many silent regions and occasional sounds.
- The degradation trend became noticeable around 70k steps, obvious around 80k steps, and severe by 90k steps.
- Earlier checkpoints before 60k are no longer available, so the earliest remaining comparison point is 60k.

Important interpretation:

- `zero` behavior is not the main training target, but it is useful as a diagnostic for whether "silent/no control" is being represented safely.
- `correct` vs `shuffled` is the critical diagnostic for melody learning. If the model responds almost identically to correct and shuffled melody, it is not using the time-ordered pitch contour.

## Server Diagnostic Run

Diagnostic script:

```text
stable_audio_control/scripts/diagnose_control_activity.py
```

Checkpoint:

```text
outputs/formal10s_v3/checkpoints/controlnet-step=step=90000.ckpt
```

Reference:

```text
audios/Then.mp3
```

Command shape:

```bash
python3 -u stable_audio_control/scripts/diagnose_control_activity.py \
  --ckpt-path outputs/formal10s_v3/checkpoints/controlnet-step=step=90000.ckpt \
  --reference-audio audios/Then.mp3 \
  --output-csv outputs/diagnostics/control_activity_step90000_full.csv \
  --output-report outputs/diagnostics/control_activity_step90000_full.report.txt \
  --output-pairwise-csv outputs/diagnostics/control_activity_step90000_full.pairwise.csv \
  --num-control-layers 12 \
  --melody-feature cqt \
  --cqt-backend librosa \
  --seconds-total 10 \
  --demo-control-variants "correct,shuffled,zero" \
  --control-scale 1.0 \
  --diagnostic-timestep 0.5 \
  --diagnostic-seed 0 \
  --no-prefer-ema \
  --model-half
```

## Key Evidence

### 1. Correct and shuffled are almost identical at denoiser level

```text
forward_delta correct/shuffled rms ratio = 1.0017
forward_delta correct/shuffled cosine = 0.9955
forward_delta correct/shuffled relative_l2_difference = 0.0945
```

This is the strongest evidence so far.

`forward_delta` is measured as:

```text
denoiser output with control_scale=1
-
denoiser output with control_scale=0
```

The result means the trained ControlNet causes almost the same denoiser perturbation for the correct melody and for the time-shuffled/reversed melody. A model that is truly following time-ordered pitch contour should not treat these two controls as nearly equivalent.

Conclusion supported by this evidence:

```text
The 90k checkpoint is largely insensitive to melody order/contour.
```

### 2. ControlNet residual is active throughout time

```text
correct forward_delta frame_active_ratio = 1.0
shuffled forward_delta frame_active_ratio = 1.0
zero forward_delta frame_active_ratio = 1.0
```

For `correct`, the frame-wise residual is not sparse:

```text
correct forward_delta frame_rms_min = 0.15356964
correct forward_delta frame_rms_max = 0.43519363
```

This supports the "continuous residual" part of the listening observation. It does not by itself prove "mid/low-frequency" spectral content; that still needs a generated-audio spectral diagnostic.

### 3. Zero audio is not null control

```text
zero melody_control zero_value_ratio = 0.0000
zero melody_control unique_value_count = 4
zero melody_control frame_active_ratio = 1.0
```

The all-zero reference waveform is converted by CQT top-k into nonzero pitch indices rather than padding/null tokens. After the melody encoder:

```text
correct control_input rms = 0.57097369
shuffled control_input rms = 0.55218214
zero control_input rms = 2.11254096
```

So `zero` is not "no control". It is a strong, fixed pseudo-pitch control. This explains why `zero` can produce a distinct failure mode instead of simply disabling ControlNet.

### 4. Zero differs from correct/shuffled, but this is not the main target

```text
forward_delta correct/zero cosine = 0.4205
forward_delta shuffled/zero cosine = 0.4172
forward_delta correct/zero relative_l2_difference = 0.9465
forward_delta shuffled/zero relative_l2_difference = 0.9486
```

This means `zero` drives ControlNet in a different direction. However, the core melody failure is not "zero is bad"; the core melody failure is:

```text
correct and shuffled are nearly indistinguishable.
```

## Current Hypothesis

The current root-cause hypothesis is not a single-point bug. It is a training-and-representation shortcut:

```text
CQT/top-k melody control
  -> mostly exposes pitch activity, pitch-index distribution, and dense time-aligned structure
  -> MelodyControlEncoder maps it into a control input not guaranteed to align with the VAE latent/audio semantics
  -> the training objective only pairs correct control with the target audio
  -> no direct penalty exists for treating shuffled control similarly to correct control
  -> the ControlNet branch learns a persistent "music activity / residual energy" perturbation
  -> long training strengthens this residual
  -> inference with correct control produces continuous smooth mid/low-frequency output rather than pitch-contour following
```

This hypothesis is supported by the 90k diagnostic, but it is not yet proven as the unique cause.

What is strongly supported:

- The failed checkpoint does not meaningfully distinguish `correct` from `shuffled`.
- ControlNet produces a continuous residual across all frames.
- `zero` audio is a strong pseudo-control, not a null control.
- `control_scale=0` being normal localizes the audible failure to the ControlNet residual path rather than the base model.

What is not yet fully proven:

- The exact reason the audible artifact concentrates in mid/low frequencies.
- Whether the failure already exists at 60k or emerges between 60k and 90k.
- Whether the same correct/shuffled collapse appears for many references and timesteps, or only for `Then.mp3` at `t=0.5`.

## Validation Plan

### A. Check the earliest remaining checkpoint

Run the same diagnostic on 60k:

```bash
python3 -u stable_audio_control/scripts/diagnose_control_activity.py \
  --ckpt-path outputs/formal10s_v3/checkpoints/controlnet-step=step=60000.ckpt \
  --reference-audio audios/Then.mp3 \
  --output-csv outputs/diagnostics/control_activity_step60000_full.csv \
  --output-report outputs/diagnostics/control_activity_step60000_full.report.txt \
  --output-pairwise-csv outputs/diagnostics/control_activity_step60000_full.pairwise.csv \
  --num-control-layers 12 \
  --melody-feature cqt \
  --cqt-backend librosa \
  --seconds-total 10 \
  --demo-control-variants "correct,shuffled,zero" \
  --control-scale 1.0 \
  --diagnostic-timestep 0.5 \
  --diagnostic-seed 0 \
  --no-prefer-ema \
  --model-half
```

Interpretation:

- If 60k already has `forward_delta correct/shuffled cosine` near 1.0, the control representation/training objective likely failed early.
- If 60k separates correct/shuffled but 90k does not, the later training phase caused collapse.

### B. Run multi-reference and multi-timestep diagnostics

Use several clear piano references and timesteps:

```text
t = 0.2, 0.5, 0.8
variants = correct, shuffled, zero, null/disabled
```

Expected useful evidence:

- Melody-following model: correct/shuffled deltas should differ consistently.
- Activity-shortcut model: correct/shuffled deltas remain highly similar across references and timesteps.

### C. Add a real null/disabled variant

Do not treat zero waveform as null control.

Needed diagnostic variant:

```text
null/disabled = no control_input passed, or explicit control_scale=0 baseline
```

This separates two questions:

- What happens with no ControlNet?
- What happens with a silent waveform that the CQT extractor converts into pseudo pitch?

### D. Add generated-audio spectral diagnostics

To prove the "mid/low-frequency" part instrumentally, compare generated outputs:

```text
control_scale=0
correct
shuffled
zero
null
```

Metrics:

- low/mid/high band energy ratio
- mel-spectrogram energy distribution
- voiced/silence ratio
- melody similarity against original reference

## Prevention Plan for `piano_clear`

Do not start another long formal training run until the following guardrails are in place.

### 1. Mask low-energy CQT frames, not only metadata padding

The current padding-mask zeroing handles dataset padding, but real quiet/weak/silent audio frames can still become pseudo pitch. Add an energy threshold path so low-energy CQT frames are set to token `0` before `MelodyControlEncoder`.

Implementation note on 2026-06-06:

- `CQTTopKExtractor` now computes per-frame CQT energy and sets low-energy frames to token `0` before returning pitch tokens.
- The same patch fixed a CQT channel/time interleave bug: the intended layout is `[L0, R0, L1, R1, ...]` with time kept on the final axis. The previous reshape could mix time frames into the channel axis, which would corrupt the melody contour before it ever reached ControlNet.

Goal:

```text
silence/near-silence -> token 0 -> padding embedding -> no pseudo melody control
```

### 2. Do not use `zero` as the main no-control baseline

Use:

```text
control_scale=0
null/disabled control_input
```

Keep `zero` only as a robustness diagnostic.

Implementation note on 2026-06-06:

- Training demos now support `null`/`disabled`/`none` as a real no-melody-control variant.
- `zero` remains available, but it means "zero waveform passed through CQT", not "no control".

### 3. Train short and stop early on correct/shuffled collapse

For about 5k-6k clean 10-second piano clips:

```text
batch_size=8
steps_per_epoch ~= 700-750
20k steps ~= 27 epochs
60k steps ~= 80+ epochs
```

Avoid another 60k-90k blind run. Start with:

```text
max_steps = 10k-20k
learning_rate = 1e-5 to 2e-5
demo_every = 500 or 1000
ckpt_every = 1000 or 2000
```

Early stop condition:

```text
forward_delta correct/shuffled cosine remains near 1.0
or
correct and shuffled demos remain perceptually/metric-wise similar
```

Implementation note on 2026-06-06:

- Training demos can write `demo_control_diagnostics.csv`, including `control_input` stats, `forward_delta` stats, correct/shuffled pairwise cosine, generated-audio silence ratio, and low/mid/high spectral energy ratios.
- `--demo-stop-on-collapse true` requests training stop when correct/shuffled `forward_delta_cosine` crosses `--demo-collapse-cosine-threshold`.

### 4. Treat correct/shuffled separation as the primary success gate

The primary training success criterion should be:

```text
correct follows the reference melody
shuffled does not follow the original reference melody
control_scale=1 does not flatten output into continuous residual sound
```

Loss alone is not a sufficient indicator.

## Why The Original Paper Likely Did Not Expose The Same Failure

This project should not assume the paper had none of these problems. The paper may simply not report all failed ablations. However, several differences make the paper less likely to hit the same failure mode, or more likely to catch it.

### 1. Much larger and more diverse training data

The paper reports four public datasets:

```text
MTG, FMA, MTT, WikiMuTe
```

After processing, it reports:

```text
59,955 recordings
2,239.7 hours
instrumental filtering with PANNs
captions generated with SALMONN-13B where needed
```

Our `piano_clear` subset is much cleaner for melody, but far smaller. A small 5k-6k dataset can be repeated many times quickly; long training can overfit the ControlNet residual path.

### 2. The paper's training objective is paired with a masking curriculum from the start

The paper explicitly identifies the risk that precise melody prompts become a compressed target-audio shortcut. Its progressive curriculum masking:

- starts with all melody prompts masked,
- gradually exposes melody frames,
- keeps top-1 while randomly masking/shuffling top-2 to top-4 after the full-mask phase.

This does not guarantee success, but it is designed to reduce direct reconstruction shortcuts. If our schedule, implementation details, or checkpoint continuation differ, the shortcut can still dominate.

### 3. The paper evaluates melody controllability directly

The paper reports melody accuracy and compares against MusicGen melody baselines. That kind of metric is closer to our current `correct/shuffled` concern than loss curves are.

Our earlier runs relied heavily on loss, listening, and later demo similarity CSVs. The 90k diagnostic shows why loss is insufficient: a model can produce a strong ControlNet residual without using the correct pitch contour.

### 4. The paper may not use `zero waveform` as a no-control diagnostic

Our `zero` issue is partly a diagnostic artifact of feeding silent waveform through top-k CQT and expecting it to behave like null control. The paper discusses masking melody prompts and empty melody prompts, but that is conceptually closer to token masking than to extracting CQT from a zero waveform.

Therefore:

```text
zero waveform -> CQT top-k -> pseudo pitch
```

is our implementation/diagnostic hazard, not necessarily a paper-reported setup.

### 5. The paper's exact implementation details may differ

The paper describes pitch-specific embeddings and convolutional downsampling into latent melody prompts. Our implementation now has `MelodyControlEncoder`, but failures can still come from details such as:

- low-energy frame handling,
- padding/mask semantics,
- exact curriculum schedule,
- optimizer/scheduler behavior,
- checkpoint continuation,
- number of repeated epochs,
- inference EMA/CFG policy,
- lack of null-control diagnostics.

The paper's published results show that their final recipe can work, not that every partial reimplementation will avoid shortcuts.

## Working Interpretation

The current safest interpretation is:

```text
The 90k formal10s_v3 checkpoint has learned a dense ControlNet residual that does not distinguish correct melody from shuffled melody. The audible continuous mid/low-frequency artifact is likely this dense residual overpowering the base model. The exact spectral concentration still needs generated-audio spectral verification.
```

For future training, the priority is not to tune `control_scale` after the fact. The priority is to prevent the ControlNet branch from learning a content-insensitive continuous residual in the first place.
