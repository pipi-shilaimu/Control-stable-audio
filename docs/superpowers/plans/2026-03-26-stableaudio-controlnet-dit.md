# StableAudio ControlNet-DiT (Melody + Text Editing) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `stabilityai/stable-audio-open-1.0` (DiT backbone) with a ControlNet-style control branch to enable melody- and text-controlled music editing, following *“Editing Music with Melody and Text: Using ControlNet for Diffusion Transformer”* (PDF in repo root).

**Architecture:** Keep the pretrained StableAudio Open DiT frozen; add a trainable ControlNet branch by cloning the first `N` Transformer blocks and injecting their outputs into the frozen blocks via zero-initialized linear layers. Use a top-`k` CQT representation (stereo, top-4 per channel ⇒ 8 indices per frame) as the melody control signal. Stabilize training with a progressive curriculum masking strategy over time (frame-wise masking + pitch-wise masking/shuffle for top-2..4).

**Tech Stack:** Python, PyTorch, `stable_audio_tools` (model + training wrappers), `pytorch_lightning`, `torchaudio` (I/O + highpass biquad), one of `nnAudio` (preferred, GPU-friendly) or `librosa` (CPU) for CQT, `safetensors`.

---

## Reference Notes (from the PDF)

- ControlNet for Transformer: clone first `N` blocks, add `zero_linear(output_copy_i)` into frozen stream before block `i+1`.
- Melody representation: compute CQT with 128 bins; per frame keep top-4 bins per stereo channel; interleave ⇒ `c ∈ R^{8 × T*fk}`, values in `1..128`.
- Preprocess melody extraction: biquad high-pass filter cutoff at Middle C (`261.2 Hz`) before CQT.
- Training: freeze DiT, train only ControlNet + melody-prompt-to-latent layers; optimizer AdamW (`lr=5e-5`) with InverseLR (`power=0.5`).
- Inference: DPM-Solver++ ~250 steps, CFG scale ~7 applied to **text** guidance (melody control stays “always-on”).

Keep the extracted text around for quick grep:
- `tmp/pdfs/editing_music_controlnet_dit_extracted.txt`

---

## File/Module Layout (new code to add)

Create a small local package for the new functionality (avoid editing `.venv/site-packages` directly):

- Create: `stable_audio_control/__init__.py`
- Create: `stable_audio_control/melody/cqt_topk.py` (top-k CQT extraction)
- Create: `stable_audio_control/melody/masking.py` (curriculum masking)
- Create: `stable_audio_control/melody/conditioner.py` (learned embedding + Conv1D downsampler)
- Create: `stable_audio_control/models/control_transformer.py` (ControlNet injection for `ContinuousTransformer`)
- Create: `stable_audio_control/models/control_dit.py` (ControlNet-DiT wrapper compatible with `stable_audio_tools` training/inference)
- Create: `stable_audio_control/data/custom_metadata.py` (dataset metadata hook to compute melody control from the target audio)
- Create: `scripts/train_controlnet_dit.py` (training entrypoint)
- Create: `scripts/generate_melody_edit.py` (inference entrypoint)

Optional (recommended):
- Create: `tests/test_cqt_topk.py`
- Create: `tests/test_masking.py`
- Create: `tests/test_control_transformer_shapes.py`

---

### Task 1: Pin down the integration strategy (model + training)

**Files:**
- Create: `stable_audio_control/models/control_dit.py`
- Modify (optional): `demo.py`

- [ ] **Step 1: Decide how the melody control enters the network**
  - Recommended: melody control enters **only** the ControlNet branch (so the frozen DiT keeps its text-only behavior).
  - Implementation consequence: the ControlNet branch gets an extra `control_input` tensor aligned to the DiT latent sequence length.

- [ ] **Step 2: Decide what to fine-tune**
  - Freeze everything from the base pretrained model (`stable_audio_tools`).
  - Train:
    - cloned control blocks (`N = depth/2`, e.g. 12 for a 24-block DiT),
    - `zero_linear` layers (one per cloned block),
    - melody prompt embedding + Conv1D downsampler.

- [ ] **Step 3: Define a clean forward signature**
  - Must work with `stable_audio_tools.training.diffusion.DiffusionCondTrainingWrapper`, which expects:
    - `conditioning = diffusion.conditioner(metadata, device)`
    - `diffusion(noised_latents, t, cond=conditioning, ...)`

---

### Task 2: Implement top-k CQT extraction (stereo, top-4 per channel)

**Files:**
- Create: `stable_audio_control/melody/cqt_topk.py`
- Test: `tests/test_cqt_topk.py`

- [ ] **Step 1: Add a CQT backend**
  - Prefer `nnAudio` (GPU-capable, avoids slow CPU librosa during training).
  - Install (venv): `.\.venv\Scripts\python.exe -m pip install nnAudio`
  - If `nnAudio` is blocked/unavailable, fallback: `librosa` (CPU).

- [ ] **Step 2: Implement preprocessing**
  - Use `torchaudio.functional.highpass_biquad(waveform, sample_rate=44100, cutoff_freq=261.2)` before CQT.
  - Ensure stereo input is shaped `[2, T]` (or `[B, 2, T]` in batch).

- [ ] **Step 3: Implement CQT + top-k**
  - Compute magnitude CQT with:
    - `bins_per_octave = 12`
    - `n_bins = 128`
    - `fmin` corresponding to MIDI note 0 (~8.18 Hz)
    - `hop_length = 512`
  - For each frame and channel: choose `top_k=4` frequency bins.
  - Output indices are **1-based** in paper; keep them as `1..128` and reserve `0` for “masked”.

- [ ] **Step 4: Unit test with a synthetic sine**
  - Generate a sine near A4 (440 Hz) and verify the top bin concentrates near the expected MIDI region.
  - Keep tests tolerant (CQT implementations differ slightly).

---

### Task 3: Implement progressive curriculum masking

**Files:**
- Create: `stable_audio_control/melody/masking.py`
- Test: `tests/test_masking.py`

- [ ] **Step 1: Encode the two masking axes**
  - Frame-wise masking: randomly mask a ratio of frames.
  - Pitch-wise masking/shuffle: after the initial full-mask phase:
    - keep top-1 always,
    - top-2..top-4 are randomly masked and shuffled.

- [ ] **Step 2: Implement a schedule**
  - Paper description: early training uses large mask ratios; over time probability mass shifts to smaller ratios, but doesn’t go strictly monotonic.
  - Practical implementation (configurable):
    - sample `mask_ratio ~ Beta(a(step), b)` with `a(step)` increasing over time so ratios get smaller on average,
    - keep a small floor probability to sample large mask ratios.

- [ ] **Step 3: Deterministic hooks for tests**
  - Masking must accept a `torch.Generator` or seed for reproducibility.

---

### Task 4: Build “melody prompt → latent control tensor” conditioner (trainable)

**Files:**
- Create: `stable_audio_control/melody/conditioner.py`
- Test: `tests/test_control_transformer_shapes.py`

- [ ] **Step 1: Define the embedding**
  - Input: integer indices `c` with shape `[B, 8, F]` where values are `0..128` (`0` = masked/pad).
  - Use `nn.Embedding(num_embeddings=129, embedding_dim=E, padding_idx=0)`.

- [ ] **Step 2: Downsample to match latent sequence length**
  - StableAudio Open operates in latent space; latent length is much shorter than waveform.
  - Implement Conv1D stack that maps `[B, 8, F]` → `[B, C_control, L_latent]`.
  - Keep it generic:
    - accept target length at runtime and use interpolation if needed,
    - or use strided convs tuned to typical audio/latent ratios.

- [ ] **Step 3: Apply masking only during training**
  - Conditioner should have a `training_masking=True/False` flag.
  - During inference, do **no masking** and no shuffling.

---

### Task 5: Implement ControlNet injection for `ContinuousTransformer`

**Files:**
- Create: `stable_audio_control/models/control_transformer.py`
- Test: `tests/test_control_transformer_shapes.py`

- [ ] **Step 1: Implement `ControlNetContinuousTransformer`**
  - Inputs:
    - `x` (the DiT input sequence, already projected to token space),
    - `control` (latent melody control tensor aligned to `x`),
    - `context` (text features for cross-attention),
    - `prepend_embeds` (timing/global conditioning used by StableAudio).
  - Internals:
    - reference the frozen `ContinuousTransformer` from the pretrained model,
    - create trainable clones of the first `N` blocks and load their weights from the frozen blocks,
    - create `zero_linear` modules (`nn.Linear(dim, dim)` with zero init) to inject into the frozen hidden stream.

- [ ] **Step 2: Inject at the right place**
  - For each layer `i < N`:
    - run `x_control = control_block_i(x_control, ...)`
    - run `x_frozen = frozen_block_i(x_frozen, ...)`
    - set `x_frozen = x_frozen + zero_linear_i(x_control)`
  - For `i >= N`: run only frozen blocks.

- [ ] **Step 3: Verify gradients**
  - Assert frozen blocks have `requires_grad=False`.
  - Assert control branch + zero linears + melody conditioner have grads.

---

### Task 6: Wrap into a ControlNet-DiT model compatible with `stable_audio_tools`

**Files:**
- Create: `stable_audio_control/models/control_dit.py`

- [ ] **Step 1: Build `ControlNetDiTWrapper`**
  - Start from a loaded pretrained model:
    - `model, model_config = stable_audio_tools.get_pretrained_model("stabilityai/stable-audio-open-1.0")`
  - Extract the underlying `DiTWrapper` (see `模型结构_易读.md` for nesting).
  - Replace the internal transformer with `ControlNetContinuousTransformer` while preserving:
    - timestep embedding path,
    - text cross-attention,
    - prepend conditioning,
    - CFG behavior (CFG on text only).

- [ ] **Step 2: Keep the public interface stable**
  - The final object used by training should still behave like a `ConditionedDiffusionModelWrapper`:
    - `diffusion.conditioner(metadata, device)` returns a dict of conditioning tensors.
    - the model forward consumes those tensors.

- [ ] **Step 3: Decide how conditioning IDs map**
  - Continue using:
    - `prompt` as cross-attention (`t5`),
    - `seconds_start` / `seconds_total` as timing conditioners (as-is),
  - Add:
    - `melody_control` as the melody indices or latent tensor.

---

### Task 7: Dataset metadata hook (derive melody from the target audio)

**Files:**
- Create: `stable_audio_control/data/custom_metadata.py`
- Create: `configs/dataset_audio_dir_with_melody.json` (optional but recommended)

- [ ] **Step 1: Implement `get_custom_metadata(info, audio)`**
  - Input `audio` is the cropped training waveform (already padded/cropped, stereo).
  - Output dict must include:
    - `prompt` (existing caption or a placeholder for local testing),
    - `melody_control` (top-k CQT indices) or `__audio__` extra audio fields if needed.

- [ ] **Step 2: Wire it into dataset config**
  - Use `docs/datasets.md` pattern with `"custom_metadata_module": "stable_audio_control/data/custom_metadata.py"`.

---

### Task 8: Training entrypoint (fine-tune ControlNet + melody conditioner)

**Files:**
- Create: `scripts/train_controlnet_dit.py`
- Create: `configs/train_controlnet_dit.json` (recommended)

- [ ] **Step 1: Build the model**
  - Load pretrained StableAudio Open.
  - Convert to ControlNet-DiT wrapper.
  - Freeze base weights.

- [ ] **Step 2: Configure optimizer/scheduler**
  - AdamW, `lr=5e-5`, weight decay as needed.
  - InverseLR scheduler with `power=0.5` (match paper).

- [ ] **Step 3: Use `DiffusionCondTrainingWrapper`**
  - Keep diffusion objective `v` (already the default for StableAudio Open configs).
  - Integrate melody masking via the melody conditioner (so it’s applied consistently).

- [ ] **Step 4: Smoke test on a tiny local dataset**
  - Use a small `audio_dir` dataset (even 10–20 `.wav` files) + dummy captions.
  - Run a few hundred steps to verify:
    - loss decreases,
    - model exports,
    - inference works.

Run example:
```powershell
.\.venv\Scripts\python.exe scripts/train_controlnet_dit.py --train-config configs/train_controlnet_dit.json --dataset-config configs/dataset_audio_dir_with_melody.json
```

---

### Task 9: Inference entrypoint (melody + text editing)

**Files:**
- Create: `scripts/generate_melody_edit.py`
- Modify (optional): `main.py` or `demo.py`

- [ ] **Step 1: CLI arguments**
  - `--prompt "..."`, `--melody_wav path.wav`, `--seconds_total 30`, `--seed 123`, `--steps 250`.

- [ ] **Step 2: Ensure CFG is applied only to text**
  - Keep the base model’s CFG path for cross-attn conditioning.
  - Do not “drop out” melody control during inference.

- [ ] **Step 3: Save output**
  - Write `output.wav` and optionally intermediate latents for debugging.

---

### Task 10: Verification & minimal tests

**Files:**
- Create: `tests/test_cqt_topk.py`
- Create: `tests/test_masking.py`
- Create: `tests/test_control_transformer_shapes.py`

- [ ] **Step 1: Shape tests (CPU)**
  - Random inputs through:
    - top-k CQT pipeline (on a short waveform),
    - masking,
    - conditioner output shape alignment,
    - ControlNet transformer forward.

- [ ] **Step 2: One forward pass smoke**
  - Build model, run forward with dummy conditioning and random latents.

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

---

## Execution Notes / Known Risks

- **CQT dependency:** `torchaudio` here does not include CQT; plan assumes `nnAudio` or `librosa`.
- **Compute cost:** computing CQT on-the-fly for every batch can be expensive; consider caching (pre-encode) once the pipeline works.
- **Length alignment:** CQT frame rate vs latent sequence length must be handled carefully (interpolation or conv downsampling).
- **CFG interactions:** keep melody conditioning independent of text CFG (match paper).

