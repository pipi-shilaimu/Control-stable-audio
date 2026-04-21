# Train Control Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and validate a one-step training smoke script for `ControlNetContinuousTransformer + ControlConditionedDiffusionWrapper` before full training.

**Architecture:** Reuse `stabilityai/stable-audio-open-1.0` loading path, wrap with `build_control_wrapper(...)`, freeze non-target parameters, then run one manual `DiffusionCondTrainingWrapper.training_step` + `backward` + `optimizer.step`. Keep control injection active by augmenting conditioner output with synthetic `melody_control`.

**Tech Stack:** Python, PyTorch, PyTorch Lightning wrapper classes in `stable_audio_tools`, local `stable_audio_control` modules.

---

### Task 1: Baseline Verification

**Files:**
- Modify: `agent.md` requirements only (no code changes)
- Test: `scripts/smoke_control_injection_stableaudio_open-1.py`, `scripts/smoke_control_dit_wrapper.py`

- [x] **Step 1: Run control injection smoke script**
- [x] **Step 2: Confirm zero-init diff ~= 0**
- [x] **Step 3: Run control wrapper smoke script**
- [x] **Step 4: Confirm post-perturb diff > 0**

### Task 2: Implement Minimal Training Smoke Script

**Files:**
- Create: `scripts/train_control_smoke.py`
- Test: `scripts/train_control_smoke.py`

- [x] **Step 1: Build script skeleton and imports**
- [x] **Step 2: Add model loading + control wrapper construction**
- [x] **Step 3: Add strict freeze policy for trainable target modules**
- [x] **Step 4: Add minimal batch creation (`reals + metadata`)**
- [x] **Step 5: Add one-step `training_step + backward + optimizer.step`**
- [x] **Step 6: Add finite loss / freeze checks / mismatch-safe diagnostics**

### Task 3: Execute and Report

**Files:**
- Create: `docs/Control-net-notes/train_control_smoke_report_2026-04-21.zh-CN.md`
- Test: `scripts/train_control_smoke.py`

- [x] **Step 1: Run `scripts/train_control_smoke.py`**
- [x] **Step 2: Capture key outputs and pass/fail evidence**
- [x] **Step 3: Write Go/No-Go report with unresolved risks (if any)**
