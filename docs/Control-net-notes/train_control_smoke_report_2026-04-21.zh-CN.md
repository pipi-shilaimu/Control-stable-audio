# StableAudio ControlNet 开训前最小验证报告（2026-04-21）

## 1. 运行命令

### Step 0：环境导入检查

```powershell
.\.venv\Scripts\python.exe -c "import stable_audio_tools; import stable_audio_control.models.control_transformer; import stable_audio_control.models.control_dit; print('IMPORT_OK')"
```

### Step 1：既有 smoke 基线

```powershell
.\.venv\Scripts\python.exe scripts/smoke_control_injection_stableaudio_open-1.py
.\.venv\Scripts\python.exe scripts/smoke_control_dit_wrapper.py
```

### Step 2：最小训练烟测（新增）

```powershell
.\.venv\Scripts\python.exe scripts/train_control_smoke.py
```

## 2. 关键输出摘录

### 2.1 导入检查

- `IMPORT_OK`

### 2.2 既有 smoke（控制注入链路）

- `smoke_control_injection_stableaudio_open-1.py`
  - `diff norm: 0.0`
- `smoke_control_dit_wrapper.py`
  - `zero-init diff norm: 0.0`
  - `after-perturb diff norm: 1.989155650138855`
  - `Smoke PASS: control_dit wrapper path is active and behaving as expected.`

### 2.3 新增训练烟测（单步训练）

- `loss: 17.825542449951172`
- `loss_is_finite: True`
- `trainable_param_count: 92053056`
- `trainable_with_grad_count: 30`
- `frozen_with_grad_count: 0`
- `trainable_probe: base_wrapper.model.model.transformer.zero_linears.0.weight delta: 235.9254608154297`
- `frozen_probe: base_wrapper.model.model.timestep_features.weight delta: 0.0`
- `Smoke PASS: one training step completed with finite loss and expected freeze behavior.`

## 3. 验证项对照（A~E）

- A：控制注入基线通过  
  - 结果：通过（zero-init 差值为 0；微扰后差值 > 0）
- B：`train_control_smoke.py` 单步训练通过  
  - 结果：通过（`training_step + backward + optimizer.step` 完整执行）
- C：loss 为有限数  
  - 结果：通过（`loss_is_finite: True`）
- D：参数冻结符合预期  
  - 结果：通过（`frozen_with_grad_count: 0`，冻结探针参数 `delta: 0.0`）
- E：可复现命令和关键输出已记录  
  - 结果：通过（见本报告第 1/2 节）

## 4. 本次新增脚本说明

- 新增脚本：`scripts/train_control_smoke.py`
- 功能覆盖：
  - 加载 `stabilityai/stable-audio-open-1.0`
  - 通过 `build_control_wrapper(...)` 构建控制包装器
  - 显式冻结非目标参数，仅放开 `control_layers` / `zero_linears` / `control_projector`
  - 构造最小 batch（`reals + metadata`，含 `prompt/seconds_start/seconds_total/padding_mask`）
  - 使用 `DiffusionCondTrainingWrapper` 执行 1 次 `training_step + backward + optimizer.step`
  - 打印 trainable 参数统计、loss finite 状态、冻结校验结果、探针参数更新

## 5. 风险与备注

- 运行日志出现 `flash_attn not installed` 与 `triton not found` 提示，但不影响本次最小训练烟测通过。
- `torch.cuda.amp.autocast(...)` FutureWarning 来自依赖库内部实现，不影响本次结论。

## 6. Go / No-Go 结论

**结论：Go（可进入下一轮正式训练准备）**

依据：A~E 全部通过，且最小训练链路已验证可执行、loss 有限、冻结策略有效。
