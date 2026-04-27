# `stable_audio_control/scripts/train_control_smoke.py` 使用与实现说明

## 1. 文档目的

本文档说明 `stable_audio_control/scripts/train_control_smoke.py` 的设计目标、执行流程、输出判据与排查方法，帮助你在正式长训前快速确认：

- ControlNet 包装路径可训练；
- 冻结策略符合预期；
- `DiffusionCondTrainingWrapper` 的单步训练闭环可执行；
- loss 数值有限（非 NaN/Inf）。

该脚本是“开训前最小烟测”，不是正式训练脚本。

---

## 2. 适用场景

在以下场景优先运行本脚本：

1. 刚改过 `stable_audio_control/models/control_dit.py` 或 `control_transformer.py`；
2. 调整了冻结策略或可训练参数范围；
3. 环境迁移后需要确认最小训练链路仍可用；
4. 准备启动长时间训练前，需要做 Go/No-Go 快速判定。

---

## 3. 运行前条件

1. 已创建并可用本仓库虚拟环境（`.venv`）。
2. 可访问 `stabilityai/stable-audio-open-1.0`（联网或本地缓存）。
3. `stable_audio_tools` 与本地 `stable_audio_control` 模块可导入。

推荐先做导入检查：

```powershell
.\.venv\Scripts\python.exe -c "import stable_audio_tools; import stable_audio_control.models.control_transformer; import stable_audio_control.models.control_dit; print('IMPORT_OK')"
```

---

## 4. 如何运行

```powershell
.\.venv\Scripts\python.exe stable_audio_control/scripts/train_control_smoke.py
```

脚本自动选择设备：
- 有 CUDA 用 `cuda`；
- 否则回退 `cpu`。

---

## 5. 脚本做了什么（逐步解释）

### Step A：加载与包装模型

1. 加载 `stabilityai/stable-audio-open-1.0`；
2. 调用 `build_control_wrapper(...)`：
   - 挂接 `ControlNetContinuousTransformer`；
   - 返回 `ControlConditionedDiffusionWrapper`。

### Step B：注入最小 `melody_control`

脚本内部使用 `MelodyControlAugmenter` 包装原 conditioner，在每次前向时额外注入合成的 `melody_control` 张量。  
目的：确保烟测确实走到控制分支，而不是“看似跑通、实际绕过控制注入”。

### Step C：处理 Lazy 参数初始化

`control_projector` 是 `LazyLinear`，若未 materialize 就直接冻结/建优化器会报错。  
脚本通过 `initialize_lazy_parameters(...)` 先用 dummy 条件走一次 `_extract_control_input(...)`，显式完成参数初始化。

### Step D：应用冻结策略

`apply_freeze_policy(...)` 先全量冻结，再仅放开以下模块：

- `control_layers`
- `zero_linears`
- `control_projector`

这样能把训练更新限制在控制分支，避免误训练主干。

### Step E：构造最小 batch

`build_minimal_batch(...)` 构造：

- `reals`: `[B, 2, T]` 随机音频（最小闭环用）；
- `metadata`：包含
  - `prompt`
  - `seconds_start`
  - `seconds_total`
  - `padding_mask`

满足 `DiffusionCondTrainingWrapper.training_step` 的最小输入要求。

### Step F：执行单步训练闭环

脚本手动执行一轮：

1. `training_step(...)`
2. `loss.backward()`
3. `optimizer.step()`

并检查：

- loss 是否 finite；
- 冻结参数是否出现梯度；
- 训练参数是否发生更新；
- 冻结探针参数是否保持不变。

---

## 6. 输出字段说明

脚本关键输出示例（不同机器数值会略有差异）：

- `trainable_param_count`: 可训练参数总数
- `trainable_tensor_count`: 可训练参数张量数量
- `loss`: 本步 loss 数值
- `loss_is_finite`: 是否是有限值
- `trainable_with_grad_count`: 有梯度的可训练参数数量
- `frozen_with_grad_count`: 有梯度的冻结参数数量（应为 0）
- `trainable_probe ... delta`: 训练探针参数更新量（应 > 0）
- `frozen_probe ... delta`: 冻结探针参数更新量（应 = 0）

成功标志：

```text
Smoke PASS: one training step completed with finite loss and expected freeze behavior.
```

---

## 7. Go / No-Go 判定建议

### Go（可继续正式训练准备）

同时满足：

1. 脚本正常结束；
2. `loss_is_finite: True`；
3. `frozen_with_grad_count: 0`；
4. `trainable_probe delta > 0`；
5. `frozen_probe delta == 0`；
6. 出现 `Smoke PASS`。

### No-Go（先修再训）

任一条件不满足即 No-Go，常见包括：

- `training_step failed: ...`（shape/dtype/device/batch mismatch）；
- loss 为 NaN/Inf；
- 冻结参数出现梯度；
- 训练参数未被更新。

---

## 8. 常见问题与排查

### 1) `Attempted to use an uninitialized parameter`

通常是 `LazyLinear` 未初始化导致。  
确认没有删除 `initialize_lazy_parameters(...)` 这一步。

### 2) `Batch mismatch` / shape 错误

优先检查：

- `metadata["padding_mask"]` 长度是否与 `reals` 音频长度一致；
- `melody_control` 的 batch 维和长度维是否被 wrapper 正确对齐；
- 自定义改动后是否仍保留 `build_minimal_batch(...)` 约束。

### 3) `loss` 非有限值

可先减小：

- `audio_length`
- `learning_rate`

并确认 dtype/device 一致。

### 4) 冻结策略异常

如果 `frozen_with_grad_count > 0`，说明冻结边界被破坏。  
重点复查 `apply_freeze_policy(...)` 是否被改动，或新模块是否被误加入优化器参数。

---

## 9. 与验证报告的关系

本脚本的实际运行证据与结论，可参考：

- `docs/Control-net-notes/train_control_smoke_report_2026-04-21.zh-CN.md`

建议流程：

1. 先跑本脚本；
2. 将关键输出追加到当日 smoke 报告；
3. 再做下一轮正式训练配置。
