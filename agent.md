# StableAudio ControlNet 开始训练前的最小验证 Agent 指南

## 目标

在**不进入长时间正式训练**的前提下，完成一套“最小训练可行性验证（smoke）”，确认当前 `ControlNetContinuousTransformer + ControlConditionedDiffusionWrapper` 方案可以安全进入训练阶段。

---

## 必读文档（先读再动手）

1. 总体注意事项（本项目当前结论与风险）  
   - `docs/Control-net-notes/controlnet_continuoustransformer_注意事项清单.zh-CN.md`
2. StableAudio 训练主流程  
   - `docs/Stable-audio-notes/training-pipeline.md`
3. Diffusion 调用链与条件路由  
   - `docs/Stable-audio-notes/diffusion-deep-dive.md`
4. 常见训练报错排查  
   - `docs/Stable-audio-notes/troubleshooting.md`
5. 当前控制模块实现（必须读代码）  
   - `stable_audio_control/models/control_transformer.py`
   - `stable_audio_control/models/control_dit.py`
6. 现有 smoke 脚本（复用）  
   - `stable_audio_control/scripts/smoke_control_injection_stableaudio_open-1.py`
   - `stable_audio_control/scripts/smoke_control_dit_wrapper.py`

---

## 任务边界（必须遵守）

- 这是“开训前验证任务”，**不是**正式训练任务。
- 只做最小闭环：前向、参数冻结、单步 training_step 可执行。
- 不做大规模数据准备，不做长时训练，不做超参搜索。

---

## 执行步骤（按顺序）

### Step 0：环境与依赖检查

- 激活环境：`.\\.venv\\Scripts\\activate.ps1`
- 基础导入检查（可用一行 python）：
  - `stable_audio_tools`
  - `stable_audio_control.models.control_transformer`
  - `stable_audio_control.models.control_dit`
- 若 HuggingFace 模型无法下载，优先使用本地缓存；本任务不要求联网新下载成功。

### Step 1：先跑已有两个 smoke（确认当前基线）

- 运行：
  - `.\\.venv\\Scripts\\python.exe stable_audio_control/scripts/smoke_control_injection_stableaudio_open-1.py`
  - `.\\.venv\\Scripts\\python.exe stable_audio_control/scripts/smoke_control_dit_wrapper.py`
- 预期：
  - zero-init 下 `diff` 接近 0
  - 微扰 `zero_linear` 后 `diff` > 0

### Step 2：补一个“最小训练烟测脚本”

- 新建脚本：`stable_audio_control/scripts/train_control_smoke.py`
- 脚本职责（最小可行）：
  1. 加载 `stabilityai/stable-audio-open-1.0`
  2. 用 `build_control_wrapper(...)` 包装模型
  3. 显式冻结非目标参数（至少保证 base transformer 不训练）
  4. 构造最小 batch（`reals + metadata`，包含 `prompt/seconds_start/seconds_total/padding_mask`）
  5. 构建 `DiffusionCondTrainingWrapper`
  6. 仅执行 1 次 `training_step` + `backward` + `optimizer.step()`
  7. 打印：
     - trainable 参数数量
     - loss 是否 finite
     - 是否出现 shape/dtype/batch mismatch

### Step 3：最小训练验证集（必须全部通过）

- 验证项 A：控制注入基线通过（Step 1 两个脚本）
- 验证项 B：`train_control_smoke.py` 单步训练通过（无异常）
- 验证项 C：loss 为有限数（非 NaN/Inf）
- 验证项 D：参数冻结符合预期（非目标参数未参与训练）
- 验证项 E：记录可复现命令和关键输出

### Step 4：输出交付（用于下一轮正式训练）

- 产出一份简短报告（可放 `docs/Control-net-notes/`）：
  - 运行命令
  - 关键输出
  - 是否达到开训条件
  - 未解决风险（若有）

---

## 开训判定标准（Go / No-Go）

**Go（可进入下一对话做正式训练）**：  
- 上述验证项 A~E 全部通过。

**No-Go（先修再训）**：  
- 任一 smoke 失败；或
- 单步 training_step 报 shape/dtype/device 错；或
- loss 非有限值；或
- 冻结范围不符合预期。

---

## 建议下一对话开场提示词（给执行 Agent）

> 请按 `agent.md` 执行“开始训练前最小验证任务”，先读文档再实施。目标是完成 `stable_audio_control/scripts/train_control_smoke.py` 并跑通最小验证集，最后给出 Go/No-Go 结论和证据。
