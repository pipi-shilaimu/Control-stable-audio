# StableAudio ControlNet-DiT（旋律 + 文本编辑）实现计划

> **面向智能体执行者：** 必须使用子技能：使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 按任务逐步执行本计划。步骤使用勾选框（`- [ ]`）语法用于追踪进度。

**目标：** 在 `stabilityai/stable-audio-open-1.0`（DiT 主干）上增加一个 ControlNet 风格的控制分支，实现“旋律 + 文本”双条件的音乐编辑能力，思路来自仓库根目录 PDF：*“Editing Music with Melody and Text: Using ControlNet for Diffusion Transformer”*。

**架构：** 冻结预训练的 StableAudio Open DiT；通过克隆前 `N` 个 Transformer block 作为可训练的 ControlNet 分支，并通过“零初始化线性层（zero-initialized linear layers）”把控制分支的输出注入到冻结分支对应层中。旋律控制信号使用 top-`k` CQT 表示（立体声、每个声道每帧 top-4 ⇒ 每帧 8 个索引）。训练稳定性通过随时间推进的“渐进式课程学习遮罩（progressive curriculum masking）”实现（frame-wise 遮罩 + 对 top-2..4 做 pitch-wise 遮罩/打乱）。

**技术栈：** Python、PyTorch、`stable_audio_tools`（模型 + 训练 wrapper）、`pytorch_lightning`、`torchaudio`（I/O + biquad 高通）、CQT 计算使用 `nnAudio`（优先，GPU 友好）或 `librosa`（CPU）、`safetensors`。

---

## 参考要点（来自 PDF）

- Transformer 版 ControlNet：克隆前 `N` 个 block，在冻结分支进入第 `i+1` 个 block 之前，将 `zero_linear(output_copy_i)` 加入冻结分支 hidden stream。
- 旋律表示：计算 128 个 bin 的 CQT；对每一帧、每个立体声声道保留 top-4 bin；再交错（interleave）⇒ `c ∈ R^{8 × T*fk}`，取值为 `1..128`。
- 旋律提取预处理：在计算 CQT 之前，先做 biquad 高通滤波，截止频率为 Middle C（`261.2 Hz`）。
- 训练：冻结 DiT，仅训练 ControlNet + “旋律 prompt → latent”的相关层；优化器 AdamW（`lr=5e-5`）+ InverseLR（`power=0.5`）。
- 推理：DPM-Solver++ 约 250 步，CFG scale 约 7，并且 CFG 只作用于 **文本** 引导（旋律控制保持一直开启）。

保留已抽取的 PDF 文本便于快速 grep：
- `tmp/pdfs/editing_music_controlnet_dit_extracted.txt`

---

## 文件 / 模块结构（需要新增的代码）

为新功能创建一个小的本地包（避免直接修改 `.venv/site-packages`）：

- 新建：`stable_audio_control/__init__.py`
- 新建：`stable_audio_control/melody/cqt_topk.py`（top-k CQT 提取）
- 新建：`stable_audio_control/melody/masking.py`（课程学习遮罩策略）
- 新建：`stable_audio_control/melody/conditioner.py`（可学习 embedding + Conv1D 下采样器）
- 新建：`stable_audio_control/models/control_transformer.py`（为 `ContinuousTransformer` 注入 ControlNet）
- 新建：`stable_audio_control/models/control_dit.py`（ControlNet-DiT wrapper，兼容 `stable_audio_tools` 训练 / 推理）
- 新建：`stable_audio_control/data/custom_metadata.py`（dataset metadata hook：从目标音频计算旋律控制）
- 新建：`scripts/train_controlnet_dit.py`（训练入口）
- 新建：`scripts/generate_melody_edit.py`（推理入口）

可选（推荐）：
- 新建：`tests/test_cqt_topk.py`
- 新建：`tests/test_masking.py`
- 新建：`tests/test_control_transformer_shapes.py`

---

### 任务 1：确定集成策略（模型 + 训练）

**文件：**
- 新建：`stable_audio_control/models/control_dit.py`
- 修改（可选）：`demo.py`

- [ ] **步骤 1：决定旋律控制如何进入网络**
  - 推荐：旋律控制 **仅** 进入 ControlNet 分支（这样冻结的 DiT 能保持“纯文本”行为）。
  - 实现影响：ControlNet 分支需要额外接收一个与 DiT latent 序列长度对齐的 `control_input` 张量。

- [ ] **步骤 2：决定微调哪些参数**
  - 冻结基础预训练模型（`stable_audio_tools`）中的所有参数。
  - 训练以下部分：
    - 克隆出来的 control blocks（`N = depth/2`，例如 24 层 DiT 则 N=12），
    - `zero_linear` 层（每个克隆 block 对应一个），
    - 旋律 prompt 的 embedding + Conv1D 下采样器。

- [ ] **步骤 3：定义一个干净的 forward 接口**
  - 必须兼容 `stable_audio_tools.training.diffusion.DiffusionCondTrainingWrapper`，它默认期望：
    - `conditioning = diffusion.conditioner(metadata, device)`
    - `diffusion(noised_latents, t, cond=conditioning, ...)`

#### 任务 1 补充：接口设计规格（V1，可直接实现）

**A. 训练/推理对外调用保持不变（兼容现有 wrapper）**

- 外部调用仍使用：
  - `conditioning = diffusion.conditioner(metadata, device)`
  - `diffusion(noised_latents, t, cond=conditioning, cfg_dropout_prob=..., **extra_args)`
- 不要求调用方显式传 `control_input`（由 `control_dit.py` 从 `cond` 中自动提取并转换）。
- 可选新增运行时参数：
  - `control_scale: float = 1.0`（训练默认 `1.0`，推理可调）

**B. `cond` 字段契约（新增 `melody_control`）**

- 约定新增 id：`melody_control`。
- `cond["melody_control"]` 的返回结构与 `stable_audio_tools` 一致，仍为二元组语义：`[tensor, mask]`。
- `tensor` 的标准输入形状（推荐）：
  - `[B, C_melody, L_melody]`（channels-first，便于与 `input_concat`/音频特征习惯对齐）
- `mask` 在 V1 可为 `None`（后续可扩展为帧级掩码）。

**C. 形状转换规则（由 `control_dit.py` 统一负责）**

- 从 `cond["melody_control"][0]` 读取 `melody_tensor` 后，执行：
  1. 时长对齐：`L_melody -> L_x`（其中 `L_x = x.shape[-1]`），默认最近邻插值；
  2. 维度变换：`[B, C_melody, L_x] -> [B, L_x, C_melody]`；
  3. 可学习投影：`Linear(C_melody, dim_in)`；
  4. 得到 `control_input: [B, L_x, dim_in]`，传入 `ControlNetContinuousTransformer`。
- `ControlNetContinuousTransformer` 不再负责外部特征形状推断；它只消费最终标准形状：
  - `control_input: [B, seq, dim_in]`

**D. 模块职责边界（避免后续职责混乱）**

- `stable_audio_control/models/control_transformer.py`
  - 只做 Transformer 层内 ControlNet 注入逻辑（control branch + zero-linear 注入）
  - 不做 metadata 解析，不做 CQT，不做 `cond` 路由
- `stable_audio_control/models/control_dit.py`
  - 负责把 `cond` 映射到模型可消费的参数（尤其是 `melody_control -> control_input`）
  - 负责 CFG 情况下 `control_input` 的 batch 对齐
- `stable_audio_control/melody/*`
  - 负责旋律特征生成与训练时遮罩策略

**E. CFG 对齐规则（必须实现）**

- 当 DiT 进入 batch CFG 路径时，`x/t` 会扩为 `2B`（条件分支 + 无条件分支）。
- `control_input` 必须同步扩为 `2B`，否则会 batch 维不匹配。
- V1 默认策略：
  - `control_input_cfg = torch.cat([control_input, control_input], dim=0)`
  - 即无条件分支也复制同一 control（先保证稳定与兼容）
- V2 可扩展策略：
  - 支持 `negative_melody_control` 或对无条件分支置零控制

**F. 可训练参数范围（V1）**

- 冻结：
  - 预训练 StableAudio Open 的 base DiT、原始 `ContinuousTransformer` 主干、pretransform、原有 text/timing conditioner
- 训练：
  - `control_layers`
  - `zero_linears`
  - melody feature projector（`C_melody -> dim_in`）
  - 后续加入的 melody embedding + Conv1D 下采样器

**G. `control_dit.py` 建议签名（V1）**

```python
class ControlConditionedDiffusionWrapper(nn.Module):
    def __init__(
        self,
        base_wrapper,                      # get_pretrained_model 返回对象
        control_id: str = "melody_control",
        default_control_scale: float = 1.0,
        melody_channels: int = 8,
        control_interp_mode: str = "nearest",
    ): ...

    def forward(
        self,
        x: torch.Tensor,                   # [B, C_latent, L]
        t: torch.Tensor,                   # [B]
        cond: dict,                        # conditioner 输出
        cfg_dropout_prob: float = 0.0,
        control_scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor: ...
```

**H. 任务 1 的最小验收标准（进入任务 2 前必须满足）**

- Smoke-1：zero-init adapter 下，`control_scale=0` 与 `control_scale>0` 输出差异为 0（或数值近似 0）。
- Smoke-2：轻微扰动任意一个 `zero_linear` 后，输出差异变为非零。
- Smoke-3：通过 `DiffusionCondTrainingWrapper` 跑 1 个 batch 时不发生 shape/batch mismatch。

---

### 任务 2：实现 top-k CQT 提取（立体声，每声道 top-4）

**文件：**
- 新建：`stable_audio_control/melody/cqt_topk.py`
- 测试：`tests/test_cqt_topk.py`

- [ ] **步骤 1：加入 CQT 后端**
  - 优先使用 `nnAudio`（支持 GPU，训练时避免 CPU librosa 太慢）。
  - 安装（venv）：`.\.venv\Scripts\python.exe -m pip install nnAudio`
  - 如果 `nnAudio` 受限/不可用，则退回使用 `librosa`（CPU）。

- [ ] **步骤 2：实现预处理**
  - 在 CQT 之前使用 `torchaudio.functional.highpass_biquad(waveform, sample_rate=44100, cutoff_freq=261.2)`。
  - 确保立体声音频输入形状是 `[2, T]`（或 batch 形式 `[B, 2, T]`）。

- [ ] **步骤 3：实现 CQT + top-k**
  - 计算 CQT 幅度谱（magnitude CQT），参数为：
    - `bins_per_octave = 12`
    - `n_bins = 128`
    - `fmin` 对应 MIDI note 0（约 `8.18 Hz`）
    - `hop_length = 512`
  - 对每个声道、每一帧选择 `top_k=4` 个频率 bin。
  - 论文里索引是 **从 1 开始（1-based）**；这里保持为 `1..128`，并保留 `0` 作为“被遮罩/填充（masked）”。

- [ ] **步骤 4：用合成正弦波做单元测试**
  - 生成接近 A4（440 Hz）的正弦波，验证 top bin 集中在预期 MIDI 区间附近。
  - 测试要允许一定误差（不同 CQT 实现会有细微差异）。

---

### 任务 3：实现渐进式课程学习遮罩（progressive curriculum masking）

**文件：**
- 新建：`stable_audio_control/melody/masking.py`
- 测试：`tests/test_masking.py`

- [ ] **步骤 1：实现两种遮罩维度**
  - Frame-wise 遮罩：随机遮罩一定比例的帧。
  - Pitch-wise 遮罩/打乱：在初始全遮罩阶段之后：
    - top-1 永远保留，
    - top-2..top-4 随机遮罩并随机打乱。

- [ ] **步骤 2：实现一个可配置的遮罩调度**
  - 论文描述：训练早期遮罩比例大；随着训练推进，采样到小遮罩比例的概率逐渐增大，但不是严格单调下降。
  - 工程实现（可配置）：
    - 采样 `mask_ratio ~ Beta(a(step), b)`，并让 `a(step)` 随训练变大，从而平均遮罩比例更小，
    - 同时保留一个很小的下限概率，仍能采样到较大的遮罩比例。

- [ ] **步骤 3：提供可复现的测试接口**
  - 遮罩函数必须支持传入 `torch.Generator` 或 seed，保证可复现。

---

### 任务 4：实现“melody prompt → latent 控制张量”的 conditioner（可训练）

**文件：**
- 新建：`stable_audio_control/melody/conditioner.py`
- 测试：`tests/test_control_transformer_shapes.py`

- [ ] **步骤 1：定义 embedding**
  - 输入：整数索引 `c`，形状 `[B, 8, F]`，取值范围 `0..128`（其中 `0` 表示 masked/pad）。
  - 使用 `nn.Embedding(num_embeddings=129, embedding_dim=E, padding_idx=0)`。

- [ ] **步骤 2：下采样到与 latent 序列长度匹配**
  - StableAudio Open 在 latent 空间运行；latent 长度会显著短于 waveform。
  - 实现一个 Conv1D stack，将 `[B, 8, F]` 映射到 `[B, C_control, L_latent]`。
  - 设计保持通用：
    - 运行时接受 target length，必要时用插值对齐，
    - 或使用针对常见 audio/latent 比例调好的 stride conv。

- [ ] **步骤 3：仅在训练时应用遮罩**
  - conditioner 需要 `training_masking=True/False` 开关。
  - 推理阶段 **不做遮罩**，也不做 shuffle。

---

### 任务 5：为 `ContinuousTransformer` 实现 ControlNet 注入

**文件：**
- 新建：`stable_audio_control/models/control_transformer.py`
- 测试：`tests/test_control_transformer_shapes.py`

- [ ] **步骤 1：实现 `ControlNetContinuousTransformer`**
  - 输入：
    - `x`（DiT 输入序列，已投影到 token 空间），
    - `control`（与 `x` 对齐的 latent 旋律控制张量），
    - `context`（文本 cross-attention 特征），
    - `prepend_embeds`（StableAudio 使用的 timing/global conditioning）。
  - 内部实现：
    - 引用预训练模型中冻结的 `ContinuousTransformer`，
    - 克隆前 `N` 个 block 作为可训练分支，并从冻结分支加载对应权重初始化，
    - 创建 `zero_linear`（`nn.Linear(dim, dim)`，并做零初始化）把控制分支注入到冻结分支 hidden stream。

- [ ] **步骤 2：在正确位置注入**
  - 对每个层 `i < N`：
    - 先跑 `x_control = control_block_i(x_control, ...)`
    - 再跑 `x_frozen = frozen_block_i(x_frozen, ...)`
    - 然后 `x_frozen = x_frozen + zero_linear_i(x_control)`
  - 对 `i >= N`：只跑冻结分支 block。

- [ ] **步骤 3：验证梯度**
  - 断言冻结 block 的参数 `requires_grad=False`。
  - 断言控制分支 + zero linear + melody conditioner 都能拿到梯度。

---

### 任务 6：封装成兼容 `stable_audio_tools` 的 ControlNet-DiT 模型

**文件：**
- 新建：`stable_audio_control/models/control_dit.py`

- [ ] **步骤 1：构建 `ControlNetDiTWrapper`**
  - 从加载好的预训练模型开始：
    - `model, model_config = stable_audio_tools.get_pretrained_model("stabilityai/stable-audio-open-1.0")`
  - 提取底层 `DiTWrapper`（嵌套层级可参考 `模型结构_易读.md`）。
  - 将内部 transformer 替换为 `ControlNetContinuousTransformer`，同时保留：
    - timestep embedding 路径，
    - 文本 cross-attention，
    - prepend conditioning，
    - CFG 行为（CFG 只作用于文本）。

- [ ] **步骤 2：保持公共接口稳定**
  - 最终用于训练的对象仍应表现得像 `ConditionedDiffusionModelWrapper`：
    - `diffusion.conditioner(metadata, device)` 返回 conditioning tensors 的 dict。
    - model forward 消费这些 conditioning tensors。

- [ ] **步骤 3：决定 conditioning IDs 的映射**
  - 继续使用：
    - `prompt` 作为 cross-attention（`t5`），
    - `seconds_start` / `seconds_total` 作为 timing conditioner（保持现状），
  - 新增：
    - `melody_control` 作为旋律索引或其对应 latent 张量。

---

### 任务 7：Dataset metadata hook（从目标音频导出旋律控制）

**文件：**
- 新建：`stable_audio_control/data/custom_metadata.py`
- 新建：`configs/dataset_audio_dir_with_melody.json`（可选但推荐）

- [ ] **步骤 1：实现 `get_custom_metadata(info, audio)`**
  - 输入 `audio` 是训练时裁剪后的 waveform（已经 pad/crop，立体声）。
  - 输出 dict 至少包含：
    - `prompt`（已有 caption，或本地测试用占位符），
    - `melody_control`（top-k CQT 索引）；如有需要也可用 `__audio__` 返回额外音频字段。

- [ ] **步骤 2：接入 dataset config**
  - 参考 `docs/datasets.md` 的写法，配置 `"custom_metadata_module": "stable_audio_control/data/custom_metadata.py"`。

---

### 任务 8：训练入口（只微调 ControlNet + melody conditioner）

**文件：**
- 新建：`scripts/train_controlnet_dit.py`
- 新建：`configs/train_controlnet_dit.json`（推荐）

- [ ] **步骤 1：构建模型**
  - 加载预训练 StableAudio Open。
  - 转换为 ControlNet-DiT wrapper。
  - 冻结 base 权重。

- [ ] **步骤 2：配置优化器 / 调度器**
  - AdamW，`lr=5e-5`，weight decay 视情况设置。
  - InverseLR scheduler，`power=0.5`（对齐论文）。

- [ ] **步骤 3：使用 `DiffusionCondTrainingWrapper`**
  - 保持 diffusion objective 为 `v`（StableAudio Open 配置中通常已是默认）。
  - 通过 melody conditioner 集成 masking（保证遮罩逻辑一致地作用在训练中）。

- [ ] **步骤 4：用极小本地数据集做 smoke test**
  - 用一个很小的 `audio_dir` 数据集（即使只有 10–20 个 `.wav`）+ dummy captions。
  - 跑几百 step 验证：
    - loss 下降，
    - 模型能导出，
    - 推理能跑通。

运行示例：
```powershell
.\.venv\Scripts\python.exe scripts/train_controlnet_dit.py --train-config configs/train_controlnet_dit.json --dataset-config configs/dataset_audio_dir_with_melody.json
```

---

### 任务 9：推理入口（旋律 + 文本编辑）

**文件：**
- 新建：`scripts/generate_melody_edit.py`
- 修改（可选）：`main.py` 或 `demo.py`

- [ ] **步骤 1：CLI 参数**
  - `--prompt "..."`、`--melody_wav path.wav`、`--seconds_total 30`、`--seed 123`、`--steps 250`。

- [ ] **步骤 2：确保 CFG 只作用于文本**
  - 保持 base model 的 CFG 路径用于 cross-attn conditioning。
  - 推理阶段不要对旋律控制做“drop out”。

- [ ] **步骤 3：保存输出**
  - 写出 `output.wav`，并可选保存中间 latents 便于调试。

---

### 任务 10：验证与最小化测试

**文件：**
- 新建：`tests/test_cqt_topk.py`
- 新建：`tests/test_masking.py`
- 新建：`tests/test_control_transformer_shapes.py`

- [ ] **步骤 1：Shape 测试（CPU）**
  - 用随机输入贯通：
    - top-k CQT pipeline（短 waveform），
    - masking，
    - conditioner 输出 shape 对齐，
    - ControlNet transformer forward。

- [ ] **步骤 2：单次 forward 的 smoke**
  - 构建模型，用 dummy conditioning + 随机 latents 跑一遍 forward。

运行：
```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

---

## 执行说明 / 已知风险

- **CQT 依赖：** 当前环境的 `torchaudio` 不包含 CQT，本计划假设使用 `nnAudio` 或 `librosa`。
- **计算成本：** 每个 batch 在线计算 CQT 可能很贵；流程跑通后建议考虑缓存（pre-encode）来降成本。
- **长度对齐：** CQT 的帧率与 latent 序列长度的对齐需要小心处理（插值或 Conv 下采样）。
- **CFG 交互：** 旋律 conditioning 保持独立于文本 CFG（对齐论文做法）。

