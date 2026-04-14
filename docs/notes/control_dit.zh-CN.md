# `control_dit.py` 说明文档

本文档对应文件：`stable_audio_control/models/control_dit.py`  
目标读者：正在基于 `stable_audio_tools` 做 ControlNet-DiT 扩展的开发者。

---

## 1. 这个模块解决什么问题

`control_dit.py` 主要解决两件事：

1. 不改 `stable_audio_tools` 源码（不改 `.venv`）的前提下，把预训练 StableAudio Open 的内部 `ContinuousTransformer` 替换为 `ControlNetContinuousTransformer`。
2. 保持训练接口兼容 `DiffusionCondTrainingWrapper`，并把 `cond["melody_control"]` 自动转换成 `control_input` 传给 ControlNet 分支。

一句话总结：  
`control_transformer.py` 负责“注入逻辑”，`control_dit.py` 负责“接口对接”。

---

## 2. 模块包含的对象

`stable_audio_control/models/control_dit.py` 提供 3 个对外入口：

1. `ControlConditionedDiffusionWrapper`
2. `attach_controlnet_transformer(...)`
3. `build_control_wrapper(...)`

它们分别解决：

- **`ControlConditionedDiffusionWrapper`**  
  兼容训练/推理 wrapper 的调用习惯，接管 `melody_control -> control_input` 这条链路。

- **`attach_controlnet_transformer(...)`**  
  就地替换 `base_wrapper.model.model.transformer` 为 `ControlNetContinuousTransformer`。

- **`build_control_wrapper(...)`**  
  一步做完“替换 transformer + 构建 control wrapper”。

---

## 3. 关键数据契约（最重要）

### 3.1 外部调用保持不变

训练侧仍可使用原习惯：

```python
conditioning = diffusion.conditioner(metadata, device)
output = diffusion(noised_latents, t, cond=conditioning, cfg_dropout_prob=...)
```

### 3.2 新增条件键

- 默认控制键：`control_id = "melody_control"`
- 从 `cond` 读取方式：`cond["melody_control"]`

### 3.3 `melody_control` 支持的输入形状

`ControlConditionedDiffusionWrapper._extract_control_input()` 支持：

- `[B, C, L]`（推荐）
- `[B, L, C]`
- `[C, L]`（会自动补 batch 维）

然后内部会统一变成：

- `control_input: [B, L_target, dim_in]`

其中：

- `L_target = x.shape[-1]`（当前 latent 序列长度）
- `dim_in = transformer.project_in.in_features`

---

## 4. 自动处理逻辑

`control_dit.py` 会自动做这些事：

1. 从 `cond` 提取 `melody_control`
2. 长度对齐（插值到 `x` 的目标长度）
3. 维度规范化（统一到 `[B, L, C]`）
4. 通过 `LazyLinear` 投影到 `dim_in`
5. 调用 base wrapper 时透传：
   - `control_input=...`
   - `control_scale=...`

同时会从 `cond_for_base` 中移除 `melody_control`，避免其进入原始 cross/prepend/input-concat 路由。

---

## 5. 与 `control_transformer.py` 的分工

- `control_dit.py`  
  负责上层接口和条件张量预处理（取值、插值、投影、传参）。

- `control_transformer.py`  
  负责层内注入（control branch、zero-linear 注入、CFG batch 对齐等）。

不要把 CQT、masking、metadata 解析放进 `control_transformer.py`，这会打乱职责边界。

---

## 6. 常用用法

### 6.1 快速构建包装器

```python
from stable_audio_tools import get_pretrained_model
from stable_audio_control.models import build_control_wrapper

base_model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
model = build_control_wrapper(
    base_wrapper=base_model,
    num_control_layers=2,
    control_id="melody_control",
    default_control_scale=1.0,
    freeze_base=True,
)
```

### 6.2 最小前向（随机 `melody_control`）

```python
cond = model.conditioner([{
    "prompt": "test prompt",
    "seconds_start": 0,
    "seconds_total": 5,
}], device)

cond["melody_control"] = [torch.randn(1, 8, 128, device=device), None]

out = model(
    x, t,
    cond=cond,
    cfg_scale=1.0,
    cfg_dropout_prob=0.0,
    control_scale=0.5,
)
```

---

## 7. 已有冒烟测试脚本

推荐先跑：

- `scripts/smoke_control_injection.py`  
  纯离线 toy transformer，验证注入机制本身。

- `scripts/smoke_control_injection_stableaudio_open-1.py`  
  真实 StableAudio Open 注入 smoke。

- `scripts/smoke_control_dit_wrapper.py`  
  专门验证 `control_dit.py` 接口链路（含 `melody_control` 提取与转换）。

---

## 8. 常见问题与排查

### Q1: `diff norm` 为 0，是不是失败了？

不一定。  
如果 `zero_linears` 是零初始化，这是预期行为：初始阶段不改变原模型输出。

### Q2: 为什么出现 batch 维不匹配？

常见原因是 CFG 场景下 batch 扩成 `2B`。  
`control_transformer.py` 已包含自动 batch 对齐逻辑；如果仍报错，优先检查传入的 `control_input` 实际 shape。

### Q3: 为什么 `get_pretrained_model` 偶尔超时？

通常是 HuggingFace 资源（特别是 T5 tokenizer/model）缓存或网络问题。  
与 `control_dit.py` 本身无关。

### Q4: `melody_control` 放哪一层生成更合理？

建议：

- dataset/custom metadata 侧生成原始 melody 特征（如 top-k CQT）
- `control_dit.py` 只负责接口转换，不做信号算法

---

## 9. 后续扩展建议（V2）

1. 支持 `negative_melody_control`（给 CFG 无条件分支单独控制）
2. 将 `control_projector` 从单层线性扩展为更强特征头（例如 LN + MLP）
3. 引入显式 `melody_mask` 并在控制分支中使用
4. 增加针对 `control_dit` 的单元测试（shape、dtype、cfg batch 对齐）

