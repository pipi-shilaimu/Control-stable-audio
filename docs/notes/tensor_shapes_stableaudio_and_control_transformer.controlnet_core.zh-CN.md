# ControlNet 关键变量（精简版）

本文档从 `tensor_shapes_stableaudio_and_control_transformer.zh-CN.md` 提炼，只保留实现 ControlNet 时必须关注的变量与形状。

## 1. 符号（只保留必需）

- `B`: batch size
- `T`: latent 时间长度（DiT 输入长度，不是 waveform 采样点）
- `S`: token 序列长度（`patch_size=1` 时通常 `S=T`）
- `C`: latent 通道数（StableAudio Open 1.0 常见 `64`）
- `D`: transformer 隐层维度（StableAudio Open 1.0 常见 `1536`）
- `dim_in`: `ContinuousTransformer.project_in` 的输入维度

## 2. 你要盯住的核心张量

### DiT 侧（进入 transformer 前）

- `x`（latent 主输入）: `[B, C, T]`
- `input_concat_cond`（如果使用 input-concat）: `[B, C_concat, T_concat]`
  - DiT 内部会先把 `T_concat` 插值到 `T`，然后通道拼接
  - 拼接后：`[B, C + C_concat, T]`

### ContinuousTransformer 侧（token 视图）

- `x_tokens`（进入 `ContinuousTransformer`）: `[B, S, dim_in]`
- `prepend_embeds`（若有）: `[B, S_pre, D]`
- `context`（cross-attn 文本条件）: `[B, S_text, D_ctx]`

### ControlNetContinuousTransformer 侧（你自己的模块）

- `x`: `[B, S, dim_in]`
- `control_input`: `[B, S, dim_in]`  ← **最重要**
- `control_scale`: `float`
- `x_base = project_in(x)`: `[B, S, D]`
- `x_ctrl = project_in(control_input)`: `[B, S, D]`
- `zero_linears[i]`: `Linear(D, D)`（zero-init）
- 每层注入后 `x_base` 仍保持: `[B, S_total, D]`

## 3. 与 StableAudio Open 对接时必须对齐的变量

- 替换点：`model.model.model.transformer`
- `dim_in` 来源：`base_transformer.project_in.in_features`
- `control_input.shape[-1]` 必须等于 `dim_in`
- `control_input.shape[1]`（`S`）必须与当前 token 序列长度一致（通常与 latent `T` 对齐）

## 4. 三个最关键的配置变量

- `num_control_layers`
  - 含义：前多少层进行 ControlNet 注入
  - 约束：`0 <= num_control_layers <= len(base.layers)`

- `default_control_scale` / `control_scale`
  - 含义：控制分支注入强度
  - 备注：zero-init 阶段就算 `control_scale > 0`，输出也可能不变（预期）

- `freeze_base()`
  - 含义：冻结预训练主干，只训练 control branch 与 adapter
  - 目的：保持基模能力，减少训练不稳定

## 5. 你最容易踩的 5 个坑

1. 把 `control_input` 传成 `[B, C, T]`（错），正确是 `[B, S, dim_in]`  
2. `control_input.shape[-1] != dim_in` 导致投影/层输入不匹配  
3. `prepend_embeds` 最后一维不等于 `D`  
4. 把 `input_concat_cond` 写成 `[B, T, C_concat]`（错序）  
5. 看到 `diff norm == 0` 就以为没生效（zero-init adapter 下这是正常行为）  

## 6. 最小检查清单（每次改完先看）

- `type(model.model.model.transformer)` 是否已替换成 `ControlNetContinuousTransformer`
- `dim_in == base_transformer.project_in.in_features`
- `control_input.shape == [B, S, dim_in]`
- zero-init 下：`diff norm` 接近 `0`
- 微小扰动 `zero_linear` 后：`diff norm` 应变成 `> 0`

