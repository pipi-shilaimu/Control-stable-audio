# StableAudio + Control Transformer 张量形状说明

本文档描述两部分的张量参数形状：
- `stable_audio_tools`（以 `ConditionedDiffusionModelWrapper -> DiTWrapper -> DiffusionTransformer -> ContinuousTransformer` 调用链为主）
- `stable_audio_control/models/control_transformer.py`（`ControlNetContinuousTransformer`）

## 1. 符号约定

- `B`: batch size
- `C`: latent 通道数（StableAudio Open 1.0 常见是 `64`）
- `T`: latent 时间长度（不是 waveform 采样点）
- `S`: token 序列长度（在 `patch_size=1` 时通常 `S=T`）
- `D`: transformer 隐层维度（StableAudio Open 1.0 常见是 `1536`）
- `S_text`: 文本 token 序列长度（例如 T5 的 token 长度）
- `D_text`: 文本条件维度（project 前）
- `S_pre`: prepend 条件 token 长度

---

## 2. 总体调用链与形状流向

典型前向调用（条件扩散）：

1. `ConditionedDiffusionModelWrapper.forward(x, t, cond, **kwargs)`
2. 路由 `cond` 到 `cross_attn_cond / input_concat_cond / prepend_cond / global_cond`
3. `DiTWrapper.forward(...)`
4. `DiffusionTransformer.forward(...)`
5. `DiffusionTransformer._forward(...)`
6. `ContinuousTransformer.forward(...)`

主干 latent 输入输出：
- `x`: `[B, C, T]`
- 输出 `output`: `[B, C, T]`

---

## 3. `stable_audio_tools` 关键参数形状

### 3.1 ConditionedDiffusionModelWrapper

文件：`.venv/Lib/site-packages/stable_audio_tools/models/diffusion.py`

- `forward(self, x, t, cond, **kwargs)`
  - `x`: `[B, C, T]`
  - `t`: `[B]`（连续时间/噪声步）
  - `cond`: 字典，值通常是 `(tensor, mask)` 对

- `get_conditioning_inputs(...)` 的路由结果：
  - `cross_attn_cond`: `[B, S_text_total, D_text]`
    - 多个 `cross_attn_cond_ids` 在序列维拼接（`dim=1`）
  - `cross_attn_mask`: `[B, S_text_total]`（或对应布尔 mask）
  - `global_cond`: `[B, D_global_total]`
    - 多个 `global_cond_ids` 在通道维拼接（`dim=-1`）
  - `input_concat_cond`: `[B, C_concat_total, T_concat]`
    - 多个 `input_concat_ids` 在通道维拼接（`dim=1`）
  - `prepend_cond`: `[B, S_pre_total, D_pre]`
    - 多个 `prepend_cond_ids` 在序列维拼接（`dim=1`）
  - `prepend_cond_mask`: `[B, S_pre_total]`

### 3.2 DiTWrapper

文件：`.venv/Lib/site-packages/stable_audio_tools/models/diffusion.py`

- `DiTWrapper.forward(...)` 关键参数
  - `x`: `[B, C, T]`
  - `t`: `[B]`
  - `cross_attn_cond`: `[B, S_text, D_text]` 或 `None`
  - `input_concat_cond`: `[B, C_concat, T_concat]` 或 `None`
  - `global_cond`: `[B, D_global]` 或 `None`
  - `prepend_cond`: `[B, S_pre, D_pre]` 或 `None`
  - 其余 CFG 参数是标量或小 tuple，不是大张量

### 3.3 DiffusionTransformer._forward

文件：`.venv/Lib/site-packages/stable_audio_tools/models/dit.py`

输入：
- `x`: `[B, C, T]`
- `cross_attn_cond`: `[B, S_text, D_text]`（之后会投影到模型内部维度）
- `input_concat_cond`: `[B, C_concat, T_concat]`
- `global_embed`: `[B, D_global]`
- `prepend_cond`: `[B, S_pre, D_pre]`
- `prepend_cond_mask`: `[B, S_pre]`

内部关键变化：

1. `input_concat_cond` 对齐长度后拼接到 `x`
- 若 `T_concat != T`，先插值到 `T`
- `x = cat([x, input_concat_cond], dim=1)`  
  -> 变成 `[B, C + C_concat, T]`

2. `timestep_embed`
- 从 `t:[B]` 生成 `timestep_embed:[B, D]`

3. `preprocess_conv` 后，转为 token 视图
- `x` 仍是 `[B, C_in, T]`
- `rearrange(x, "b c t -> b t c")` -> `[B, T, C_in]`
- 若 `patch_size > 1`，再重排为 `[B, S, C_in*patch_size]`

4. 调用 `self.transformer(...)`
- 输入 token：`[B, S, dim_in]`
- `context`（cross attn）: `[B, S_text, D_context]`
- `prepend_embeds`: `[B, S_pre_eff, D]`

5. 输出恢复回 latent
- `output` 从 `[B, S, dim_out]` 还原为 `[B, C, T]`

### 3.4 ContinuousTransformer.forward

文件：`.venv/Lib/site-packages/stable_audio_tools/models/transformer.py`

输入：
- `x`: `[B, S, dim_in]`
- `prepend_embeds`: `[B, S_pre, D]` 或 `None`
- `global_cond`: `[B, D_global]` 或 `None`
- `context`: `[B, S_ctx, D_ctx]`（通过 `**kwargs` 传入每个 block）

内部：

1. `project_in`
- `[B, S, dim_in] -> [B, S, D]`

2. prepend 拼接（若有）
- `[B, S + S_pre, D]`

3. memory tokens（若 `num_memory_tokens > 0`）
- `[B, S + S_pre + S_mem, D]`

4. 层循环 `for layer in self.layers`
- 每层输入输出形状都保持 `[B, S_total, D]`

5. 去掉 memory tokens，`project_out`
- `[B, S_no_mem, D] -> [B, S_no_mem, dim_out]`

---

## 4. `control_transformer.py` 关键参数形状

文件：`stable_audio_control/models/control_transformer.py`

类：`ControlNetContinuousTransformer`

### 4.1 构造期

- `base_transformer.dim = D`
- `zero_linears[i]`: `Linear(D, D)`（零初始化）
- `control_layers[i]`: 复制 `base.layers[i]`，每层输入输出都是 `[B, S_total, D]`

### 4.2 forward 输入

- `x`: `[B, S, dim_in]`（与 `ContinuousTransformer.forward` 相同）
- `prepend_embeds`: `[B, S_pre, D]` 或 `None`
- `global_cond`: `[B, D_global]` 或 `None`
- `control_input`: `[B, S, dim_in]` 或 `None`
- `control_scale`: 标量 float

### 4.3 forward 形状流

1. 主干与控制分支都先做 `project_in`
- `x_base = project_in(x)` -> `[B, S, D]`
- `ctrl = project_in(control_input)` -> `[B, S, D]`

2. prepend / memory 长度对齐
- 若主干拼了 prepend 或 memory token，`ctrl` 前面补零 token
- 对齐后两者都为 `[B, S_total, D]`

3. 控制分支初始化
- `x_ctrl = x_base + ctrl * control_scale` -> `[B, S_total, D]`

4. 层循环注入（前 N 层）
- `x_ctrl = control_layer_i(x_ctrl)` -> `[B, S_total, D]`
- `x_base = base_layer_i(x_base)` -> `[B, S_total, D]`
- 注入：`x_base = x_base + zero_linear_i(x_ctrl)` -> `[B, S_total, D]`

5. 后续层（>=N）仅主干
- 形状保持 `[B, S_total, D]`

6. 收尾
- 去 memory token -> `[B, S_no_mem, D]`
- `project_out` -> `[B, S_no_mem, dim_out]`

### 4.4 与真实 StableAudio Open 对接时的形状

当它被挂载到 `model.model.model.transformer` 时：
- 来自 DiT 的 token 输入通常是 `[B, T, C_in]`（`patch_size=1` 时 `S=T`）
- 因此 `control_input` 也必须是 `[B, T, C_in]`

常见做法是从 latent 长度 `T` 构造控制输入：
- `control_input = randn(B, T, dim_in)`
- 或从 melody 特征映射/下采样得到同样长度

---

## 5. 常见 shape 错误与排查

1. 把 `control_input` 当成 `[B, C, T]` 传入控制 transformer
- 实际需要 `[B, S, dim_in]`

2. `prepend_embeds` 末维不等于 `D`
- 代码里会直接 `ValueError`

3. `input_concat_cond` 用错维度
- 应是 `[B, C_concat, T]`，不是 `[B, T, C_concat]`

4. `input_concat_cond` 长度和 `x` 不一致
- DiT 会插值对齐到 `T`，但你要确认这是否符合控制语义

5. 看到 `diff norm == 0` 误以为没生效
- 在 zero-init adapter 下这是预期行为

---

## 6. 快速对照（最常用）

- latent 主输入 `x`（DiT）: `[B, C, T]`
- text 条件 `cross_attn_cond`: `[B, S_text, D_text]`
- input-concat 条件: `[B, C_concat, T]`
- prepend 条件: `[B, S_pre, D_pre]`
- transformer token 输入（进入 `ContinuousTransformer`）: `[B, S, dim_in]`
- `control_input`（你的模块）: `[B, S, dim_in]`

