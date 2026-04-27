# ControlNetContinuousTransformer 使用文档（StableAudio Open）

本文档针对 `stable_audio_control/models/control_transformer.py` 中的 `ControlNetContinuousTransformer`，并以 `stable_audio_control/scripts/smoke_control_injection_stableaudio_open-1.py` 为使用参考，补充其在 `stable_audio_tools` 调用链中的实际接入方式。

## 1. 这个类解决什么问题

`ControlNetContinuousTransformer` 是对 `stable_audio_tools.models.transformer.ContinuousTransformer` 的轻量封装，目标是：

- 不改动上游 `stable_audio_tools` 源码（无需修改 `.venv/site-packages`）。
- 保留预训练主干（`base_transformer`）的结构和行为。
- 通过“复制前 `N` 层 + 零初始化线性注入（`zero_linears`）”实现 ControlNet 风格控制分支。
- 在训练早期尽量不扰动原模型（因为 `zero_linears` 全零初始化）。

## 2. 类接口说明

文件位置：`stable_audio_control/models/control_transformer.py`

### 2.1 构造函数

```python
ControlNetContinuousTransformer(
    base_transformer: nn.Module,
    num_control_layers: int = 1,
    default_control_scale: float = 1.0,
)
```

- `base_transformer`：
  - 期望是 `ContinuousTransformer` 实例。
  - 必须具备 `.layers`、`.dim`、`.project_in`、`.project_out` 等属性。
- `num_control_layers`：
  - 控制分支覆盖前多少层 transformer block。
  - 取值范围必须在 `[0, len(base_transformer.layers)]`，否则抛 `ValueError`。
- `default_control_scale`：
  - 当 `forward(..., control_scale=None)` 时使用的默认控制强度。

初始化时会：

- 深拷贝 `base.layers[:num_control_layers]` 到 `self.control_layers`。
- 创建 `self.zero_linears`（每层 `Linear(D, D)`）并做零初始化。

### 2.2 `freeze_base()`

```python
freeze_base() -> None
```

- 冻结 `base_transformer` 全部参数（`requires_grad=False`）。
- 常见训练策略是：仅训练 `control_layers` 和 `zero_linears`。

### 2.3 `forward(...)`

```python
forward(
    x,
    prepend_embeds=None,
    global_cond=None,
    return_info=False,
    use_checkpointing=True,
    exit_layer_ix=None,
    *,
    control_input=None,
    control_scale=None,
    **kwargs,
)
```

核心参数：

- `x`: `Tensor[B, S, dim_in]`
  - 与原 `ContinuousTransformer` 输入一致。
- `control_input`: `Tensor[B, S, dim_in]` 或 `None`
  - 必须与 `x` 的 shape 对齐（同 batch、同 token 数、同 `dim_in`）。
  - 若为 `None`，行为退化为原始 `base_transformer`。
- `control_scale`: `float` 或 `None`
  - 控制分支强度；为 `None` 时用 `default_control_scale`。
- `prepend_embeds`: `Tensor[B, P, D]` 或 `None`
- `global_cond`: `Tensor[B, G]` 或 `None`

返回值：

- 默认返回 transformer 输出张量。
- `return_info=True` 时返回 `(output, info)`，其中 `info["hidden_states"]` 与 base 语义对齐。

实现细节（理解行为的关键）：

1. `x_base = base.project_in(x)`，`ctrl = base.project_in(control_input)`。
2. 若存在 prepend/memory token，会给 `ctrl` 前置补零，保证与 `x_base` 序列长度一致。
3. 前 `N` 层执行：
   - `x_ctrl = control_layer_i(x_ctrl)`
   - `x_base = base_layer_i(x_base)`
   - `x_base = x_base + zero_linear_i(x_ctrl)`（注入点）
4. 后续层只走主干 `base_layer`。
5. 末尾对齐 base：去掉 memory token，再 `project_out`。

## 3. 在 stable_audio_tools 中是如何透传到这个类的

以 `stable_audio_tools` 的 DiT 路径为例（StableAudio Open 默认常用）：

1. `ConditionedDiffusionModelWrapper.forward(...)`
   - 调用 `self.model(x, t, **self.get_conditioning_inputs(cond), **kwargs)`。
   - 这里的 `**kwargs` 会继续下传。
2. `DiTWrapper.forward(...)`
   - 调用 `self.model(..., **kwargs)`，继续下传。
3. `DiffusionTransformer.forward(...)`
   - 调用 `_forward(..., **kwargs)`。
4. `DiffusionTransformer._forward(...)`
   - 调用 `self.transformer(..., **extra_args, **kwargs)`。
5. 当你已替换 `self.transformer` 为 `ControlNetContinuousTransformer` 时，
   - `control_input`、`control_scale` 这两个自定义参数就会进入 `ControlNetContinuousTransformer.forward(...)`。

这也是 `stable_audio_control/scripts/smoke_control_injection_stableaudio_open-1.py` 能直接在 `model(...)` 里传 `control_input/control_scale` 的原因。

## 4. 参考脚本接入步骤（最小可运行）

参考：`stable_audio_control/scripts/smoke_control_injection_stableaudio_open-1.py`

### 步骤 1：加载预训练 StableAudio Open

```python
from stable_audio_tools import get_pretrained_model
model, _ = get_pretrained_model("stabilityai/stable-audio-open-1.0")
```

### 步骤 2：替换内部 transformer

```python
base_transformer = model.model.model.transformer
control_transformer = ControlNetContinuousTransformer(
    base_transformer=base_transformer,
    num_control_layers=2,
    default_control_scale=0.5,
)
control_transformer.freeze_base()
model.model.model.transformer = control_transformer
```

### 步骤 3：准备条件与输入

```python
cond = model.conditioner(batch_metadata, device)

bsz = 1
latent_len = 128
io_channels = model.io_channels
dtype = next(model.parameters()).dtype

x = torch.randn((bsz, io_channels, latent_len), device=device, dtype=dtype)
t = torch.full((bsz,), 0.5, device=device, dtype=dtype)

dim_in = base_transformer.project_in.in_features
control_input = torch.randn((bsz, latent_len, dim_in), device=device, dtype=dtype)
```

说明：上面写法与 smoke 脚本一致，默认假设 `patch_size=1`（即 `S=T`）。
若你的模型配置是 `patch_size>1`，则应把 `control_input` 的第二维改为 token 长度 `S=T/patch_size`。

### 步骤 4：前向调用（透传控制参数）

```python
out = model(
    x,
    t,
    cond=cond,
    cfg_scale=1.0,
    cfg_dropout_prob=0.0,
    control_input=control_input,
    control_scale=0.5,
)
```

## 5. Shape 约束（最容易踩坑）

通过 DiT 路径时，约束如下：

- 扩散模型主输入：`x` 是 `[B, C, T]`。
- 进入 transformer 前会 token 化为 `[B, S, dim_in]`：
  - `patch_size=1` 时通常 `S=T`
  - `patch_size=p>1` 时通常 `S=T/p`
- `control_input` 必须与该 token 形状一致，即 `[B, S, dim_in]`。

实操建议：

- 先从 `base_transformer.project_in.in_features` 读取 `dim_in`。
- 若你自己构造控制特征（melody/f0/节奏），最终一定要对齐到 `S` 和 `dim_in` 两个维度。

## 6. 为什么 smoke 里可能看到 `diff norm == 0`

脚本里有如下注释结论：即使 `control_scale > 0`，初始阶段差值也可能是 `0`。

原因：

- `zero_linears` 全零初始化，意味着注入残差初始近似为 0。
- 这不是“接线失败”，而是 ControlNet 风格稳定初始化的预期。

如果要快速验证注入通路是否打通，可临时微扰某个 `zero_linear` 权重（脚本里有示例注释）。

## 7. 常见错误与排查

1. `num_control_layers` 越界
- 现象：构造时报 `ValueError`。
- 处理：确保 `0 <= num_control_layers <= len(base_transformer.layers)`。

2. `control_input` 维度错误（把 `[B, C, T]` 直接传进来）
- 现象：`project_in` 维度不匹配。
- 处理：按 token 视角构造 `[B, S, dim_in]`。

3. `prepend_embeds` 末维不等于模型 `D`
- 现象：报错 `prepend_embeds last dim must be ...`。
- 处理：投影到 transformer 隐层维度后再传入。

4. 误判“控制分支没生效”
- 现象：`control_scale` 改了，输出几乎不变。
- 处理：先确认是否仍是 zero-init 阶段；可按脚本建议临时扰动权重做链路验证。

## 8. 建议的训练参数分组

如果你只想训练控制分支：

- 冻结：`base_transformer`（调用 `freeze_base()`）。
- 训练：`control_layers` + `zero_linears`。

可用如下逻辑检查参数是否符合预期：

```python
for n, p in model.named_parameters():
    if "transformer.base" in n:
        assert not p.requires_grad
```

---

补充说明：本仓库已有更细的调用链与形状推导文档，可结合阅读：

- `docs/notes/callchain_controlnet_shape_map.zh-CN.md`
- `docs/notes/tensor_shapes_stableaudio_and_control_transformer.zh-CN.md`
