# StableAudio ControlNet 替换 `ContinuousTransformer` 注意事项清单

> 记录日期：2026-04-14  
> 适用范围：当前仓库 `stable_audio_control/models/control_transformer.py` 与 `stable_audio_control/models/control_dit.py` 的接入方案（就地替换 transformer + 旁路 control 输入）

## 1) 当前方案是否稳当（结论）

- 当前替换路径在工程上是可行的，核心调用链已打通：`ConditionedDiffusionModelWrapper -> DiTWrapper -> DiffusionTransformer -> transformer`。
- `control_input=None` 时会退回原始主干路径，能保证不传控制条件时行为一致。
- `zero_linear` 全零初始化策略合理，符合 ControlNet 训练初期“尽量不扰动基模”的原则。

## 2) 最容易踩坑的技术点

- **输入形状约束**
  - `control_input` 目标形状是 `[B, S, dim_in]`（token 视角），不是 `[B, C, T]`。
  - 若 `patch_size > 1`，`S` 不是 `T`，需要显式对齐 token 长度。
- **CFG 批次扩展**
  - DiT 在 CFG 路径会把 batch 变成 `2B`；`control_input` 必须同步扩展，否则 batch mismatch。
- **长度对齐语义**
  - 当前采用 `nearest` 插值对齐序列长度，训练前需确认这是否符合旋律控制语义（避免节拍/音高信息被粗糙拉伸）。
- **dtype/device 对齐**
  - 需与模型参数 dtype 一致；半精度下尤其要避免控制分支张量残留在 fp32 或 CPU。

## 3) 训练前必须确认的策略点

- **冻结范围**
  - `freeze_base()` 仅冻结 transformer 主干；默认训练 wrapper 会优化 `self.diffusion.parameters()` 全量参数。
  - 必须明确是否还要冻结 pretransform、原生 conditioner、DiT 其他模块，否则训练成本和不稳定风险会升高。
- **可训练参数集合**
  - 建议只训练：`control_layers`、`zero_linears`、`control_projector`（和后续 melody conditioner）。
  - 建议显式打印 trainable 参数清单做一次核对。
- **conditioning 路由**
  - 当前是旁路 `melody_control -> control_input`，不走 `*_cond_ids` 主链；这能工作，但要自行维护与训练/推理脚本的一致性。

## 4) 与训练生态集成的风险点

- **Demo 回调可能“看不到”控制效果**
  - 默认 demo 多走 `get_conditioning_inputs` 路径；若未把 `melody_control` 在 demo 流程中注入 forward，demo 结果可能失真。
- **配置一致性**
  - `sample_size`、`min_input_length`、pretransform 下采样比、patch size 必须一致，否则高概率 shape 错误。
- **pre-encoded 模式**
  - 若使用预编码数据，注意 pretransform scale 与 latent 长度/掩码同步。

## 5) 开训前最小验收清单（建议）

- Smoke-1：`control_scale=0` 与 `control_scale>0` 在 zero-init 下输出近似一致（差值接近 0）。
- Smoke-2：微扰任一 `zero_linear` 后输出差值应变为非零（证明控制路径有效）。
- Smoke-3：走一次 `DiffusionCondTrainingWrapper.training_step`，确认无 batch/shape/dtype 错误。
- Smoke-4：确认 optimizer 实际只覆盖预期参数集合（非目标模块无梯度或未被优化）。

## 6) 建议的推进顺序（仅记录，不执行）

1. 先固化冻结策略与参数分组。
2. 再对齐训练与 demo 的 control 注入路径。
3. 最后做小数据集短步数训练 smoke（先看 loss 曲线与 demo 可控性，再长训）。
