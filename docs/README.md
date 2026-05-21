# StableAudio ControlNet 文档中心

## 项目概览

本仓库在 StableAudio Open（DiT backbone）上接入 ControlNet-DiT 风格的旋律控制分支。两条 melody control 特征线：

- **CQT**（默认）：基于 CQT top-k 离散控制特征
- **Chromagram**（新增）：基于 12 维 pitch-class chromagram dense 控制特征

完整实验链路：数据导出 → 模型包装 → 训练 → 生成对比 → 旋律相似度评分。

## 推荐阅读路径

### 路径 A：第一次理解仓库

1. [稳定音频工具架构总览](Stable-audio-notes/architecture-overview.md)
2. [概念—代码对照表](controlnet/concept_code_map.md) — StableAudio 核心概念与源码位置
3. [张量形状说明](controlnet/tensor_shapes.md) — 调用链中关键张量的形状约束
4. [调用链 + ControlNet 位置 + 形状图](controlnet/callchain_shape_map.md)

### 路径 B：理解 ControlNet-DiT 改造

1. [注意事项清单](controlnet/注意事项清单.md) — 替换 ContinuousTransformer 的风险、策略和验收清单
2. [ControlNet 关键变量（精简版）](controlnet/tensor_shapes_controlnet_core.md)
3. 代码入口：[control_transformer.py](../stable_audio_control/models/control_transformer.py)
4. 代码入口：[control_dit.py](../stable_audio_control/models/control_dit.py)
5. 使用文档：[control_transformer 使用说明](controlnet/code-docs/control_transformer使用.md)
6. 使用文档：[control_dit 说明](controlnet/code-docs/control_dit.md)

### 路径 C：准备数据与训练

1. [脚本命令自查表](controlnet/脚本命令自查表.md)
2. 数据导出：[dataset_load.py](../stable_audio_control/data/dataset_load.py)
3. Metadata：[song_describer_metadata.py](../stable_audio_control/data/song_describer_metadata.py)
4. 训练入口：[train_controlnet_dit.py](../stable_audio_control/scripts/train_controlnet_dit.py)
5. 训练文档：[train_controlnet_dit 使用说明](controlnet/code-docs/train_controlnet_dit.md)

### 路径 D：验证、smoke 与生成对比

1. [开训前最小验证报告](controlnet/smoke_report_2026-04-21.md)
2. 训练 smoke 文档：[train_control_smoke 使用说明](controlnet/code-docs/train_control_smoke.md)
3. 生成对比：[compare_controlnet_generation.py](../stable_audio_control/scripts/compare_controlnet_generation.py)
4. 旋律相似度：[compare_melody_similarity.py](../stable_audio_control/scripts/compare_melody_similarity.py)
5. 随机批量：[compare_controlnet_generation_random.py](../stable_audio_control/scripts/compare_controlnet_generation_random.py)

### 路径 E：理解旋律特征与可视化

1. [cqt_topk 模块](controlnet/code-docs/cqt_topk.md)
2. FFT/CQT 可视化：[fft_cqt_visual_compare.py](../scripts/fft_cqt_visual_compare.py)
3. CQT 提取器：[cqt_topk.py](../stable_audio_control/melody/cqt_topk.py)
4. Chromagram 提取器：[chromagram.py](../stable_audio_control/melody/chromagram.py)
5. 特征工厂：[extractors.py](../stable_audio_control/melody/extractors.py)

### 路径 F：只想跑推理或 UI

1. [推理与 UI](Stable-audio-notes/inference-and-ui.md)
2. [run_gradio.py](../run_gradio.py)
3. [demo.py](../demo.py)

### 路径 G：训练排障

1. [训练流程](Stable-audio-notes/training-pipeline.md)
2. [训练日志字段字典](Stable-audio-notes/training-log-metrics.md)
3. [排障手册](Stable-audio-notes/troubleshooting.md)
4. [Diffusion 深潜](Stable-audio-notes/diffusion-deep-dive.md)

### 路径 H：功能扩展

1. [扩展手册](Stable-audio-notes/extension-playbook.md)
2. [Diffusion 深潜](Stable-audio-notes/diffusion-deep-dive.md)

## 文档地图

### 上游参考（stable-audio-tools 原始英文文档）

| 文件 | 主题 |
|------|------|
| [upstream/autoencoders.md](upstream/autoencoders.md) | Autoencoder 模型、训练、loss、bottleneck |
| [upstream/conditioning.md](upstream/conditioning.md) | Conditioning 类型、conditioner 配置 |
| [upstream/datasets.md](upstream/datasets.md) | 数据集配置（本地、S3、pre-encoded） |
| [upstream/diffusion.md](upstream/diffusion.md) | Diffusion 模型与训练配置 |
| [upstream/pre_encoding.md](upstream/pre_encoding.md) | 预编码流程 |
| [upstream/pretransforms.md](upstream/pretransforms.md) | Pretransform 配置 |

### Stable Audio 中文笔记

| 文件 | 作用 |
|------|------|
| [architecture-overview.md](Stable-audio-notes/architecture-overview.md) | 架构总览 |
| [training-pipeline.md](Stable-audio-notes/training-pipeline.md) | 训练流程 |
| [training-log-metrics.md](Stable-audio-notes/training-log-metrics.md) | 训练日志字段字典 |
| [diffusion-deep-dive.md](Stable-audio-notes/diffusion-deep-dive.md) | Diffusion 深潜 |
| [inference-and-ui.md](Stable-audio-notes/inference-and-ui.md) | 推理与 UI |
| [extension-playbook.md](Stable-audio-notes/extension-playbook.md) | 扩展手册 |
| [troubleshooting.md](Stable-audio-notes/troubleshooting.md) | 排障手册 |

### ControlNet 改造笔记

| 文件 | 作用 |
|------|------|
| [脚本命令自查表.md](controlnet/脚本命令自查表.md) | 常用脚本命令、自查点和使用示例 |
| [concept_code_map.md](controlnet/concept_code_map.md) | 概念和代码位置对照表 |
| [callchain_shape_map.md](controlnet/callchain_shape_map.md) | 调用链、位置和关键形状 |
| [tensor_shapes.md](controlnet/tensor_shapes.md) | 张量形状详细说明 |
| [tensor_shapes_controlnet_core.md](controlnet/tensor_shapes_controlnet_core.md) | 关键变量精简版 |
| [注意事项清单.md](controlnet/注意事项清单.md) | 替换注意事项 |
| [smoke_report_2026-04-21.md](controlnet/smoke_report_2026-04-21.md) | 最小验证报告 |
| [callchain_flowchart.svg](controlnet/callchain_flowchart.svg) | 调用链流程图 |

### 代码模块文档

| 文件 | 对应源码 |
|------|----------|
| [code-docs/control_transformer使用.md](controlnet/code-docs/control_transformer使用.md) | ControlNetContinuousTransformer |
| [code-docs/control_dit.md](controlnet/code-docs/control_dit.md) | ControlConditionedDiffusionWrapper |
| [code-docs/cqt_topk.md](controlnet/code-docs/cqt_topk.md) | CQT top-k 旋律控制特征提取 |
| [code-docs/train_control_smoke.md](controlnet/code-docs/train_control_smoke.md) | 训练烟测脚本 |
| [code-docs/train_controlnet_dit.md](controlnet/code-docs/train_controlnet_dit.md) | ControlNet-DiT 正式训练 |

### 补充笔记

| 文件 | 内容 |
|------|------|
| [notes/论文提炼_ControlNet-DiT音乐编辑.md](notes/论文提炼_ControlNet-DiT音乐编辑.md) | 核心论文提炼 |
| [notes/模型结构_易读.md](notes/模型结构_易读.md) | 模型结构易读版 |
| [notes/模型结构.md](notes/模型结构.md) | 模型结构原始整理 |
| [notes/常用命令.md](notes/常用命令.md) | 常用命令集合 |

## 按任务找文件

| 任务 | 优先入口 |
|------|----------|
| 理解整体架构 | [architecture-overview.md](Stable-audio-notes/architecture-overview.md) |
| 理解训练流程 | [training-pipeline.md](Stable-audio-notes/training-pipeline.md) |
| 排查训练/推理错误 | [troubleshooting.md](Stable-audio-notes/troubleshooting.md) |
| 查看脚本命令和自查点 | [脚本命令自查表.md](controlnet/脚本命令自查表.md) |
| 理解 ControlNet 注入点 | [callchain_shape_map.md](controlnet/callchain_shape_map.md) |
| 查 shape 约束 | [tensor_shapes.md](controlnet/tensor_shapes.md) |
| 看 ControlNet transformer 代码 | [control_transformer.py](../stable_audio_control/models/control_transformer.py) |
| 看 diffusion wrapper 接入 | [control_dit.py](../stable_audio_control/models/control_dit.py) |
| 看 CQT 旋律特征提取 | [cqt_topk.py](../stable_audio_control/melody/cqt_topk.py) |
| 看 chromagram 旋律特征提取 | [chromagram.py](../stable_audio_control/melody/chromagram.py) |
| 跑最小训练 smoke | [train_control_smoke.py](../stable_audio_control/scripts/train_control_smoke.py) |
| 跑 ControlNet-DiT 训练 | [train_controlnet_dit.py](../stable_audio_control/scripts/train_controlnet_dit.py) |
| 跑 base/control 生成对比 | [compare_controlnet_generation.py](../stable_audio_control/scripts/compare_controlnet_generation.py) |
| 做旋律相似度比较 | [compare_melody_similarity.py](../stable_audio_control/scripts/compare_melody_similarity.py) |
| 跑 Gradio UI | [run_gradio.py](../run_gradio.py) |

## 论文资料

| 文件 | 用途 |
|------|------|
| [Editing music with melody and text...](../Editing%20music%20with%20melody%20and%20text_using%20controlnet%20for%20diffusion%20transformer.pdf) | 核心参考论文 |
| [ControlNet 原始论文](../Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.pdf) | ControlNet 原始思想 |
| [论文提炼](notes/论文提炼_ControlNet-DiT音乐编辑.md) | 核心论文的中文提炼 |
