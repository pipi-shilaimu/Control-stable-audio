# stable-audio-tools 中文文档索引

## 读者对象
面向希望基于 `stable-audio-tools` 做训练、微调、推理或二次开发的工程师。

## 本文覆盖范围
本套文档基于当前仓库源码组织方式，重点解释“代码如何流动”，而不是重复配置项逐字段说明。  
配置字段的细节仍建议配合英文原始文档阅读：
- [Diffusion](../diffusion.md)
- [Autoencoders](../autoencoders.md)
- [Datasets](../datasets.md)
- [Conditioning](../conditioning.md)

## 3分钟速读版
- 先看“文档地图”，确定你当前是训练、推理、扩展还是排障场景。
- 首次接触仓库，建议顺序是：架构总览 -> 训练流程 -> Diffusion 深潜。
- 只想尽快出声音频：优先看推理与 UI，再看排障手册。
- 要改代码而不是只用代码：优先看扩展手册和 Diffusion 深潜。

## 文档地图

| 文档 | 作用 | 适合什么时候看 |
| --- | --- | --- |
| [架构总览](./architecture-overview.md) | 看清入口脚本、工厂分发、训练包装器、数据与条件控制的全局关系 | 第一次接触仓库 |
| [训练流程](./training-pipeline.md) | 从 `train.py` 出发理解训练生命周期、各类 wrapper 和 demo 回调 | 要开始训练/微调 |
| [训练日志字段字典](./training-log-metrics.md) | 解读 `train/*` 与 `val/*` 指标、识别异常信号 | 要判断训练是否健康 |
| [Diffusion 深潜](./diffusion-deep-dive.md) | 深入 `models/diffusion.py` 与 `training/diffusion.py`，包含调用时序图 | 要改 diffusion 内核或排查复杂问题 |
| [推理与 UI](./inference-and-ui.md) | 理解生成 API、采样器参数、Gradio 路由和模型加载路径 | 要跑本地推理或改 UI |
| [扩展手册](./extension-playbook.md) | 新增 `model_type`、条件控制、数据源模式时该改哪些文件 | 要做功能扩展 |
| [排障手册](./troubleshooting.md) | 快速定位配置断言、shape 对齐、预编码、dtype/device 常见问题 | 训练或推理报错时 |

## 推荐阅读路径

### 路径 A：先跑通训练再理解细节
1. [训练流程](./training-pipeline.md)
2. [训练日志字段字典](./training-log-metrics.md)
3. [排障手册](./troubleshooting.md)
4. [Diffusion 深潜](./diffusion-deep-dive.md)
5. [架构总览](./architecture-overview.md)

### 路径 B：先读架构再做扩展
1. [架构总览](./architecture-overview.md)
2. [Diffusion 深潜](./diffusion-deep-dive.md)
3. [扩展手册](./extension-playbook.md)
4. [推理与 UI](./inference-and-ui.md)

### 路径 C：只关心推理与生成
1. [推理与 UI](./inference-and-ui.md)
2. [排障手册](./troubleshooting.md)

## 关键入口清单
- 训练入口：[`train.py`](../../train.py)
- Gradio 入口：[`run_gradio.py`](../../run_gradio.py)
- 预编码入口：[`pre_encode.py`](../../pre_encode.py)
- 模型解包入口：[`unwrap_model.py`](../../unwrap_model.py)

## 术语约定
- 条件控制：conditioning
- 预变换：pretransform
- 训练包装器：training wrapper（PyTorch Lightning `LightningModule`）
- 预编码数据集：pre-encoded dataset
