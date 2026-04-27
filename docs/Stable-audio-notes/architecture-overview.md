# stable-audio-tools 架构总览

## 读者对象
希望快速建立仓库心智模型的开发者。

## 本文覆盖范围
聚焦代码结构、调度关系和数据流转。  
不展开每个配置字段的完整字典（详见英文文档）。

## 3分钟速读版
- 入口只有四个：`train.py`、`run_gradio.py`、`pre_encode.py`、`unwrap_model.py`。
- 所有模型创建都经过 `models/factory.py`，所有训练包装器都经过 `training/factory.py`。
- 数据进入模型前统一走 `data/dataset.py`，条件控制统一走 `models/conditioners.py`。
- 推理主链路是 `inference/generation.py` + `inference/sampling.py`。

## 1. 一句话架构图

```text
CLI 脚本
  ├─ train.py / run_gradio.py / pre_encode.py / unwrap_model.py
  ↓
配置文件 (model_config + dataset_config)
  ↓
工厂分发
  ├─ stable_audio_tools/models/factory.py
  └─ stable_audio_tools/training/factory.py
  ↓
核心模型与训练包装器
  ├─ models/*.py
  └─ training/*.py
  ↓
数据与条件控制
  ├─ data/dataset.py
  └─ models/conditioners.py
  ↓
采样与推理
  ├─ inference/generation.py
  └─ inference/sampling.py
```

## 2. 四个入口脚本分别做什么

| 入口脚本 | 关键动作 | 代码锚点 |
| --- | --- | --- |
| [`train.py`](../../train.py) | 读取配置，构建 dataloader、model、training wrapper、demo callback，最后 `Trainer.fit` | `main()` |
| [`run_gradio.py`](../../run_gradio.py) | 构建并启动 Gradio UI，支持 HF 预训练或本地 ckpt | `main(args)` |
| [`pre_encode.py`](../../pre_encode.py) | 用自编码器预编码数据集，输出 `.npy + .json` 元数据 | `PreEncodedLatentsInferenceWrapper`、`main(args)` |
| [`unwrap_model.py`](../../unwrap_model.py) | 将训练包装器 checkpoint 导出为可直接推理/迁移的“解包模型” | 脚本主逻辑 + `export_model()` |

## 3. 模型构建分发链路

顶层入口是 [`create_model_from_config`](../../stable_audio_tools/models/factory.py)。

`model_type` 到构造函数的映射：

| model_type | 构造函数 |
| --- | --- |
| `autoencoder` | `create_autoencoder_from_config` |
| `diffusion_uncond` | `create_diffusion_uncond_from_config` |
| `diffusion_cond` / `diffusion_cond_inpaint` | `create_diffusion_cond_from_config` |
| `diffusion_autoencoder` | `create_diffAE_from_config` |
| `lm` | `create_audio_lm_from_config` |

关键点：
- 预变换通过 [`create_pretransform_from_config`](../../stable_audio_tools/models/factory.py) 构建。
- 条件控制通过 [`create_multi_conditioner_from_conditioning_config`](../../stable_audio_tools/models/conditioners.py) 构建。
- 条件扩散模型在 [`ConditionedDiffusionModelWrapper`](../../stable_audio_tools/models/diffusion.py) 内把业务条件映射成模型输入张量。

## 4. 训练包装器分发链路

入口是 [`create_training_wrapper_from_config`](../../stable_audio_tools/training/factory.py)。

对应关系：

| model_type | 训练包装器 |
| --- | --- |
| `autoencoder` | `AutoencoderTrainingWrapper` |
| `diffusion_uncond` | `DiffusionUncondTrainingWrapper` |
| `diffusion_cond` / `diffusion_cond_inpaint` | `DiffusionCondTrainingWrapper`（或 ARC 分支） |
| `diffusion_autoencoder` | `DiffusionAutoencoderTrainingWrapper` |
| `lm` | `AudioLanguageModelTrainingWrapper` |

Demo 回调由 [`create_demo_callback_from_config`](../../stable_audio_tools/training/factory.py) 分发，训练时按 `demo_every` 产出音频和频谱图。

## 5. 数据与元数据如何进入模型

统一入口是 [`create_dataloader_from_config`](../../stable_audio_tools/data/dataset.py)。

支持三种主模式：
- `audio_dir`：本地音频目录，按采样率重采样并 pad/crop。
- `pre_encoded`：读取预编码 latent（`.npy`）和同名元数据（`.json`）。
- `wds`/`s3`：WebDataset tar 流。

元数据流转重点：
1. `dataset.py` 产出 `(audio_or_latent, metadata)`。
2. `metadata` 进入 conditioner，形成各类条件张量。
3. 对于条件扩散，`ConditionedDiffusionModelWrapper.get_conditioning_inputs()` 负责拼接：
   - `cross_attn_cond`
   - `global_cond`
   - `input_concat_cond`
   - `prepend_cond`

## 6. 推理路径核心关系

条件与无条件扩散的推理入口在 [`inference/generation.py`](../../stable_audio_tools/inference/generation.py)：
- `generate_diffusion_uncond`
- `generate_diffusion_cond`
- `generate_diffusion_cond_inpaint`

这些入口最终调用 [`inference/sampling.py`](../../stable_audio_tools/inference/sampling.py) 的采样器（如 `sample_k`、`sample_rf`），并在需要时通过 pretransform 做 latent/audio 映射。

## 7. 三个稳定的架构约束

1. `model_config` 决定模型结构和训练包装器类型；`dataset_config` 决定样本来源与元数据策略。  
2. 训练时可以使用 wrapper checkpoint；推理通常使用“解包后”checkpoint。  
3. 若启用 pretransform，样本长度与最小输入长度需要满足下采样/patch 对齐约束。

## 8. 继续阅读
- Diffusion 详细调用链： [Diffusion 深潜](./diffusion-deep-dive.md)
- 训练执行细节： [训练流程](./training-pipeline.md)
- 推理与 UI： [推理与 UI](./inference-and-ui.md)
- 扩展改造： [扩展手册](./extension-playbook.md)
- 报错排查： [排障手册](./troubleshooting.md)
