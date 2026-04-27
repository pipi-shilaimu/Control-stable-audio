# stable-audio-tools 训练流程

## 读者对象
要启动训练、断点续训或微调模型的开发者。

## 本文覆盖范围
从 `train.py` 入口解释训练生命周期，以及 autoencoder/diffusion/lm 三类训练包装器的职责分工。

## 3分钟速读版
- `train.py` 做四件事：读配置、建数据、建模型/包装器、交给 `pl.Trainer.fit`。
- checkpoint 默认是“训练包装器”格式，推理前通常需要先 `unwrap_model.py` 解包。
- diffusion 训练里最关键的开关是 `diffusion_objective`、`timestep_sampler`、`pre_encoded`。
- demo 回调能最早暴露“模型在学什么”，建议优先盯住 demo 音频与频谱。

## 1. 训练主流程（`train.py`）

训练主流程在 [`train.py`](../../train.py) 的 `main()`：

1. 解析命令行参数并设置随机种子。  
2. 读取 `model_config` 与 `dataset_config`。  
3. 调用 [`create_dataloader_from_config`](../../stable_audio_tools/data/dataset.py) 创建训练集（和可选验证集）。  
4. 调用 [`create_model_from_config`](../../stable_audio_tools/models/factory.py) 构建模型。  
5. 调用 [`create_training_wrapper_from_config`](../../stable_audio_tools/training/factory.py) 构建训练包装器。  
6. 调用 [`create_demo_callback_from_config`](../../stable_audio_tools/training/factory.py) 绑定 demo 回调。  
7. 初始化 `pl.Trainer`，执行 `trainer.fit(...)`。

训练时你看到的 checkpoint 默认是“包装器 checkpoint”（包含优化器状态、EMA 等训练态信息）。

## 2. 训练包装器的职责边界

### 2.1 `AutoencoderTrainingWrapper`
位置：[`stable_audio_tools/training/autoencoders.py`](../../stable_audio_tools/training/autoencoders.py)

关键职责：
- 负责编码器/解码器前向与重建损失计算。
- 可选判别器训练（手动优化，`automatic_optimization = False`）。
- 维护可选 EMA 模型。
- 支持 warmup、蒸馏（teacher model）、latent mask 等策略。

实现锚点：
- `configure_optimizers`
- `training_step`
- `validation_step`
- `export_model`

### 2.2 `Diffusion*TrainingWrapper`
位置：[`stable_audio_tools/training/diffusion.py`](../../stable_audio_tools/training/diffusion.py)

主要类：
- `DiffusionUncondTrainingWrapper`
- `DiffusionCondTrainingWrapper`
- `DiffusionAutoencoderTrainingWrapper`

关键职责：
- 对输入（音频或 latent）加噪，构造 denoise 目标。
- 根据 `diffusion_objective` 选择目标定义（如 `v`、`rectified_flow`、`rf_denoiser`）。
- 条件扩散里调用 conditioner 获取条件张量并交给 diffusion wrapper。
- 支持 `timestep_sampler`、inpainting、padding mask、EMA。

### 2.3 `AudioLanguageModelTrainingWrapper`
位置：[`stable_audio_tools/training/lm.py`](../../stable_audio_tools/training/lm.py)

关键职责：
- 将音频（或预编码输入）转换为离散 codebook token。
- 调用 `compute_logits` 并按 codebook 计算交叉熵。
- 记录 perplexity 与分 codebook 指标。
- 管理可选 EMA。

## 3. Demo 回调如何工作

`training/factory.py` 会按 `model_type` 分配 demo callback。  
这些 callback 通常在 `on_train_batch_end` 按 `demo_every` 触发：

- 生成若干音频样本
- 保存到本地临时文件
- 通过 logger 记录音频和频谱图

常见类：
- `AutoencoderDemoCallback`
- `DiffusionUncondDemoCallback`
- `DiffusionCondDemoCallback`
- `DiffusionCondInpaintDemoCallback`
- `AudioLanguageModelDemoCallback`

## 4. 常见训练命令模板

### 4.1 从零训练
```bash
python3 ./train.py \
  --dataset-config /path/to/dataset_config.json \
  --model-config /path/to/model_config.json \
  --name my_train_run
```

### 4.2 从包装器 checkpoint 续训
```bash
python3 ./train.py \
  --dataset-config /path/to/dataset_config.json \
  --model-config /path/to/model_config.json \
  --ckpt-path /path/to/wrapped.ckpt \
  --name my_resume_run
```

### 4.3 用解包模型初始化新训练（常见微调方式）
```bash
python3 ./train.py \
  --dataset-config /path/to/dataset_config.json \
  --model-config /path/to/model_config.json \
  --pretrained-ckpt-path /path/to/unwrapped.ckpt \
  --name my_finetune_run
```

### 4.4 解包 checkpoint
```bash
python3 ./unwrap_model.py \
  --model-config /path/to/model_config.json \
  --ckpt-path /path/to/wrapped.ckpt \
  --name exported_model
```

## 5. `pre_encoded` 训练的关键对齐点

当你使用预编码数据集时，需要同时满足：

1. 数据集配置是 `dataset_type = "pre_encoded"`。  
2. 模型训练配置里 `training.pre_encoded = true`。  
3. pretransform 的下采样和 latent 长度与 `sample_size`、裁剪策略一致。  

如果这三者不一致，最常见的是长度、mask、scale 不对齐导致训练异常。

## 6. 训练前快速核对清单

1. `model_type` 是否与期望 wrapper 匹配。  
2. `training.demo` 是否包含该 wrapper 需要的字段（例如条件扩散常见 `num_demos`、`demo_cfg_scales`）。  
3. 如果使用 `--pretransform-ckpt-path`，checkpoint 是否与模型 pretransform 结构匹配。  
4. 多卡时策略、精度和 batch 是否与硬件匹配。  

## 7. 相关文档
- 架构关系： [架构总览](./architecture-overview.md)
- 训练指标解读： [训练日志字段字典](./training-log-metrics.md)
- Diffusion 内核细节： [Diffusion 深潜](./diffusion-deep-dive.md)
- 推理与采样： [推理与 UI](./inference-and-ui.md)
- 典型报错： [排障手册](./troubleshooting.md)
- 英文字段说明： [Diffusion](../diffusion.md), [Autoencoders](../autoencoders.md), [Datasets](../datasets.md)
