# `stable_audio_control/scripts/train_controlnet_dit.py` 使用与实现说明

## 1. 文档目的

本文档说明 `stable_audio_control/scripts/train_controlnet_dit.py` 的定位、运行方式、关键参数和内部流程，帮助你从“单步 smoke”过渡到“可持续正式训练”。

适用读者：

- 已经跑通过 `train_control_smoke.py`；
- 希望开始多步训练（`Trainer.fit`）；
- 正在做 StableAudio Open + ControlNet + melody control 的实验迭代。

---

## 2. 脚本定位

`train_controlnet_dit.py` 是本仓库当前的正式训练入口（ControlNet 方向）。

与 `stable_audio_control/scripts/train_control_smoke.py` 的关系：

- `train_control_smoke.py`：最小闭环验证（单步训练，主要用于 Go/No-Go）。
- `train_controlnet_dit.py`：配置驱动的多步训练（使用 Lightning `Trainer.fit`）。

核心能力：

1. 加载 `stabilityai/stable-audio-open-1.0` 预训练模型。
2. 挂接 `ControlNetContinuousTransformer`（通过 `build_control_wrapper(...)`）。
3. 在每个训练 step 里，把当前 batch 的真实音频在线提取为 top-k CQT，并注入 `cond["melody_control"]`。
4. 仅训练控制分支相关参数（`control_layers` / `zero_linears` / `melody_encoder`，以及浮点控制 fallback 的 `control_projector`）。
5. 使用 `pytorch_lightning.Trainer` 执行持续训练。

---

## 3. 运行前条件

1. 虚拟环境可用：`.venv`
2. 可导入：
   - `stable_audio_tools`
   - `stable_audio_control.models`
   - `stable_audio_control.melody.cqt_topk`
3. 能访问预训练模型：
   - `stabilityai/stable-audio-open-1.0`（联网或本地缓存）
4. 已准备 dataset config JSON（`--dataset-config` 必填）
5. 数据是 waveform 路径（当前脚本不支持 `pre_encoded=True`）

快速导入检查示例：

```powershell
.\.venv\Scripts\python.exe -c "import stable_audio_tools; import stable_audio_control.models; import stable_audio_control.melody.cqt_topk; print('IMPORT_OK')"
```

---

## 4. 如何运行

最小命令：

```powershell
.\.venv\Scripts\python.exe stable_audio_control/scripts/train_controlnet_dit.py --dataset-config <path/to/dataset_config.json>
```

建议先做短程试跑（确认数据和链路都通）：

```powershell
.\.venv\Scripts\python.exe stable_audio_control/scripts/train_controlnet_dit.py `
  --dataset-config <path/to/dataset_config.json> `
  --max-steps 50 `
  --batch-size 1 `
  --num-workers 2 `
  --limit-train-batches 1.0
```

从 checkpoint 续训：

```powershell
.\.venv\Scripts\python.exe stable_audio_control/scripts/train_controlnet_dit.py `
  --dataset-config <path/to/dataset_config.json> `
  --ckpt-path <path/to/last.ckpt>
```

---

## 5. 参数说明（按类别）

### 5.1 必填参数

- `--dataset-config`
  - 数据集配置 JSON 路径。
  - 内部传给 `create_dataloader_from_config(...)`。

### 5.2 训练控制参数

- `--max-steps`（默认 `1000`）
- `--batch-size`（默认 `1`）
- `--num-workers`（默认 `2`）
- `--learning-rate`（默认 `None`）
  - 若不传：优先读 `model_config["training"]["learning_rate"]`
  - 若配置也没有：回退到 `5e-5`
- `--use-ema`（默认 `true`）
- `--accumulate-grad-batches`（默认 `1`）
- `--gradient-clip-val`（默认 `0.0`）
- `--log-every-n-steps`（默认 `10`）
- `--limit-train-batches`（默认 `1.0`）
- `--ckpt-path`（默认 `None`）

### 5.3 Lightning 设备与精度参数

- `--accelerator`（默认 `auto`）
- `--devices`（默认 `1`）
- `--precision`（默认 `16-mixed`）
  - 若当前无 CUDA 且不是 `32-true`，脚本会自动切到 `32-true` 并打印提示。
- `--default-root-dir`（默认 `outputs/train_controlnet_dit`）

### 5.4 ControlNet 参数

- `--num-control-layers`（默认 `2`）
- `--control-id`（默认 `melody_control`）
- `--default-control-scale`（默认 `1.0`）
- `--freeze-base`（默认 `true`）
- `--melody-embedding-dim`（默认 `64`）
- `--melody-hidden-dim`（默认 `256`）
- `--melody-conv-layers`（默认 `2`）

### 5.5 CQT 参数

- `--top-k`（默认 `4`）
- `--n-bins`（默认 `128`）
- `--bins-per-octave`（默认 `12`）
- `--fmin-hz`（默认 `8.175798915643707`，MIDI 0）
- `--hop-length`（默认 `512`）
- `--highpass-cutoff-hz`（默认 `261.2`）
- `--cqt-backend`（默认 `auto`，可选 `auto|nnaudio|librosa`）

说明：`--use-ema`、`--freeze-base` 等布尔参数用字符串解析，推荐显式写 `true`/`false`。

---

## 6. 脚本内部流程（逐步）

### Step A：解析参数与设定随机种子

- 入口：`build_arg_parser()` + `set_seed(...)`
- 同时设置 `random`、`torch`、`pl.seed_everything(...)`。

### Step B：加载 dataset config

- 校验 `--dataset-config` 路径存在，否则直接报错退出。
- 读取 JSON 给 dataloader 使用。

### Step C：加载并包装预训练模型

1. `get_pretrained_model(...)` 加载 StableAudio Open。
2. `build_control_wrapper(...)` 挂接控制分支：
   - `ControlNetContinuousTransformer`
   - `ControlConditionedDiffusionWrapper`

### Step D：构建 CQT 提取器并替换 conditioner

1. 根据 CLI 构造 `CQTTopKExtractor`。
2. 创建 `MelodyControlAugmenter` 包装原 conditioner。
3. 在每个 batch 上从真实 waveform 提取 `melody_control` 并写入 `cond`。

### Step E：初始化控制头 + 冻结策略

1. `initialize_lazy_parameters(...)` 显式检查整数 CQT 路径的 `melody_encoder`，并 materialize 浮点 fallback 的 `control_projector`。
2. `apply_control_only_freeze_policy(...)`：
   - 先冻结所有参数；
   - 再只放开：
     - `control_layers`
     - `zero_linears`
     - `melody_encoder`
     - `control_projector`

### Step F：创建 dataloader

- 调用 `create_dataloader_from_config(...)`。
- 注意：上游实现启用了 `persistent_workers=True`，脚本会把 `num_workers` 下限提升到 `1`。

### Step G：创建训练包装器

- `MelodyAwareDiffusionCondTrainingWrapper` 继承自 `DiffusionCondTrainingWrapper`。
- 在 `training_step/validation_step` 前，先把当前 batch waveform 写入 melody augmenter。

### Step H：创建 Trainer 并执行 `fit`

- 通过 `pl.Trainer(...)` 注入训练参数。
- 输出关键运行信息（模型名、数据配置路径、sample_size、可训练参数样例）。
- 调用：

```python
trainer.fit(training_wrapper, train_dataloaders=train_dl, ckpt_path=args.ckpt_path)
```

---

## 7. 输出与成功判据

脚本启动后建议优先检查：

1. 是否打印：
   - `model_name=...`
   - `dataset_config=...`
   - `trainable_name_samples=...`
2. 训练日志中 `train/loss` 是否为有限值。
3. 显存占用和 step 速度是否符合预期。

若出现持续 NaN/Inf，先缩小学习率、缩短音频长度或降低 batch size 进行定位。

---

## 8. 当前已知限制

1. 不支持 `pre_encoded=True` 路线
- 原因：当前脚本在 step 内需要真实 waveform 来在线提 CQT。
- 触发点：`MelodyAwareDiffusionCondTrainingWrapper._set_melody_batch(...)` 会显式报错。

2. CQT 在线提取有额外算力开销
- `librosa` 后端尤其慢，建议训练优先使用 `nnAudio`（`--cqt-backend auto` 或 `nnaudio`）。

3. 训练效果高度依赖数据 metadata 质量
- 需要稳定的 `prompt`、`seconds_*`、`padding_mask` 等字段。
- 具体参考 `docs/datasets.md`。

---

## 9. 常见问题与排查

### 1) `Dataset config not found`

- 原因：`--dataset-config` 路径错误。
- 处理：检查绝对/相对路径是否在当前工作目录可见。

### 2) `No trainable parameters left...`

- 原因：冻结策略后没有可训练参数（通常是包装对象不符合预期或初始化过程被改动）。
- 处理：确认 `build_control_wrapper(...)` 正常返回，`melody_encoder` 已创建，且 `control_projector` fallback 已 materialize。

### 3) `pre_encoded=True is not supported...`

- 原因：你走了 latent 数据模式。
- 处理：先切回 waveform 数据训练；若必须 pre-encoded，需要额外实现离线 melody 对齐方案。

### 4) `No available CQT backend`

- 原因：`nnAudio` 和 `librosa` 都不可用。
- 处理：安装至少一种；正式训练建议 `nnAudio`。

### 5) DataLoader worker 报错或卡住

- 可先试：
  - `--num-workers 1`
  - 减小 `--batch-size`
  - 确认 dataset config 中 `custom_metadata_module` 可导入。

---

## 10. 与相关文档的关系

- CQT 细节：`docs/Control-net-notes/doc/cqt_topk.zh-CN.md`
- 开训前最小烟测：`docs/Control-net-notes/doc/train_control_smoke_usage.zh-CN.md`
- 数据配置规范：`docs/datasets.md`

建议顺序：

1. 先跑 `stable_audio_control/scripts/train_control_smoke.py`（确保链路通）。
2. 再用本文脚本做短程正式训练（几十步）。
3. 最后再扩到长程训练与系统性调参。
