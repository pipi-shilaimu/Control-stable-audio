# `stable_audio_control/scripts/train_controlnet_dit.py` 使用与实现说明

## 1. 文档目的

本文档说明 `stable_audio_control/scripts/train_controlnet_dit.py` 的定位、运行方式、关键参数和内部流程。

适用读者：
- 已经跑通过 `train_control_smoke.py`
- 希望开始多步训练（`Trainer.fit`）
- 正在做 StableAudio Open + ControlNet + melody control 的实验迭代

---

## 2. 脚本定位

`train_controlnet_dit.py` 是本仓库当前的正式训练入口（ControlNet 方向）。

核心能力：
1. 加载 `stabilityai/stable-audio-open-1.0` 预训练模型
2. 挂接 `ControlNetContinuousTransformer`（通过 `build_control_wrapper(...)`）
3. 在每个训练 step 里，把当前 batch 的真实音频在线提取为 top-k CQT，并注入 `cond["melody_control"]`
4. 仅训练控制分支相关参数（`control_layers` / `zero_linears` / `melody_encoder`）
5. 使用 `pytorch_lightning.Trainer` 执行持续训练

---

## 3. 运行前条件

1. 虚拟环境可用
2. 可导入：`stable_audio_tools`、`stable_audio_control.models`、`stable_audio_control.melody.cqt_topk`
3. 能访问预训练模型 `stabilityai/stable-audio-open-1.0`
4. 已准备 dataset config JSON（`--dataset-config` 必填）

---

## 4. 如何运行

最小命令：
```bash
python3 stable_audio_control/scripts/train_controlnet_dit.py --dataset-config <path/to/dataset_config.json>
```

建议先做短程试跑：
```bash
python3 stable_audio_control/scripts/train_controlnet_dit.py \
  --dataset-config <path/to/dataset_config.json> \
  --max-steps 50 --batch-size 1 --num-workers 1
```

30 条音频 overfit 训练（首次验证 ControlNet 效果）：
```bash
python3 stable_audio_control/scripts/train_controlnet_dit.py \
    --dataset-config stable_audio_control/data/mtg_jamendo/dataset_config_train_30.json \
    --max-steps 50000 \
    --batch-size 6 \
    --num-control-layers 12 \
    --precision bf16-mixed \
    --learning-rate 1e-4 \
    --log-every-n-steps 10 \
    --default-root-dir outputs/overfit_piano_30 \
    --ckpt-at-step 25000 \
    --demo-every 500 \
    > train.log 2>&1 &
```

从 checkpoint 续训：
```bash
python3 stable_audio_control/scripts/train_controlnet_dit.py \
  --dataset-config <path/to/dataset_config.json> \
  --ckpt-path <path/to/last.ckpt>
```

---

## 5. 参数说明（按类别）

### 5.1 训练控制参数

- `--max-steps`（默认 `1000`）
- `--batch-size`（默认 `1`）
- `--num-workers`（默认 `2`）
- `--learning-rate`（默认 `None`，回退 5e-5）
- `--use-ema`（默认 `true`）
- `--accumulate-grad-batches`（默认 `1`）
- `--log-every-n-steps`（默认 `10`）
- `--limit-train-batches`（默认 `1.0`）

### 5.2 Lightning 设备与精度参数

- `--accelerator`（默认 `auto`）
- `--devices`（默认 `1`）
- `--precision`（默认 `16-mixed`）
- `--default-root-dir`（默认 `outputs/train_controlnet_dit`）

### 5.3 ControlNet 参数

- `--num-control-layers`（默认 `12`）
- `--control-id`（默认 `melody_control`）
- `--default-control-scale`（默认 `1.0`）
- `--freeze-base`（默认 `true`）
- `--melody-embedding-dim`（默认 `64`）
- `--melody-hidden-dim`（默认 `256`）
- `--melody-conv-layers`（默认 `2`）

### 5.4 CQT 参数

- `--top-k`（默认 `4`）
- `--n-bins`（默认 `128`）
- `--bins-per-octave`（默认 `12`）
- `--fmin-hz`（默认 `8.18`，MIDI 0）
- `--hop-length`（默认 `512`）
- `--highpass-cutoff-hz`（默认 `261.2`）
- `--cqt-backend`（默认 `auto`，可选 `auto|nnaudio|librosa`）

### 5.5 VAE Encoder 确定性开关

- `--deterministic-encode`（默认 `false`）
  - `false`：使用原生 `pretransform.encode()`（含 VAE bottleneck 随机采样），匹配 backbone 训练与推理分布。正常训练用这个。
  - `true`：跳过 VAE bottleneck，直接用 encoder 的 mean 输出。仅用于过拟合调试。

### 5.6 Demo 生成回调

- `--demo-every`（默认 `0` = 禁用）
  - 每隔 N 步生成 demo 音频，自动注入旋律控制条件（CQT → melody_encoder → control_input）
  - 生成 3 个 cfg_scale（3, 6, 9）各一段 wav + TensorBoard mel spectrogram
  - demo 使用当前 batch 的真实音频提取旋律（和训练同源）
  - 用于训练中判断 ControlNet 的控制质量。示例：`--demo-every 500`

### 5.7 验证与 Checkpoint 参数

- `--val-dataset-config`（默认 `None`）
- `--val-check-interval`（默认 `100`）
- `--ckpt-at-step`（默认 `0`，`0` = 禁用）
- `--ckpt-path`（默认 `None`）
- `--sigint-save`（默认 `true`，Ctrl+C 时保存 checkpoint）

---

## 6. 脚本内部流程

1. 解析参数 + 设置随机种子
2. 加载 dataset config
3. 加载预训练模型 + 挂接 ControlNet wrapper
4. 构建 CQT 提取器 + 替换 conditioner 为 `MelodyControlAugmenter`
5. 初始化 `control_projector` 等 lazy 参数 + 冻结 backbone
6. 创建 `create_dataloader_from_config` 下的 DataLoader
7. 创建 `MelodyAwareDiffusionCondTrainingWrapper`（继承 `DiffusionCondTrainingWrapper`）
8. 可选：构建验证 DataLoader + `ModelCheckpoint`
9. 启动 `Trainer.fit`

---

## 7. 判断训练效果

**不要用 `train/loss` 的绝对值判断 ControlNet 效果。** 扩散模型的 per-step loss 方差极大（不同 t 下最优 loss 从 0 到 1.0），VAE 随机采样进一步增加波动。

正确方式：
1. **Demo 音频**：`--demo-every 500` 定期生成，耳朵听 3 个 cfg_scale 的音频。旋律是否接近原曲？
2. **生成对比**：用 checkpoint 跑 `control_scale=1.0` vs `0.0`，对比旋律相似度
3. **TensorBoard**：观察 `train/loss` 的平滑趋势（不是绝对值），以及 demo mel spectrogram

---

## 8. 已知限制

1. demo 回调不使用 EMA model——EMA 是原始 DiT，不通 `control_input` 注入路径
2. CQT 在线提取有算力开销，`librosa` 后端慢，建议 `--cqt-backend nnaudio`
3. 训练效果依赖 metadata 质量（`prompt`、`seconds_*`、`padding_mask`）

---

## 9. 与相关文档的关系

- CQT 细节：`cqt_topk.md`
- 开训前最小烟测：`train_control_smoke.md`
- 排障：`noob-docs/10-翻车指南.md`（重点关注 VAE encoder 确定性相关条目）
- Demo 回调实现：`stable_audio_control/inference/control_demo_callback.py`