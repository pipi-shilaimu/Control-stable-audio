# train_controlnet_dit.py 完整参数参考

> 最后一次更新：2026-05-31  对应文件：`stable_audio_control/scripts/train_controlnet_dit.py`

---

## 必需参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `--dataset-config` | `str` | 数据集配置 JSON 路径。由 `organize_dataset.py` 或手写生成。 |

## 基础训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model-name` | `str` | `stabilityai/stable-audio-open-1.0` | HuggingFace 预训练模型 ID |
| `--seed` | `int` | `0` | 随机种子 |
| `--batch-size` | `int` | `1` | 每 GPU batch size |
| `--num-workers` | `int` | `2` | DataLoader 工作进程数 |
| `--max-steps` | `int` | `1000` | 最大训练步数 |
| `--accumulate-grad-batches` | `int` | `1` | 梯度累积批次数。实际 batch = `batch_size × accumulate_grad_batches × devices` |
| `--gradient-clip-val` | `float` | `0.0` | 梯度裁剪阈值，0 = 不裁剪 |
| `--log-every-n-steps` | `int` | `10` | 每 N 步打印一次 loss |
| `--limit-train-batches` | `float` | `1.0` | 每 epoch 使用多少比例的训练数据。`0.1` = 只用 10% |

## 学习率与优化器

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--learning-rate` | `float` | `None` | 覆盖 model_config 中的学习率。`None` = 使用 model_config 原值（SAO 默认 `5e-5`） |
| `--use-ema` | `bool` | `true` | 是否启用 EMA（指数滑动平均）。推理时 EMA 模型更稳定 |

## 训练长度（二选一）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--seconds-total` | `float` | `None` | 训练音频长度（秒）。如 `10.0`。**推荐**，比写采样点数直观 |
| `--sample-size` | `int` | `None` | 训练音频长度（采样点）。如 `441000` = 10s@44100Hz |

两个参数互斥。都不传时使用 model_config 的 `sample_size`（SAO = `2097152` = 47.55s）。

最终值会自动对齐到 `min_input_length`（= `downsampling_ratio` = 2048）的整数倍。

## 输出与日志

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--default-root-dir` | `str` | `outputs/train_controlnet_dit` | 输出根目录。checkpoint 和 demo 都放这里 |
| `--precision` | `str` | `16-mixed` | PyTorch Lightning 精度。`16-mixed`、`bf16-mixed`、`32-true` |
| `--accelerator` | `str` | `auto` | Lightning accelerator。`auto`、`cpu`、`gpu` |
| `--devices` | `int` | `1` | GPU 数量 |

## 断点续训与保存

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--ckpt-path` | `str` | `None` | 从指定 checkpoint 恢复训练（含 optimizer/scheduler 状态） |
| `--ckpt-at-step` | `int` | `0` | 在指定 global_step 保存一个快照。`0` = 禁用 |
| `--sigint-save` | `bool` | `true` | Ctrl+C 时自动保存 checkpoint 到 `{default_root_dir}/checkpoints/interrupted-step-{step}.ckpt` |

> ModelCheckpoint 每 8000 步自动保存一次（`save_last=True`，`save_top_k=-1`）。

## 验证

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--val-dataset-config` | `str` | `None` | 验证集配置 JSON。不传则跳过验证 |
| `--val-check-interval` | `int` | `100` | 每 N 步跑一次验证。仅当 `--val-dataset-config` 提供时生效 |

## ControlNet 架构

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num-control-layers` | `int` | `12` | 控制分支复制的 transformer 层数。越大控制力越强、显存越多。过拟合/小数据建议 `4-8` |
| `--control-id` | `str` | `melody_control` | conditioning 字典中控制信号的 key |
| `--default-control-scale` | `float` | `1.0` | 训练时默认的控制强度 |
| `--freeze-base` | `bool` | `true` | 是否冻结 base model。`false` = 全参数训练（谨慎） |
| `--deterministic-encode` | `bool` | `false` | 是否绕过 VAE bottleneck 随机采样，用 encoder mean。**仅用于过拟合调试，正常训练必须 `false`** |

## MelodyControlEncoder 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--melody-embedding-dim` | `int` | `64` | pitch embedding 维度 |
| `--melody-hidden-dim` | `int` | `256` | 卷积隐藏层维度 |
| `--melody-conv-layers` | `int` | `2` | 卷积层数 |

## Demo 生成（训练中诊断）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--demo-every` | `int` | `0` | 每 N 步生成 demo。`0` = 禁用 |
| `--demo-steps` | `int` | `100` | Demo 采样步数。越少越快，越多质量越好 |
| `--demo-cfg-scales` | `str` | `"3,6,9"` | 逗号分隔的 CFG scale 值。如 `"7"` 或 `"5,7,9"` |
| `--demo-control-scales` | `str` | `"0,0.1,0.3,0.6,1.0"` | 逗号分隔的 control_scale 值。生成 `N_cfg × N_control × N_variants` 个样本 |
| `--demo-control-variants` | `str` | `"correct,shuffled, zero"` | 逗号分隔的对照类型。`correct`=正确旋律，`shuffled`=错位/反转旋律，`zero`=全零旋律 |
| `--demo-control-audio` | `str` | `None` | 固定音频文件作为控制源。`None` = 用当前 batch 的音频 |
| `--demo-prompt` | `str` | `None` | 固定文本 prompt。`None` = 用当前 batch 的 metadata prompt |

### Demo 参数解读

Demo 生成的总样本数 = `len(demo_cfg_scales) × len(demo_control_scales) × len(demo_control_variants)`。

示例：
```bash
--demo-cfg-scales "7" \
--demo-control-scales "0,0.6,1.0" \
--demo-control-variants "correct,shuffled,zero"
```
→ 每轮 demo 生成 `1 × 3 × 3 = 9` 个 wav 文件。

**诊断标准**：
- `correct` 有明显旋律跟随，`zero` 没有 → ControlNet 在工作
- `correct` 和 `shuffled` 旋律明显不同 → 模型在"听"控制，不是盲加残差
- 三个 variant 没区别 → 控制分支未学会听从控制信号

## 旋律特征：CQT

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--melody-feature` | `str` | `cqt` | 旋律特征类型。`cqt` 或 `chromagram` |
| `--top-k` | `int` | `4` | 每通道保留的最大 bin 数 |
| `--n-bins` | `int` | `128` | CQT 频率 bin 总数 |
| `--bins-per-octave` | `int` | `12` | 每八度音程的 bin 数 |
| `--fmin-hz` | `float` | `8.176` | CQT 最低频率（Hz） |
| `--hop-length` | `int` | `512` | CQT 帧移 |
| `--highpass-cutoff-hz` | `float` | `261.2` | 高通滤波截止频率（≈ 中央 C） |
| `--cqt-backend` | `str` | `auto` | CQT 后端。`auto` → `nnaudio`（GPU）→ `librosa`（CPU fallback） |

## 旋律特征：Chromagram（备选）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--chroma-bins` | `int` | `12` | 半音阶 bins |
| `--chroma-n-fft` | `int` | `2048` | STFT 窗口大小 |

**使用 chromagram 时需加 `--melody-feature chromagram`。**

---

## 常用命令组合

### 过拟合（单 batch，快速验证）

```bash
python3 stable_audio_control/scripts/train_controlnet_dit.py \
    --dataset-config data/overfit/dataset_config_train.json \
    --seconds-total 10 \
    --batch-size 1 \
    --num-control-layers 8 \
    --learning-rate 5e-5 \
    --max-steps 1500 \
    --demo-every 100 \
    --demo-cfg-scales "7" \
    --demo-control-scales "0,0.6,1.0" \
    --demo-control-variants "correct,shuffled,zero" \
    --demo-steps 100 \
    --precision bf16-mixed \
    --default-root-dir outputs/overfit_test
```

### 正式训练（10s 音频）

```bash
python3 stable_audio_control/scripts/train_controlnet_dit.py \
    --dataset-config data/dataset_config_train.json \
    --seconds-total 10 \
    --batch-size 4 \
    --num-control-layers 8 \
    --learning-rate 5e-5 \
    --max-steps 50000 \
    --demo-every 2000 \
    --demo-cfg-scales "7" \
    --demo-control-scales "0,0.6,1.0" \
    --demo-control-variants "correct,shuffled,zero" \
    --demo-steps 100 \
    --precision bf16-mixed \
    --default-root-dir outputs/formal_training \
    --ckpt-at-step 5000
```

### 从断点恢复

```bash
python3 stable_audio_control/scripts/train_controlnet_dit.py \
    --dataset-config data/dataset_config_train.json \
    --seconds-total 10 \
    --batch-size 4 \
    --num-control-layers 8 \
    --learning-rate 5e-5 \
    --max-steps 50000 \
    --ckpt-path outputs/formal_training/checkpoints/interrupted-step-12000.ckpt
```
