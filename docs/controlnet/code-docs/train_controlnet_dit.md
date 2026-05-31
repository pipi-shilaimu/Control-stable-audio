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
    --seconds-total 10 \
    --max-steps 50000 \
    --batch-size 6 \
    --num-control-layers 12 \
    --precision bf16-mixed \
    --learning-rate 1e-4 \
    --log-every-n-steps 10 \
    --default-root-dir outputs/overfit_piano_30 \
    --ckpt-at-step 25000 \
    --demo-every 500 \
    --demo-control-audio audio/Then.mp3 \
    --demo-prompt "instrumental piano melody, clear lead melody, no vocals" \
    --demo-cfg-scales 6 \
    --demo-steps 30 \
    --demo-control-scales 0,0.1,0.3,0.6,1.0 \
    --demo-control-variants correct,shuffled,zero \
    > train.log 2>&1 &
```

如果训练数据来自 `formal10s`、`audio_10s` 这类 10 秒切片，建议显式加 `--seconds-total 10`。否则脚本会沿用 Stable Audio Open 配置里的 `sample_size=2097152`，在 44100 Hz 下约为 47.55 秒，10 秒音频会被补成大量静音，ControlNet 很容易学到“平滑/静音偏置”而不是旋律控制。

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

### 5.6 训练长度参数

- `--seconds-total`（默认 `None`）
  - 用秒指定训练、验证、demo 都使用的有效音频长度。
  - 例如 10 秒切片建议传 `--seconds-total 10`。
  - 脚本会按 `model.min_input_length` 向上对齐，所以 44100 Hz、10 秒通常会变成 `442368` samples，约 `10.030s`。
- `--sample-size`（默认 `None`）
  - 直接用采样点数指定有效长度。
  - 和 `--seconds-total` 互斥。

直觉上，`sample_size` 就是训练时给每条音频发的“固定答题纸长度”。真实音频比它短，就会被补 0；真实音频比它长，就会被裁掉。本项目现在会把 train dataloader、val dataloader、demo callback 都统一到同一个 `effective_train_sample_size`，避免训练和 demo 使用不同长度。

启动日志会打印：

```text
sample_rate=44100, batch_size=...
model_config_sample_size=2097152
effective_train_sample_size=442368 (10.0300s, source=--seconds-total)
min_input_length=2048
demo_cfg_scales=[6.0]
demo_steps=30
```

这里的 `model_config_sample_size` 是预训练模型原始配置，不一定等于本次训练实际使用的长度；真正要看的是 `effective_train_sample_size`。

### 5.7 Demo 生成回调

- `--demo-every`（默认 `0` = 禁用）
  - 每隔 N 步生成 demo 音频，自动注入旋律控制条件（CQT → melody_encoder → control_input）
  - 默认生成多个 `cfg_scale`、`control_scale`、`control_variant` 组合的 wav + TensorBoard mel spectrogram
  - 默认 `control_scale` 扫描：`0.0,0.1,0.3,0.6,1.0`
  - demo 默认使用当前 batch 的真实音频提取旋律；如果传 `--demo-control-audio`，则使用指定音频提取旋律
  - 用于训练中判断 ControlNet 的控制质量。示例：`--demo-every 500`
- `--demo-cfg-scales`（默认 `3,6,9`）
  - demo 专用的 CFG scale 列表。
  - 第一次 smoke 建议先用 `--demo-cfg-scales 6`，避免每轮 demo 生成过多样本。
- `--demo-steps`（默认 `100`）
  - 每次 demo 采样的扩散步数。
  - 第一次 smoke 建议先用 `--demo-steps 30`，链路跑通后再升回更高步数听质量。
- `--demo-control-scales`（默认 `0,0.1,0.3,0.6,1.0`）
  - demo 专用的 `control_scale` 扫描列表。
  - `0.0` 用来观察关闭控制分支时的 baseline；`0.1`、`0.3`、`0.6`、`1.0` 用来观察控制强度上升后是否更贴近旋律。
- `--demo-control-variants`（默认 `correct,shuffled,zero`）
  - `correct`：使用正确旋律控制。
  - `shuffled`：使用错误/错位旋律控制。若没有外部控制音频，会在 batch 内错配；若传了 `--demo-control-audio`，会用时间反转制造错误控制。
  - `zero`：使用全 0 控制，检查模型是不是忽略了 melody control。
- `--demo-control-audio`（默认 `None`）
  - 指定 demo 里“提取旋律控制”的音频。
  - 适合固定一首你关心的旋律做长期观察，例如 `--demo-control-audio audio/Then.mp3`。
  - 脚本会加载该音频、必要时重采样到模型采样率、转成 stereo、裁剪/补齐到 `effective_train_sample_size`，再复制到 `num_demos`。
- `--demo-prompt`（默认 `None`）
  - 指定 demo 生成时使用的固定文本 prompt。
  - 不传时沿用当前训练 batch metadata 里的 `prompt`；如果 metadata 没有 prompt，则回退到 `"music"`。
  - 当你同时传 `--demo-control-audio` 时，建议也传 `--demo-prompt`，否则旋律控制来自固定音频，但文本 prompt 仍会跟随当前训练 batch 变化。

两个容易混淆的字段：

- `control_variant` 不是“源文件名”，而是 demo 消融实验的控制源变体：`correct`、`shuffled`、`zero`。
- `stem` 是输出文件名的主体，不含 `.wav` 后缀。例如 `demo_cfg_7_control_0p3_correct_step_00000500` 是 stem，最终文件是 `demo_cfg_7_control_0p3_correct_step_00000500.wav`。

demo 成本粗略等于：

```text
len(demo_cfg_scales) * len(demo_control_scales) * len(demo_control_variants)
```

默认配置是 `3 * 5 * 3 = 45` 次采样。第一次训练想快速确认链路是否通，可以用：

```text
--demo-cfg-scales 6
--demo-steps 30
--demo-control-scales 0,1
--demo-control-variants correct,zero
--demo-prompt "instrumental piano melody, clear lead melody, no vocals"
```

这样每轮 demo 只有 `1 * 2 * 2 = 4` 次采样，适合 smoke；确认无 shape/device/保存问题后再扩大网格。

### 5.8 验证与 Checkpoint 参数

- `--val-dataset-config`（默认 `None`）
- `--val-check-interval`（默认 `100`）
- `--ckpt-at-step`（默认 `0`，`0` = 禁用）
- `--ckpt-path`（默认 `None`）
- `--sigint-save`（默认 `true`，Ctrl+C 时保存 checkpoint）

### 5.9 下一轮架构消融计划（尚未实现）

> 本节是训练恢复路线图，不代表当前脚本已经支持这些参数。实际命令里不要提前传下面的计划参数，除非代码已经实现。

当前 ControlNet 分支的关键风险是 `x_ctrl = ctrl`：控制分支只看 CQT melody 特征，不看 base branch 当前正在去噪的 latent 状态。直觉上，这像副驾驶只看导航截图，却看不到车速、方向盘和路况。

下一轮建议把 control branch 输入方式做成可配置消融：

```text
--control-branch-input control_only
--control-branch-input detached_base_plus_control
--control-branch-input base_plus_control
```

含义：

- `control_only`：当前旧行为，`x_ctrl = ctrl`。
- `detached_base_plus_control`：优先实验，`x_ctrl = x_base.detach() + ctrl`。
- `base_plus_control`：第二消融，`x_ctrl = x_base + ctrl`。

推荐第一轮只试 `detached_base_plus_control`。`detach()` 的作用是借用 base 当前状态给 control branch 做参照，但不让控制分支反向改变 base 状态的计算图。base 参数虽然冻结，但这个边界更干净，也更接近“control learns to guide base”的设计意图。

建议短训设置：

```text
--seconds-total 10
--control-branch-input detached_base_plus_control
--demo-cfg-scales 6
--demo-steps 30
--demo-control-scales 0,0.3,1
--demo-control-variants correct,shuffled,zero
```

不要用旧 checkpoint 直接判断这个改动。旧 checkpoint 是在 `control_only` 输入分布下训练出来的；改变输入方式后，应新开输出目录训练 2000-4000 步，观察 correct / shuffled / zero 是否拉开。

### 5.10 参考实现差异：先记录，不立刻照搬

这轮训练失败后，有两个来自 `stable-audio-controlnet` 参考实现的差异值得保留在路线图里，但它们不应该和 `--seconds-total 10`、demo 消融、`x_ctrl = x_base.detach() + ctrl` 同时改。

**差异 B：control 条件是否对齐到 VAE latent 空间。**

当前实现把 CQT top-k index 送进 `MelodyControlEncoder`，得到的 `ctrl` 更像旋律符号 embedding。参考实现通常让 control audio 经过同一个 frozen VAE / pretransform，因此 control condition 天然落在 base DiT 熟悉的 latent/token 空间。这个差异可能导致当前 `ctrl` 对 DiT 来说语义不够对齐，进而学成平滑残差，而不是旋律跟随。

短期不建议直接替换成完整 guide-audio pretransform 路线，因为这会改变数据输入、推理接口、demo callback 和训练目标。更稳的长期折中是 `melody-to-latent adapter`：用 `(audio, CQT)` 配对数据蒸馏一个 pitch-relevant 的 adapter，让 CQT 特征更接近 DiT 能消费的 control residual 空间，但不要强迫 CQT 重建完整 VAE latent。

**差异 C：control residual 在 layer 前还是 layer 后注入。**

当前结构更接近：

```python
x_base = base_layer(x_base, ...)
x_base = x_base + zero_linear[layer_ix](x_ctrl) * control_scale
```

这属于“base layer 先处理，再用 control residual 事后修正”。参考实现更偏向在进入对应 transformer block 前把 control embedding 混进状态，让 attention / MLP 在该层计算时就已经看见控制。它可能更强，但也是单独变量。

建议消融顺序：

1. 先做 `detached_base_plus_control`，只改变 control branch 输入。
2. 如果 correct / shuffled / zero 仍无区分度，再单独试 layer 前注入。
3. 不要把输入方式、注入时机、CQT padding mask、latent adapter 一次性全改，否则听感结果无法归因。

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
1. **Demo 音频**：`--demo-every 500` 定期生成，听 `correct`、`shuffled`、`zero` 三类控制变体。
2. **控制强度扫描**：重点比较 `control_scale=0.0,0.1,0.3,0.6,1.0`。如果 `1.0` 明显更平滑但不更像参考旋律，说明控制残差可能在破坏音质，而不是学会旋律。
3. **消融判断**：如果 `correct`、`shuffled`、`zero` 听起来差不多，ControlNet 很可能还没有学到 melody-specific control。
4. **固定旋律源**：用 `--demo-control-audio <path>` 固定一首你关心的旋律，不要每次都跟随训练 batch 随机变化。
5. **TensorBoard**：观察 `train/loss` 的平滑趋势（不是绝对值），以及 demo mel spectrogram。

如果 `--seconds-total 10` 后 correct / shuffled / zero 仍然听不出差异，下一步不要马上长训，而是优先做 `x_ctrl = x_base.detach() + ctrl` 的短训消融。只有这个实验仍失败时，再继续处理 CQT padding mask、energy 特征或 melody-to-latent adapter。

---

## 8. 已知限制

1. demo 回调不使用 EMA model——EMA 是原始 DiT，不通 `control_input` 注入路径
2. CQT 在线提取有算力开销，`librosa` 后端慢，建议 `--cqt-backend nnaudio`
3. 训练效果依赖 metadata 质量（`prompt`、`seconds_*`、`padding_mask`）
4. 如果 10 秒切片未加 `--seconds-total 10` 或合适的 `--sample-size`，模型会按约 47.55 秒读取样本，大量 padding 会污染训练信号。
5. `control_scale=1.0` 听起来“更有影响”不等于控制学好了，必须配合 `correct/shuffled/zero` 消融判断。

---

## 9. 与相关文档的关系

- CQT 细节：`cqt_topk.md`
- 开训前最小烟测：`train_control_smoke.md`
- 排障：`noob-docs/10-翻车指南.md`（重点关注 VAE encoder 确定性相关条目）
- Demo 回调实现：`stable_audio_control/inference/control_demo_callback.py`
