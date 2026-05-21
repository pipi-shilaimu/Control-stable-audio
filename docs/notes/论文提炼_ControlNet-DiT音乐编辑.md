# 论文提炼：Editing Music with Melody and Text

更新日期：2026-04-27

本文档整理根目录论文 [Editing music with melody and text_using controlnet for diffusion transformer.pdf](<Editing music with melody and text_using controlnet for diffusion transformer.pdf>) 的核心内容，并结合当前仓库实现状态，提炼出项目还要完成的点、已经完成的点、整体思路与关键机制。

已抽取的论文文本位于 [tmp/pdfs/editing_music_controlnet_dit_pypdf.txt](pdfs/editing_music_controlnet_dit_pypdf.txt)。本次抽取结果显示 PDF 共 5 页，正文、图 1 文本、表 I/II 与参考文献均可读。

## 1. 项目初步理解

根据 [导航.md](../README.md)，当前仓库围绕 StableAudio Open 做 ControlNet-DiT 改造，目标是实现“文本 + 旋律”双条件音乐编辑。推荐阅读路径 B 已经把本论文列为 ControlNet-DiT 改造的第一入口，后续对应到实现计划、注意事项、核心模型代码与训练脚本。

当前代码主线如下：

| 模块 | 位置 | 当前职责 |
| --- | --- | --- |
| ControlNet Transformer | [stable_audio_control/models/control_transformer.py](stable_audio_control/models/control_transformer.py) | 包装 StableAudio 的 `ContinuousTransformer`，复制前 N 层作为控制分支，通过 zero-linear 注入冻结主干 |
| Diffusion wrapper 接入 | [stable_audio_control/models/control_dit.py](stable_audio_control/models/control_dit.py) | 从 `cond["melody_control"]` 取控制张量，投影成 `control_input`，透传给 ControlNet transformer |
| top-k CQT | [stable_audio_control/melody/cqt_topk.py](stable_audio_control/melody/cqt_topk.py) | 从立体声音频中提取每声道 top-4 CQT pitch index，形成 8 通道旋律提示 |
| 正式训练入口 | [stable_audio_control/scripts/train_controlnet_dit.py](stable_audio_control/scripts/train_controlnet_dit.py) | 加载 StableAudio Open，构建 ControlNet wrapper，在线提取 CQT，冻结非控制参数，启动 Lightning 训练 |
| 训练 smoke | [stable_audio_control/scripts/train_control_smoke.py](stable_audio_control/scripts/train_control_smoke.py) | 最小训练闭环验证 |
| smoke 报告 | [smoke 报告](../controlnet/smoke_report_2026-04-21.md) | 记录 control 注入与单步训练通过情况 |

StableAudio Open 的基础结构可以概括为：

```text
waveform [B, 2, T]
  -> pretransform / audio autoencoder
latent [B, C_latent, L]
  -> DiTWrapper / DiffusionTransformer
  -> ContinuousTransformer, 24 transformer blocks
  -> denoised latent
  -> pretransform.decode()
waveform output
```

文本条件通过 T5 conditioner 进入 cross-attention；时长条件 `seconds_start` / `seconds_total` 通过 number conditioner 进入 timing/global/prepend 相关路径。当前 ControlNet 改造采用旁路方式新增 `melody_control`，不把旋律控制塞进原始 text/timing conditioning 路由。

## 2. 论文要解决的问题

论文关注的是可控音乐生成与音乐编辑，尤其是用文本控制全局风格/语义，用旋律提示控制时间变化的音高结构。它认为现有方法主要有两类瓶颈。

第一类是表示和结构瓶颈。很多音乐扩散模型使用 Mel-spectrogram 表示和 UNet 架构。Mel-spectrogram 输出长度常被 2D 卷积、下采样倍率和固定谱图尺寸限制，变长生成不够自然；最终还要依赖 vocoder 从谱图还原 waveform，可能引入额外失真。UNet 对长程音乐结构建模也不如 Transformer 适合。

第二类是旋律提示的信息不足。Music ControlNet、DITTO 等方法常用 12 pitch-class chromagram 作为旋律条件。这种表示只保留一个八度内的 pitch class，丢失绝对音高和跨八度变化；如果每帧只取 argmax，还会丢失多轨音乐中同时出现的多个显著音高。因此模型可能只能保留旋律轮廓，而不能精确保留实际旋律与和声细节。

论文的核心目标是：在保持 StableAudio Open 长音频、变长生成、文本控制能力的基础上，引入更精确的旋律控制，使模型能完成 melody + text guided music editing。

## 3. 论文总体思路

论文方法可以分成四层：

1. **使用 StableAudio Open / DiT 作为预训练主干**

   论文采用 StableAudio Open 作为 backbone。它基于 latent audio codec 和 Diffusion Transformer，可以生成较长的 stereo audio，并通过 timing conditioning 支持 variable-length generation。

2. **给 DiT 增加 ControlNet-Transformer 分支**

   原始 ControlNet 设计依赖 UNet 的 encoder-decoder 和 skip connection，不适合直接套到 DiT。论文借鉴 PixArt-δ 中的 ControlNet-Transformer：复制前 N 个 Transformer block 作为可训练控制分支，冻结原主干，通过 zero-initialized linear 将控制分支输出逐层注入冻结主干 hidden stream。

3. **用 top-k CQT 作为旋律提示**

   论文不用 12 pitch-class chroma，而是对左右声道分别计算 128-bin CQT，每帧每声道取 top-4 最显著 pitch bin，再交错成 8 通道旋律提示。这样既保留绝对音高，也能表达多轨/多音高信息。

4. **用 progressive curriculum masking 平衡旋律与文本**

   如果旋律提示过于精确，模型容易学成“根据旋律提示重建目标音频”，文本 prompt 的控制力会下降。论文用渐进式课程遮罩训练：早期全部遮罩旋律，让模型先保留文本生成能力；之后逐步暴露更多旋律帧，并对 top-2 到 top-4 进行随机遮罩和打乱，让模型学习鲁棒使用旋律而不是死记重建。

## 4. 关键机制详解

### 4.1 ControlNet for Diffusion Transformer

论文中的 DiT-ControlNet 结构与当前仓库目标高度一致：

```text
输入 noisy latent + text/timing condition
  -> 冻结 DiT 主干 block 1
  -> 冻结 DiT 主干 block 2
  -> ...

旋律 latent control
  -> 复制的 control block 1
  -> zero linear
  -> 加到主干 block 1 输出

  -> 复制的 control block 2
  -> zero linear
  -> 加到主干 block 2 输出

  -> ...
```

更具体地，对第 `i` 层：

```text
x_control = control_block_i(x_control, condition)
x_base = frozen_block_i(x_base, condition)
x_base = x_base + zero_linear_i(x_control)
```

zero-linear 的意义是稳定初始化。初始时 `zero_linear_i` 权重和 bias 全为 0，所以刚接入 ControlNet 时不会破坏预训练 StableAudio 的原始行为。训练过程中，zero-linear 和 control blocks 逐渐学会把旋律控制残差注入主干。

论文训练策略是冻结 DiT 主干，只微调 ControlNet 和旋律 prompt 转 latent 的结构。

### 4.2 top-k CQT 旋律表示

论文 top-k CQT pipeline 如下：

```text
stereo music waveform
  -> left/right channel CQT
  -> each channel: 128 bins, hop length 512, fmin = MIDI note 0
  -> each frame/channel keep top-4 pitch bins
  -> interleave left/right:
     [L0, R0, L1, R1, L2, R2, L3, R3]
  -> melody prompt c, shape [8, frames]
```

关键参数：

| 参数 | 论文设定 | 当前仓库默认 |
| --- | --- | --- |
| 声道 | stereo left/right | stereo `[B, 2, T]` |
| CQT bins | 128 | `n_bins=128` |
| bins per octave | 12 | `bins_per_octave=12` |
| fmin | MIDI note 0, 约 8.18 Hz | `fmin_hz=8.175798915643707` |
| hop length | 512 | `hop_length=512` |
| 每声道 top-k | 4 | `top_k=4` |
| 输出通道 | 8 | `[B, 2K, F]` |
| pitch index | `1..128` | 代码中 `topk_idx + 1` |
| mask/pad 预留 | 0 | 当前代码保留 0 |
| CQT 前高通 | Middle C, 261.2 Hz | `highpass_cutoff_hz=261.2` |

top-k CQT 相比 chroma 的优势：

- 保留绝对音高，不把所有八度折叠到 12 个 pitch class。
- 每帧保留多个显著 pitch，可以表达多轨或和声信息。
- 仍然是人可理解的 pitch index，可以从音频提取，也可以由乐谱或手工旋律构造。

### 4.3 Melody prompt 到 latent melody prompt

论文不是直接把 `[8, F]` 的整数索引当连续值输入模型，而是：

```text
top-k CQT index c, values in 1..128
  -> pitch-specific trainable embeddings
  -> high-dimensional latent melody prompt
  -> 1D conv layers downsample / align
  -> ControlNet input shape
```

这个设计让模型学习每个 pitch index 的可训练语义，而不是把 pitch index 当普通数值。比如 index 60 和 61 不是普通的数值大小关系，而是两个相邻音高；embedding 可以学习更适合模型的音高空间。

当前仓库还没有完整实现这个 embedding + Conv1D conditioner。当前 [control_dit.py](stable_audio_control/models/control_dit.py) 中的 `control_projector` 是 `LazyLinear`，它能把 CQT 通道投影到 transformer `dim_in`，但还不是论文中更明确的离散 pitch embedding + Conv 下采样结构。

### 4.4 Progressive curriculum masking

论文认为精确旋律提示有副作用：

- 旋律提示可能包含 timbre 等额外信息，变成目标音频的压缩版本。
- 旋律提示与目标音频时间对齐，模型可能绕过文本 prompt，直接学“旋律到音频重建”。
- 这样虽然编辑指标可能好看，但模型文本生成能力会下降，无法真正平衡 text 与 melody。

因此论文提出 progressive curriculum masking：

| 阶段 | 策略 | 目的 |
| --- | --- | --- |
| 训练初期 | 全部 melody prompt masked | 让模型先学空旋律条件下的文本生成 |
| 训练推进 | frame-wise mask ratio 逐渐降低，但非严格单调 | 逐步增强旋律使用能力，同时保留对缺失旋律的鲁棒性 |
| full-mask 后 | top-1 保留，top-2/top-3/top-4 随机 mask 和 shuffle | 防止模型依赖完整多音高信息做重建 |
| 推理阶段 | 不做训练遮罩 | 使用完整旋律控制 |

这个机制是论文能同时保留 text-to-music generation 和 melody editing ability 的关键。

### 4.5 CFG 与推理

论文推理设置：

- sampler：DPM-Solver++
- steps：250
- CFG scale：7
- CFG 只作用于 global text prompt control
- melody control 保持开启，不作为无条件分支被 drop 掉

这意味着推理时不能把旋律控制像文本 prompt 一样在 unconditional branch 中置空，否则会弱化旋律控制或引入分支不一致。当前仓库的 `ControlNetContinuousTransformer` 已经处理 CFG batch 变成 `2B` 时的 control batch 对齐，但还需要在正式推理脚本中明确“文本 CFG、旋律常开”的策略。

## 5. 实验设置与结果

### 5.1 数据

论文训练数据来自四个公开音乐数据集：

- MTG
- FMA
- MTT
- WikiMuTe

处理流程：

1. 目标是 instrumental music，因此用 PANNs tagger 过滤掉包含 vocal 的片段。
2. 因为很多数据集只有 tags 而没有自由文本描述，用 SALMONN-13B 为音频生成多个 captions。
3. 最终得到 59,955 条 recordings，总计 2,239.7 小时高质量 text-music paired data。
4. 评估使用 Song Describer dataset 的 no-singing subset。

### 5.2 模型与训练

论文使用 StableAudio Open checkpoint，其中包含 DiT 和 autoencoder。ControlNet 克隆半个预训练 DiT，也就是 24 层中的 12 层。

训练参数：

| 项 | 论文设置 |
| --- | --- |
| audio sample rate | 44.1 kHz |
| audio channels | stereo |
| CQT hop length | 512 |
| CQT bins | 128 |
| fmin | MIDI note 0, 8.18 Hz |
| objective | v-objective |
| optimizer | AdamW |
| learning rate | `5e-5` |
| scheduler | InverseLR, power `0.5` |
| frozen | DiT 主干 |
| trainable | ControlNet + melody prompt extraction/latent conversion |
| hardware | 4 x V100 |
| batch size | 每 GPU 8 |
| training time | 约 3 天 |

### 5.3 评测任务与指标

论文评测两个任务：

- Text-to-music：只给文本 prompt 生成音乐。
- Music editing：给目标音频提取 melody prompt，再结合文本 prompt 生成音乐。

客观指标：

| 指标 | 用途 |
| --- | --- |
| Melody accuracy | 评估生成音频 pitch class 与输入 melody control 的 frame-wise 对齐 |
| FDopenl3 | 评估生成音频与真实参考分布差异 |
| KLpasst | 评估音频语义/标签分布差异 |
| CLAP score | 评估生成音频与文本 prompt 的一致性 |

主观指标：

| 指标 | 用途 |
| --- | --- |
| TF | text fit |
| OVL | overall quality |

### 5.4 主要结果

论文与 MusicGen-stereo-melody、MusicGen-stereo-melody-large 对比。核心结果：

- 论文模型在 music editing 中 melody accuracy 达到 `56.6%`，高于 MusicGen-stereo-melody 的 `42.3%` 和 large 的 `44.7%`。
- Text-to-music 虽比原 StableAudio Open 有一定退化，但仍明显优于 MusicGen baseline。
- 主观 MOS 中，论文模型在 text-to-music 和 music editing 两个任务上的 text fit 与 overall quality 均优于 MusicGen baseline。

消融结论：

- 用 cross-attention 注入旋律虽然更符合 Transformer 直觉，生成质量也可能不错，但 melody accuracy 很低，几乎退化成纯 text-to-music。
- 不用 masking strategy 时，editing 指标可能好，但模型更像在做目标音频重建，text-to-music 能力会变差。
- 最终模型的关键是平衡：既能保留文本生成能力，又能完成旋律控制编辑。

## 6. 当前仓库已完成的点

### 6.1 ControlNet-DiT 注入主链路已完成

[ControlNetContinuousTransformer](stable_audio_control/models/control_transformer.py) 已经实现：

- 校验 base transformer 有 `.layers`。
- 深拷贝前 `num_control_layers` 个 block 到 `control_layers`。
- 为每个 control layer 创建零初始化 `zero_linears`。
- 提供 `freeze_base()` 冻结预训练主干。
- 支持 `control_input=None` 时退回原 base 行为。
- 支持 CFG batch 扩展时重复 control input。
- 支持 control input 长度不一致时插值对齐。
- 在每个控制层后执行 `x_base = x_base + zero_linear_i(x_ctrl)`。

这部分基本对应论文的 ControlNet-Transformer 核心机制。

### 6.2 StableAudio wrapper 接入已完成

[ControlConditionedDiffusionWrapper](stable_audio_control/models/control_dit.py) 已经实现：

- 保持 StableAudio 原训练/推理调用风格。
- 从 `cond["melody_control"]` 中取旋律控制。
- 支持 `[B, C, L]`、`[B, L, C]`、`[C, L]` 输入。
- 插值到当前 latent 长度。
- 转成 `[B, L, C]` 后用 `LazyLinear` 投影到 transformer `dim_in`。
- 调用 base wrapper 时透传 `control_input` 和 `control_scale`。
- 从 `cond_for_base` 移除 `melody_control`，避免它进入原始 condition routing。

### 6.3 top-k CQT 提取已完成

[CQTTopKExtractor](stable_audio_control/melody/cqt_topk.py) 已经实现论文主要设定：

- stereo input: `[2, T]` 或 `[B, 2, T]`
- high-pass biquad, cutoff `261.2 Hz`
- CQT 后端：优先 `nnAudio`，否则 `librosa`
- 128 bins, 12 bins per octave, fmin MIDI note 0
- 每声道每帧 top-4
- index 从 `1..128`
- `0` 保留给 mask/pad
- 左右声道交错输出 `[L0, R0, L1, R1, ...]`

当前环境里 `librosa` 和 `torchaudio` 可用，`nnAudio` 不可用，因此实际会走 CPU fallback。

### 6.4 训练脚本雏形已完成

[train_controlnet_dit.py](stable_audio_control/scripts/train_controlnet_dit.py) 已经覆盖：

- 加载 `stabilityai/stable-audio-open-1.0`。
- 构建 ControlNet wrapper。
- 创建 CQT extractor。
- 用 `MelodyControlAugmenter` 在线从 batch waveform 提取 melody control。
- 包装 `DiffusionCondTrainingWrapper`，在 training/validation step 前设置 batch audio。
- 显式冻结非控制参数。
- 只放开 `control_layers`、`zero_linears`、`control_projector`。
- 构造 Lightning Trainer。

### 6.5 smoke 验证已通过

[train_control_smoke_report_2026-04-21.zh-CN.md](../controlnet/smoke_report_2026-04-21.md) 记录：

- `smoke_control_injection_stableaudio_open-1.py`: `diff norm: 0.0`
- `smoke_control_dit_wrapper.py`: `zero-init diff norm: 0.0`
- 微扰 zero-linear 后：`after-perturb diff norm: 1.989155650138855`
- 单步训练：`loss_is_finite: True`
- 冻结参数无梯度：`frozen_with_grad_count: 0`
- 结论：Go，可进入正式训练准备

这说明当前工程接线、zero-init 行为、冻结策略和最小训练闭环都已经验证过。

## 7. 当前仓库尚未完成的点

### 7.1 Progressive curriculum masking 未实现

计划文件要求新增：

- `stable_audio_control/melody/masking.py`
- `tests/test_masking.py`

当前这两个文件不存在。后续需要实现：

- frame-wise masking
- pitch-wise masking
- 初期 full-mask phase
- mask ratio 随训练推进逐步降低但保留随机性
- top-1 保留
- top-2/top-3/top-4 随机遮罩和 shuffle
- 可复现的 `torch.Generator` 或 seed
- 训练启用、推理关闭

这是论文平衡文本和旋律控制的关键机制，应优先补齐。

### 7.2 论文版 melody conditioner 未实现

计划文件要求新增：

- `stable_audio_control/melody/conditioner.py`

当前还没有 `Embedding + Conv1D downsample` 结构。现有 `control_projector` 是线性投影，能跑通控制链路，但还不完全等价于论文的 pitch-specific embedding 机制。

后续建议实现：

```text
input: LongTensor [B, 8, F], values 0..128
  -> nn.Embedding(129, E, padding_idx=0)
  -> reshape / merge top-k and channel dimensions
  -> Conv1D stack
  -> interpolate or stride-align to target latent length
  -> output [B, C_control, L_latent] or [B, L_latent, dim_in]
```

### 7.3 正式推理入口未实现

计划文件要求新增：

- `stable_audio_control/scripts/generate_melody_edit.py`

当前还没有面向 melody + text editing 的 CLI。后续需要支持：

- `--prompt`
- `--melody-wav`
- `--seconds-total`
- `--seed`
- `--steps`
- `--cfg-scale`
- `--control-scale`
- checkpoint 加载
- 输出 wav 保存
- CFG 只作用于文本，旋律控制保持开启

### 7.4 Dataset metadata / 缓存链路未实现

计划文件要求新增：

- `stable_audio_control/data/custom_metadata.py`
- 可选 dataset config

当前训练脚本在线从 batch waveform 提取 CQT。这个方式适合 smoke 或小规模验证，但正式训练会有性能压力，尤其在没有 `nnAudio` 的情况下会走 `librosa` CPU 路径。

后续建议：

- 支持 custom metadata hook 从裁剪后的 audio 生成 melody control。
- 支持离线缓存 top-k CQT。
- 明确缓存与 random crop、seconds_start、sample_rate 的关系。
- 避免 pre-encoded latent 模式下丢失 waveform 导致无法提取 melody。

### 7.5 训练配置还未完全对齐论文

论文设置是 ControlNet 克隆 12 层，即 24 层 DiT 的一半。当前训练脚本默认：

```text
--num-control-layers 2
```

这更像省资源 smoke 默认。正式复现实验应改为：

```text
--num-control-layers 12
```

还需要确认：

- AdamW 是否按论文设置使用。
- learning rate 是否为 `5e-5`。
- InverseLR scheduler `power=0.5` 是否真实配置。
- batch size、梯度累积、precision 与硬件能力是否匹配。
- EMA 是否与训练/导出流程兼容。

### 7.6 评测与消融未实现

论文评估和消融目前还没有仓库脚本对应。后续需要补：

- melody accuracy
- FDopenl3
- KLpasst
- CLAP score
- subjective MOS 流程或至少本地试听评测表
- MusicGen baseline 对比
- cross-attention injection ablation
- without masking strategy ablation
- text-to-music 和 music editing 双任务评估

### 7.7 测试覆盖不足

当前只有 [tests/test_train_controlnet_dit.py](tests/test_train_controlnet_dit.py)，主要检查训练脚本参数默认值。后续建议新增：

- `tests/test_cqt_topk.py`
- `tests/test_masking.py`
- `tests/test_melody_conditioner.py`
- `tests/test_control_transformer_shapes.py`
- `tests/test_control_dit_wrapper.py`

重点覆盖：

- CQT 输出 shape、dtype、值域 `1..128`
- mask 后是否出现 `0`
- top-1 是否保留
- control input batch/length/dtype/device 对齐
- CFG batch `B -> 2B` 是否正确
- zero-init 输出差异是否为 0
- 扰动 zero-linear 后控制路径是否非零
- 冻结参数是否无梯度

## 8. 建议推进顺序

### 阶段 1：补齐论文关键训练机制

1. 新增 `stable_audio_control/melody/masking.py`。
2. 新增 `tests/test_masking.py`。
3. 新增 `stable_audio_control/melody/conditioner.py`。
4. 将 `train_controlnet_dit.py` 的 `MelodyControlAugmenter` 改为使用 conditioner + masking。
5. 确保 masking 只在训练时启用，验证/推理关闭。

### 阶段 2：加强训练可用性

1. 明确正式训练配置，尤其 `num_control_layers=12`。
2. 对齐 AdamW、`lr=5e-5`、InverseLR `power=0.5`。
3. 支持 `nnAudio` 或 CQT 缓存。
4. 增加训练日志中的 trainable 参数统计、control scale、mask ratio。
5. 小数据集跑短步数，确认 loss、梯度和导出正常。

### 阶段 3：补推理闭环

1. 新增 `generate_melody_edit.py`。
2. 输入 melody wav，提取 top-k CQT。
3. 加载训练 checkpoint。
4. 用文本 prompt + melody control 生成音频。
5. 确保 CFG 只作用于文本。
6. 保存 output wav 和可选中间控制图。

### 阶段 4：补评测与消融

1. 实现 melody accuracy。
2. 接 CLAP score。
3. 视资源补 FDopenl3 / KLpasst。
4. 准备 MusicGen baseline。
5. 做三组消融：ours、without masking、cross-attention injection。
6. 整理评测报告。

## 9. 简化版任务清单

| 优先级 | 任务 | 状态 | 说明 |
| --- | --- | --- | --- |
| P0 | 读论文并抽取内容 | 已完成 | 文本已抽取到 `tmp/pdfs/editing_music_controlnet_dit_pypdf.txt` |
| P0 | ControlNet-DiT 注入 | 已完成 | `control_transformer.py` 已实现 |
| P0 | wrapper 接入 StableAudio | 已完成 | `control_dit.py` 已实现 |
| P0 | top-k CQT 提取 | 已完成 | `cqt_topk.py` 已实现 |
| P0 | 最小训练 smoke | 已完成 | 报告结论为 Go |
| P0 | progressive masking | 未完成 | 论文关键机制，建议下一步优先做 |
| P0 | melody embedding + Conv conditioner | 未完成 | 当前只有 LazyLinear projector |
| P1 | 正式推理 CLI | 未完成 | 需要 `generate_melody_edit.py` |
| P1 | dataset metadata / CQT 缓存 | 未完成 | 正式训练性能需要 |
| P1 | 训练配置对齐论文 | 部分完成 | 默认 control layers 仍是 2，正式应设 12 |
| P2 | 评测指标与 baseline | 未完成 | 论文结果复现必需 |
| P2 | 消融实验 | 未完成 | masking/cross-attention 结论需要验证 |
| P2 | 测试覆盖 | 部分完成 | 当前只有训练脚本参数测试 |

## 10. 最短技术路线

如果目标是尽快把仓库推进到“接近论文实现”的状态，建议按下面路线做：

```text
当前已完成：
StableAudio Open
  + ControlNetContinuousTransformer
  + ControlConditionedDiffusionWrapper
  + top-k CQT extractor
  + smoke train

下一步：
top-k CQT index [B, 8, F]
  -> progressive masking
  -> pitch embedding + Conv1D conditioner
  -> control_input [B, L, dim_in]
  -> 12-layer ControlNet branch
  -> formal training
  -> melody + text inference CLI
  -> evaluation
```

一句话总结：当前仓库已经把 ControlNet-DiT 的“骨架”和“接线”搭起来了；真正决定论文效果的下一块，是把旋律提示从简单线性投影升级为“top-k CQT + 课程遮罩 + embedding/Conv latent conditioner”，并补齐推理和评测闭环。
