# 论文提炼：Editing Music with Melody and Text

更新日期：2026-06-03

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

## 7. 当前仓库仍需补齐的论文关键点

本节已按 2026-06-02 的本地代码状态更新。旧版文档中部分“未实现”判断已经过时：`conditioner.py` 已存在，`--num-control-layers` 默认已是 12，训练长度修正和 demo sweep 已经完成，离线 batch 推理脚本也已经支持 control variants。

### 7.1 Progressive curriculum masking 已实现

当前仓库已新增 `stable_audio_control/melody/masking.py`，并把训练期 progressive curriculum masking 接入 `MelodyControlAugmenter`。已覆盖：

- early full-mask phase
- frame-wise mask ratio schedule
- full-mask 后 top-1 保留
- top-2/top-3/top-4 随机遮罩和 shuffle
- 训练启用，验证/推理关闭

这是论文平衡文本控制和旋律控制的关键机制。后续正式训练应使用 `--melody-mask true`，并在日志里确认 full-mask steps、schedule steps 与 mask ratios。

### 7.2 CQT padding mask 归零已实现

`CQTTopKExtractor` 会把真实 pitch index 变成 `1..128`，`0` 只保留给 mask/pad。问题是静音或 padding 区域并不会天然输出 `0`，top-k 仍可能选出低频伪 pitch token。

当前训练链路已把 data utils 里的 waveform-level `padding_mask` 对齐到 CQT 帧率，并在进入 `MelodyControlEncoder` 前把 padded frames 置为 `0`。这比继续调 control scale 更基础，因为它直接决定模型看到的是“无旋律”还是“伪旋律”。

### 7.3 MelodyControlEncoder 已实现，但需要 masked/padded 输入测试

旧判断“论文版 melody conditioner 未实现”已经不准确。当前 [stable_audio_control/melody/conditioner.py](stable_audio_control/melody/conditioner.py) 已经实现：

```text
top-k CQT LongTensor [B, 8, F], values 0..128
  -> nn.Embedding(129, E, padding_idx=0)
  -> merge channel and embedding dimensions
  -> Conv1D stack
  -> interpolate to target latent length
  -> output [B, L, dim_in]
```

剩余风险不是“没有 conditioner”，而是需要补测试和训练链路验证：

- `padding_idx=0` 在 full-mask / padding-mask 输入下是否稳定。
- CQT padded frames 是否真的先被置为 `0`。
- dtype/device/target length 对齐是否覆盖。
- 完全空旋律输入是否不会产生异常或伪控制。

### 7.4 推理入口已存在，EMA/CFG 诊断已补齐

旧判断“正式推理入口未实现”需要修正。当前已有：

- `stable_audio_control/scripts/batch_generate_control.py`
- `stable_audio_control/scripts/compare_controlnet_generation.py`
- `stable_audio_control/scripts/compare_controlnet_generation_random.py`

真正需要优先处理的是训练 demo 与离线推理是否走同一类权重路径。训练 demo 使用 online `ControlConditionedDiffusionWrapper`；离线脚本默认 `prefer_ema=True`，可能出现 online melody/control routing 加 EMA DiT/ControlNet 权重的 hybrid。`batch_generate_control.py` 现在会打印 EMA overlay 诊断、`ema_missing_keys` / `ema_unexpected_keys` 数量，并在使用 EMA 时提示第一步应跑 `--no-prefer-ema`，而不是直接怀疑训练完全失败。

CFG 策略也已在离线 batch inference 日志中明确：文本 CFG 可以走 conditional/unconditional 分支，但 melody control 应保持常开；显式关闭控制应使用 `control_scale=0`。

### 7.5 Dataset metadata / CQT 缓存仍是中期工程项

当前训练脚本在线从 batch waveform 提取 CQT。这个方式适合 smoke 或小规模验证，但正式训练会有性能压力，尤其在没有 `nnAudio` 的情况下会走 `librosa` CPU 路径。

后续建议：

- 支持 custom metadata hook 从裁剪后的 audio 生成 melody control。
- 支持离线缓存 top-k CQT。
- 明确缓存与 random crop、seconds_start、sample_rate 的关系。
- 避免 pre-encoded latent 模式下丢失 waveform 导致无法提取 melody。

这项重要，但不应排在 progressive masking 和 padding mask 前面。

### 7.6 训练配置大体对齐，剩余是 scheduler/EMA/日志细节

论文设置是 ControlNet 克隆 12 层，即 24 层 DiT 的一半。当前训练脚本默认已经是：

```text
--num-control-layers 12
```

训练长度也已经支持 `--seconds-total 10`，避免 10 秒数据被 47.55 秒 StableAudio 默认 sample size 污染。

剩余需要确认：

- AdamW / InverseLR 是否完全按论文配置生效。
- learning rate 默认和正式训练命令是否使用 `5e-5`。
- EMA 是否与训练 demo / 离线推理一致。
- 日志是否记录 effective sample size、mask ratio、control variants、EMA 使用状态。

### 7.7 评测与消融已有最小闭环，重指标后移

论文完整评估包括 melody accuracy、FDopenl3、KLpasst、CLAP、主观 MOS。Phase 1 已补两条轻量评估路径：训练 demo 生成时即时写 `demo_melody_similarity.csv`，以及事后用 `evaluate_control_variants.py` 扫描 `correct/shuffled/zero` 生成结果并输出 `melody_control_report.csv`。它们能回答：

- `correct` 是否比 `shuffled` / `zero` 更接近参考旋律。
- `control_scale` 增大是否提高旋律跟随，而不是只让输出变平。
- `--prefer-ema` 与 `--no-prefer-ema` 的离线推理是否行为不同。

当前轻量指标基于 CQT top-k overlap 与 top-1 pitch accuracy，不声称复现完整论文评估。重指标和 MusicGen baseline 可以放到 Phase 3。

最小用法示例：

训练 demo 即时评分默认开启；如需显式指定 CSV：

```bash
python3 stable_audio_control/scripts/train_controlnet_dit.py \
  ... \
  --demo-every 8000 \
  --demo-melody-similarity true \
  --demo-melody-similarity-csv demo_melody_similarity.csv
```

训练时会在 `<default-root-dir>/demo_melody_similarity.csv` 追加每个 demo 的 `cfg_scale`、`control_scale`、`variant`、`cqt_top1_score`、`cqt_topk_score` 等字段。`zero` variant 没有有效参考旋律，会被记录为 skip，不参与 top-1 判断。

事后扫描已生成 WAV：

```bash
python3 stable_audio_control/scripts/evaluate_control_variants.py \
  --reference-audio audios/Then.mp3 \
  --generated-dir outputs/debug_infer_no_ema \
  --seconds-total 10 \
  --melody-feature cqt \
  --cqt-backend auto \
  --output-csv outputs/debug_infer_no_ema/melody_control_report.csv
```

判读方式：如果同一 seed / control_scale 下 `correct` 的 top-1 accuracy 明显高于 `shuffled` 和 `zero`，说明 melody control 至少在推理输出中留下了可测信号；如果三者接近，优先检查 checkpoint、EMA 路径、control scale 与 demo/inference 是否同配置。

## 8. 当前建议推进顺序

### Phase 1：论文关键训练机制补齐（最高优先级）

1. 已新增 `stable_audio_control/melody/masking.py`。
2. 已新增 `tests/test_melody_masking.py`。
3. 已实现 progressive curriculum masking。
4. 已实现 CQT padding mask 归零。
5. 已将 masking/padding zeroing 接入 `MelodyControlAugmenter`，并确保只在训练启用课程遮罩。
6. 已补 inference EMA/CFG 诊断说明和测试。
7. 已补轻量 melody-control evaluation gate。

### Phase 2：正式训练验证 / ControlNet 训练恢复

1. 容器同步代码，确认 `--seconds-total`、`--demo-control-variants`、`--demo-prompt`、`--melody-mask` 都存在。
2. 跑 `--seconds-total 10 --max-steps 1 --demo-every 0` smoke。
3. 跑 2000-4000 step fresh training。
4. 听并评估 `correct/shuffled/zero` demo。
5. 若训练 demo 有旋律但离线推理没有，优先跑 `--no-prefer-ema`。

### Phase 3：论文级评测与实验管理

1. 扩展 melody accuracy。
2. 接 CLAP score。
3. 视资源补 FDopenl3 / KLpasst。
4. 准备 MusicGen baseline。
5. 做 masking / no masking / cross-attention 或其他结构消融。
6. 整理评测报告和实验追踪。

### Phase 4：工程整理与新特征探索

1. 整理硬编码和配置入口。
2. 支持 CQT 缓存。
3. 新增 pitch contour / HPSS 等 melody control 特征。

## 9. 简化版任务清单

| 优先级 | 任务 | 状态 | 说明 |
| --- | --- | --- | --- |
| P0 | 读论文并抽取内容 | 已完成 | 文本已抽取到 `tmp/pdfs/editing_music_controlnet_dit_pypdf.txt` |
| P0 | ControlNet-DiT 注入 | 已完成 | `control_transformer.py` 已实现 |
| P0 | wrapper 接入 StableAudio | 已完成 | `control_dit.py` 已实现 |
| P0 | top-k CQT 提取 | 已完成 | `cqt_topk.py` 已实现 |
| P0 | melody embedding + Conv conditioner | 已完成基础实现 | `conditioner.py` 已存在，剩余 masked/padded 输入测试 |
| P0 | 12 层 ControlNet 默认 | 已完成 | `--num-control-layers` 默认已是 12 |
| P0 | 训练长度修正 | 已完成 | `--seconds-total` / `--sample-size` |
| P0 | demo control sweep | 已完成 | `correct/shuffled/zero` + control scales + fixed prompt |
| P0 | batch inference variants | 已完成 | `--demo-control-variants "correct/zero/shuffle"` |
| P0 | progressive masking | 已完成 | Phase 1 第一优先级 |
| P0 | CQT padding mask 归零 | 已完成 | Phase 1 第一优先级 |
| P0 | EMA/CFG 推理策略诊断 | 已完成 | 解释 demo 好但离线推理差的优先排查项 |
| P1 | 轻量 melody-control evaluation | 已完成 | 用于 Phase 2 训练判定 |
| P2 | CQT 缓存 / dataset metadata | 未完成 | 性能与规模化训练需要 |
| P2 | 论文重评测与 baseline | 未完成 | CLAP/FD/KL/MusicGen |

## 10. 最短技术路线

如果目标是尽快把仓库推进到“更接近论文且能解释训练失败”的状态，建议按下面路线做：

```text
当前已完成：
StableAudio Open
  + ControlNetContinuousTransformer
  + ControlConditionedDiffusionWrapper
  + top-k CQT extractor
  + MelodyControlEncoder
  + 12-layer ControlNet default
  + 10-second effective training length
  + correct/shuffled/zero demo diagnostics

Phase 1 立即补：
top-k CQT index [B, 8, F]
  -> padding_mask 对齐到 CQT 帧并置 0
  -> progressive curriculum masking
  -> MelodyControlEncoder
  -> control_input [B, L, dim_in]
  -> ControlNet branch
  -> training demo / offline inference EMA diagnostic
  -> lightweight melody-control metric

Phase 2 再训练：
fresh 10s training
  -> correct/shuffled/zero demo
  -> no-EMA offline inference comparison
  -> decide whether需要更大架构 ablation
```

一句话总结：当前仓库已经不只是“骨架和接线”，而是已经具备了可训练链路、论文式离散 melody conditioner、训练长度修正、demo 消融、progressive masking、padding-safe CQT、EMA/CFG 推理诊断和轻量 melody-control 评估。现在 Phase 1 剩余重点是补强 `MelodyControlEncoder` 在全 0 mask、padding 与静音输入下的边界测试；随后进入 fresh 10s diagnostic training，用 `correct/shuffled/zero` 与 `--no-prefer-ema` 对照判断控制分支是否真的工作。
