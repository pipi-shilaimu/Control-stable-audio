# StableAudio 概念—代码对照表（基础为主，源码为辅）

目标：用“能复述的一句话 + 直觉类比 + 典型张量形状”先把概念立住；需要时再用“源码入口”做验证。

> 约定符号：`B`=batch，`C`=channels（音频声道或特征通道），`T`=waveform 采样点维度，`L`=latent 时间长度，`S`=token 序列长度，`D`=embedding 维度。  
> 本仓库的说明文档在 `docs/`；真实实现代码主要在 `.venv/Lib/site-packages/stable_audio_tools/`（安装到 venv 的包源码）。

---

## 0) 你可以怎么用这份表

- 先从 **A. Pretransform/Latent** 看起（你当前正打开 `docs/pretransforms.md:1`）。
- 遇到术语不熟：直接在本页 `Ctrl+F` 搜索中英文关键字。
- 想验证“我理解对不对”：点同一行的“源码入口”，看最小实现（通常 1 个函数/类）。

---

## 1) 总览：StableAudio Open 的端到端 pipeline（文字版）

1. 数据集加载 waveform（裁剪/补零/生成 mask）  
2. `pretransform.encode()`：waveform → latent sequence  
3. 扩散模型（DiT）在 latent 上做去噪（带 conditioning）  
4. `pretransform.decode()`：latent → waveform  

结构上可以从 `模型结构_易读.md:13` 开始看：顶层 `DiTWrapper`，并挂载 `conditioner` 与 `pretransform`。

---

## A) Pretransform / Autoencoder / Latent（15 条）

| # | 概念（中 / 英） | 一句话定义（能复述） | 直觉/类比（1–2 句） | 常见形状 | 入门读物（docs） | 源码入口（最短路径） | 常见坑点（1 条） |
|---:|---|---|---|---|---|---|---|
| 1 | 预变换 / Pretransform | 把 waveform 可逆地映射到更短的 latent 表示（encode/decode） | 像“音频编解码器/Tokenizer”，让后面的模型在更短序列上工作 | `encode: [B,2,T]→[B,C_latent,L]` | `docs/pretransforms.md:1` | `stable_audio_tools.models.diffusion.DiffusionModelWrapper` 会挂载 `pretransform`：`.venv/Lib/site-packages/stable_audio_tools/models/diffusion.py:34` | latent 的“时间长度 L”不是秒数，也不是谱图帧数 |
| 2 | Autoencoder pretransform | 用预训练 autoencoder 做 pretransform | 先压缩再生成/编辑，生成模型更省显存和算力 | 同 #1 | `docs/pretransforms.md:8` | autoencoder 文档入口：`docs/autoencoders.md:1` | 压缩带来信息瓶颈，极高频细节可能不可逆 |
| 3 | Autoencoder（自编码器） | encoder 压缩，decoder 还原 | 目标是“可压缩但尽量可还原”的表示空间 | `encoder: [B,2,T]→[B,D,L]` | `docs/autoencoders.md:1` |（概念为主）`stable_audio_tools.models.autoencoders`（包内） | 不要把 autoencoder 当“扩散模型的一部分”；它通常是冻结的 pretransform |
| 4 | Latent / latent sequence | waveform 的压缩序列表示 | 更短的时间维 + 更多的通道维，适合 Transformer 建模 | `[B,C_latent,L]` | `docs/autoencoders.md:4` | 训练 wrapper 会在前向里调用 encode（条件允许时）：`.venv/Lib/site-packages/stable_audio_tools/training/diffusion.py:360` | latent 的数值分布可能会被缩放（见 latent rescaling） |
| 5 | 下采样比 / downsampling_ratio | waveform 与 latent 长度的固定比例关系 | “每个 latent 步代表多少 waveform 采样点” | `L ≈ T / ratio` | `docs/autoencoders.md:21` |（读配置概念即可） | 不同 checkpoint 可能 ratio 不同，做“时间对齐”前要先确认 |
| 6 | Latent rescaling | 对 latent 做缩放以匹配训练分布 | 类似把 latent 标准化到更稳定的数值范围 | 仍是 `[B,C,L]` | `docs/pretransforms.md:27` |（读 docs 足够） | 预编码（pre_encoded）时若跳过 encode，仍可能要手动应用同样缩放 |
| 7 | VAE bottleneck | 在 latent 空间引入概率采样（均值/方差） | 让 latent 更“高斯化”，扩散更容易学 | `encode` 额外返回 KL 等信息 | `docs/autoencoders.md:234` |（概念为主） | VAE 会牺牲部分重建质量换来“latent 更规则” |
| 8 | `io_channels`（扩散 I/O 通道） | 扩散模型输入/输出的基础通道数 | 在 latent 空间里，通道数不等于音频声道数 | `x: [B,io_channels,L]` | `docs/diffusion.md:17` | `DiffusionTransformer(io_channels=...)`：`.venv/Lib/site-packages/stable_audio_tools/models/dit.py:12` | 把 `io_channels` 当成“立体声 2 通道”是常见误会 |
| 9 | `sample_rate` | waveform 每秒采样点数 | 决定所有时域/频域操作的“时间尺度” | `sr=44100` 等 |（论文设定常用） | dataset 会强制重采样：`.venv/Lib/site-packages/stable_audio_tools/data/dataset.py:803`（create_dataloader_from_config 传入 sample_rate） | 训练/推理 sample_rate 不一致会导致控制信号（如 CQT）失配 |
|10| `sample_size` | 训练/推理时模型看到的固定样本长度（采样点数） | 近似等同于“每次训练的秒数窗口” | `T = sample_size` | `docs/datasets.md:4`（结合 dataset config） | dataloader 构造需要 sample_size：`.venv/Lib/site-packages/stable_audio_tools/data/dataset.py:803` | sample_size 太长会显著增加内存；太短会限制长程结构学习 |
|11| Padding / padding_mask | 短音频补零并给出 mask | 让 batch 形状统一，同时告诉模型“哪些是有效音频” | `padding_mask: [T]` 或 `[L]` | `docs/datasets.md:45`（pre_encoded 里经常出现） | collation 使用 list 包装，训练时 stack：`.venv/Lib/site-packages/stable_audio_tools/training/diffusion.py:356` | mask 的尺度（waveform vs latent）必须与当前训练输入一致 |
|12| Crop（裁剪） | 长音频裁剪出固定窗口 | 把长音频拆成训练样本片段 | `audio: [2,T]` | `docs/datasets.md:9` | SampleDataset 在 `__getitem__` 里 pad/crop 并产出 info：`.venv/Lib/site-packages/stable_audio_tools/data/dataset.py:205` | pre_encoded + random_crop 时 `seconds_start` 可能不更新（docs 已提示） |
|13| Stereo / 立体声 | 左右两个声道 | 可看作 2 条相关序列（也可能差异很大） | `[B,2,T]` | `docs/autoencoders.md:7`（示例语境） |（数据层）`create_dataloader_from_config(... audio_channels=2 ...)`：`.venv/Lib/site-packages/stable_audio_tools/data/dataset.py:803` | 旋律特征若按声道提取，需要明确 L/R 的处理方式 |
|14| Pre-encoded latents | 把 latents 提前算好并存盘训练 | 以磁盘换算力，训练吞吐更高 | `latents: [B,C,L]` | `docs/pre_encoding.md:1`、`docs/datasets.md:45` | `PreEncodedDataset` 在同文件：`.venv/Lib/site-packages/stable_audio_tools/data/dataset.py:265` | 对 StableAudio Open 这类依赖 `seconds_start` 的模型要谨慎（random_crop 限制） |
|15| Pretransform vs Conditioner | pretransform 是 waveform↔latent 的“信号变换”；conditioner 是 metadata→tensor 的“条件翻译器” | 一个改变输入空间，一个提供控制信号 | pretransform 处理 `[B,2,T]`；conditioner 处理 dict | `docs/pretransforms.md:1`、`docs/conditioning.md:29` | `MultiConditioner`：`.venv/Lib/site-packages/stable_audio_tools/models/conditioners.py:639` | 很多人会把“旋律特征提取”放错层：它既可以在 dataset 层，也可以做成 conditioner |

---

## B) Conditioning / Conditioner（15 条）

| # | 概念（中 / 英） | 一句话定义（能复述） | 直觉/类比（1–2 句） | 常见形状 | 入门读物（docs） | 源码入口（最短路径） | 常见坑点（1 条） |
|---:|---|---|---|---|---|---|---|
|16| Conditioning（条件控制） | 用额外信号控制生成/编辑的方向 | “给模型一张说明书/约束”而不是改变数据本身 | 多种形状并存 | `docs/conditioning.md:1` | 条件路由发生在 wrapper：`.venv/Lib/site-packages/stable_audio_tools/models/diffusion.py:100` | 只看概念不看形状约定，很容易把 tensor 接错维度 |
|17| Conditioner（条件器） | 把人类可读 metadata 转成 `(tensor, mask)` | 相当于“翻译器”：prompt/秒数→向量/序列 | 取决于类型 | `docs/conditioning.md:29` | `Conditioner` 基类：`.venv/Lib/site-packages/stable_audio_tools/models/conditioners.py:15` | 输出 mask 的语义依赖具体 conditioner，别默认都是 1 |
|18| Conditioning ID | metadata dict 的 key（如 `prompt`） | 像字段名：决定哪个 conditioner 来处理 | 字符串 key | `docs/conditioning.md:31` | `MultiConditioner.forward` 逐 key 查找：`.venv/Lib/site-packages/stable_audio_tools/models/conditioners.py:657` | 训练时 metadata 缺 key 会直接报错（除非配置 default_keys） |
|19| MultiConditioner | 多个 conditioner 的容器，批量处理 metadata | 一次性把 `prompt/seconds_*` 都编码出来 | 输出 dict：`id→(tensor,mask)` | `docs/conditioning.md:29` | `class MultiConditioner`：`.venv/Lib/site-packages/stable_audio_tools/models/conditioners.py:639` | 你的 metadata 经过 dataloader collate 后常会多一层 list，需要注意 unwrap 逻辑 |
|20| Cross-attention conditioning | 用 cross-attention 让音频 token “看”文本 token | 像阅读理解：音频 token 查询文本上下文 | `[B,S_text,D]` | `docs/conditioning.md:7` | 路由与拼接：`.venv/Lib/site-packages/stable_audio_tools/models/diffusion.py:145` | mask 可能在某些实现里被临时禁用（要以源码为准） |
|21| Cross-attention mask | 指示哪些 text token 有效 | 处理 padding token / 可变长输入 | `[B,S_text]` | `docs/conditioning.md:10` |（实现依赖 transformer 内核） | 你以为 mask 生效 ≠ 真生效（遇到 flash-attn/flex-attn 兼容问题时会被禁用） |
|22| Global conditioning | 一条向量影响整段序列 | 像“全局风格旋钮” | `[B,D]` | `docs/conditioning.md:12` | wrapper 拼接 global：`.venv/Lib/site-packages/stable_audio_tools/models/diffusion.py:165` | global 有时会被实现成 prepend token（见 DiT） |
|23| Prepend conditioning | 把条件 token 拼在输入 token 前面 | 像“在句首加前缀提示词”，让 self-attn 自己消化 | `[B,S_pre,D]` | `docs/conditioning.md:17` | `ContinuousTransformer` 拼接：`.venv/Lib/site-packages/stable_audio_tools/models/transformer.py:799` | prepend 序列长度增加，会影响注意力计算成本 |
|24| Input concatenation | 在 channel 维拼接、在时间维对齐的条件 | 像给每个时间步多加一组特征通道（很适合旋律/掩码） | `[B,C_cond,L]` | `docs/conditioning.md:24` | DiT 内插值+cat：`.venv/Lib/site-packages/stable_audio_tools/models/dit.py:231`（forward 入口），实现段落见 `.venv/Lib/site-packages/stable_audio_tools/models/dit.py:160`（以文件内为准） | 条件序列长度不匹配时会插值，插值方式可能影响控制精度 |
|25| Conditioning config | 声明有哪些 conditioner、输出维度等 | 像“表单 schema”：决定训练时需要哪些字段 | JSON config | `docs/conditioning.md:44` | 构建函数：`.venv/Lib/site-packages/stable_audio_tools/models/conditioners.py:686` | `cond_dim` 会统一所有 conditioner 输出维度，别忘了这一点 |
|26| `cond_dim` | 统一条件 embedding 的维度 | 让 cross-attn/global 等能共享维度接口 | `D=768` 等 | `docs/conditioning.md:59` | `create_multi_conditioner_from_conditioning_config`：`.venv/Lib/site-packages/stable_audio_tools/models/conditioners.py:695` | 某些 conditioner 本来输出维度不同，会被 project 到 cond_dim |
|27| T5 conditioner | 用冻结 T5 编码文本 prompt | 像把句子变成 token embedding 序列 | `[B,S_text,D]` | `docs/conditioning.md:66` |（外部依赖 transformers；此处先理解概念） | 文本长度受 max_length 限制，过长会截断 |
|28| Number conditioner | 把秒数等连续数值编码成 embedding | 像把标量“秒数”变成可学习向量 | `[B,1,D]` 或 squeeze 后 `[B,D]` | `docs/conditioning.md:141` | `NumberConditioner`：`.venv/Lib/site-packages/stable_audio_tools/models/conditioners.py:46` | 连续数值会被 clamp 到 `[min_val,max_val]`，越界会失真 |
|29| `seconds_start` / `seconds_total` | StableAudio Open 的时间条件 | 告诉模型“从哪里开始、总共多长”，帮助变长生成 | 标量→embedding | `docs/conditioning.md:141` | dataset 里会写入：`.venv/Lib/site-packages/stable_audio_tools/data/dataset.py:230` | pre_encoded + random_crop 场景下 `seconds_start` 可能不准（docs 警告） |
|30| Condition routing（条件路由） | 把多个 id 按类型拼成模型所需输入 | 像把“表单字段”汇总成 4 类入口：cross/global/prepend/input | 多种 | `docs/diffusion.md:20` | `get_conditioning_inputs`：`.venv/Lib/site-packages/stable_audio_tools/models/diffusion.py:137` | 注意 cat 的维度：cross/prepend 在序列维 cat，global/input-concat 在通道维 cat |

---

## C) Transformer / Attention（15 条）

| # | 概念（中 / 英） | 一句话定义（能复述） | 直觉/类比（1–2 句） | 常见形状 | 入门读物（docs） | 源码入口（最短路径） | 常见坑点（1 条） |
|---:|---|---|---|---|---|---|---|
|31| Token / 序列建模 | 把时间序列变成 token 序列做注意力 | 像把音频切成一串“字/词”去理解长程关系 | `x: [B,S,D]` | `docs/diffusion.md:132` | `ContinuousTransformer.forward`：`.venv/Lib/site-packages/stable_audio_tools/models/transformer.py:796` | 不要把 token 等同于“音符”；这里 token 是连续 latent patch |
|32| Self-attention | 序列内部的信息聚合 | 每个 token 都能“看”全序列（或局部窗口） | `Q,K,V: [B,S,D]` |（概念） | TransformerBlock 内调用 self-attn：`.venv/Lib/site-packages/stable_audio_tools/models/transformer.py:703` | 注意力成本 ~ `S^2`，序列长会爆显存 |
|33| Cross-attention | 主序列对条件序列做注意力 | “音频 token 去查询文本 token” | `x:[B,S,D] ctx:[B,S_ctx,D_ctx]` | `docs/conditioning.md:7` | `if context is not None ...`：`.venv/Lib/site-packages/stable_audio_tools/models/transformer.py:688` | ctx 的维度/长度与主序列不同是正常的，别强行对齐长度 |
|34| `context`（cross-attn 输入） | cross-attn 的条件序列张量 | 就是“文本特征序列”这类东西 | `[B,S_ctx,D_ctx]` | `docs/conditioning.md:10` | `context = None` 参数：`.venv/Lib/site-packages/stable_audio_tools/models/transformer.py:448` | context 为空时 cross-attn 不会执行（这对 CFG uncond 分支很关键） |
|35| `prepend_embeds` | 前缀 token 序列 | 像在句首加特殊 token（提示/全局条件） | `[B,S_pre,D]` | `docs/conditioning.md:17` | 拼接发生在：`.venv/Lib/site-packages/stable_audio_tools/models/transformer.py:817` | prepend 会改变 token 索引位置，调试对齐时要记住它存在 |
|36| Positional encoding | 给 token 加位置信息 | 让模型知道“先后顺序” | `[B,S,D]` 加到 x 上 |（概念） | `RotaryEmbedding` 入口：`.venv/Lib/site-packages/stable_audio_tools/models/transformer.py:750` | 位置编码选择会影响长序列泛化能力（但先理解概念即可） |
|37| RoPE / RotaryEmbedding | 旋转位置编码 | 一种把位置信息混进 Q/K 的方式 | 与注意力头相关 |（概念） | `RotaryEmbedding` 类：`.venv/Lib/site-packages/stable_audio_tools/models/transformer.py:84` | 对齐问题常出现在“序列长度变化”与“插值/缩放” |
|38| Attention heads / 多头 | 把注意力拆成多个子空间 | 类似多视角并行观察序列关系 | head 维度 `dim_heads` |（概念） | 注意力里计算 `num_heads`：`.venv/Lib/site-packages/stable_audio_tools/models/transformer.py:455` | head 数太多/太少都是权衡：容量 vs 速度 |
|39| `dim_in` / `project_in` | 把输入特征投影到 transformer 维度 | 像把“原始特征空间”变换到“模型内部语言” | `[B,S,dim_in]→[B,S,D]` | `docs/diffusion.md:132` | `project_in`：`.venv/Lib/site-packages/stable_audio_tools/models/transformer.py:747` | 输入维度不匹配时别硬改数据，应该改投影层或配置 |
|40| LayerNorm | 稳定训练的归一化层 | 类似“每个 token 的特征标准化” | `[B,S,D]` |（概念） | TransformerBlock 里 pre_norm：`模型结构_易读.md:20`（结构视图） | norm 放置位置（Pre-LN vs Post-LN）会影响训练稳定性 |
|41| FeedForward / MLP | 每 token 的非线性变换 | 类似“对每个 token 做更复杂的特征变换” | `[B,S,D]→[B,S,D]` |（概念） | 结构可从 `模型结构_易读.md:39` 看 | FFN 参数量巨大，是模型容量的重要来源 |
|42| Sliding window attention（可选） | 限制注意力只看局部窗口 | 用局部近似换取显存/速度 | 仍是 `[B,S,D]` |（概念） | `sliding_window` 参数：`.venv/Lib/site-packages/stable_audio_tools/models/transformer.py:736` | 窗口太小会丢失长程结构（音乐里很重要） |
|43| Patch / patch_size | 把时间维分块成 patch token | 类似把 1D 序列按块打包，减少 token 数 | `S ≈ L/patch_size` | `docs/diffusion.md:34`（prepend 仅 DiT） | DiT 里 `patch_size`：`.venv/Lib/site-packages/stable_audio_tools/models/dit.py:15` | patch_size>1 会改变时间对齐，做控制信号对齐前要确认 |
|44| Conv1D（卷积） | 在时间维做局部滤波/下采样/上采样 | 像“可学习滤波器组”，擅长局部模式 | `[B,C,T]` |（概念） | DiT 的 preprocess conv：`.venv/Lib/site-packages/stable_audio_tools/models/dit.py:120`（同文件） | Conv stride 会改变长度，和 CQT/latent 对齐密切相关 |
|45| Downsample / Upsample | 改变序列分辨率 | 多尺度处理，省算力/提表达 | 长度变化 |（概念） | 模块集合在 blocks：`.venv/Lib/site-packages/stable_audio_tools/models/diffusion.py:9`（imports） | 任何下采样都会引入对齐问题；记住你要对齐的是“模型实际输入长度” |

---

## D) Diffusion / Training / Data Pipeline（15 条）

| # | 概念（中 / 英） | 一句话定义（能复述） | 直觉/类比（1–2 句） | 常见形状 | 入门读物（docs） | 源码入口（最短路径） | 常见坑点（1 条） |
|---:|---|---|---|---|---|---|---|
|46| Diffusion（扩散/去噪） | 学习把噪声逐步还原为数据 | “从随机噪声雕刻出音频” | `x_t: [B,C,L]` | `docs/diffusion.md:1` | 训练 wrapper：`.venv/Lib/site-packages/stable_audio_tools/training/diffusion.py:214` | 别把 diffusion 当成单步网络，它是“采样过程+网络”的系统 |
|47| Timestep `t` | 噪声强度/扩散阶段的连续标量 | 像“进度条”，决定当前该去噪多少 | `[B]` |（概念） | DiT forward 参数：`.venv/Lib/site-packages/stable_audio_tools/models/dit.py:231` | `t` 的分布（如何采样）会影响训练稳定性 |
|48| Noise schedule（alpha/sigma） | 把 `t` 映射到噪声与信号的权重 | 控制噪声注入强度 | `alpha(t),sigma(t)` |（概念） | v-objective 里：`.venv/Lib/site-packages/stable_audio_tools/models/dit.py:12`（类） | 不同 objective/schedule 不可混用，需要全链路一致 |
|49| v-objective | 预测 v（信号/噪声的组合）作为训练目标 | 常被认为更稳定、更适合高质量采样 | 输出仍是 `[B,C,L]` |（论文背景） | `DiffusionTransformer` 内 objective 分支：`.venv/Lib/site-packages/stable_audio_tools/models/dit.py:12` | 如果你改 objective，采样器/训练目标也要一起改 |
|50| Loss（MSE） | 预测值与目标的均方误差 | “方向对齐”的基本监督 | 标量 | `docs/diffusion.md:44` | `MSELoss` 被训练 wrapper 使用：`.venv/Lib/site-packages/stable_audio_tools/training/diffusion.py:270` | padding 区域若不 mask，会把静音补零当成学习信号 |
|51| EMA | 维护参数的指数滑动平均用于采样/导出 | 像“更平滑的版本”，采样更稳 | 参数级 | `docs/diffusion.md:49` | `EMA` 使用：`.venv/Lib/site-packages/stable_audio_tools/training/diffusion.py:239` | 训练看 online loss，推理/导出常用 EMA 权重 |
|52| AdamW | 带权重衰减的 Adam | 训练 transformer 常用优化器 | 参数级 | `docs/diffusion.md:69` | `configure_optimizers`：`.venv/Lib/site-packages/stable_audio_tools/training/diffusion.py:318` | weight_decay 不是“L2 正则”的完全等价替代（但先用默认理解即可） |
|53| Scheduler（InverseLR） | 随训练步衰减学习率 | 让后期收敛更稳 | 标量 | `docs/diffusion.md:69` | scheduler 创建：`.venv/Lib/site-packages/stable_audio_tools/training/utils.py`（需要时再查） | 学习率策略与 batch size/数据规模强相关 |
|54| CFG（classifier-free guidance） | 用 cond/uncond 两次前向增强条件强度 | “用差分放大条件影响” | 输出 `[B,C,L]` | `docs/diffusion.md:109`（demo_cfg_scales 相关） | DiT forward CFG 逻辑在同文件：`.venv/Lib/site-packages/stable_audio_tools/models/dit.py:231`（forward 入口） | CFG scale 太大会过拟合 prompt，太小则控制弱 |
|55| Negative prompt / uncond | uncond 分支的条件输入（空/负向） | CFG 的对照组 | 与 conditioning 同形状 |（概念） | wrapper 里 negative 分支构造：`.venv/Lib/site-packages/stable_audio_tools/models/diffusion.py:199` | 有些条件类型（例如 input-concat）不一定支持 negative（要看具体模型实现） |
|56| Sampler steps | 采样迭代步数 | 越多通常越精细但越慢 | 整数 | `docs/diffusion.md:98`（demo_steps） | `stable_audio_tools.inference.generation.generate_diffusion_cond`（外部入口） | 训练步数和采样步数不是一回事，别混淆 |
|57| audio_dir dataset | 从本地目录递归读取音频 | MVP 最容易跑通的方式 | `(audio, info)` | `docs/datasets.md:9` | dataloader 构造：`.venv/Lib/site-packages/stable_audio_tools/data/dataset.py:803` | prompt 等元数据需要通过 custom_metadata_module 补齐 |
|58| WebDataset（wds/s3） | tar shard + json 元数据的数据格式 | 更适合大规模/分布式训练 | `(audio, json)` | `docs/datasets.md:28` | `WebDatasetDataLoader` 在同文件（需要时再看）：`.venv/Lib/site-packages/stable_audio_tools/data/dataset.py:643` | JSON schema 要对齐 conditioner 的 IDs（尤其是 `prompt`） |
|59| Custom metadata module | 用 Python 钩子从 info/audio 生成额外 metadata | 最适合在线提取旋律特征（CQT/top-k） | metadata dict | `docs/datasets.md:70` | 加载与调用：`.venv/Lib/site-packages/stable_audio_tools/data/dataset.py:826`、`.venv/Lib/site-packages/stable_audio_tools/data/dataset.py:242` | Windows 多进程 dataloader 下，模块导入/依赖要更小心（可先 num_workers=0） |
|60| Collation / batch 结构 | 把样本列表打包成 batch（audio tensor + metadata list） | 你训练时拿到的 metadata 往往还是 “list[dict]” | `batch = (reals, metadata)` |（概念） | `collation_fn`：`.venv/Lib/site-packages/stable_audio_tools/data/dataset.py:628`；训练 step 入口：`.venv/Lib/site-packages/stable_audio_tools/training/diffusion.py:332` | metadata 在 batch 里通常不 stack；要让 `MultiConditioner` 自己逐样本编码 |

---

## 附录 1：术语速查（超短）

- **Waveform**：时域音频采样序列（`[B,2,T]`）。
- **Latent**：压缩后的连续序列表示（`[B,C,L]`）。
- **Pretransform**：waveform↔latent 的固定变换（常为 autoencoder）。
- **Conditioner**：metadata→(tensor,mask) 的翻译器。
- **Cross-attention**：主序列 attend 条件序列（常用在文本提示）。
- **Prepend**：把条件 token 拼到输入 token 之前。
- **Input-concat**：把时间对齐的条件拼到通道维。
- **CFG**：用 cond/uncond 两次前向做 guidance。

---

## 附录 2：建议学习路径（基础为主）

1) 读 `docs/pretransforms.md:1` → 理解 pretransform 的定位与类型。  
2) 读 `docs/autoencoders.md:1` → 建立“latent sequence/下采样比”的直觉。  
3) 读 `docs/conditioning.md:4` → 牢记 4 种 conditioning 的形状约定。  
4) 读 `docs/diffusion.md:20` → 认识 `cross_attention_cond_ids/global_cond_ids/prepend_cond_ids/input_concat_ids`。  
5) 用 `模型结构_易读.md:13` 把模块树连起来（DiTWrapper/conditioner/pretransform）。  
6) 需要验证时再看 `.venv/.../stable_audio_tools/models/diffusion.py:137`（条件路由）。  
7) 再看 `.venv/.../stable_audio_tools/models/transformer.py:817`（prepend 真实拼接点）。  
8) 最后看 `.venv/.../stable_audio_tools/training/diffusion.py:332`（训练 step 里 batch 结构怎么走）。  

