# `stable_audio_control/melody/cqt_topk.py` 说明文档

本文档对应文件：`stable_audio_control/melody/cqt_topk.py`  
目标读者：正在做 StableAudio ControlNet 旋律控制链路开发、调试与训练的开发者。

---

## 1. 这个模块解决什么问题

`cqt_topk.py` 的职责是把立体声音频转成可学习、可控、离散化的旋律条件张量，供下游 `melody_control` 使用。

在本仓库 ControlNet-DiT 路线中，它承担的是“旋律特征提取器”角色，而不是模型结构本身：

- 它不负责 ControlNet 注入逻辑（那是 `control_transformer.py` 的职责）。
- 它不负责 `cond -> control_input` 的投影与对齐（那是 `control_dit.py` 的职责）。
- 它只负责从波形提取 top-k CQT 旋律索引。

一句话：  
`cqt_topk.py` 把 `[B, 2, T]` 音频变成 `[B, 2K, F]` 的离散旋律提示（默认 `K=4`，即 8 通道）。

---

## 2. 与论文方法的对齐关系

仓库根目录论文 *Editing Music with Melody and Text: Using ControlNet for Diffusion Transformer* 的关键设定，在本实现中有直接映射：

1. 旋律表示：CQT 128 bins
- 论文给出 CQT 采用 128 个频率 bins，对齐 128 个 MIDI 音高。
- 代码默认：`n_bins=128`, `bins_per_octave=12`, `fmin_hz=8.175798915643707`（MIDI note 0）。

2. 每帧每声道取 top-4
- 论文描述双声道分别取每帧最显著 4 个 pitch，再合并。
- 代码默认：`top_k=4`，在 `dim=2`（频率 bin 维）做 `torch.topk`。

3. 高通预处理
- 论文描述 CQT 前采用以 Middle C 为截止的高通滤波（261.2 Hz）。
- 代码默认：`highpass_cutoff_hz=261.2`，调用 `torchaudio.functional.highpass_biquad(...)`。

4. 1-based pitch index
- 论文中旋律像素值在 `1..128`。
- 代码先拿到 `0..127` 的 top-k bin 索引，再执行 `+1`，得到 `1..128`，并保留 `0` 作为 mask/pad 预留值。

5. 立体声交错（interleave）
- 论文图示中有 L/R 交错构造旋律提示。
- 代码输出顺序为 `[L0, R0, L1, R1, ..., L(K-1), R(K-1)]`，形状为 `[B, 2K, F]`。

---

## 3. 模块结构总览

`cqt_topk.py` 对外包含两个核心对象：

1. `CQTTopKConfig`
- 参数配置 dataclass，定义采样率、CQT 参数、top-k 数量和后端策略。

2. `CQTTopKExtractor`
- 真正执行提取流程的类，主入口是 `extract(audio)`。

默认参数如下：

```python
CQTTopKConfig(
    sample_rate: int,                 # 必填
    fmin_hz: float = 8.175798915643707,
    highpass_cutoff_hz: float = 261.2,
    n_bins: int = 128,
    bins_per_octave: int = 12,
    hop_length: int = 512,
    top_k: int = 4,
    backend: Literal["auto","nnaudio","librosa"] = "auto",
)
```

---

## 4. 端到端数据流

`extract(audio)` 的内部流程可概括为 5 步。

### Step 1: 输入规范化 `_normalize_audio`

接受两种输入形状：

- `[2, T]`（单条立体声）
- `[B, 2, T]`（批量立体声）

不满足以下条件会抛错：

- 声道数必须是 `2`
- `T >= hop_length`
- `top_k <= n_bins`

规范化后统一为 `[B, 2, T]`。

### Step 2: 高通滤波 `_highpass`

把 `[B, 2, T]` 展平为 `[B*2, T]` 后调用：

```python
torchaudio.functional.highpass_biquad(
    waveform=flattened,
    sample_rate=config.sample_rate,
    cutoff_freq=config.highpass_cutoff_hz,
)
```

滤波后再 reshape 回 `[B, 2, T]`。

### Step 3: CQT 计算（`nnAudio` 或 `librosa`）

后端由 `_resolve_backend` 决定：

- `backend="auto"`：优先 `nnAudio`，不可用时回退 `librosa`
- `backend="nnaudio"`：强制 `nnAudio`，缺依赖时报错
- `backend="librosa"`：强制 `librosa`，缺依赖时报错

输出统一成 CQT 幅度谱形状：

- `magnitude: [B, 2, n_bins, F]`

其中 `F` 是 CQT 帧数，和输入长度、`hop_length` 及后端实现细节有关。

### Step 4: 逐帧逐声道 top-k

在频率维 `dim=2` 上执行：

```python
_, topk_idx = torch.topk(magnitude, k=top_k, dim=2, largest=True, sorted=True)
```

得到 `topk_idx: [B, 2, K, F]`，随后做 `+1`，范围从 `0..127` 变成 `1..128`。

### Step 5: L/R 交错并返回

先拆左右声道：

- `left = topk_idx[:, 0, :, :]`  -> `[B, K, F]`
- `right = topk_idx[:, 1, :, :]` -> `[B, K, F]`

再交错重排：

- 输出 `interleaved: [B, 2K, F]`
- 通道顺序：`L0, R0, L1, R1, ...`
- 返回 dtype：`torch.long`

---

## 5. 输入输出契约（给下游模块看的）

### 5.1 输入契约

- 音频必须是立体声（2 声道）。
- `sample_rate` 必须与输入音频实际采样率一致。
- 输入长度不得短于 `hop_length`。

### 5.2 输出契约

`extract(audio)` 返回：

- 类型：`torch.LongTensor`
- 形状：`[B, 2K, F]`（默认 `[B, 8, F]`）
- 值域：`1..n_bins`（默认 `1..128`）
- 语义：每个时间帧内，每个交错通道对应一个“排名后的显著 pitch bin”

### 5.3 与 `control_dit.py` 的衔接

`ControlConditionedDiffusionWrapper._extract_control_input()` 支持 `[B, C, L]` 输入。  
`cqt_topk.py` 返回的 `[B, 2K, F]` 可直接作为 `melody_control` 的 `tensor` 放入：

```python
cond["melody_control"] = [melody_indices, None]
```

下游会自动完成：

1. 长度对齐（`F -> target_len`）
2. `dtype` 转浮点
3. `LazyLinear` 投影到 transformer 的 `dim_in`

这也是 `stable_audio_control/scripts/train_control_smoke.py` 的实际接法。

---

## 6. 关键实现细节和设计取舍

### 6.1 为什么保留 `0` 不用

`+1` 后返回 `1..128`，保留 `0` 给后续 mask/padding 语义，便于课程学习遮罩或批处理补齐。

### 6.2 为什么 `topk` 使用 `sorted=True`

保证每帧内输出顺序稳定（top-1 到 top-k）。  
这让后续的“保留 top-1、扰动 top-2..4”这类 masking 策略更容易实现。

### 6.3 为什么 `nnAudio` 优先

`nnAudio` 在 PyTorch 图内运行、对 GPU 更友好；`librosa` 路径需要 CPU 循环和 NumPy 转换，更适合兜底或离线处理。

### 6.4 `nnAudio` 模块缓存

`_build_nnaudio(...)` 内部会缓存 `self._nnaudio_cqt`，避免每次重复构建 CQT 模块，减少开销。

---

## 7. 最小使用示例

### 7.1 独立调用

```python
import torch
from stable_audio_control.melody.cqt_topk import CQTTopKConfig, CQTTopKExtractor

cfg = CQTTopKConfig(
    sample_rate=44100,
    top_k=4,
    backend="auto",  # nnAudio 优先, 无则 librosa
)
extractor = CQTTopKExtractor(cfg)

# [B, 2, T]
audio = torch.randn(2, 2, 44100)
melody_idx = extractor.extract(audio)

print(melody_idx.shape)  # [2, 8, F]
print(melody_idx.dtype)  # torch.int64
```

### 7.2 注入 `melody_control`

```python
conditioning = model.conditioner(metadata, device)
melody_idx = extractor.extract(audio_batch).to(device)
conditioning["melody_control"] = [melody_idx, None]

out = model(
    x, t,
    cond=conditioning,
    cfg_dropout_prob=0.0,
)
```

---

## 8. 常见错误与排查

1. `Expected [2, T] or [B, 2, T]`
- 原因：输入不是立体声或维度顺序不对。
- 处理：确认输入 shape，必要时先转为 `[B, 2, T]`。

2. `Audio length (...) must be >= hop_length (...)`
- 原因：音频太短。
- 处理：加长输入或降低 `hop_length`。

3. `top_k (...) must be <= n_bins (...)`
- 原因：配置越界。
- 处理：调小 `top_k` 或增大 `n_bins`。

4. `No available CQT backend`
- 原因：`nnAudio` 和 `librosa` 都不可用。
- 处理：安装至少一种依赖，训练场景优先 `nnAudio`。

5. 训练效果异常但代码不报错
- 高概率是“采样率不一致”导致旋律索引语义偏移。
- 确认 dataset 重采样后的 `sample_rate` 与 `CQTTopKConfig.sample_rate` 一致。

---

## 9. 性能与工程建议

1. 在线提取成本
- 每步都算 CQT 会显著增加训练开销，尤其在 `librosa` 后端。
- 规模训练建议考虑缓存（pre-encode）或 custom metadata 中做策略优化。

2. 批量与设备
- `nnAudio` 路径可直接吃 GPU Tensor，更适合训练在线提取。
- `librosa` 路径会走 CPU，注意 DataLoader 吞吐。

3. 下游对齐
- `cqt_topk.py` 的时间长度是 CQT 帧率尺度，和 latent token 长度通常不相等。
- 由 `control_dit.py` 统一做插值对齐是当前仓库推荐边界。

---

## 10. 当前边界与后续扩展

当前文件只做“特征提取”。论文里提到的 progressive curriculum masking 不在本文件内实现，建议放在独立 `melody/masking.py`（与现有模块职责分离一致）。

可扩展方向：

1. 支持可选输出幅值（不仅是索引）用于更细粒度控制。
2. 增加单元测试（纯音定位、shape 合法性、1-based 索引约束）。
3. 在 custom metadata 流中接入真实音频而非合成噪声，统一训练链路。
