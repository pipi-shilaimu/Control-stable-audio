# stable-audio-tools 训练日志字段字典

## 读者对象
需要解读训练日志、定位异常趋势、快速判断训练状态的开发者。

## 本文覆盖范围
基于当前训练包装器实现，整理常见 `train/*`、`val/*` 和相关 debug 指标的含义。  
重点覆盖：
- [`stable_audio_tools/training/diffusion.py`](../../stable_audio_tools/training/diffusion.py)
- [`stable_audio_tools/training/autoencoders.py`](../../stable_audio_tools/training/autoencoders.py)
- [`stable_audio_tools/training/lm.py`](../../stable_audio_tools/training/lm.py)

## 3分钟速读版
- 先看 `train/loss` 的整体趋势，再看每个子损失是否彼此“拉扯”。
- diffusion 常看：`train/loss`、`train/std_data`、`train/lr`、`val/loss_*`、`val/avg_loss`。
- autoencoder 常看：`train/loss`、`train/latent_std`、`train/data_std`、`train/gen_lr`、`train/disc_lr`。
- LM 常看：`train/cross_entropy`、`train/perplexity`，以及每个 codebook 的 `cross_entropy_q*`。

## 1. 快速判读顺序

1. 先看是否出现 NaN/Inf。  
2. 再看学习率是否按预期变化。  
3. 再看总损失趋势是否稳定下降或至少震荡收敛。  
4. 最后看分项指标是否出现单点爆炸（某一项突然放大）。  

## 2. Diffusion 训练日志字段

### 2.1 无条件扩散（`DiffusionUncondTrainingWrapper`）

| 字段 | 含义 | 异常信号 |
| --- | --- | --- |
| `train/loss` | 当前 step 总损失（通常是 MSE 聚合） | 长时间不下降或突然爆炸 |
| `train/std_data` | 当前训练输入（audio 或 latent）的标准差 | 过小可能输入塌缩，过大可能归一化不稳 |
| `train/<loss_name>` | 动态子损失项（例如 `mse_loss`） | 某子项独自飙升通常指向该路径异常 |

### 2.2 条件扩散（`DiffusionCondTrainingWrapper`）

| 字段 | 含义 | 异常信号 |
| --- | --- | --- |
| `train/loss` | 条件扩散总损失 | 训练初期可高，但应逐步稳定 |
| `train/std_data` | diffusion 输入标准差 | 长期异常波动说明输入处理不稳 |
| `train/lr` | 当前优化器学习率 | 与 scheduler 预期不一致需排查配置 |
| `train/<loss_name>` | 动态子损失（常见 `mse_loss`） | 与总损失走势明显背离需排查 |
| `model/loss_all_x.x` | 按 sigma bucket 聚合的 debug loss（仅 `log_loss_info=true`） | 某些 bucket 长期异常高，常见于时间采样分布或目标定义问题 |

### 2.3 条件扩散验证指标

| 字段 | 含义 | 异常信号 |
| --- | --- | --- |
| `val/loss_0.1` ... `val/loss_0.9` | 固定 timestep 的验证 MSE | 某些 timestep 特别差，说明分布学习不均衡 |
| `val/avg_loss` | 所有验证 timestep 的平均损失 | 持续上升通常表示过拟合或训练不稳定 |

### 2.4 扩散自编码器（`DiffusionAutoencoderTrainingWrapper`）

| 字段 | 含义 | 异常信号 |
| --- | --- | --- |
| `train/loss` | 总损失 | 长期不降或突然跳变 |
| `train/std_data` | 输入标准差 | 异常低/高表示输入尺度问题 |
| `train/latent_std` | latent 标准差 | 持续塌缩接近 0 通常不是好信号 |
| `train/<loss_name>` | 动态子损失（含 bottleneck 相关项） | 某项主导总损失需调权重或检查实现 |

## 3. Autoencoder 训练日志字段

### 3.1 训练期常见字段（`AutoencoderTrainingWrapper`）

| 字段 | 含义 | 异常信号 |
| --- | --- | --- |
| `train/loss` | 生成器分支总损失（在生成器 step 记录） | 波动过大可能是判别器过强或学习率过高 |
| `train/latent_std` | latent 标准差 | 接近 0 可能 latent collapse |
| `train/data_std` | 输入数据标准差 | 不稳定说明输入预处理不一致 |
| `train/gen_lr` | 生成器学习率 | 与配置不一致需查 scheduler |
| `train/disc_lr` | 判别器学习率（在判别器 step 记录） | 长期偏离预期可能调度错误 |
| `train/<loss_name>` | 动态损失（如 `mrstft_loss`, `loss_adv`, `feature_matching_loss`, `kl_loss` 等） | 某项持续主导可导致听感单一或失真 |

### 3.2 验证期字段

| 字段 | 含义 | 异常信号 |
| --- | --- | --- |
| `val/<eval_key>` | 验证指标（取决于 `eval_loss_config`，如 `stft`, `mel`, `sisdr`, `pesq`） | 某指标恶化但训练损失正常，可能是感知质量退化 |

## 4. LM 训练日志字段

### 4.1 训练期字段（`AudioLanguageModelTrainingWrapper`）

| 字段 | 含义 | 异常信号 |
| --- | --- | --- |
| `train/loss` | 与 `train/cross_entropy` 一致（当前实现） | 持续上升需优先检查数据/token 化 |
| `train/cross_entropy` | 交叉熵损失 | 长期高位说明语言建模未收敛 |
| `train/perplexity` | 困惑度（`exp(cross_entropy)`） | 长期居高不下说明建模质量欠佳 |
| `train/lr` | 学习率 | 不符合预期调度需排查 optimizer 配置 |
| `cross_entropy_q{k}` | 第 k 个 codebook 的交叉熵 | 某个 codebook 显著偏高可能该量化器学习困难 |
| `perplexity_q{k}` | 第 k 个 codebook 困惑度 | 单个 quantizer 异常常见于 token 分布偏斜 |

## 5. 异常信号与优先排查建议

### 5.1 出现 NaN/Inf
1. 先降低学习率和 batch。  
2. 检查半精度是否触发数值不稳定（先回到全精度复现）。  
3. 检查输入是否存在异常值（损坏音频、全零 latent、mask 异常）。  

### 5.2 训练损失下降但 demo 听感变差
1. 检查子损失权重是否失衡（例如对抗损失或感知损失过弱/过强）。  
2. 对照 `demo_cfg_scales` 看是否 CFG 过大导致伪影。  
3. 查看验证指标是否与训练损失分离。  

### 5.3 `train/std_data` 或 `train/latent_std` 异常
1. 检查预处理和归一化链路。  
2. 检查 `pre_encoded` 路径下 scale 是否正确应用。  
3. 检查数据集中是否混入异常样本。  

## 6. 采集与比对建议

1. 同时保留 `train/*`、`val/*`、demo 音频三条信号，避免只看单一曲线。  
2. 每次关键改动（objective/sampler/loss 权重）前后各跑一段短实验做 A/B。  
3. 保留最小“健康实验”配置，用于快速判断新改动是否引入回归。  

## 7. 相关文档
- 训练流程： [训练流程](./training-pipeline.md)
- Diffusion 内核： [Diffusion 深潜](./diffusion-deep-dive.md)
- 报错排查： [排障手册](./troubleshooting.md)
