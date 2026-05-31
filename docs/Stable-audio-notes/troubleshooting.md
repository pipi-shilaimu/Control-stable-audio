# stable-audio-tools 排障手册

## 读者对象
训练或推理过程中遇到报错，需要快速定位根因的开发者。

## 本文覆盖范围
基于当前代码中最常见的失败点，给出“症状 -> 原因 -> 处理建议”。

## 3分钟速读版
- 启动即报错通常是配置字段缺失或结构不匹配，先核对 factory 断言。
- 训练中 shape 错误通常来自 `sample_size` 与下采样/patch 约束不对齐。
- 推理异常先区分 checkpoint 类型：包装器 ckpt 还是解包 ckpt。
- 半精度异常先回到全精度复现，再逐步恢复 mixed precision。

## 1. 配置断言类错误

### 症状
启动阶段直接报 `assert` 失败，例如：
- `model_type must be specified in model config`
- `training config must be specified in model config`
- `Must specify diffusion config`

### 常见原因
1. `model_config` 顶层字段不完整。  
2. `model_type` 与 `model` 子树结构不匹配。  
3. diffusion 模型缺 `model.diffusion.type/config`。

### 处理建议
1. 对照 [`stable_audio_tools/models/factory.py`](../../stable_audio_tools/models/factory.py) 的分支检查 `model_type`。  
2. 对照 [`stable_audio_tools/training/factory.py`](../../stable_audio_tools/training/factory.py) 检查 `training` 字段。  
3. 对照 [`stable_audio_tools/models/diffusion.py`](../../stable_audio_tools/models/diffusion.py) 检查扩散模型必需字段。  

## 2. 长度或 shape 不匹配

### 症状
训练/推理中出现 tensor shape 错误，或 input concat 条件无法拼接。

### 常见原因
1. `sample_size` 与 pretransform 下采样比、patch/factor 约束不匹配。  
2. `input_concat_cond` 序列长度与主输入长度不同且未按预期插值。  
3. 预编码数据裁剪后 `padding_mask` 与 latent 长度不同步。  

### 处理建议
1. 检查 `model.min_input_length`（由 pretransform、UNet factors、DiT patch 大小共同决定）。  
2. 检查 [`ConditionedDiffusionModelWrapper`](../../stable_audio_tools/models/diffusion.py) 的条件拼接逻辑。  
3. 检查 [`PreEncodedDataset`](../../stable_audio_tools/data/dataset.py) 对 `padding_mask` 的裁剪同步。  

### 2.1 10 秒切片被按 47.55 秒训练

#### 症状
- 数据实际只有 10 秒，但训练启动日志里 `effective_train_sample_size` 接近 `2097152` samples，约 `47.55s`。
- demo 或生成结果听起来很平滑、很空，像在学静音，而不是学旋律。

#### 常见原因
Stable Audio Open 的默认 `model_config_sample_size=2097152`。如果训练 `formal10s`、`audio_10s` 这类 10 秒切片时没有传 `--seconds-total 10` 或合适的 `--sample-size`，短音频会被补上大量 0。对 ControlNet 来说，这等于每条样本的大部分时间都在告诉模型“这里是 padding/silence”。

#### 处理建议
1. 训练 10 秒切片时加 `--seconds-total 10`。
2. 启动后确认日志类似：`effective_train_sample_size=442368 (10.0300s, source=--seconds-total)`。
3. 如果需要精确采样点数，用 `--sample-size <N>`，但不要和 `--seconds-total` 同时传。
4. train dataloader、val dataloader、demo callback 应使用同一个 `effective_train_sample_size`，不要只改其中一个。

### 2.2 容器里没有同步到本地最新代码

#### 症状
- 容器里运行训练脚本时报：`unrecognized arguments: --seconds-total 10`。
- 或者先报：`ImportError: cannot import name 'DEFAULT_CONTROL_SCALES' from 'stable_audio_control.inference.control_demo_callback'`。
- 同时日志开头有 `flash_attn not installed`，容易误以为是依赖导致。

#### 常见原因
这通常不是 `flash_attn` 问题，而是容器 `/app/data/Control-stable-audio` 里的代码和本机工作区不同步。

两种典型不同步状态：

1. `train_controlnet_dit.py` 是新的，但 `control_demo_callback.py` 还是旧的，所以找不到 `DEFAULT_CONTROL_SCALES`。
2. `train_controlnet_dit.py` 还是旧的，所以 argparse 不认识 `--seconds-total`。

如果这些改动还只是本地未提交修改，`git pull` 不会把它们带进容器；需要提交/推送后在容器拉取，或者直接同步对应文件。

#### 处理建议
1. 在容器里先检查关键符号：

```bash
grep -n "seconds-total" stable_audio_control/scripts/train_controlnet_dit.py
grep -n "DEFAULT_CONTROL_SCALES" stable_audio_control/inference/control_demo_callback.py
```

2. 两个文件必须同时是最新版本：

```text
stable_audio_control/scripts/train_controlnet_dit.py
stable_audio_control/inference/control_demo_callback.py
```

3. 同步后再跑 smoke：

```bash
python3 stable_audio_control/scripts/train_controlnet_dit.py \
  --dataset-config stable_audio_control/data/mtg_jamendo/dataset_config_train.json \
  --default-root-dir outputs/formal10s_lenfix_smoke \
  --seconds-total 10 \
  --max-steps 1 \
  --batch-size 1 \
  --demo-every 0
```

4. 预期脚本能识别 `--seconds-total`，并打印约 `442368` samples / `10.03s` 的 effective train sample size。
5. `flash_attn not installed` 只是回退到非 Flash Attention 的提示，不是上述 CLI/import 报错的根因。

## 3. `pre_encoded` 链路不一致

### 症状
训练能启动但 loss 异常、音频质量崩坏，或推理行为与预期不符。

### 常见原因
1. 数据集是预编码数据，但 `training.pre_encoded` 没设为 `true`。  
2. 跳过编码时没有应用 pretransform scale（某些路径需要显式除以 scale）。  
3. 预编码数据并非来自当前 pretransform 版本。  

### 处理建议
1. 保证 dataset 与 training 的 `pre_encoded` 语义一致。  
2. 检查 wrapper 中“跳过编码分支”是否有 scale 处理（见 diffusion/lm 训练包装器）。  
3. 预编码数据升级后建议重做一小批样本做 smoke test。  

## 4. dtype / device 不一致

### 症状
- `Expected all tensors to be on the same device`
- 半精度下出现类型不兼容或数值不稳定

### 常见原因
1. `model_half` 后输入音频或条件张量未转到一致 dtype。  
2. conditioner 输出在 CPU，模型在 CUDA。  
3. pretransform 解码使用了与主模型不同 dtype。  

### 处理建议
1. 对齐 `next(model.parameters()).dtype` 与输入 dtype。  
2. 统一通过 wrapper/生成函数内部的 device 传递。  
3. 遇到问题先关闭半精度验证逻辑正确性，再逐步恢复。  

## 5. Demo 回调字段缺失

### 症状
训练运行到 demo 回调时报 `KeyError`（常见于 `num_demos`、`demo_cfg_scales`）。

### 常见原因
不同 `model_type` 对 `training.demo` 字段需求不同。

### 处理建议
1. 按 [`create_demo_callback_from_config`](../../stable_audio_tools/training/factory.py) 对应分支补全字段。  
2. 对新模型先用最小 demo 配置跑通，再逐步加复杂条件。  

### 5.1 ControlNet demo 消融看不出差异

#### 症状
- demo 一触发训练就长时间卡住，第一次 smoke 被 demo 采样拖慢。
- `control_scale=1.0` 比 `0.1` 更平滑或更糊，但旋律并没有更像参考音频。
- `correct`、`shuffled`、`zero` 三种 demo 听起来差不多。
- 你希望 demo 用指定旋律，但输出似乎一直跟随当前训练 batch 变化。

#### 常见原因
1. 默认 demo 网格是 `demo_cfg_scales=[3,6,9]`、`demo_control_scales=[0,0.1,0.3,0.6,1]`、`demo_control_variants=[correct,shuffled,zero]`，一轮 demo 是 `3 * 5 * 3 = 45` 次采样。
2. 默认 `--demo-steps 100`，第一次训练时很容易被 demo 成本拖住。
3. `control_scale` 只说明控制残差影响有多大，不保证影响是正确旋律。
4. 默认 demo 从当前 batch 的真实音频提取旋律，所以每次 batch 不同，参考旋律也会变。
5. 如果只听 `correct`，无法判断模型是在跟随旋律，还是对任意控制输入都产生类似扰动。

#### 处理建议
1. 第一次 smoke 先用 `--demo-cfg-scales 6 --demo-steps 30 --demo-control-scales 0,1 --demo-control-variants correct,zero`，把一轮 demo 压到 4 次采样。
2. 链路确认后，再把 `--demo-control-scales` 扩到 `0,0.1,0.3,0.6,1.0`。
3. 做正式诊断时保留 `--demo-control-variants correct,shuffled,zero`，不要只看正确控制。
4. 想固定一首旋律做长期观察时，加 `--demo-control-audio <path/to/audio>`。该音频会被重采样、转 stereo、裁剪/补齐到训练长度，并复制到 demo batch。
5. 想固定文本条件时，加 `--demo-prompt "instrumental piano melody, clear lead melody, no vocals"`。否则 demo prompt 会来自当前 batch metadata，缺失时回退为 `"music"`。
6. 如果同时使用 `--demo-control-audio`，建议也固定 `--demo-prompt`，这样跨 step 听感差异更容易归因到 control，而不是 batch prompt 变化。
7. `control_variant` 是 demo 的诊断变体，不是音频文件名：`correct` 是正确控制，`shuffled` 是错误/错位控制，`zero` 是全 0 控制。
8. `stem` 是输出文件名主体，不含 `.wav` 后缀，用来把 `cfg`、`control_scale`、`variant` 和 step 写进文件名。

### 5.2 correct / shuffled / zero 仍然没有区分度

#### 症状
- 已经用 `--seconds-total 10` 修正训练长度。
- demo 中 `correct`、`shuffled`、`zero` 三类控制仍然听起来差不多。
- `control_scale=1.0` 主要表现为压平、变糊、低动态，而不是更贴近正确旋律。

#### 常见原因
Control branch 只看纯 melody 特征，即 `x_ctrl = ctrl`，没有看见 base branch 当前正在去噪的 latent 状态。可以把这想成：控制分支拿到了旋律地图，但不知道主模型当前这团 noisy latent 长什么样、正在扩散过程的哪一步。

另一个参考实现差异是语义空间：当前 `MelodyControlEncoder` 输出的是 CQT 旋律 embedding，不是 frozen VAE / pretransform 产生的 audio latent。参考 `stable-audio-controlnet` 路线的 control condition 更天然对齐到 DiT 已熟悉的 latent/token 空间。这个差异可能让当前控制分支学到“平滑残差”，而不是可辨认的旋律跟随。

还有一个注入时机差异：当前更像 base layer 先处理，再把 `zero_linear(x_ctrl) * control_scale` 加回去；参考实现更偏向让 control embedding 在对应 layer 计算前进入状态，让 attention / MLP 更早看见控制。

#### 处理建议
1. 不要继续直接长训当前结构；先做短训消融。
2. 新增可配置模式，而不是硬改旧行为：
   - `control_only`：旧行为，`x_ctrl = ctrl`。
   - `detached_base_plus_control`：优先实验，`x_ctrl = x_base.detach() + ctrl`。
   - `base_plus_control`：第二消融，`x_ctrl = x_base + ctrl`。
3. 用新输出目录训练，不要直接拿旧 checkpoint 推理判断。旧 checkpoint 是在 `control_only` 输入分布下训练出来的。
4. 建议跑 2000-4000 步，继续使用 correct / shuffled / zero demo 判断控制是否真的被使用。
5. 如果输入消融仍失败，再单独实验 layer 前注入；不要和 CQT padding mask、adapter 同时改。
6. 如果 layer 前/后注入也不能解决，再考虑 `melody-to-latent adapter` 或完整 guide-audio pretransform 路线。

### 5.3 CQT padding 区域产生伪 pitch

#### 症状
- 训练长度已修正，但 melody control 仍然在静音或 padding 区域产生稳定低变化 token。
- demo 中 `zero` 或 `shuffled` 仍然能产生类似的平滑残差。

#### 常见原因
`CQTTopKExtractor` 会对 CQT magnitude 做 top-k，然后把 pitch index 加 1。也就是说正常输出范围是 `1..n_bins`，`0` 只是在 `MelodyControlEncoder` 中预留给 padding 的 mask token。静音 padding 不会天然变成 0，而是可能变成固定的伪 pitch index，例如 1、2、3、4。

#### 处理建议
1. 从 metadata 中读取 waveform 级别的 `padding_mask`。
2. 将 `padding_mask` 插值到 CQT 帧率，即 `melody_control.shape[-1]`。
3. 在 padding 区域把 melody index 置为 0，再交给 `MelodyControlEncoder`。
4. 如果卷积层 bias 或邻近帧仍让 padding 区域产生非零特征，再把 frame mask 传进 encoder，在卷积输出后再乘一次 mask。

## 6. checkpoint 使用错误（包装器 vs 解包模型）

### 症状
推理加载训练 checkpoint 失败，或参数键不匹配。

### 常见原因
直接拿“包装器 checkpoint”去推理，而推理期通常需要“解包模型 checkpoint”。

### 处理建议
1. 用 [`unwrap_model.py`](../../unwrap_model.py) 导出解包模型。  
2. 确认推理命令使用的 ckpt 与配置结构一致。  
3. pretransform 需要单独替换时使用 `--pretransform-ckpt-path`。  

## 7. 数据加载不稳定或样本被大量跳过

### 症状
训练吞吐低、日志频繁打印样本加载失败，或有效样本比例偏低。

### 常见原因
1. 数据中损坏文件较多。  
2. 静音过滤策略过于严格。  
3. WebDataset URL/权限问题导致读取中断。  

### 处理建议
1. 检查 [`SampleDataset.__getitem__`](../../stable_audio_tools/data/dataset.py) 的回退逻辑。  
2. 调整静音相关参数（如 `remove_silence`、`silence_threshold`）。  
3. 先用小规模样本和 `num_workers=0` 复现，确认根因。  

## 8. 快速自检命令

### 检查关键符号是否存在
```bash
rg "create_model_from_config|create_training_wrapper_from_config|create_dataloader_from_config|generate_diffusion_cond" stable_audio_tools -n
```

### 检查文档与配置入口
```bash
rg "^#|model_type|dataset_type|pre_encoded" README.md docs/*.md docs/zh/*.md -n
```

## 9. 相关文档
- 全局结构： [架构总览](./architecture-overview.md)
- 训练细节： [训练流程](./training-pipeline.md)
- 指标解释： [训练日志字段字典](./training-log-metrics.md)
- 推理细节： [推理与 UI](./inference-and-ui.md)
- 扩展策略： [扩展手册](./extension-playbook.md)
