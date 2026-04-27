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
