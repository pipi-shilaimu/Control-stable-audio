# stable-audio-tools 扩展手册

## 读者对象
需要在现有框架上新增模型能力、条件控制或数据接入方式的开发者。

## 本文覆盖范围
提供“改哪里、按什么顺序改、最小验证怎么做”的工程清单，避免只改一半导致链路断裂。

## 3分钟速读版
- 新增能力时要成对修改：模型工厂分支 + 训练工厂分支。
- 新增条件控制要打通三层：metadata 产出、conditioner 构建、`*_cond_ids` 映射。
- 新增数据模式要保证统一输出 `(audio_or_latent, metadata)`。
- 每次改动至少跑四类验证：模型构建、数据读取、单步训练、基础推理。

## 1. 新增 `model_type` 的最小改动面

### 必改清单
1. 在 [`stable_audio_tools/models/factory.py`](../../stable_audio_tools/models/factory.py) 的 `create_model_from_config` 注册新 `model_type`。  
2. 在对应模型文件实现构造函数（可参考 `create_diffusion_cond_from_config`、`create_audio_lm_from_config`）。  
3. 在 [`stable_audio_tools/training/factory.py`](../../stable_audio_tools/training/factory.py) 的 `create_training_wrapper_from_config` 增加训练包装器分支。  
4. 在同文件的 `create_demo_callback_from_config` 增加 demo 回调分支。  
5. 如果需要 UI 支持，在 [`stable_audio_tools/interface/gradio.py`](../../stable_audio_tools/interface/gradio.py) 的 `create_ui` 增加分发。  
6. 提供最小可运行模型配置样例（建议放在 `stable_audio_tools/configs/model_configs/`）。

### 可选清单
1. 在 `docs/` 对应主题页补配置说明。  
2. 在 README 补入口说明。  

## 2. 新增条件控制信号（conditioning）的改动面

### 典型路线
1. 定义 metadata 来源：  
   - 本地数据可在 `custom_metadata_module` 里生成字段。  
   - WebDataset 可在 `wds_preprocess` 之后补字段。  
2. 在 [`stable_audio_tools/models/conditioners.py`](../../stable_audio_tools/models/conditioners.py) 新增 conditioner（或复用现有 `IntConditioner`/`NumberConditioner`/`T5Conditioner` 等）。  
3. 确保 `create_multi_conditioner_from_conditioning_config(...)` 能根据配置构建该 conditioner。  
4. 在模型配置中把新 conditioner 的 `id` 放到正确的 `*_cond_ids`：  
   - `cross_attention_cond_ids`
   - `global_cond_ids`
   - `input_concat_ids`
   - `prepend_cond_ids`

### 常见遗漏
1. metadata 的 key 名和 conditioner `id` 不一致。  
2. 张量形状不符合目标条件类型（特别是 input concat 的 `[B, C, T]`）。  
3. 只改了训练配置，推理 `demo_cond` 或在线输入未同步。  

## 3. 新增数据集模式（dataset_type）的改动面

入口：[`create_dataloader_from_config`](../../stable_audio_tools/data/dataset.py)

### 必改清单
1. 新增 `dataset_type` 分支并返回统一的 dataloader 接口。  
2. 保证每个样本能产出 `(audio_or_latent, metadata)`。  
3. metadata 至少要包含下游常用字段（例如 `padding_mask`、文本条件字段等）。  
4. 若支持预编码链路，确认 latent 与 mask 长度同步裁剪。

### 兼容性建议
1. 统一在数据层做采样率和通道规范化。  
2. 对异常样本做容错回退，避免整个 worker 崩溃。  
3. 使用 `drop_last`、`persistent_workers`、`pin_memory` 时关注显存/内存压力。  

## 4. 训练包装器扩展建议

如果你新增 loss 或训练策略，优先遵循当前 wrapper 结构：

1. 在 `training_step` 中只做当前 step 需要的计算。  
2. 记录 loss 与关键统计量（例如 `std`, `lr`）便于排障。  
3. 使用 `export_model` 明确导出行为（尤其是 EMA 与非 EMA）。  
4. demo 回调里捕获异常并清理缓存，避免长训中断。  

## 5. 最小验证清单（开发阶段）

### 验证 A：模型可构建
```bash
python -c "import json; from stable_audio_tools.models import create_model_from_config; cfg=json.load(open('PATH/TO/model_config.json')); create_model_from_config(cfg); print('model ok')"
```

### 验证 B：数据可读取
```bash
python -c "import json; from stable_audio_tools.data.dataset import create_dataloader_from_config; dcfg=json.load(open('PATH/TO/dataset_config.json')); mcfg=json.load(open('PATH/TO/model_config.json')); dl=create_dataloader_from_config(dcfg,batch_size=1,sample_size=mcfg['sample_size'],sample_rate=mcfg['sample_rate'],audio_channels=mcfg.get('audio_channels',2),num_workers=0); batch=next(iter(dl)); print('data ok', type(batch))"
```

### 验证 C：训练链路可启动
```bash
python3 ./train.py \
  --dataset-config PATH/TO/dataset_config.json \
  --model-config PATH/TO/model_config.json \
  --name smoke_test_run \
  --batch-size 1 \
  --num-workers 0 \
  --checkpoint-every 10
```

### 验证 D：推理链路可启动
```bash
python3 ./run_gradio.py \
  --model-config PATH/TO/model_config.json \
  --ckpt-path PATH/TO/unwrapped_model.ckpt
```

## 6. 提交前的防回归检查

1. 训练入口、推理入口、导出入口都跑过至少一次。  
2. 新增分支在 `models/factory.py` 和 `training/factory.py` 成对出现。  
3. demo 回调字段和配置字段名称一致。  
4. 文档中新增字段有样例配置。  

## 7. 相关文档
- 全局关系： [架构总览](./architecture-overview.md)
- 训练链路： [训练流程](./training-pipeline.md)
- 推理链路： [推理与 UI](./inference-and-ui.md)
- 常见错误： [排障手册](./troubleshooting.md)
