# stable-audio-tools 推理与 UI

## 读者对象
要做本地生成、参数调优、Gradio 二次开发的开发者。

## 本文覆盖范围
解释扩散推理 API、采样器参数、pretransform 在推理中的长度与 dtype 处理，以及 Gradio 路由结构。

## 3分钟速读版
- 推理入口核心是 `generate_diffusion_uncond/cond/cond_inpaint` 三个函数。
- 有 pretransform 时，采样在 latent 空间进行，最终再 decode 回音频。
- `cfg_scale`、`steps`、`sigma_min/max` 是最常调的质量/风格参数。
- Gradio 只是壳层，真正推理逻辑仍走 `inference/generation.py`。

## 1. 模型加载路径

推理侧常用两条加载路径：

1. 从 Hugging Face 拉取预训练模型  
   入口：[`get_pretrained_model`](../../stable_audio_tools/models/pretrained.py)
2. 本地 `model_config + ckpt`  
   入口：[`create_model_from_config`](../../stable_audio_tools/models/factory.py) + checkpoint state dict

Gradio 侧统一在 [`stable_audio_tools/interface/gradio.py`](../../stable_audio_tools/interface/gradio.py) 的 `load_model(...)` 做分发。

## 2. 三个扩散生成入口

定义位置：[`stable_audio_tools/inference/generation.py`](../../stable_audio_tools/inference/generation.py)

- `generate_diffusion_uncond(...)`
- `generate_diffusion_cond(...)`
- `generate_diffusion_cond_inpaint(...)`

共同特点：
- 先根据 `sample_size` 与 pretransform 下采样比计算 latent 长度。
- 生成初始噪声。
- 按 objective 选择采样实现（`v` 或 flow 相关）。
- 若模型是 latent diffusion，最后解码回音频（可通过 `return_latents` 关闭）。

## 3. 采样器参数落到哪里

采样实现位置：[`stable_audio_tools/inference/sampling.py`](../../stable_audio_tools/inference/sampling.py)

常见函数：
- `sample_k`：k-diffusion 风格采样
- `sample_rf`：flow 相关采样
- `sample_discrete_euler` / `sample_flow_pingpong`：在训练 demo 和某些 objective 下常用

常见参数语义：
- `steps`：迭代步数
- `sigma_min` / `sigma_max`：噪声调度边界（对部分采样器有效）
- `cfg_scale`：分类器自由引导强度
- `seed`：随机种子（`-1` 通常表示随机）
- `init_audio` / `init_noise_level`：变体生成或起始参考音频

## 4. pretransform 在推理中的影响

如果模型含 pretransform：

1. 输入 `sample_size` 会被映射为 latent 长度。  
2. `init_audio` 在采样前会先编码到 latent 空间。  
3. 采样结束后再解码回音频。  

这意味着你在“秒数不变”的情况下，真正参与采样的序列长度取决于 `downsampling_ratio`（以及某些模型的 patch/factor 约束）。

## 5. 条件生成的张量流

条件扩散生成时：

1. 传入 `conditioning`（字典或字典列表）。  
2. conditioner 编码为条件张量。  
3. [`ConditionedDiffusionModelWrapper.get_conditioning_inputs`](../../stable_audio_tools/models/diffusion.py) 把多路条件整理成模型前向参数。  

支持的主条件输入形态：
- cross attention
- global embedding
- input concat
- prepend conditioning（在支持的模型上）

## 6. Gradio 路由结构

入口脚本：[`run_gradio.py`](../../run_gradio.py)  
UI 工厂：[`create_ui`](../../stable_audio_tools/interface/gradio.py)

分发逻辑（按 `model_type`）：
- `diffusion_cond` / `diffusion_cond_inpaint` -> `create_diffusion_cond_ui`
- `diffusion_uncond` -> `create_diffusion_uncond_ui`
- `autoencoder` / `diffusion_autoencoder` -> `create_autoencoder_ui`
- `lm` -> `create_lm_ui`

条件扩散 UI 细节在：
- [`stable_audio_tools/interface/interfaces/diffusion_cond.py`](../../stable_audio_tools/interface/interfaces/diffusion_cond.py)

## 7. 常见推理命令

### 7.1 直接跑预训练模型 UI
```bash
python3 ./run_gradio.py --pretrained-name stabilityai/stable-audio-open-1.0
```

### 7.2 本地模型配置 + 解包 checkpoint
```bash
python3 ./run_gradio.py \
  --model-config /path/to/model_config.json \
  --ckpt-path /path/to/model_unwrapped.ckpt
```

### 7.3 叠加 pretransform 解包 checkpoint（常见于替换解码器）
```bash
python3 ./run_gradio.py \
  --model-config /path/to/model_config.json \
  --ckpt-path /path/to/model_unwrapped.ckpt \
  --pretransform-ckpt-path /path/to/pretransform_unwrapped.ckpt
```

## 8. 推理稳定性建议

1. 先用较小步数验证链路正确，再增加 `steps`。  
2. 若启用半精度，关注 `dtype` 与 conditioner / pretransform 的一致性。  
3. 遇到 out-of-memory 时优先减小 batch 和样本长度。  
4. 条件不生效时，先检查 conditioner `id` 与 `*_cond_ids` 是否一致。  

## 9. 相关文档
- 训练端视角： [训练流程](./training-pipeline.md)
- Diffusion 内核细节： [Diffusion 深潜](./diffusion-deep-dive.md)
- 扩展策略： [扩展手册](./extension-playbook.md)
- 报错定位： [排障手册](./troubleshooting.md)
- 英文配置细节： [Diffusion](../diffusion.md), [Conditioning](../conditioning.md)
