import random
import sys
from pathlib import Path
from typing import cast

import torch

# 允许直接以脚本方式运行，无需先安装本地包。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_audio_control.models.control_transformer import ControlNetContinuousTransformer
from stable_audio_tools.models.diffusion import ConditionedDiffusionModelWrapper
from stable_audio_tools.models.transformer import ContinuousTransformer


def main() -> None:
    """
    在真实的 StableAudio Open 预训练模型上执行 smoke test。

    脚本会做什么：
    - 通过 `get_pretrained_model` 加载 `stabilityai/stable-audio-open-1.0`
    - 将内部 `ContinuousTransformer` 替换为 `ControlNetContinuousTransformer`
    - 用同一输入各跑一次前向（`control_scale=0.0` 与 `0.5`），并打印差值范数

    依赖条件：
    - 需要可用的 HuggingFace 资源（联网或本地缓存均可）。特别是 StableAudio 默认
      conditioner 使用 T5 tokenizer/model；若本地未缓存，加载时可能超时。
    """

    torch.manual_seed(0)
    random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"


    from stable_audio_tools import get_pretrained_model
    model, _model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")


    model = cast(ConditionedDiffusionModelWrapper, model)

    model = model.to(device).eval()
    base_transformer = cast(ContinuousTransformer, model.model.model.transformer) # type: ignore
    control_transformer = ControlNetContinuousTransformer(
        base_transformer=base_transformer,
        num_control_layers=2,
        default_control_scale=0.5,
    ).to(device).eval()
    control_transformer.freeze_base()
    model.model.model.transformer = control_transformer # type: ignore



    # 可选调试：验证注入路径确实会改变输出
    #
    # 在标准 ControlNet 初始化（adapter 全零初始化）下，初始阶段 control 不会生效，
    # 因此即使 `control_scale` 非零，`diff norm` 也会精确等于 0.0。
    #
    # 如果你想快速确认注入链路端到端打通，可取消下面代码块的注释。
    # 它会轻微扰动一个 zero-linear adapter，使注入残差变成非零。
    # with torch.no_grad():
    #     cast(ControlNetContinuousTransformer, model.model.model.transformer).zero_linears[0].weight.normal_(
    #         mean=0.0, std=1e-3
    #     )

    # 通过已有 conditioner 构建条件字典
    batch_metadata = [
        {
            "prompt": "test prompt",
            "seconds_start": 0,
            "seconds_total": 5,
        }
    ]
    cond = model.conditioner(batch_metadata, device)

    # 随机生成 latent 噪声和时间步
    bsz = 1
    io_channels = model.io_channels
    latent_len = 128

    dtype = next(model.parameters()).dtype
    x = torch.randn((bsz, io_channels, latent_len), device=device, dtype=dtype)
    t = torch.full((bsz,), 0.5, device=device, dtype=dtype)

    # control_input 必须与 transformer 输入形状一致: [B, seq, dim_in]
    dim_in = base_transformer.project_in.in_features
    control_input = torch.randn((bsz, latent_len, dim_in), device=device, dtype=dtype)  # type: ignore
    
    #with torch.no_grad():
    
    #    control_transformer.zero_linears[0].weight.normal_(mean=0.0, std=1e-3)
    with torch.no_grad():
        out0 = model(
            x,
            t,
            cond=cond,
            cfg_scale=1.0,
            cfg_dropout_prob=0.0,
            control_input=control_input,
            control_scale=0.0,
        )
        out1 = model(
            x,
            t,
            cond=cond,
            cfg_scale=1.0,
            cfg_dropout_prob=0.0,
            control_input=control_input,
            control_scale=0.5,
        )

    print("device:", device, "dtype:", dtype)
    print("out0:", tuple(out0.shape), "norm:", float(out0.float().norm()))
    print("out1:", tuple(out1.shape), "norm:", float(out1.float().norm()))
    print("diff norm:", float((out1 - out0).float().norm()))


if __name__ == "__main__":
    main()
