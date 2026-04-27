import random
import sys
from pathlib import Path

import torch

# 允许直接以脚本方式运行，无需先安装本地包。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_audio_control.models.control_transformer import ControlNetContinuousTransformer


def main() -> None:
    torch.manual_seed(0)
    random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from stable_audio_tools.models.transformer import ContinuousTransformer
    bsz = 1
    seq = 64
    dim_in = 32
    dim = 128
    dim_out = 32

    base_transformer = ContinuousTransformer(
        dim=dim,
        depth=4,
        dim_in=dim_in,
        dim_out=dim_out,
        dim_heads=32,
        cross_attend=True,
        cond_token_dim=64,
        global_cond_dim=None,
        causal=False,
    ).to(device).eval()

    control_transformer = ControlNetContinuousTransformer(
        base_transformer=base_transformer,
        num_control_layers=2,
        default_control_scale=0.5,
    ).to(device).eval()
    control_transformer.freeze_base()

    dtype = next(control_transformer.parameters()).dtype

    x = torch.randn((bsz, seq, dim_in), device=device, dtype=dtype)
    control_input = torch.randn((bsz, seq, dim_in), device=device, dtype=dtype)
    context = torch.randn((bsz, 16, 64), device=device, dtype=dtype)

    def run_pair(tag: str) -> None:
        with torch.no_grad():
            out_no_control = control_transformer(
                x,
                context=context,
                control_input=control_input,
                control_scale=0.0,
                use_checkpointing=False,
            )
            out_with_control = control_transformer(
                x,
                context=context,
                control_input=control_input,
                control_scale=0.5,
                use_checkpointing=False,
            )

        diff = (out_with_control - out_no_control).float()
        print(f"\n== {tag} ==")
        print("out_no_control:", tuple(out_no_control.shape), "norm:", float(out_no_control.float().norm()))
        print("out_with_control:", tuple(out_with_control.shape), "norm:", float(out_with_control.float().norm()))
        print("diff norm:", float(diff.norm()))

    print("device:", device, "dtype:", dtype)

    # 在标准 ControlNet 初始化（zero-linears 全 0）下，模型输出会完全一致。
    # 这是预期行为，也是零初始化设计的核心目的。
    run_pair("零初始化适配器（预期 diff=0）")

    # 为了验证注入路径端到端可用，轻微扰动一个 adapter。
    with torch.no_grad():
        control_transformer.zero_linears[0].weight.normal_(mean=0.0, std=1e-3)
    run_pair("微扰 adapter 后（预期 diff>0）")
    # 基于真实模型的 smoke test（使用 get_pretrained_model）在：
    # `stable_audio_control/scripts/smoke_control_injection_stableaudio_open-1.py`


if __name__ == "__main__":
    main()
