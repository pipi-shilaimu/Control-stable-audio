import copy
import typing as tp

import torch
from torch import nn


class ControlNetContinuousTransformer(nn.Module):
    """
    `stable_audio_tools.models.transformer.ContinuousTransformer` 的最小 ControlNet 风格封装。

    设计目标（便于 smoke test）：
    - 保持预训练 transformer 主干结构不变（通常由调用方冻结）。
    - 增加一个可训练的控制分支，复制前 N 个 transformer block。
    - 通过零初始化的 Linear 层把控制分支特征注入到主干流中。

    该实现无需修改 `.venv` / site-packages：
    加载 StableAudio Open 模型后，将

        model.model.model.transformer = ControlNetContinuousTransformer(base_transformer, ...)

    然后在 diffusion model 的 forward 里透传额外参数：

        control_input: Tensor[batch, seq, dim_in]（与传给 transformer 的 `x` 形状一致）
        control_scale: float
    """

    def __init__(
        self,
        base_transformer: nn.Module,
        num_control_layers: int = 1,
        default_control_scale: float = 1.0,
    ) -> None:
        super().__init__()

        if not hasattr(base_transformer, "layers"):
            raise TypeError("base_transformer must have a .layers attribute (ContinuousTransformer expected).")

        self.base = base_transformer
        self.num_control_layers = int(num_control_layers)
        self.default_control_scale = float(default_control_scale)

        if self.num_control_layers < 0 or self.num_control_layers > len(self.base.layers):
            raise ValueError(
                f"num_control_layers must be in [0, {len(self.base.layers)}], got {self.num_control_layers}"
            )

        # 复制前 N 个 block 作为可训练控制分支。
        # 这里使用深拷贝：初始权重与主干一致，但后续训练参数彼此独立。
        self.control_layers = nn.ModuleList(
            [copy.deepcopy(self.base.layers[i]) for i in range(self.num_control_layers)]
        )

        # 用零初始化线性层把控制分支输出注入主干流。
        # 零初始化保证训练初期几乎不改变原模型行为，便于稳定接入。
        dim = getattr(self.base, "dim", None)
        if dim is None:
            raise TypeError("base_transformer is missing .dim (ContinuousTransformer expected).")

        self.zero_linears = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.num_control_layers)])
        for layer in self.zero_linears:
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def freeze_base(self) -> None:
        """便捷方法：冻结预训练 base transformer 的全部参数。"""
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        x: torch.Tensor,
        prepend_embeds: tp.Optional[torch.Tensor] = None,
        global_cond: tp.Optional[torch.Tensor] = None,
        return_info: bool = False,
        use_checkpointing: bool = True,
        exit_layer_ix: tp.Optional[int] = None,
        *,
        control_input: tp.Optional[torch.Tensor] = None,
        control_scale: tp.Optional[float] = None,
        **kwargs,
    ):
        """
        参数说明：
            x: Tensor[batch, seq, dim_in]，与 ContinuousTransformer 输入一致。
            prepend_embeds: 可选 Tensor[batch, prepend_seq, dim]。
            global_cond: 可选 Tensor[batch, global_dim]。
            control_input: 可选 Tensor[batch, seq, dim_in]，用于驱动控制分支。
        """

        # 未提供控制输入，或控制层数为 0 时，行为与原始 base 完全一致。
        if control_input is None or self.num_control_layers == 0:
            return self.base(
                x,
                prepend_embeds=prepend_embeds,
                global_cond=global_cond,
                return_info=return_info,
                use_checkpointing=use_checkpointing,
                exit_layer_ix=exit_layer_ix,
                **kwargs,
            )

        if control_scale is None:
            control_scale = self.default_control_scale
        control_scale = float(control_scale)

        # 对齐到模型参数 dtype，避免混合精度下出现类型不一致。
        model_dtype = next(self.parameters()).dtype
        x = x.to(model_dtype)
        control_input = control_input.to(model_dtype)

        # 与 base 返回结构保持一致：仅在需要时收集中间层隐藏态。
        info = {"hidden_states": []} if return_info else None

        # --- 与 base transformer 保持一致的输入预处理 ---
        x_base = self.base.project_in(x)

        if prepend_embeds is not None:
            prepend_embeds = prepend_embeds.to(model_dtype)
            prepend_length, prepend_dim = prepend_embeds.shape[1:]
            if prepend_dim != x_base.shape[-1]:
                raise ValueError(f"prepend_embeds last dim must be {x_base.shape[-1]}, got {prepend_dim}")
            x_base = torch.cat((prepend_embeds, x_base), dim=-2)
        else:
            prepend_length = 0

        if getattr(self.base, "num_memory_tokens", 0) > 0:
            memory_tokens = self.base.memory_tokens.expand(x_base.shape[0], -1, -1)
            x_base = torch.cat((memory_tokens, x_base), dim=1)
            memory_length = memory_tokens.shape[1]
        else:
            memory_length = 0

        # 旋转位置编码 / 绝对位置编码：与 base 处理路径一致。
        if getattr(self.base, "rotary_pos_emb", None) is not None:
            rotary_pos_emb = self.base.rotary_pos_emb.forward_from_seq_len(x_base.shape[1])
        else:
            rotary_pos_emb = None

        if getattr(self.base, "use_sinusoidal_emb", False) or getattr(self.base, "use_abs_pos_emb", False):
            x_base = x_base + self.base.pos_emb(x_base)

        # 全局条件向量编码（供 block 内部 adaLN 风格条件调制使用）。
        if global_cond is not None and getattr(self.base, "global_cond_embedder", None) is not None:
            global_cond = global_cond.to(model_dtype)
            global_cond = self.base.global_cond_embedder(global_cond)

        # --- 构建控制分支输入 ---
        ctrl = self.base.project_in(control_input)

        # control_input 只对应原始 x token，需要在前面补零以对齐 prepend + memory token。
        # 这样 ctrl 与 x_base 在序列长度上严格一致，可逐 token 相加。
        if prepend_length > 0:
            ctrl = torch.cat(
                [torch.zeros((ctrl.shape[0], prepend_length, ctrl.shape[2]), device=ctrl.device, dtype=ctrl.dtype), ctrl],
                dim=1,
            )
        if memory_length > 0:
            ctrl = torch.cat(
                [torch.zeros((ctrl.shape[0], memory_length, ctrl.shape[2]), device=ctrl.device, dtype=ctrl.dtype), ctrl],
                dim=1,
            )

        # 控制流从与主干同构的表示出发，再叠加外部控制信号。
        x_ctrl = x_base + ctrl * control_scale

        # 层循环：前 N 层执行“控制支路 + 主干支路 + zero_linear 注入”，其余层仅走主干。
        for layer_ix, base_layer in enumerate(self.base.layers):
            if layer_ix < self.num_control_layers:
                ctrl_layer = self.control_layers[layer_ix]

                if use_checkpointing:
                    # checkpoint 分别包裹控制层与主干层，以减少显存占用。
                    x_ctrl = torch.utils.checkpoint.checkpoint(
                        ctrl_layer,
                        x_ctrl,
                        rotary_pos_emb=rotary_pos_emb,
                        global_cond=global_cond,
                        self_attention_flash_sliding_window=getattr(self.base, "sliding_window", None),
                        use_reentrant=False,
                        **kwargs,
                    )
                    x_base = torch.utils.checkpoint.checkpoint(
                        base_layer,
                        x_base,
                        rotary_pos_emb=rotary_pos_emb,
                        global_cond=global_cond,
                        self_attention_flash_sliding_window=getattr(self.base, "sliding_window", None),
                        use_reentrant=False,
                        **kwargs,
                    )
                else:
                    x_ctrl = ctrl_layer(
                        x_ctrl,
                        rotary_pos_emb=rotary_pos_emb,
                        global_cond=global_cond,
                        self_attention_flash_sliding_window=getattr(self.base, "sliding_window", None),
                        **kwargs,
                    )
                    x_base = base_layer(
                        x_base,
                        rotary_pos_emb=rotary_pos_emb,
                        global_cond=global_cond,
                        self_attention_flash_sliding_window=getattr(self.base, "sliding_window", None),
                        **kwargs,
                    )

                # 把控制分支输出经过零初始化映射后注入主干。
                x_base = x_base + self.zero_linears[layer_ix](x_ctrl)
            else:
                if use_checkpointing:
                    x_base = torch.utils.checkpoint.checkpoint(
                        base_layer,
                        x_base,
                        rotary_pos_emb=rotary_pos_emb,
                        global_cond=global_cond,
                        self_attention_flash_sliding_window=getattr(self.base, "sliding_window", None),
                        use_reentrant=False,
                        **kwargs,
                    )
                else:
                    x_base = base_layer(
                        x_base,
                        rotary_pos_emb=rotary_pos_emb,
                        global_cond=global_cond,
                        self_attention_flash_sliding_window=getattr(self.base, "sliding_window", None),
                        **kwargs,
                    )

            if return_info:
                info["hidden_states"].append(x_base)

            # 提前退出语义与 base 对齐：返回前先移除 memory token。
            if exit_layer_ix is not None and layer_ix == exit_layer_ix:
                x_base = x_base[:, memory_length:, :]
                if return_info:
                    return x_base, info
                return x_base

        # 末尾行为与 base 对齐：移除 memory token 后再投影输出。
        x_base = x_base[:, memory_length:, :]
        x_base = self.base.project_out(x_base)

        if return_info:
            return x_base, info
        return x_base
