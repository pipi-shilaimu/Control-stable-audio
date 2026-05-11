from __future__ import annotations

import unittest

import torch
from torch import nn

from stable_audio_control.models.control_transformer import ControlNetContinuousTransformer


class AddOneBlock(nn.Module):
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x + 1.0


class TinyContinuousTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dim = 2
        self.project_in = nn.Linear(2, 2, bias=False)
        self.project_out = nn.Linear(2, 2, bias=False)
        self.layers = nn.ModuleList([AddOneBlock()])
        self.num_memory_tokens = 0
        self.rotary_pos_emb = None
        self.use_sinusoidal_emb = False
        self.use_abs_pos_emb = False
        self.global_cond_embedder = None
        self.sliding_window = None
        nn.init.eye_(self.project_in.weight)
        nn.init.eye_(self.project_out.weight)

    def forward(
        self,
        x: torch.Tensor,
        prepend_embeds=None,
        global_cond=None,
        return_info: bool = False,
        use_checkpointing: bool = True,
        exit_layer_ix=None,
        **kwargs,
    ):
        x = self.project_in(x)
        hidden_states = []
        for layer in self.layers:
            x = layer(x)
            hidden_states.append(x)
        x = self.project_out(x)
        if return_info:
            return x, {"hidden_states": hidden_states}
        return x


class ControlNetContinuousTransformerTests(unittest.TestCase):
    def test_control_scale_zero_disables_control_residual_injection(self) -> None:
        base = TinyContinuousTransformer()
        model = ControlNetContinuousTransformer(base, num_control_layers=1)
        with torch.no_grad():
            nn.init.eye_(model.zero_linears[0].weight)
            model.zero_linears[0].bias.zero_()

        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        control_input = torch.tensor([[[9.0, 8.0], [7.0, 6.0]]])

        expected = base(x, use_checkpointing=False)
        actual = model(
            x,
            control_input=control_input,
            control_scale=0.0,
            use_checkpointing=False,
        )

        torch.testing.assert_close(actual, expected)

    def test_control_scale_scales_control_residual_not_base_control_input(self) -> None:
        base = TinyContinuousTransformer()
        model = ControlNetContinuousTransformer(base, num_control_layers=1)
        with torch.no_grad():
            nn.init.eye_(model.zero_linears[0].weight)
            model.zero_linears[0].bias.zero_()

        x = torch.tensor([[[1.0, 2.0]]])
        control_input = torch.tensor([[[9.0, 8.0]]])

        without_control = model(
            x,
            control_input=control_input,
            control_scale=0.0,
            use_checkpointing=False,
        )
        half_control = model(
            x,
            control_input=control_input,
            control_scale=0.5,
            use_checkpointing=False,
        )
        full_control = model(
            x,
            control_input=control_input,
            control_scale=1.0,
            use_checkpointing=False,
        )

        torch.testing.assert_close(full_control - without_control, 2.0 * (half_control - without_control))


if __name__ == "__main__":
    unittest.main()
