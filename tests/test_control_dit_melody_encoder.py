from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_audio_control.models.control_dit import ControlConditionedDiffusionWrapper
from stable_audio_control.models.control_transformer import ControlNetContinuousTransformer


class TinyBaseTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.dim = 16
        self.project_in = nn.Linear(4, 16, bias=False)
        self.project_out = nn.Linear(16, 4, bias=False)


class TinyBaseWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        transformer = ControlNetContinuousTransformer(TinyBaseTransformer(), num_control_layers=0)
        self.model = SimpleNamespace(model=SimpleNamespace(transformer=transformer))
        self.conditioner = nn.Identity()
        self.io_channels = 4
        self.sample_rate = 44_100
        self.diffusion_objective = "v"
        self.pretransform = None
        self.cross_attn_cond_ids = []
        self.global_cond_ids = []
        self.input_concat_ids = []
        self.prepend_cond_ids = []
        self.min_input_length = 1
        self.dist_shift = 0.0


class ControlDitMelodyEncoderTests(unittest.TestCase):
    def test_integer_melody_control_uses_melody_encoder(self) -> None:
        wrapper = ControlConditionedDiffusionWrapper(
            TinyBaseWrapper(),
            melody_channels=8,
            melody_embedding_dim=8,
            melody_hidden_dim=16,
            melody_conv_layers=1,
        )
        cond = {"melody_control": [torch.randint(1, 129, (2, 8, 9), dtype=torch.long), None]}

        control_input = wrapper._extract_control_input(
            cond=cond,
            target_len=5,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        self.assertIsNotNone(control_input)
        assert control_input is not None
        self.assertEqual(tuple(control_input.shape), (2, 5, 4))
        self.assertTrue(torch.isfinite(control_input).all())


if __name__ == "__main__":
    unittest.main()
