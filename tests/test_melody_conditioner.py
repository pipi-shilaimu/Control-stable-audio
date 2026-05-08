from __future__ import annotations

import unittest
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_audio_control.melody.conditioner import MelodyControlEncoder


class MelodyControlEncoderTests(unittest.TestCase):
    def test_encoder_maps_pitch_indices_to_target_length_control_input(self) -> None:
        encoder = MelodyControlEncoder(
            num_pitch_bins=128,
            melody_channels=8,
            embedding_dim=16,
            hidden_dim=32,
            output_dim=64,
            conv_layers=2,
        )
        melody = torch.randint(1, 129, (2, 8, 11), dtype=torch.long)
        melody[:, :, 0] = 0

        control = encoder(melody, target_len=7, dtype=torch.float32, device=torch.device("cpu"))

        self.assertEqual(tuple(control.shape), (2, 7, 64))
        self.assertEqual(control.dtype, torch.float32)
        self.assertTrue(torch.isfinite(control).all())

    def test_encoder_accepts_already_transposed_melody_tensor(self) -> None:
        encoder = MelodyControlEncoder(
            num_pitch_bins=128,
            melody_channels=8,
            embedding_dim=8,
            hidden_dim=16,
            output_dim=32,
            conv_layers=1,
        )
        melody = torch.randint(1, 129, (2, 13, 8), dtype=torch.long)

        control = encoder(melody, target_len=5, dtype=torch.float32, device=torch.device("cpu"))

        self.assertEqual(tuple(control.shape), (2, 5, 32))

    def test_encoder_uses_zero_as_padding_index(self) -> None:
        encoder = MelodyControlEncoder(
            num_pitch_bins=128,
            melody_channels=8,
            embedding_dim=8,
            hidden_dim=16,
            output_dim=32,
            conv_layers=1,
        )

        self.assertEqual(encoder.pitch_embedding.padding_idx, 0)
        self.assertTrue(torch.equal(encoder.pitch_embedding.weight[0], torch.zeros(8)))

    def test_encoder_rejects_pitch_indices_outside_configured_range(self) -> None:
        encoder = MelodyControlEncoder(
            num_pitch_bins=128,
            melody_channels=8,
            embedding_dim=8,
            hidden_dim=16,
            output_dim=32,
            conv_layers=1,
        )
        melody = torch.full((1, 8, 4), 129, dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "0..128"):
            encoder(melody, target_len=4, dtype=torch.float32, device=torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
