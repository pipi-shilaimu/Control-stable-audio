from __future__ import annotations

import unittest
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_audio_control.melody.masking import (
    MelodyMaskingConfig,
    apply_progressive_melody_mask,
    zero_melody_frames_from_energy,
    zero_melody_frames_from_padding_mask,
)


class MelodyMaskingTests(unittest.TestCase):
    def test_disabled_or_eval_mode_keeps_melody_unchanged(self) -> None:
        melody = torch.arange(1, 1 + 8 * 4, dtype=torch.long).reshape(1, 8, 4)

        disabled = apply_progressive_melody_mask(
            melody,
            global_step=0,
            config=MelodyMaskingConfig(enabled=False, full_mask_steps=10),
            training=True,
        )
        eval_mode = apply_progressive_melody_mask(
            melody,
            global_step=0,
            config=MelodyMaskingConfig(enabled=True, full_mask_steps=10),
            training=False,
        )

        self.assertTrue(torch.equal(disabled, melody))
        self.assertTrue(torch.equal(eval_mode, melody))
        self.assertIsNot(disabled, melody)
        self.assertIsNot(eval_mode, melody)

    def test_full_mask_phase_sets_every_pitch_token_to_zero(self) -> None:
        melody = torch.arange(1, 1 + 8 * 4, dtype=torch.long).reshape(1, 8, 4)

        masked = apply_progressive_melody_mask(
            melody,
            global_step=99,
            config=MelodyMaskingConfig(enabled=True, full_mask_steps=100),
            training=True,
        )

        self.assertTrue(torch.equal(masked, torch.zeros_like(melody)))

    def test_schedule_end_preserves_top1_and_masks_secondary_pitch_channels(self) -> None:
        melody = torch.tensor(
            [
                [
                    [10, 11, 12],
                    [20, 21, 22],
                    [30, 31, 32],
                    [40, 41, 42],
                    [50, 51, 52],
                    [60, 61, 62],
                    [70, 71, 72],
                    [80, 81, 82],
                ]
            ],
            dtype=torch.long,
        )

        masked = apply_progressive_melody_mask(
            melody,
            global_step=10,
            config=MelodyMaskingConfig(
                enabled=True,
                full_mask_steps=0,
                schedule_steps=10,
                frame_mask_ratio_start=0.0,
                frame_mask_ratio_end=0.0,
                secondary_mask_prob=1.0,
                secondary_shuffle_prob=0.0,
                preserve_top1=True,
            ),
            training=True,
        )

        self.assertTrue(torch.equal(masked[:, 0:2, :], melody[:, 0:2, :]))
        self.assertTrue(torch.equal(masked[:, 2:, :], torch.zeros_like(melody[:, 2:, :])))

    def test_frame_mask_ratio_zeros_whole_frames_after_full_mask_phase(self) -> None:
        melody = torch.arange(1, 1 + 8 * 5, dtype=torch.long).reshape(1, 8, 5)

        masked = apply_progressive_melody_mask(
            melody,
            global_step=10,
            config=MelodyMaskingConfig(
                enabled=True,
                full_mask_steps=0,
                schedule_steps=10,
                frame_mask_ratio_start=1.0,
                frame_mask_ratio_end=1.0,
                secondary_mask_prob=0.0,
                secondary_shuffle_prob=0.0,
                preserve_top1=True,
            ),
            training=True,
        )

        self.assertTrue(torch.equal(masked, torch.zeros_like(melody)))

    def test_secondary_shuffle_is_reproducible_with_generator(self) -> None:
        melody = torch.tensor(
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [10, 11, 12, 13],
                    [20, 21, 22, 23],
                    [30, 31, 32, 33],
                    [40, 41, 42, 43],
                    [50, 51, 52, 53],
                    [60, 61, 62, 63],
                ]
            ],
            dtype=torch.long,
        )
        config = MelodyMaskingConfig(
            enabled=True,
            full_mask_steps=0,
            schedule_steps=1,
            frame_mask_ratio_start=0.0,
            frame_mask_ratio_end=0.0,
            secondary_mask_prob=0.0,
            secondary_shuffle_prob=1.0,
            preserve_top1=True,
        )

        first = apply_progressive_melody_mask(
            melody,
            global_step=1,
            config=config,
            training=True,
            generator=torch.Generator().manual_seed(123),
        )
        second = apply_progressive_melody_mask(
            melody,
            global_step=1,
            config=config,
            training=True,
            generator=torch.Generator().manual_seed(123),
        )

        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first[:, 0:2, :], melody[:, 0:2, :]))
        self.assertFalse(torch.equal(first[:, 2:, :], melody[:, 2:, :]))

    def test_padding_mask_zeroes_padded_cqt_frames(self) -> None:
        melody = torch.ones((1, 8, 4), dtype=torch.long)
        padding_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.float32)

        masked = zero_melody_frames_from_padding_mask(
            melody,
            padding_mask=padding_mask,
            audio_num_samples=8,
        )

        self.assertTrue(torch.equal(masked[:, :, :2], torch.ones_like(masked[:, :, :2])))
        self.assertTrue(torch.equal(masked[:, :, 2:], torch.zeros_like(masked[:, :, 2:])))

    def test_missing_padding_mask_keeps_melody_unchanged(self) -> None:
        melody = torch.ones((1, 8, 4), dtype=torch.long)

        masked = zero_melody_frames_from_padding_mask(
            melody,
            padding_mask=None,
            audio_num_samples=8,
        )

        self.assertTrue(torch.equal(masked, melody))
        self.assertIsNot(masked, melody)

    def test_energy_mask_zeroes_silent_and_low_energy_cqt_frames(self) -> None:
        melody = torch.arange(1, 1 + 8 * 4, dtype=torch.long).reshape(1, 8, 4)
        frame_energy = torch.tensor([[0.0, 0.001, 0.1, 1.0]], dtype=torch.float32)

        masked = zero_melody_frames_from_energy(
            melody,
            frame_energy=frame_energy,
            min_energy_ratio=0.05,
            min_energy_abs=1e-6,
        )

        self.assertTrue(torch.equal(masked[:, :, :2], torch.zeros_like(masked[:, :, :2])))
        self.assertTrue(torch.equal(masked[:, :, 2:], melody[:, :, 2:]))

    def test_energy_mask_zeroes_all_frames_when_audio_is_silent(self) -> None:
        melody = torch.ones((1, 8, 4), dtype=torch.long)
        frame_energy = torch.zeros((1, 4), dtype=torch.float32)

        masked = zero_melody_frames_from_energy(
            melody,
            frame_energy=frame_energy,
            min_energy_ratio=0.01,
            min_energy_abs=1e-8,
        )

        self.assertTrue(torch.equal(masked, torch.zeros_like(melody)))


if __name__ == "__main__":
    unittest.main()
