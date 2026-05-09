from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_audio_control.melody.chromagram import ChromagramConfig, ChromagramExtractor
from stable_audio_control.melody.extractors import build_melody_extractor, melody_control_channels


def stereo_sine(*, sample_rate: int = 22_050, seconds: float = 0.25, frequency: float = 440.0) -> torch.Tensor:
    t = torch.linspace(0.0, seconds, int(sample_rate * seconds), dtype=torch.float32)
    wave = 0.5 * torch.sin(2.0 * torch.pi * frequency * t)
    return torch.stack([wave, wave], dim=0)


class ChromagramExtractorTests(unittest.TestCase):
    def test_extract_returns_12_channel_float_control_tensor(self) -> None:
        extractor = ChromagramExtractor(
            ChromagramConfig(sample_rate=22_050, hop_length=256, n_fft=1024, n_chroma=12)
        )

        chroma = extractor.extract(stereo_sine())

        self.assertEqual(chroma.ndim, 3)
        self.assertEqual(chroma.shape[0], 1)
        self.assertEqual(chroma.shape[1], 12)
        self.assertGreater(chroma.shape[2], 0)
        self.assertEqual(chroma.dtype, torch.float32)
        self.assertTrue(torch.isfinite(chroma).all())
        self.assertGreaterEqual(float(chroma.min()), 0.0)
        self.assertLessEqual(float(chroma.max()), 1.0)

    def test_extract_folds_a440_to_a_pitch_class(self) -> None:
        extractor = ChromagramExtractor(
            ChromagramConfig(sample_rate=22_050, hop_length=256, n_fft=1024, n_chroma=12)
        )

        chroma = extractor.extract(stereo_sine(frequency=440.0))
        mean_chroma = chroma[0].mean(dim=1)

        # librosa chroma bins are C, C#, ..., A, A#, B.
        self.assertEqual(int(mean_chroma.argmax().item()), 9)

    def test_rejects_non_stereo_input(self) -> None:
        extractor = ChromagramExtractor(ChromagramConfig(sample_rate=22_050))

        with self.assertRaisesRegex(ValueError, "Expected \\[2, T\\]"):
            extractor.extract(torch.zeros(1, 1024))

    def test_melody_extractor_factory_exposes_chromagram_channels(self) -> None:
        extractor = build_melody_extractor(
            feature="chromagram",
            sample_rate=22_050,
            top_k=4,
            n_bins=128,
            bins_per_octave=12,
            fmin_hz=8.175798915643707,
            hop_length=256,
            highpass_cutoff_hz=261.2,
            cqt_backend="auto",
            chroma_bins=12,
            chroma_n_fft=1024,
        )

        self.assertIsInstance(extractor, ChromagramExtractor)
        self.assertEqual(melody_control_channels("chromagram", top_k=4, chroma_bins=12), 12)
        self.assertEqual(melody_control_channels("cqt", top_k=4, chroma_bins=12), 8)


if __name__ == "__main__":
    unittest.main()
