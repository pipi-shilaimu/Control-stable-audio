from __future__ import annotations

import unittest
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_audio_control.melody.cqt_topk import CQTTopKConfig, CQTTopKExtractor


class CQTTopKExtractorTests(unittest.TestCase):
    def test_extract_disables_autocast_and_uses_float32_for_feature_extraction(self) -> None:
        extractor = CQTTopKExtractor(
            CQTTopKConfig(
                sample_rate=44_100,
                hop_length=4,
                n_bins=8,
                top_k=2,
                backend="librosa",
            )
        )
        calls: dict[str, object] = {}

        def fake_highpass(audio: torch.Tensor) -> torch.Tensor:
            calls["highpass_dtype"] = audio.dtype
            calls["highpass_autocast"] = torch.is_autocast_enabled(audio.device.type)
            return audio

        def fake_cqt(audio: torch.Tensor) -> torch.Tensor:
            calls["cqt_dtype"] = audio.dtype
            calls["cqt_autocast"] = torch.is_autocast_enabled(audio.device.type)
            return torch.arange(1 * 2 * 8 * 3, dtype=torch.float32).reshape(1, 2, 8, 3)

        extractor._highpass = fake_highpass  # type: ignore[method-assign]
        extractor._cqt_with_librosa = fake_cqt  # type: ignore[method-assign]

        audio = torch.randn(1, 2, 16, dtype=torch.bfloat16)
        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
            melody = extractor.extract(audio)

        self.assertEqual(calls["highpass_dtype"], torch.float32)
        self.assertEqual(calls["cqt_dtype"], torch.float32)
        self.assertFalse(calls["highpass_autocast"])
        self.assertFalse(calls["cqt_autocast"])
        self.assertEqual(melody.dtype, torch.long)
        self.assertEqual(tuple(melody.shape), (1, 4, 3))

    def test_extract_zeroes_low_energy_frames_before_returning_pitch_tokens(self) -> None:
        extractor = CQTTopKExtractor(
            CQTTopKConfig(
                sample_rate=44_100,
                hop_length=4,
                n_bins=4,
                top_k=1,
                backend="librosa",
                silence_threshold_ratio=0.5,
                silence_threshold_abs=1e-6,
            )
        )

        def fake_highpass(audio: torch.Tensor) -> torch.Tensor:
            return audio

        def fake_cqt(audio: torch.Tensor) -> torch.Tensor:
            magnitude = torch.zeros((1, 2, 4, 3), dtype=torch.float32)
            magnitude[:, :, 2, 1] = 0.01
            magnitude[:, 0, 1, 2] = 5.0
            magnitude[:, 1, 3, 2] = 7.0
            return magnitude

        extractor._highpass = fake_highpass  # type: ignore[method-assign]
        extractor._cqt_with_librosa = fake_cqt  # type: ignore[method-assign]

        melody = extractor.extract(torch.ones((1, 2, 16), dtype=torch.float32))

        self.assertTrue(torch.equal(melody[:, :, :2], torch.zeros_like(melody[:, :, :2])))
        torch.testing.assert_close(melody[0, :, 2], torch.tensor([2, 4], dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
