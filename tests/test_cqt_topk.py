from __future__ import annotations

import unittest

import torch

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


if __name__ == "__main__":
    unittest.main()
