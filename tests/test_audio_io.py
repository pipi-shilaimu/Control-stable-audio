from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf
import torch
import torchaudio


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import stable_audio_control.audio_io as audio_io


class AudioIOFallbackTests(unittest.TestCase):
    def test_torchaudio_load_falls_back_to_soundfile_on_torchcodec_failure(self) -> None:
        original_load = torchaudio.load
        module = importlib.reload(audio_io)

        def fail_with_torchcodec(*args, **kwargs):
            raise RuntimeError("Could not load libtorchcodec")

        with TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "tone.wav"
            expected = np.stack(
                [
                    np.linspace(-0.25, 0.25, num=64, dtype=np.float32),
                    np.linspace(0.25, -0.25, num=64, dtype=np.float32),
                ],
                axis=1,
            )
            sf.write(audio_path, expected, 16_000, format="WAV", subtype="FLOAT")

            try:
                torchaudio.load = fail_with_torchcodec
                module.install_torchaudio_load_fallback()

                audio, sample_rate = torchaudio.load(str(audio_path), format="wav")
            finally:
                torchaudio.load = original_load
                importlib.reload(module)

        self.assertEqual(sample_rate, 16_000)
        self.assertEqual(tuple(audio.shape), (2, 64))
        torch.testing.assert_close(audio, torch.from_numpy(expected).T)


if __name__ == "__main__":
    unittest.main()
