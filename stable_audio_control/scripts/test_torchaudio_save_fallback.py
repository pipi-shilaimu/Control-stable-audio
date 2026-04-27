from pathlib import Path
import tempfile
import sys

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_audio_control.audio_io import install_torchaudio_save_fallback


def main() -> None:
    install_torchaudio_save_fallback()

    output_path = Path(tempfile.gettempdir()) / "stableaudio_torchcodec_fallback_test.wav"
    if output_path.exists():
        output_path.unlink()

    audio = torch.zeros((1, 1600), dtype=torch.int16)
    torchaudio.save(str(output_path), audio, 16000)

    assert output_path.exists(), f"Expected output file to exist: {output_path}"
    assert output_path.stat().st_size > 44, "Saved wav looks too small to be valid."
    print("PASS:", output_path)


if __name__ == "__main__":
    main()
