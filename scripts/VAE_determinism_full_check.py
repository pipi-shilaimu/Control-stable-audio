"""全面验证 VAE encoder 非确定性来源。

测试五个维度，每组重复 N 次取统计：
v1: 默认设置 + 长音频 (baseline)
v2: 默认设置 + 短音频 (避开 chunked encoding)
v3: cudnn.deterministic + 长音频
v4: cudnn.deterministic + 短音频
v5: 真实音频 (MTG Jamendo piano)

用法：python scripts/VAE_determinism_full_check.py [--repeat 100]
"""
import sys
import argparse
import torch
import torchaudio
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--repeat", type=int, default=50, help="每组测试重复次数")
args = parser.parse_args()
NUM_REPEATS = args.repeat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stable_audio_control.audio_io import install_torchaudio_load_fallback
install_torchaudio_load_fallback()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(False)

from stable_audio_tools import get_pretrained_model

print("Loading model...")
model, config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
pretransform = model.pretransform.to("cuda").eval()
print("Done.\n")
for _ in range(1000):
    long_audio = torch.randn(1, 2, 2097152, device="cuda")
    short_audio = torch.randn(1, 2, 65536, device="cuda")

    audio_path = str(Path(__file__).resolve().parents[1] / "stable_audio_control/data/mtg_jamendo/train_30/0095400_seg000.wav")
    real_audio, sr = torchaudio.load(audio_path)
    real_audio = real_audio.unsqueeze(0).to("cuda")[:, :, :65536]

    tests = [
        ("baseline (long)",          long_audio,  False),
        ("baseline (short)",         short_audio, False),
        ("deterministic (long)",     long_audio,  True),
        ("deterministic (short)",    short_audio, True),
        ("real audio (short)",       real_audio,  True),
    ]

    for name, audio, use_deterministic in tests:
        if use_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True


        with torch.no_grad():
            l1 = pretransform.encode(audio)
            l2 = pretransform.encode(audio)
        diff = (l1 - l2).abs().max().item()
        print(f"[{name:25s}] max diff: {diff:.6f}")