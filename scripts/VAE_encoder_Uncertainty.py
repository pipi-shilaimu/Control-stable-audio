"""独立验证：stable_audio_tools VAE encoder 非确定性。不依赖本项目任何代码。"""
from sympy import im
import torch
import torchaudio
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stable_audio_control.audio_io import install_torchaudio_load_fallback
install_torchaudio_load_fallback()
from stable_audio_tools import get_pretrained_model
torch.backends.cudnn.deterministic = True   # 关掉 CUDA 非确定性算子
torch.backends.cudnn.benchmark = False      # 关掉自动调优（它会引入非确定性）
model, config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
pretransform = model.pretransform
pretransform = pretransform.to("cuda")
audio_path = r"C:\PROJECT\StableAudio\stable_audio_control\data\mtg_jamendo\train_30\0095400_seg000.wav"
real_audio, sr = torchaudio.load(audio_path)
real_audio = real_audio.unsqueeze(0).to("cuda")[:, :, :65536]  # 取前 1.5s
# 同一段随机音频，encode 两次
NUM_REPEATS = 20  # 改这个数字控制重复次数
for _ in range(NUM_REPEATS):
    audio = real_audio

    with torch.no_grad():
        raw = pretransform.model.encoder(audio)
        latent_1 = pretransform.encode(audio)
        latent_2 = pretransform.encode(audio)

    diff = (latent_1 - latent_2).abs().max().item()
    print(f"Same input, two encodes → max latent diff: {diff:.6f}")
    print(f"diff: {diff}")