
import soundfile as sf
import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

# Download model
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

model = model.to(device)

# Set up text and timing conditioning
conditioning = [{
    "prompt": "Lofi,study,chill,relaxed,lazybass, slow anafternoon,ambient, drum and Smooth, 60 BPM",
    "seconds_start": 0,
    "seconds_total": 60
}]

# Generate stereo audio
my_seed = random.randint(0, 2**31 - 1)
output = generate_diffusion_cond(
    model,
    steps=100,
    cfg_scale=7,
    conditioning=conditioning,
    sample_size=sample_size,
    sigma_min=0.3,
    sigma_max=500,
    sampler_type="dpmpp-3m-sde",
    device=device,
    seed=my_seed
)

# Rearrange audio batch to a single sequence
output = rearrange(output, "b d n -> d (b n)")

# Peak normalize, clip, convert to int16, and save to file
output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
#torchaudio.save("output.wav", output, sample_rate,backend="soundfile")
audio_data = output.squeeze().cpu().numpy().T
sf.write("output.wav", audio_data, sample_rate)
print("音频已成功保存为 output.wav")
