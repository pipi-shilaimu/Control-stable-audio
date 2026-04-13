from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_audio_control.test.SimpleModel import SimpleModel

model = SimpleModel("TestModel")
x = torch.randn(1, 10)
y = model(x)
print("x:", x)
print("y:", y)
