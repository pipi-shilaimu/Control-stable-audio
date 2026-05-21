filepath = r"C:\PROJECT\StableAudio\stable_audio_control\scripts\train_controlnet_dit.py"
with open(filepath) as f:
    lines = f.readlines()
new_lines = []
for i, line in enumerate(lines):
    if 176 <= i <= 181:
        continue
    if i == 718 and "enable_progress_bar" in line:
        continue
    new_lines.append(line)
with open(filepath, "w") as f:
    f.writelines(new_lines)
import py_compile
py_compile.compile(filepath, doraise=True)
print("Syntax OK")
