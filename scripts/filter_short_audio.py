"""删除切分后产生的末尾碎片文件（远小于正常 10 秒音频）。"""
import os
from pathlib import Path

TARGET_DIR = Path(r"stable_audio_control\data\mtg_jamendo_full\audio\no_vocals_10s")
MIN_SIZE_BYTES = 50_000  # 小于 50KB 的视为碎片（正常 10s MP3 ~400KB）

if not TARGET_DIR.exists():
    print(f"目录不存在: {TARGET_DIR}")
    exit(1)

audio_files = sorted(TARGET_DIR.glob("*.mp3")) + sorted(TARGET_DIR.glob("*.wav"))

deleted = []
for f in audio_files:
    size = f.stat().st_size
    if size < MIN_SIZE_BYTES:
        os.remove(f)
        deleted.append((f.name, size))
        print(f"  删除 {f.name}  ({size} bytes)")

print(f"\n共删除 {len(deleted)} 个碎片文件")