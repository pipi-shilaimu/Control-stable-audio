#!/usr/bin/env python
"""从 mtg_jamendo_full 中筛出不含 voice 的 MP3 到 no_vocals/ 子目录。"""

import json, os, shutil
from pathlib import Path

d = Path(__file__).resolve().parent / "mtg_jamendo_full"
m = json.loads((d / "manifests/train.json").read_text("utf-8"))

out = d / "audio/no_vocals"
out.mkdir(parents=True, exist_ok=True)

count = 0
for fn, entry in m.items():
    if "---voice" in entry.get("tags", ""):
        continue
    src = d / "audio" / fn
    if src.exists():
        shutil.copy2(str(src), str(out / fn))
        count += 1

print(f"Copied {count} no-vocal tracks to {out}")
