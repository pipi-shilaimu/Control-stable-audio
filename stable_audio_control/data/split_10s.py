#!/usr/bin/env python
"""Cut audio files into non-overlapping 10s MP3 segments using ffmpeg.

Usage:
    python split_10s.py --audio-dir mtg_jamendo_full/audio/no_vocals
    python split_10s.py  # defaults to mtg_jamendo/train
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

TARGET_SR = 44100
SEG_DURATION = 10

SCRIPT_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument("--audio-dir", type=str, default=str(SCRIPT_DIR / "mtg_jamendo/train"))
parser.add_argument("--manifest", type=str, default=None)
args = parser.parse_args()

ad = Path(args.audio_dir)
if not ad.is_absolute():
    ad = Path.cwd() / ad

manifest = None
if args.manifest is not None:
    mpath = Path(args.manifest)
    if not mpath.is_absolute():
        mpath = Path.cwd() / mpath
    manifest = json.loads(mpath.read_text("utf-8"))

segdir = ad.with_name(ad.name + "_10s")
segdir.mkdir(parents=True, exist_ok=True)

nm = {}
cnt = 0
files = sorted(ad.iterdir())

for src in files:
    if not src.is_file():
        continue
    fn = src.name
    entry = manifest.get(fn) if manifest else None

    # Get duration
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(src)],
        capture_output=True, text=True, timeout=30,
    )
    try:
        duration = float(probe.stdout.strip())
    except (ValueError, TypeError):
        print("  WARN " + fn + ": couldn't read duration, skipping")
        continue

    total_segs = int(duration // SEG_DURATION)
    if total_segs == 0:
        print("  WARN " + fn + " too short (" + str(round(duration, 1)) + "s)")
        continue

    base = fn.rsplit(".", 1)[0]
    out_pattern = str(segdir / (base + "_seg%03d.mp3"))

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src),
         "-c", "copy",
         "-f", "segment", "-segment_time", str(SEG_DURATION),
         "-reset_timestamps", "1",
         out_pattern],
        capture_output=True, timeout=300,
    )

    # Build manifest entries from generated files
    for seg_idx in range(total_segs):
        sfn = base + "_seg" + str(seg_idx).zfill(3) + ".mp3"
        if (segdir / sfn).exists():
            if entry is not None:
                e = dict(entry)
                e["source_segment"] = fn + ":" + str(seg_idx * SEG_DURATION) + "s"
                nm[sfn] = e

    print(fn + " (" + str(int(duration)) + "s) -> " + str(total_segs) + " segs")
    cnt += total_segs

if manifest:
    mfr = segdir.parent / "manifests"
    mfr.mkdir(parents=True, exist_ok=True)
    (mfr / "train_10s.json").write_text(json.dumps(nm, indent=2, ensure_ascii=False), "utf-8")
    print("Manifest: " + str(mfr / "train_10s.json"))

print("Total: " + str(cnt) + " segments -> " + str(segdir))