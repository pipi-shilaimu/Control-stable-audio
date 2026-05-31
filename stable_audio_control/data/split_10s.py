#!/usr/bin/env python
"""Cut audio files into non-overlapping 10s MP3 segments using ffmpeg.

Usage:
    python split_10s.py --audio-dir G:\PROJECT\StableAudio\audios
    python split_10s.py --audio-dir G:\PROJECT\StableAudio\audios --output-dir G:\PROJECT\StableAudio\audios_10s
"""

import argparse
import subprocess
from pathlib import Path

SEG_DURATION = 10

parser = argparse.ArgumentParser()
parser.add_argument("--audio-dir", type=str, required=True, help="Directory containing source audio files")
parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: {audio_dir}_10s)")
args = parser.parse_args()

audio_dir = Path(args.audio_dir)
output_dir = Path(args.output_dir) if args.output_dir else audio_dir.with_name(audio_dir.name + "_10s")
output_dir.mkdir(parents=True, exist_ok=True)

total_segments = 0
files = sorted(audio_dir.iterdir())

for src in files:
    if not src.is_file():
        continue
    fn = src.name

    # Get duration via ffprobe
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(src)],
        capture_output=True, text=True, timeout=30,
    )
    try:
        duration = float(probe.stdout.strip())
    except (ValueError, TypeError):
        print(f"  WARN {fn}: couldn't read duration, skipping")
        continue

    n_segs = int(duration // SEG_DURATION)
    if n_segs == 0:
        print(f"  WARN {fn} too short ({duration:.1f}s)")
        continue

    base = fn.rsplit(".", 1)[0]
    out_pattern = str(output_dir / (base + "_seg%03d.mp3"))

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src),
         "-c", "copy",
         "-f", "segment", "-segment_time", str(SEG_DURATION),
         "-reset_timestamps", "1",
         out_pattern],
        capture_output=True, timeout=300,
    )

    print(f"{fn} ({int(duration)}s) -> {n_segs} segs")
    total_segments += n_segs

print(f"\nDone: {total_segments} segments -> {output_dir}")