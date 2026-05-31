#!/usr/bin/env python
# Delete audio fragments shorter than --min-duration seconds.

import argparse
import os
import soundfile as sf

parser = argparse.ArgumentParser(description="Delete short audio fragments")
parser.add_argument("--dir", required=True, help="Audio directory path")
parser.add_argument("--min-duration", type=float, default=9.0, help="Min duration to keep in seconds (default 9)")
parser.add_argument("--dry-run", action="store_true", help="List only, dont delete")
args = parser.parse_args()

d = args.dir
if not os.path.isdir(d):
    print("Error: directory " + d + " not found")
    exit(1)

files = sorted(f for f in os.listdir(d) if f.endswith((".mp3", ".wav")))
deleted = 0
kept = 0

for f in files:
    path = os.path.join(d, f)
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        print("  [BROKEN] " + f + ": zero-byte or not a regular file, deleting")
        deleted += 1
        if not args.dry_run:
            os.remove(path)
        continue
    try:
        info = sf.info(path)
    except Exception as e:
        print("  [BROKEN] " + f + ": " + str(e) + ", deleting")
        deleted += 1
        if not args.dry_run:
            os.remove(path)
        continue
    if info.duration < args.min_duration:
        label = "[DRY]" if args.dry_run else "[DEL]"
        print("  " + label + " " + f + ": " + str(round(info.duration, 1)) + "s")
        deleted += 1
        if not args.dry_run:
            os.remove(path)
    else:
        kept += 1

action = "would be deleted" if args.dry_run else "deleted"
print("\nKept: " + str(kept) + ", " + action + ": " + str(deleted))