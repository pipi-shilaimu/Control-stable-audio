import json, os
from pathlib import Path
import soundfile as sf
import librosa
ROOT = Path("C:/PROJECT/StableAudio")
ad = ROOT / "stable_audio_control/data/mtg_jamendo/train"
manifest = json.loads((ROOT / "stable_audio_control/data/mtg_jamendo/manifests/train.json").read_text("utf-8"))
segdir = ROOT / "stable_audio_control/data/mtg_jamendo/train_10s"
segdir.mkdir(parents=True, exist_ok=True)
TARGET_SR = 44100; SEG_FRAMES = 10 * TARGET_SR
nm = {}; cnt = 0
for fn, entry in sorted(manifest.items()):
    src = ad / fn
    if not src.exists(): continue
    a, sr = librosa.load(str(src), sr=None, mono=False)
    if a.ndim == 1: a = a[None, :]
    if a.shape[0] == 1: a = a.repeat(2, axis=0)
    if sr != TARGET_SR: a = librosa.resample(a, orig_sr=sr, target_sr=TARGET_SR)
    tf = a.shape[1]
    n = 0
    for start in range(0, tf, SEG_FRAMES):
        if start + SEG_FRAMES > tf: continue
        seg = a[:, start:start+SEG_FRAMES]
        sfn = fn.replace(".wav", f"_seg{n:03d}.wav")
        sf.write(str(segdir/sfn), seg.T, TARGET_SR, format="WAV", subtype="PCM_16")
        e = dict(entry); e["source_segment"] = f"{fn}:{start//TARGET_SR}s"
        nm[sfn] = e; n += 1
    if n == 0: print(f"  WARN {fn} too short ({tf/TARGET_SR:.1f}s)")
    else: print(f"{fn} ({tf/TARGET_SR:.0f}s) -> {n} segs"); cnt += n
mdir = segdir.parent / "manifests"
mdir.mkdir(parents=True, exist_ok=True)
(mdir / "train_10s.json").write_text(json.dumps(nm, indent=2), "utf-8")
print(f"\nTotal: {cnt} segments")
