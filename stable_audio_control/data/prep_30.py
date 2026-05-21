import json, shutil, os
from pathlib import Path

ROOT = Path("C:/PROJECT/StableAudio")
src_dir = ROOT / "stable_audio_control/data/mtg_jamendo/train_10s"
manifest = json.loads((ROOT / "stable_audio_control/data/mtg_jamendo/manifests/train_10s.json").read_text("utf-8"))

picked = [
    "0095400_seg000.wav", "0116600_seg000.wav", "0116600_seg010.wav",
    "0202200_seg000.wav", "0399500_seg000.wav", "0501300_seg000.wav",
    "0633200_seg000.wav", "0661300_seg000.wav", "0752300_seg000.wav",
    "0757500_seg000.wav", "0816200_seg000.wav", "0847200_seg000.wav",
    "0903600_seg000.wav", "0920000_seg000.wav", "1028900_seg000.wav",
    "1041500_seg000.wav", "1066200_seg000.wav", "1105300_seg000.wav",
    "1116400_seg000.wav", "1158200_seg000.wav", "1158200_seg015.wav",
    "1173500_seg000.wav", "1211600_seg000.wav", "1300500_seg000.wav",
    "1300500_seg020.wav", "1357400_seg000.wav", "1374300_seg000.wav",
    "1385300_seg000.wav", "1396500_seg000.wav", "1416200_seg000.wav",
]

dst_dir = ROOT / "stable_audio_control/data/mtg_jamendo/train_30"
man_dir = ROOT / "stable_audio_control/data/mtg_jamendo/manifests"
dst_dir.mkdir(parents=True, exist_ok=True)

# Clean old
for f in os.listdir(dst_dir):
    if f.endswith(".wav"):
        os.remove(dst_dir / f)

new_manifest = {}
for fn in picked:
    src = src_dir / fn
    if not src.exists():
        print(f"MISSING {fn}")
        continue
    shutil.copy2(str(src), str(dst_dir / fn))
    new_manifest[fn] = manifest[fn]
    print(f"  {fn}")

(man_dir / "train_30.json").write_text(json.dumps(new_manifest, indent=2), "utf-8")
print(f"\nCopied {len(new_manifest)} files")

# Write dataset_config
mm = ROOT / "stable_audio_control/data/mtg_jamendo_metadata.py"
cp = ROOT / "stable_audio_control/data/mtg_jamendo/dataset_config_train_30.json"
cp.write_text(json.dumps({
    "dataset_type": "audio_dir",
    "datasets": [{
        "id": "mtg_jamendo_piano_30",
        "path": str(dst_dir.resolve()),
        "custom_metadata_module": str(mm.resolve()),
    }],
    "random_crop": False,
}, indent=2), "utf-8")
print(f"Config: {cp}")
