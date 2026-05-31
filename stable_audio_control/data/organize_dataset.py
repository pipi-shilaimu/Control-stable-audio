#!/usr/bin/env python
"""一键组织数据集：给定音频目录和 prompt JSON，生成 metadata 模块可用的目录结构和 config。"""
import argparse, json, shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--audio-dir", type=str, required=True)
parser.add_argument("--manifest", type=str, required=True)
args = parser.parse_args()

audio_dir = Path(args.audio_dir)
manifest_path = Path(args.manifest)

# 加载 manifest
manifest = json.loads(manifest_path.read_text("utf-8"))
audio_files = {f.name for f in audio_dir.iterdir() if f.is_file()}
missing = [k for k in manifest if k not in audio_files]
if missing:
    print(f"[WARN] {len(missing)} manifest entries have no audio file")

# 输出路径：manifests/{audio_dir上一级名}.json, dataset_config 放 audio_dir 同级
dataset_root = audio_dir.parent.parent
manifest_dir = dataset_root / "manifests"
manifest_target = manifest_dir / f"{audio_dir.parent.name}.json"
config_target = audio_dir.parent / "dataset_config_train.json"

manifest_dir.mkdir(parents=True, exist_ok=True)
if manifest_path.resolve() != manifest_target.resolve():
    shutil.copy2(str(manifest_path), str(manifest_target))
    print(f"manifest -> {manifest_target}")
else:
    print(f"manifest already in place: {manifest_target}")

config = {
    "dataset_type": "audio_dir",
    "datasets": [{
        "id": audio_dir.parent.name,
        "path": str(audio_dir).replace(chr(92), "/"),
        "custom_metadata_module": "stable_audio_control/data/mtg_jamendo_metadata.py",
    }],
    "random_crop": False,
}
config_target.write_text(json.dumps(config, indent=2, ensure_ascii=False), "utf-8")
print(f"config  -> {config_target}")
print("Done. 用 --dataset-config " + str(config_target) + " 启动训练。")