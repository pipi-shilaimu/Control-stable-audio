#!/usr/bin/env python
"""MTG-Jamendo incremental downloader.

Batch workflow: download tars -> extract -> build manifest -> upload -> upload -> manually delete -> repeat.
--count N downloads the next N unprocessed tars.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import requests
from tqdm import tqdm

# Paths
ROOT = Path(__file__).resolve().parent
DATASET_REPO = ROOT / "mtg-jamendo-dataset"
DOWNLOADS_DIR = ROOT / "mtg_jamendo_downloads"
OUTPUT_DIR = ROOT / "mtg_jamendo_full"

GIDS_FILE = DATASET_REPO / "data/download/raw_30s_audio_gids.txt"
SHA256_TARS_FILE = DATASET_REPO / "data/download/raw_30s_audio_sha256_tars.txt"
SHA256_TRACKS_FILE = DATASET_REPO / "data/download/raw_30s_audio_sha256_tracks.txt"
TSV_FILE = DATASET_REPO / "data/raw_30s_cleantags.tsv"
META_FILE = DATASET_REPO / "data/raw.meta.tsv"
STATE_FILE = OUTPUT_DIR / ".download_state.json"

CHUNK_SIZE = 512 * 1024
# State management
def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text("utf-8"))
    return {"completed_tars": [], "last_tar_index": -1}

def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), "utf-8")

# Load metadata: tags + song/artist names
def load_track_metadata() -> dict[str, dict]:
    tracks: dict[str, dict] = {}
    with open(TSV_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            path = row["PATH"]
            tracks[path] = {"track_id": row["TRACK_ID"], "tags": row["TAGS"]}
    meta_lookup: dict[str, dict] = {}
    with open(META_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            meta_lookup[row["TRACK_ID"]] = row
    for t in tracks.values():
        m = meta_lookup.get(t["track_id"])
        if m:
            t["track_name"] = m.get("TRACK_NAME", "")
            t["artist_name"] = m.get("ARTIST_NAME", "")
    return tracks

def make_prompt(tags_str: str) -> str:
    parts = []
    for token in tags_str.split("\t"):
        token = token.strip()
        if not token:
            continue
        if "---" in token:
            parts.append(token.split("---", 1)[1])
        else:
            parts.append(token)
    if not parts:
        return "instrumental music"
    s = ", ".join(parts)
    if "instrument" not in s:
        s += " instrumental"
    return s

# Download helpers
def load_tar_list() -> list[dict]:
    gid_map: dict[str, str] = {}
    with open(GIDS_FILE, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                gid_map[parts[1]] = parts[0]
    tars: list[dict] = []
    with open(SHA256_TARS_FILE, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                tars.append({"filename": parts[1], "sha256": parts[0], "file_id": gid_map.get(parts[1], "")})
    return tars

def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def download_from_mtg_fast(filename: str, output: str) -> None:
    url = f"https://cdn.freesound.org/mtg-jamendo/raw_30s/audio/{filename}"
    output_path = Path(output)
    tmp_file = None
    try:
        res = requests.get(url, stream=True, timeout=30)
        res.raise_for_status()
        total = int(res.headers.get("Content-Length", 0))
        with tempfile.NamedTemporaryFile(prefix=output_path.name, dir=output_path.parent, delete=False) as tmp:
            tmp_file = tmp.name
            with tqdm(total=total, unit="B", unit_scale=True, desc=filename) as pbar:
                for chunk in res.iter_content(chunk_size=CHUNK_SIZE):
                    tmp.write(chunk)
                    pbar.update(len(chunk))
        shutil.move(tmp_file, output)
        tmp_file = None
    except Exception:
        if tmp_file and os.path.exists(tmp_file):
            os.unlink(tmp_file)
        raise

def download_from_gdrive(file_id: str, output: str) -> None:
    import gdown
    gdown.download(id=file_id, output=output, quiet=False)
# Extract tar -> manifest
def extract_and_build_manifest(tar_path: str, audio_dir: Path, tracks_meta: dict[str, dict], sha256_track_map: dict[str, str]) -> dict[str, dict]:
    manifest: dict[str, dict] = {}
    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".mp3"):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            content = f.read()
            actual_sha = hashlib.sha256(content).hexdigest()
            expected_sha = sha256_track_map.get(member.name)
            if expected_sha and actual_sha != expected_sha:
                print(f"  WARN {member.name}: SHA256 mismatch, skipping", file=sys.stderr)
                continue
            meta = tracks_meta.get(member.name)
            if meta is None:
                continue
            track_num = meta["track_id"].replace("track_", "")
            out_filename = f"{track_num}.mp3"
            out_path = audio_dir / out_filename
            out_path.write_bytes(content)
            prompt = make_prompt(meta["tags"])
            manifest[out_filename] = {
                "prompt": prompt, "source": "mtg_jamendo", "tags": meta["tags"],
                "track_name": meta.get("track_name", ""),
                "artist_name": meta.get("artist_name", ""),
                "track_id": meta["track_id"],
            }
    return manifest

# Status
def show_status() -> None:
    state = load_state()
    tars = load_tar_list()
    done = len(state["completed_tars"])
    total = len(tars)
    pct = done / total * 100 if total > 0 else 0
    print(f"进度: {done}/{total} 个 tar ({pct:.1f}%)")
    if done < total:
        print(f"下一个待下载: {tars[done]['filename']}")
    if state["completed_tars"]:
        few = state["completed_tars"][:5]
        print(f"已完成: {', '.join(few)}{' ...' if len(state['completed_tars']) > 5 else ''}")

def reset_state() -> None:
    if STATE_FILE.exists():
        STATE_FILE.unlink()
    print("状态已重置。下次运行将从 tar-00 开始。")
def run_batch(count: int, download_from: str) -> int:
    state = load_state()
    tars = load_tar_list()
    start_idx = state["last_tar_index"] + 1
    if start_idx >= len(tars):
        print("所有 tar 已下载完成！")
        return 0
    batch_tars = tars[start_idx : start_idx + count]
    tracks_meta = load_track_metadata()
    sha256_track_map: dict[str, str] = {}
    with open(SHA256_TRACKS_FILE, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                sha256_track_map[parts[1]] = parts[0]
    audio_dir = OUTPUT_DIR / "audio"
    manifest_dir = OUTPUT_DIR / "manifests"
    if audio_dir.exists():
        print(f"错误: 输出目录 {audio_dir} 已存在，说明上一批还没上传完。")
        print("请先上传并删除该目录，再跑下一批。")
        return 0
    if manifest_dir.exists():
        print(f"错误: 输出目录 {manifest_dir} 已存在，说明上一批还没上传完。")
        print("请先上传并删除该目录，再跑下一批。")
        return 0
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    all_manifest: dict[str, dict] = {}
    for idx, tar_info in enumerate(batch_tars):
        filename = tar_info["filename"]
        tar_path = str(DOWNLOADS_DIR / filename)
        print(f"\n{'='*60}")
        print(f"  [{filename}]  第 {start_idx + idx + 1}/{len(tars)} 个 tar")
        print(f"{'='*60}")
        print(f"  下载中...")
        if download_from == "gdrive":
            download_from_gdrive(tar_info["file_id"], tar_path)
        else:
            download_from_mtg_fast(filename, tar_path)
        print(f"  校验 SHA256...")
        actual_sha = compute_sha256(tar_path)
        if actual_sha != tar_info["sha256"]:
            os.unlink(tar_path)
            print(f"  SHA256 校验失败，已删除。请重试。", file=sys.stderr)
            return len(all_manifest)
        print(f"  校验通过")
        print(f"  解压并生成 manifest...")
        manifest = extract_and_build_manifest(tar_path, audio_dir, tracks_meta, sha256_track_map)
        all_manifest.update(manifest)
        print(f"  本批累计: {len(all_manifest)} 首曲目")
        state["completed_tars"].append(filename)
        state["last_tar_index"] = start_idx + idx
        save_state(state)
    manifest_path = manifest_dir / "train.json"
    manifest_path.write_text(json.dumps(dict(sorted(all_manifest.items())), indent=2, ensure_ascii=False), "utf-8")
    mm_path = ROOT / "mtg_jamendo_metadata.py"
    config_path = OUTPUT_DIR / "dataset_config_train.json"
    config_path.write_text(json.dumps({
        "dataset_type": "audio_dir",
        "datasets": [{"id": "mtg_jamendo_full", "path": str(audio_dir.resolve()), "custom_metadata_module": str(mm_path.resolve())}],
        "random_crop": True,
    }, indent=2, ensure_ascii=False), "utf-8")
    print(f"\n{'='*60}")
    print(f"  本批处理完成: {len(all_manifest)} 首曲目")
    print(f"  Manifest: {manifest_path}")
    print(f"  Config: {config_path}")
    print(f"  MP3 目录: {audio_dir.resolve()}")
    print(f"{'='*60}")
    print(f"  下一步:")
    print(f"  1. 上传 {OUTPUT_DIR.resolve()} 整个目录到服务器")
    print(f"  2. 上传后删掉本地 {OUTPUT_DIR.resolve()}\audio 和 manifests")
    print(f"  3. 再跑 python download_mtg.py --count {count} 继续下一批")
    return len(all_manifest)

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MTG-Jamendo 增量下载器")
    parser.add_argument("--count", type=int, default=None, help="这次下载几个 tar（每个约 5GB）")
    parser.add_argument("--from", default="mtg-fast", choices=["mtg-fast", "gdrive"], dest="download_from", help="下载源")
    parser.add_argument("--status", action="store_true", help="查看进度")
    parser.add_argument("--reset", action="store_true", help="重置进度")
    return parser

def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.status:
        show_status()
        return 0
    if args.reset:
        reset_state()
        return 0
    if args.count is None:
        print("请指定 --count N 或使用 --status / --reset")
        return 1
    if args.count <= 0:
        print("--count 必须是正数")
        return 1
    run_batch(args.count, args.download_from)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())