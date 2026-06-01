#!/usr/bin/env python
"""一键组织数据集：给定音频目录和 prompt JSON，生成 metadata 模块可用的目录结构和 config。

同时提供双向一致性校验，确保音频与 manifest 条目完全对齐。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


def validate_manifest_audio(
    audio_dir: Path,
    manifest: dict[str, Any],
    *,
    audio_extensions: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """校验音频目录和 manifest JSON 是否双向一致。

    Args:
        audio_dir: 存放音频文件的目录。
        manifest: manifest 字典，key 为音频文件名。
        audio_extensions: 要计入统计的音频扩展名（如 ('.wav', '.flac')）。
                          传入 None 则统计目录下所有文件。

    Returns:
        dict:
          - manifest_count:   manifest 条目总数
          - audio_count:      音频文件总数
          - missing_audio:    manifest 中有但磁盘缺少的文件名（排序后）
          - missing_manifest: 磁盘有但 manifest 缺少的文件名（排序后）
          - is_consistent:    双向完全匹配
    """
    # 收集音频文件名
    if audio_extensions is not None:
        exts_lower = tuple(e.lower() for e in audio_extensions)
        audio_filenames = {
            f.name for f in audio_dir.iterdir()
            if f.is_file() and f.name.lower().endswith(exts_lower)
        }
    else:
        audio_filenames = {f.name for f in audio_dir.iterdir() if f.is_file()}

    manifest_keys = set(manifest.keys())

    missing_audio = sorted(k for k in manifest_keys if k not in audio_filenames)
    missing_manifest = sorted(f for f in audio_filenames if f not in manifest_keys)

    return {
        "manifest_count": len(manifest_keys),
        "audio_count": len(audio_filenames),
        "missing_audio": missing_audio,
        "missing_manifest": missing_manifest,
        "is_consistent": len(missing_audio) == 0
        and len(missing_manifest) == 0
        and len(manifest_keys) == len(audio_filenames),
    }


def _report_validation(result: dict[str, Any]) -> list[str]:
    """根据 validate_manifest_audio 的返回值生成人类可读的报告行。"""
    lines: list[str] = []

    count_ok = result["manifest_count"] == result["audio_count"]
    lines.append(
        f"manifest 条目: {result['manifest_count']}, "
        f"音频文件: {result['audio_count']}"
        + ("  (数量一致)" if count_ok else "  ⚠ 数量不一致!")
    )

    if result["is_consistent"]:
        lines.append("✓ 双向校验通过：音频与 manifest 完全对齐。")
        return lines

    if result["missing_audio"]:
        lines.append(
            f"✗ manifest 中有 {len(result['missing_audio'])} 个条目缺少对应音频文件:"
        )
        for name in result["missing_audio"][:10]:
            lines.append(f"    - {name}")
        if len(result["missing_audio"]) > 10:
            lines.append(f"    ... 及其他 {len(result['missing_audio']) - 10} 个")

    if result["missing_manifest"]:
        lines.append(
            f"✗ 磁盘上有 {len(result['missing_manifest'])} 个音频文件缺少对应 manifest 条目:"
        )
        for name in result["missing_manifest"][:10]:
            lines.append(f"    - {name}")
        if len(result["missing_manifest"]) > 10:
            lines.append(f"    ... 及其他 {len(result['missing_manifest']) - 10} 个")

    return lines


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="组织数据集：校验音频与 manifest 双向一致，生成目录结构和 config。"
    )
    parser.add_argument("--audio-dir", type=str, required=True,
                        help="音频文件所在目录")
    parser.add_argument("--manifest", type=str, required=True,
                        help="prompt JSON 文件路径")
    parser.add_argument(
        "--strict", action="store_true", default=False,
        help="发现不一致时以非零退出码终止（默认仅打印警告）"
    )
    parser.add_argument(
        "--audio-extensions", type=str, nargs="+", default=None,
        help="要计入统计的音频扩展名，默认统计所有文件（如 --audio-extensions .wav .flac 则只统计指定扩展名）"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    audio_dir = Path(args.audio_dir)
    manifest_path = Path(args.manifest)

    if not audio_dir.is_dir():
        print(f"[ERROR] 音频目录不存在: {audio_dir}", file=sys.stderr)
        return 1
    if not manifest_path.is_file():
        print(f"[ERROR] manifest 文件不存在: {manifest_path}", file=sys.stderr)
        return 1

    # 加载 manifest
    manifest: dict[str, Any] = json.loads(manifest_path.read_text("utf-8"))

    # --- 双向一致性校验 ---
    extensions: tuple[str, ...] | None = None
    if args.audio_extensions is not None:
        extensions = tuple(args.audio_extensions)

    result = validate_manifest_audio(audio_dir, manifest, audio_extensions=extensions)

    for line in _report_validation(result):
        print(line)

    if not result["is_consistent"] and args.strict:
        print("\n[ERROR] 严格模式：音频与 manifest 不一致，终止。", file=sys.stderr)
        return 1

    # --- 组织输出 ---
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
