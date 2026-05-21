"""Custom metadata module."""
from __future__ import annotations
import json, os
from pathlib import Path
_MANIFEST_CACHE = {}
_MODULE_DIR = Path(__file__).resolve().parent
def get_custom_metadata(info, audio):
    r = str(info.get("relpath","")).replace("\\","/").lstrip("./")
    f = os.path.basename(r)
    ap = Path(str(info.get("path","")))
    mp = ap.parent.parent / "manifests" / f"{ap.parent.name}.json"
    e = os.environ.get("STABLE_AUDIO_CONTROL_METADATA_MANIFEST")
    if not mp.exists() and e: mp = Path(e)
    if not mp.exists(): mp = _MODULE_DIR / "mtg_jamendo" / "manifests" / "train.json"
    if not mp.exists(): return {"prompt": "music"}
    rs = str(mp.resolve())
    if rs not in _MANIFEST_CACHE: _MANIFEST_CACHE[rs] = json.loads(mp.read_text("utf-8"))
    m = _MANIFEST_CACHE[rs]; e = m.get(f) or m.get(r)
    if e is None: return {"prompt": "music"}
    p = e["prompt"] if isinstance(e, dict) else str(e); d = {"prompt": p}
    if isinstance(e, dict):
        for k, v in e.items():
            if k != "prompt": d[f"mtg_{k}"] = v
    return d
