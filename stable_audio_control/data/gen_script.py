"""Regenerate MTG-Jamendo piano overfit dataset."""
import json, os, tarfile, tempfile
from pathlib import Path
import librosa
import soundfile as sf

ROOT = Path("C:/PROJECT/StableAudio")
D = ROOT / "stable_audio_control"
M = D / "data/mtg-jamendo-dataset"
TAR = D / "data/mtg_jamendo_downloads/autotagging_moodtheme_audio-low-00.tar"

def load_tsv(path):
    lines = path.read_text("utf-8").strip().splitlines()
    h = lines[0].split("\t")
    return {l.split("\t")[0]: dict(zip(h, l.split("\t"))) for l in lines[1:]}

def make_prompt(s):
    t = []
    for x in s.split("\t"):
        x = x.strip()
        if "---" in x: t.append(x.split("---", 1)[1])
        else: t.append(x)
    if not t: return "instrumental music"
    if len(t) == 1: return t[0] + " instrumental"
    return ", ".join(t) + " instrumental"

def main():
    I = load_tsv(M / "data/autotagging_instrument.tsv")
    M2 = load_tsv(M / "data/raw.meta.tsv")
    ids = [i for i, r in I.items()
           if any("---piano" in t for t in r["TAGS"].split("\t"))
           and not any("---voice" in t for t in r["TAGS"].split("\t"))
           and r["PATH"].startswith("00/")]
    ids.sort()
    print(f"Piano (no voice) in 00/: {len(ids)}")
    tracks = {i: I[i] for i in ids}
    ad = D / "data/mtg_jamendo/train"
    md = D / "data/mtg_jamendo/manifests"
    ad.mkdir(parents=True, exist_ok=True)
    md.mkdir(parents=True, exist_ok=True)
    exp = {i.split("_")[1] + ".wav" for i in ids}
    for f in os.listdir(ad):
        if f.endswith(".wav") and f not in exp:
            os.remove(ad / f)
            print(f"  rm stale: {f}")
    lookup = {}
    for i, r in tracks.items():
        b, e = os.path.splitext(r["PATH"])
        lookup[b + ".low" + e] = i
    manifest = {}
    with tarfile.open(str(TAR)) as tar:
        for m in tar.getmembers():
            if m.name not in lookup: continue
            i = lookup[m.name]; r = tracks[i]
            print(f"[{m.name}] -> {i}")
            ff = tar.extractfile(m)
            if ff is None: continue
            d = ff.read()
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tmp.close()
            try:
                open(tmp.name, "wb").write(d)
                a, sr = librosa.load(tmp.name, sr=None, mono=False)
                if a.ndim == 1: a = a[None, :]
                if a.shape[0] == 1: a = a.repeat(2, axis=0)
                tc = a.T; tsr = 44100
                if sr != tsr: tc = librosa.resample(tc.T, orig_sr=sr, target_sr=tsr).T
                fn = i.split("_")[1] + ".wav"
                sf.write(str(ad / fn), tc, tsr, format="WAV", subtype="PCM_16")
                prompt = make_prompt(r["TAGS"])
                mr = M2.get(i, {})
                manifest[fn] = {"prompt": prompt, "source": "mtg_jamendo_instrument",
                    "tags": r["TAGS"], "track_name": mr.get("TRACK_NAME", ""),
                    "artist_name": mr.get("ARTIST_NAME", ""), "track_id": i}
                print(f"  {fn}: {prompt}")
            finally:
                try: os.unlink(tmp.name)
                except: pass
    (md / "train.json").write_text(json.dumps(manifest, indent=2), "utf-8")
    print(f"Manifest: {len(manifest)} entries")
    # metadata module
    mm = D / "data/mtg_jamendo_metadata.py"
    mm.write_text(
        chr(34) + chr(34) + chr(34)+ "Custom metadata module." + chr(34) + chr(34) + chr(34) + "\n"
        + "from __future__ import annotations\nimport json, os\nfrom pathlib import Path\n"
        + "_MANIFEST_CACHE = {}\n_MODULE_DIR = Path(__file__).resolve().parent\n"
        + "def get_custom_metadata(info, audio):\n"
        + "    r = str(info.get(\"relpath\",\"\")).replace(\"\\\\\",\"/\").lstrip(\"./\")\n"
        + "    f = os.path.basename(r)\n"
        + "    ap = Path(str(info.get(\"path\",\"\")))\n"
        + "    mp = ap.parent.parent / \"manifests\" / f\"{ap.parent.name}.json\"\n"
        + "    e = os.environ.get(\"STABLE_AUDIO_CONTROL_METADATA_MANIFEST\")\n"
        + "    if not mp.exists() and e: mp = Path(e)\n"
        + "    if not mp.exists(): mp = _MODULE_DIR / \"mtg_jamendo\" / \"manifests\" / \"train.json\"\n"
        + "    if not mp.exists(): return {\"prompt\": \"music\"}\n"
        + "    rs = str(mp.resolve())\n"
        + "    if rs not in _MANIFEST_CACHE: _MANIFEST_CACHE[rs] = json.loads(mp.read_text(\"utf-8\"))\n"
        + "    m = _MANIFEST_CACHE[rs]; e = m.get(f) or m.get(r)\n"
        + "    if e is None: return {\"prompt\": \"music\"}\n"
        + "    p = e[\"prompt\"] if isinstance(e, dict) else str(e); d = {\"prompt\": p}\n"
        + "    if isinstance(e, dict):\n"
        + "        for k, v in e.items():\n"
        + "            if k != \"prompt\": d[f\"mtg_{k}\"] = v\n"
        + "    return d\n",
        "utf-8",
    )
    print(f"Metadata: {mm}")
    # dataset_config
    cp = D / "data/mtg_jamendo/dataset_config_train.json"
    cp.write_text(json.dumps({
        "dataset_type": "audio_dir",
        "datasets": [{"id": "mtg_jamendo_piano_overfit",
                      "path": str(ad.resolve()),
                      "custom_metadata_module": str(mm.resolve())}],
        "random_crop": True
    }, indent=2), "utf-8")
    print(f"DONE. {len(manifest)} piano tracks")

main()
