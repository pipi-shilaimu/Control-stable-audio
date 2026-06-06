from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


DEFAULT_EXTENSIONS = ".wav,.mp3,.flac,.ogg,.m4a"
DEFAULT_INCLUDE_TAGS = ("instrument---piano",)
DEFAULT_HARD_EXCLUDE_TAGS = ("instrument---voice",)
DEFAULT_SOFT_EXCLUDE_TAGS = (
    "instrument---drums",
    "instrument---drum",
    "instrument---drummachine",
    "instrument---bass",
    "instrument---electricguitar",
    "instrument---synthesizer",
    "instrument---synth",
    "instrument---computer",
)
DEFAULT_BOOST_TAGS = (
    "genre---classical",
    "genre---easylistening",
    "genre---soundtrack",
    "genre---newage",
    "mood/theme---relaxing",
    "mood/theme---emotional",
)
DEFAULT_PROMPT_HIT_TERMS = ("piano", "pianoforte", "piano melody")
DEFAULT_PROMPT_PENALTY_TERMS = ("vocal", "voice", "singing", "sung", "lyrics", "rap")
DEFAULT_SOURCE_TSV_CANDIDATES = (
    "data/raw_30s_cleantags.tsv",
    "data/raw_30s_cleantags_50artists.tsv",
    "data/raw_30s.tsv",
    "data/autotagging.tsv",
    "data/autotagging_top50tags.tsv",
    "data/autotagging_instrument.tsv",
)


@dataclass(frozen=True)
class PianoCandidateConfig:
    include_tags: tuple[str, ...] = DEFAULT_INCLUDE_TAGS
    hard_exclude_tags: tuple[str, ...] = DEFAULT_HARD_EXCLUDE_TAGS
    soft_exclude_tags: tuple[str, ...] = DEFAULT_SOFT_EXCLUDE_TAGS
    boost_tags: tuple[str, ...] = DEFAULT_BOOST_TAGS
    prompt_hit_terms: tuple[str, ...] = DEFAULT_PROMPT_HIT_TERMS
    prompt_penalty_terms: tuple[str, ...] = DEFAULT_PROMPT_PENALTY_TERMS
    max_soft_excludes: int = 2
    min_accept_score: float = 1.0
    min_maybe_score: float = 0.35


@dataclass(frozen=True)
class MTGTrack:
    track_id: str
    artist_id: str
    album_id: str
    mtg_path: str
    duration: float
    tags: tuple[str, ...]
    track_name: str = ""
    artist_name: str = ""
    album_name: str = ""


@dataclass(frozen=True)
class PianoCandidateRow:
    rank: int
    decision: str
    tag_score: float
    track_id: str
    filename: str
    audio_path: str
    mtg_path: str
    duration: float
    artist_id: str
    album_id: str
    artist_name: str
    track_name: str
    album_name: str
    tags: str
    include_tags_hit: str
    hard_exclude_tags_hit: str
    soft_exclude_tags_hit: str
    boost_tags_hit: str
    prompt: str
    prompt_hits: str
    prompt_penalties: str
    reason: str


def _default_mtg_root() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "mtg-jamendo-dataset"


def _split_csv_tokens(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _split_tags(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        parts = value.replace(",", "\t").split("\t")
    else:
        parts = list(value)
    return tuple(dict.fromkeys(part.strip() for part in parts if part and part.strip()))


def _join_values(values: Iterable[str]) -> str:
    return ",".join(values)


def _normalized_track_id_from_digits(digits: str) -> str:
    return f"track_{int(digits):07d}"


def extract_track_id_from_name(name: str | Path) -> str | None:
    """Extract an MTG-Jamendo track id from paths like 1000113_seg000.mp3 or 82/382.mp3."""

    normalized = str(name).replace("\\", "/")
    filename = normalized.rsplit("/", 1)[-1]
    stem = Path(filename).stem

    track_match = re.search(r"track[_-](\d{1,9})", stem, flags=re.IGNORECASE)
    if track_match:
        return _normalized_track_id_from_digits(track_match.group(1))

    segment_match = re.search(r"^(\d{1,9})(?:[_-]seg\d+)?$", stem, flags=re.IGNORECASE)
    if segment_match:
        return _normalized_track_id_from_digits(segment_match.group(1))

    digit_match = re.search(r"(\d{1,9})", stem)
    if digit_match:
        return _normalized_track_id_from_digits(digit_match.group(1))
    return None


def _read_variable_tag_tsv(path: str | Path) -> dict[str, dict[str, str]]:
    tsv_path = Path(path)
    rows: dict[str, dict[str, str]] = {}
    if not tsv_path.exists():
        return rows

    lines = tsv_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return rows
    headers = lines[0].split("\t")
    fixed_headers = headers[:5]
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        track_id = parts[0]
        row = {header: parts[index] if index < len(parts) else "" for index, header in enumerate(fixed_headers)}
        row["TAGS"] = "\t".join(parts[5:]) if len(parts) > 5 else ""
        rows[track_id] = row
    return rows


def _read_meta_tsv(path: str | Path) -> dict[str, dict[str, str]]:
    meta_path = Path(path)
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        return {str(row["TRACK_ID"]): dict(row) for row in reader}


def _resolve_source_tsv(mtg_root: str | Path, source_tsv: str | Path | None = None) -> Path:
    root = Path(mtg_root)
    if source_tsv:
        candidate = Path(source_tsv)
        return candidate if candidate.is_absolute() else root / candidate
    for relative in DEFAULT_SOURCE_TSV_CANDIDATES:
        candidate = root / relative
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find an MTG tag TSV under {root}. Tried: {', '.join(DEFAULT_SOURCE_TSV_CANDIDATES)}"
    )


def load_mtg_tracks(
    mtg_root: str | Path,
    *,
    source_tsv: str | Path | None = None,
    extra_tag_tsvs: Iterable[str | Path] = (),
) -> dict[str, MTGTrack]:
    root = Path(mtg_root)
    resolved_source = _resolve_source_tsv(root, source_tsv)
    base_rows = _read_variable_tag_tsv(resolved_source)

    merged_tags: dict[str, set[str]] = {
        track_id: set(_split_tags(row.get("TAGS", ""))) for track_id, row in base_rows.items()
    }
    for extra in extra_tag_tsvs:
        extra_path = Path(extra)
        if not extra_path.is_absolute():
            extra_path = root / extra_path
        for track_id, row in _read_variable_tag_tsv(extra_path).items():
            if track_id not in base_rows:
                base_rows[track_id] = row
            merged_tags.setdefault(track_id, set()).update(_split_tags(row.get("TAGS", "")))

    meta_rows = _read_meta_tsv(root / "data" / "raw.meta.tsv")
    tracks: dict[str, MTGTrack] = {}
    for track_id, row in base_rows.items():
        meta = meta_rows.get(track_id, {})
        duration = 0.0
        try:
            duration = float(row.get("DURATION", "0") or 0.0)
        except ValueError:
            duration = 0.0
        tracks[track_id] = MTGTrack(
            track_id=track_id,
            artist_id=row.get("ARTIST_ID", ""),
            album_id=row.get("ALBUM_ID", ""),
            mtg_path=row.get("PATH", ""),
            duration=duration,
            tags=tuple(sorted(merged_tags.get(track_id, set()))),
            track_name=meta.get("TRACK_NAME", ""),
            artist_name=meta.get("ARTIST_NAME", ""),
            album_name=meta.get("ALBUM_NAME", ""),
        )
    return tracks


def _find_audio_files(
    audio_dir: str | Path,
    *,
    extensions: Iterable[str],
    recursive: bool = True,
    max_files: int | None = None,
) -> list[Path]:
    root = Path(audio_dir)
    if not root.exists():
        raise FileNotFoundError(f"Audio directory not found: {root}")
    normalized_exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    iterator = root.rglob("*") if recursive else root.glob("*")
    files = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in normalized_exts)
    if max_files is not None:
        files = files[: max(0, int(max_files))]
    return files


def _load_prompt_manifest(path: str | Path | None) -> dict[str, str]:
    if not path:
        return {}
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Prompt manifest not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    prompts: dict[str, str] = {}
    for key, value in data.items():
        prompt = ""
        if isinstance(value, Mapping):
            prompt = str(value.get("prompt") or value.get("caption") or value.get("text") or "")
        elif value is not None:
            prompt = str(value)
        prompts[str(key).replace("\\", "/").lstrip("./")] = prompt
        prompts[Path(str(key)).name] = prompt
    return prompts


def _make_prompt_from_tags(tags: Iterable[str]) -> str:
    names = []
    for tag in tags:
        clean = tag.split("---", 1)[-1].replace("_", " ").strip()
        if clean and clean not in names:
            names.append(clean)
    if not names:
        return "instrumental piano music"
    if "piano" not in names:
        names.insert(0, "piano")
    return ", ".join(names[:8]) + ", instrumental melody"


def _prompt_for_file(
    *,
    prompts: Mapping[str, str],
    audio_dir: Path | None,
    audio_path: Path | None,
    filename: str,
    tags: Iterable[str],
) -> str:
    candidates = [filename]
    if audio_dir is not None and audio_path is not None:
        try:
            candidates.append(audio_path.relative_to(audio_dir).as_posix())
        except ValueError:
            pass
    for key in candidates:
        prompt = prompts.get(key)
        if prompt:
            return prompt
    return _make_prompt_from_tags(tags)


def _terms_found(prompt: str, terms: Iterable[str]) -> tuple[str, ...]:
    lowered = prompt.lower()
    return tuple(term for term in terms if term.lower() in lowered)


def _score_track(
    *,
    tags: tuple[str, ...],
    prompt: str,
    config: PianoCandidateConfig,
) -> tuple[str, float, tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...], str]:
    tag_set = set(tags)
    include_hits = tuple(tag for tag in config.include_tags if tag in tag_set)
    hard_hits = tuple(tag for tag in config.hard_exclude_tags if tag in tag_set)
    soft_hits = tuple(tag for tag in config.soft_exclude_tags if tag in tag_set)
    boost_hits = tuple(tag for tag in config.boost_tags if tag in tag_set)
    prompt_hits = _terms_found(prompt, config.prompt_hit_terms)
    prompt_penalties = _terms_found(prompt, config.prompt_penalty_terms)

    if not include_hits:
        return (
            "reject",
            0.0,
            include_hits,
            hard_hits,
            soft_hits,
            boost_hits,
            prompt_hits,
            prompt_penalties,
            "missing required piano tag",
        )

    instrument_tags = tuple(tag for tag in tags if tag.startswith("instrument---"))
    soloish_instruments = set(config.include_tags) | {"instrument---electricpiano", "instrument---keyboard"}
    soloish_bonus = 0.15 if instrument_tags and all(tag in soloish_instruments for tag in instrument_tags) else 0.0
    score = (
        1.0
        + min(0.45, 0.15 * len(boost_hits))
        + min(0.20, 0.10 * len(prompt_hits))
        + soloish_bonus
        - 0.80 * len(hard_hits)
        - 0.25 * len(soft_hits)
        - min(0.35, 0.12 * len(prompt_penalties))
    )
    score = round(max(0.0, score), 6)

    if hard_hits:
        decision = "reject"
        reason = "hard excluded by " + _join_values(hard_hits)
    elif len(soft_hits) > int(config.max_soft_excludes):
        decision = "maybe" if score >= float(config.min_maybe_score) else "reject"
        reason = "too many accompaniment tags: " + _join_values(soft_hits)
    elif score >= float(config.min_accept_score):
        decision = "accept"
        reason = "piano candidate"
    elif score >= float(config.min_maybe_score):
        decision = "maybe"
        reason = "piano present but weak tag confidence"
    else:
        decision = "reject"
        reason = "low tag confidence"

    return (
        decision,
        score,
        include_hits,
        hard_hits,
        soft_hits,
        boost_hits,
        prompt_hits,
        prompt_penalties,
        reason,
    )


def _row_for_track(
    *,
    track: MTGTrack | None,
    track_id: str,
    filename: str,
    audio_path: str,
    prompt: str,
    config: PianoCandidateConfig,
) -> PianoCandidateRow:
    if track is None:
        return PianoCandidateRow(
            rank=0,
            decision="reject",
            tag_score=0.0,
            track_id=track_id,
            filename=filename,
            audio_path=audio_path,
            mtg_path="",
            duration=0.0,
            artist_id="",
            album_id="",
            artist_name="",
            track_name="",
            album_name="",
            tags="",
            include_tags_hit="",
            hard_exclude_tags_hit="",
            soft_exclude_tags_hit="",
            boost_tags_hit="",
            prompt=prompt,
            prompt_hits="",
            prompt_penalties="",
            reason="missing MTG metadata",
        )

    (
        decision,
        score,
        include_hits,
        hard_hits,
        soft_hits,
        boost_hits,
        prompt_hits,
        prompt_penalties,
        reason,
    ) = _score_track(tags=track.tags, prompt=prompt, config=config)
    return PianoCandidateRow(
        rank=0,
        decision=decision,
        tag_score=score,
        track_id=track.track_id,
        filename=filename,
        audio_path=audio_path,
        mtg_path=track.mtg_path,
        duration=round(float(track.duration), 3),
        artist_id=track.artist_id,
        album_id=track.album_id,
        artist_name=track.artist_name,
        track_name=track.track_name,
        album_name=track.album_name,
        tags="\t".join(track.tags),
        include_tags_hit=_join_values(include_hits),
        hard_exclude_tags_hit=_join_values(hard_hits),
        soft_exclude_tags_hit=_join_values(soft_hits),
        boost_tags_hit=_join_values(boost_hits),
        prompt=prompt,
        prompt_hits=_join_values(prompt_hits),
        prompt_penalties=_join_values(prompt_penalties),
        reason=reason,
    )


def _rank_rows(rows: list[PianoCandidateRow]) -> list[PianoCandidateRow]:
    decision_order = {"accept": 0, "maybe": 1, "reject": 2}
    ordered = sorted(
        rows,
        key=lambda row: (
            decision_order.get(row.decision, 9),
            -float(row.tag_score),
            len(_split_csv_tokens(row.soft_exclude_tags_hit)),
            row.track_id,
            row.filename,
        ),
    )
    return [replace(row, rank=index + 1) for index, row in enumerate(ordered)]


def build_piano_candidate_rows(
    *,
    mtg_root: str | Path | None = None,
    audio_dir: str | Path | None = None,
    prompt_manifest: str | Path | None = None,
    source_tsv: str | Path | None = None,
    extra_tag_tsvs: Iterable[str | Path] = (),
    extensions: Iterable[str] = _split_csv_tokens(DEFAULT_EXTENSIONS),
    recursive: bool = True,
    max_files: int | None = None,
    config: PianoCandidateConfig = PianoCandidateConfig(),
) -> list[PianoCandidateRow]:
    root = Path(mtg_root) if mtg_root is not None else _default_mtg_root()
    tracks = load_mtg_tracks(root, source_tsv=source_tsv, extra_tag_tsvs=extra_tag_tsvs)
    prompts = _load_prompt_manifest(prompt_manifest)

    rows: list[PianoCandidateRow] = []
    if audio_dir is None:
        for track_id, track in tracks.items():
            filename = Path(track.mtg_path).name or f"{track_id}.mp3"
            prompt = _make_prompt_from_tags(track.tags)
            rows.append(
                _row_for_track(
                    track=track,
                    track_id=track_id,
                    filename=filename,
                    audio_path=track.mtg_path,
                    prompt=prompt,
                    config=config,
                )
            )
        return _rank_rows(rows)

    root_audio_dir = Path(audio_dir)
    files = _find_audio_files(root_audio_dir, extensions=extensions, recursive=recursive, max_files=max_files)
    for audio_path in files:
        filename = audio_path.name
        track_id = extract_track_id_from_name(filename) or ""
        track = tracks.get(track_id)
        tags = track.tags if track is not None else ()
        prompt = _prompt_for_file(
            prompts=prompts,
            audio_dir=root_audio_dir,
            audio_path=audio_path,
            filename=filename,
            tags=tags,
        )
        rows.append(
            _row_for_track(
                track=track,
                track_id=track_id,
                filename=filename,
                audio_path=str(audio_path),
                prompt=prompt,
                config=config,
            )
        )
    return _rank_rows(rows)


def write_candidates_csv(path: str | Path, rows: list[PianoCandidateRow]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(PianoCandidateRow.__dataclass_fields__.keys())
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_candidates_jsonl(path: str | Path, rows: list[PianoCandidateRow]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as fp:
        for row in rows:
            fp.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def select_rows_for_export(
    rows: list[PianoCandidateRow],
    *,
    decisions: Iterable[str] = ("accept",),
    max_per_track: int | None = None,
    limit: int | None = None,
) -> list[PianoCandidateRow]:
    allowed = set(decisions)
    per_track_counts: dict[str, int] = {}
    selected: list[PianoCandidateRow] = []
    for row in rows:
        if row.decision not in allowed:
            continue
        if max_per_track is not None:
            used = per_track_counts.get(row.track_id, 0)
            if used >= int(max_per_track):
                continue
            per_track_counts[row.track_id] = used + 1
        selected.append(row)
        if limit is not None and len(selected) >= int(limit):
            break
    return selected


def copy_selected_audio_files(rows: list[PianoCandidateRow], copy_audio_dir: str | Path) -> int:
    output_dir = Path(copy_audio_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for row in rows:
        source = Path(row.audio_path)
        if not source.exists():
            continue
        shutil.copy2(source, output_dir / row.filename)
        copied += 1
    return copied


def write_manifest_seed(
    path: str | Path,
    rows: list[PianoCandidateRow],
    *,
    decisions: Iterable[str] = ("accept",),
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    allowed = set(decisions)
    manifest: dict[str, dict[str, object]] = {}
    for row in rows:
        if row.decision not in allowed:
            continue
        manifest[row.filename] = {
            "prompt": row.prompt or "instrumental piano melody",
            "source": "mtg_jamendo_piano_candidate",
            "track_id": row.track_id,
            "artist_id": row.artist_id,
            "album_id": row.album_id,
            "artist_name": row.artist_name,
            "track_name": row.track_name,
            "album_name": row.album_name,
            "mtg_path": row.mtg_path,
            "tags": row.tags,
            "decision": row.decision,
            "tag_score": row.tag_score,
            "soft_exclude_tags": row.soft_exclude_tags_hit,
            "boost_tags": row.boost_tags_hit,
        }
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def write_candidate_outputs(
    *,
    rows: list[PianoCandidateRow],
    output_csv: str | Path | None = None,
    output_jsonl: str | Path | None = None,
    output_manifest: str | Path | None = None,
    manifest_decisions: Iterable[str] = ("accept",),
    copy_audio_dir: str | Path | None = None,
    copy_decisions: Iterable[str] | None = None,
    export_max_per_track: int | None = None,
    export_limit: int | None = None,
) -> None:
    if output_csv:
        write_candidates_csv(output_csv, rows)
    if output_jsonl:
        write_candidates_jsonl(output_jsonl, rows)
    if output_manifest:
        manifest_rows = select_rows_for_export(
            rows,
            decisions=manifest_decisions,
            max_per_track=export_max_per_track,
            limit=export_limit,
        )
        write_manifest_seed(output_manifest, manifest_rows, decisions=manifest_decisions)
    if copy_audio_dir:
        selected_rows = select_rows_for_export(
            rows,
            decisions=tuple(copy_decisions) if copy_decisions is not None else tuple(manifest_decisions),
            max_per_track=export_max_per_track,
            limit=export_limit,
        )
        copy_selected_audio_files(selected_rows, copy_audio_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a fast MTG-Jamendo piano-candidate list before expensive melody-clear CQT scoring."
    )
    parser.add_argument("--mtg-root", type=str, default=str(_default_mtg_root()))
    parser.add_argument("--source-tsv", type=str, default=None, help="Relative to --mtg-root unless absolute.")
    parser.add_argument(
        "--extra-tag-tsvs",
        type=str,
        default="",
        help="Comma-separated extra MTG TSVs to merge tags from, relative to --mtg-root unless absolute.",
    )
    parser.add_argument("--audio-dir", type=str, default=None, help="Optional segmented audio dir, e.g. audio_10s.")
    parser.add_argument("--prompt-manifest", type=str, default=None, help="Optional JSON manifest with prompts per segment.")
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, default=None)
    parser.add_argument("--output-manifest", type=str, default=None)
    parser.add_argument("--copy-audio-dir", type=str, default=None)
    parser.add_argument(
        "--manifest-decisions",
        type=str,
        default="accept",
        help="Comma-separated decisions to include in --output-manifest, e.g. accept,maybe.",
    )
    parser.add_argument(
        "--copy-decisions",
        type=str,
        default=None,
        help="Comma-separated decisions to copy; defaults to --manifest-decisions.",
    )
    parser.add_argument("--export-max-per-track", type=int, default=None)
    parser.add_argument("--export-limit", type=int, default=None)
    parser.add_argument("--extensions", type=str, default=DEFAULT_EXTENSIONS)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-files", type=int, default=None)

    parser.add_argument("--include-tags", type=str, default=",".join(DEFAULT_INCLUDE_TAGS))
    parser.add_argument("--hard-exclude-tags", type=str, default=",".join(DEFAULT_HARD_EXCLUDE_TAGS))
    parser.add_argument("--soft-exclude-tags", type=str, default=",".join(DEFAULT_SOFT_EXCLUDE_TAGS))
    parser.add_argument("--boost-tags", type=str, default=",".join(DEFAULT_BOOST_TAGS))
    parser.add_argument("--prompt-hit-terms", type=str, default=",".join(DEFAULT_PROMPT_HIT_TERMS))
    parser.add_argument("--prompt-penalty-terms", type=str, default=",".join(DEFAULT_PROMPT_PENALTY_TERMS))
    parser.add_argument("--max-soft-excludes", type=int, default=2)
    parser.add_argument("--min-accept-score", type=float, default=1.0)
    parser.add_argument("--min-maybe-score", type=float, default=0.35)
    return parser


def config_from_args(args: argparse.Namespace) -> PianoCandidateConfig:
    return PianoCandidateConfig(
        include_tags=_split_csv_tokens(args.include_tags),
        hard_exclude_tags=_split_csv_tokens(args.hard_exclude_tags),
        soft_exclude_tags=_split_csv_tokens(args.soft_exclude_tags),
        boost_tags=_split_csv_tokens(args.boost_tags),
        prompt_hit_terms=_split_csv_tokens(args.prompt_hit_terms),
        prompt_penalty_terms=_split_csv_tokens(args.prompt_penalty_terms),
        max_soft_excludes=int(args.max_soft_excludes),
        min_accept_score=float(args.min_accept_score),
        min_maybe_score=float(args.min_maybe_score),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows = build_piano_candidate_rows(
        mtg_root=args.mtg_root,
        audio_dir=args.audio_dir,
        prompt_manifest=args.prompt_manifest,
        source_tsv=args.source_tsv,
        extra_tag_tsvs=_split_csv_tokens(args.extra_tag_tsvs),
        extensions=_split_csv_tokens(args.extensions),
        recursive=bool(args.recursive),
        max_files=args.max_files,
        config=config_from_args(args),
    )
    write_candidate_outputs(
        rows=rows,
        output_csv=args.output_csv,
        output_jsonl=args.output_jsonl,
        output_manifest=args.output_manifest,
        manifest_decisions=_split_csv_tokens(args.manifest_decisions),
        copy_audio_dir=args.copy_audio_dir,
        copy_decisions=_split_csv_tokens(args.copy_decisions) if args.copy_decisions else None,
        export_max_per_track=args.export_max_per_track,
        export_limit=args.export_limit,
    )

    counts: dict[str, int] = {}
    for row in rows:
        counts[row.decision] = counts.get(row.decision, 0) + 1
    print(f"ranked {len(rows)} MTG piano candidates -> {args.output_csv}")
    print("decision_counts=" + ", ".join(f"{key}:{counts[key]}" for key in sorted(counts)))
    if args.output_manifest:
        print(f"manifest seed -> {args.output_manifest}")
    if args.copy_audio_dir:
        print(f"copied selected audio -> {args.copy_audio_dir}")
    for row in rows[:10]:
        print(
            f"rank={row.rank} decision={row.decision} score={row.tag_score:.3f} "
            f"track={row.track_id} soft=[{row.soft_exclude_tags_hit}] file={row.filename}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
