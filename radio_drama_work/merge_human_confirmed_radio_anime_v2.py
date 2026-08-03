# -*- coding: utf-8 -*-
"""Merge frozen human-approved radio/anime clips into new, validated lists."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parent

INPUTS = [
    ("radio_v5", ROOT / "human_confirmed_broadcast_v5_train_1p5_6s.list"),
    ("anime_direct_v1", ROOT / "human_verified_anime_semantic_v1_direct.list"),
    ("anime_review_v1", ROOT / "human_verified_anime_review_batch_v1.list"),
    ("anime_boundary_rescue_v1", ROOT / "human_verified_anime_boundary_rescue_v1.list"),
]
ANIME_LIST = ROOT / "human_verified_anime_v2_train_1p5_6s.list"
COMBINED_LIST = ROOT / "human_confirmed_radio_anime_v2_train_1p5_6s.list"
REPORT = ROOT / "human_confirmed_radio_anime_v2_report.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_list(path: Path, group: str) -> list[dict]:
    rows = []
    if not path.exists():
        raise FileNotFoundError(path)
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split("|", 3)
        if len(parts) != 4:
            raise ValueError(f"{path}:{line_number}: expected 4 pipe-separated fields")
        audio, speaker, language, text = parts
        audio_path = Path(audio).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"{path}:{line_number}: {audio_path}")
        info = sf.info(str(audio_path))
        if info.samplerate != 32000 or info.channels != 1 or info.subtype != "PCM_16":
            raise ValueError(f"invalid format {audio_path}: {info}")
        if not 1.5 <= float(info.duration) <= 6.0:
            raise ValueError(f"invalid duration {audio_path}: {info.duration}")
        if not text.strip() or "/ufffd" in text:
            raise ValueError(f"invalid text {audio_path}: {text!r}")
        rows.append(
            {
                "group": group,
                "audio": str(audio_path),
                "speaker": speaker,
                "language": language,
                "text": text,
                "duration": float(info.duration),
                "sha256": sha256(audio_path),
            }
        )
    return rows


def write_list(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            f"{row['audio']}|{row['speaker']}|{row['language']}|{row['text']}/n"
            for row in rows
        ),
        encoding="utf-8",
    )


def main() -> None:
    grouped = {group: parse_list(path, group) for group, path in INPUTS}
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    rows: list[dict] = []
    for group, _path in INPUTS:
        for row in grouped[group]:
            if row["audio"] in seen_paths:
                raise ValueError(f"duplicate path: {row['audio']}")
            if row["sha256"] in seen_hashes:
                raise ValueError(f"duplicate audio hash: {row['audio']}")
            seen_paths.add(row["audio"])
            seen_hashes.add(row["sha256"])
            rows.append(row)

    anime_rows = [row for row in rows if row["group"].startswith("anime_")]
    write_list(ANIME_LIST, anime_rows)
    write_list(COMBINED_LIST, rows)

    report = {
        "version": "human_confirmed_radio_anime_v2_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": [
            {"group": group, "path": str(path.resolve()), "count": len(grouped[group]), "duration_sec": sum(row["duration"] for row in grouped[group])}
            for group, path in INPUTS
        ],
        "anime_only": {"count": len(anime_rows), "duration_sec": sum(row["duration"] for row in anime_rows), "list": str(ANIME_LIST.resolve())},
        "combined": {"count": len(rows), "duration_sec": sum(row["duration"] for row in rows), "list": str(COMBINED_LIST.resolve())},
        "unique_paths": len(seen_paths),
        "unique_hashes": len(seen_hashes),
        "format_policy": "32 kHz mono PCM16; 1.5-6.0 sec; nonempty non-FFFD text",
        "training_started": False,
        "existing_training_lists_modified": False,
        "score_90_claimed": False,
        "groups": dict(Counter(row["group"] for row in rows)),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "/n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
