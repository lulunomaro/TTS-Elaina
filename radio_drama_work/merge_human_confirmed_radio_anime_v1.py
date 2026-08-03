# -*- coding: utf-8 -*-
"""Merge the strict human-confirmed radio and anime candidate lists."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf


ROOT = Path(__file__).resolve().parent
INPUTS = (
    ROOT / "human_confirmed_broadcast_v5_train_1p5_6s.list",
    ROOT / "human_verified_anime_semantic_v1_direct.list",
)
OUT = ROOT / "human_confirmed_radio_anime_v1_train_1p5_6s.list"
REPORT = ROOT / "human_confirmed_radio_anime_v1_report.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    rows = []
    seen_paths = set()
    seen_hashes = set()
    input_counts = {}
    for source_list in INPUTS:
        lines = [
            line.strip()
            for line in source_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        input_counts[source_list.name] = len(lines)
        for line in lines:
            fields = line.split("|", 3)
            if len(fields) != 4:
                raise RuntimeError(f"Malformed list line: {line!r}")
            path = Path(fields[0]).resolve()
            text = fields[3].strip()
            if not path.is_file():
                raise FileNotFoundError(path)
            if not text or "/ufffd" in text:
                raise RuntimeError(f"Invalid text: {line!r}")
            path_key = str(path).lower()
            wave_hash = digest(path)
            if path_key in seen_paths:
                raise RuntimeError(f"Duplicate path: {path}")
            if wave_hash in seen_hashes:
                raise RuntimeError(f"Duplicate waveform: {path}")
            info = sf.info(str(path))
            if (
                info.samplerate != 32000
                or info.channels != 1
                or info.subtype != "PCM_16"
                or not 1.5 <= info.duration <= 6.0
            ):
                raise RuntimeError(f"Invalid candidate audio: {path} {info}")
            seen_paths.add(path_key)
            seen_hashes.add(wave_hash)
            rows.append(
                {
                    "path": str(path),
                    "speaker": fields[1],
                    "language": fields[2],
                    "text": text,
                    "duration": float(info.duration),
                    "sha256": wave_hash,
                    "source_list": source_list.name,
                }
            )

    if len(rows) != 100:
        raise RuntimeError(f"Expected 100 strict human-confirmed rows, got {len(rows)}")
    OUT.write_text(
        "/n".join(
            f"{row['path']}|{row['speaker']}|{row['language']}|{row['text']}"
            for row in rows
        )
        + "/n",
        encoding="utf-8",
    )
    report = {
        "version": "human_confirmed_radio_anime_v1_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": input_counts,
        "count": len(rows),
        "duration_sec": sum(row["duration"] for row in rows),
        "duration_min": sum(row["duration"] for row in rows) / 60.0,
        "list": str(OUT.resolve()),
        "checks": {
            "paths_exist": True,
            "unique_paths": True,
            "unique_waveform_hashes": True,
            "format_32k_mono_pcm16": True,
            "duration_1p5_to_6s": True,
            "valid_text": True,
            "old_lists_modified": False,
        },
        "training_started": False,
        "score_90_claimed": False,
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "/n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
