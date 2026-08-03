# -*- coding: utf-8 -*-
"""Merge prior human-confirmed audio with the 13 direct v2 blind passes."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "human_confirmed_broadcast_v2_train_1p5_6s.list"
NEW = ROOT / "human_verified_full_semantic_v2_direct.list"
OUTPUT = ROOT / "human_confirmed_broadcast_v3_train_1p5_6s.list"
REPORT = ROOT / "human_confirmed_broadcast_v3_report.json"


def read_list(path: Path, provenance: str) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("|", 3)
        if len(parts) != 4:
            raise RuntimeError(f"Malformed row {path}:{line_number}")
        rows.append(
            {
                "audio": parts[0],
                "speaker": parts[1],
                "language": parts[2],
                "text": parts[3],
                "provenance": provenance,
            }
        )
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    rows = read_list(BASE, "human_confirmed_broadcast_v2") + read_list(
        NEW, "human_verified_full_semantic_v2_direct"
    )
    seen_paths = set()
    seen_hashes = set()
    validated = []
    for row in rows:
        path = Path(row["audio"]).resolve()
        normalized = str(path).casefold()
        if normalized in seen_paths:
            raise RuntimeError(f"Duplicate path: {path}")
        seen_paths.add(normalized)
        if not path.is_file():
            raise FileNotFoundError(path)
        info = sf.info(str(path))
        if (
            info.samplerate != 32000
            or info.channels != 1
            or info.subtype != "PCM_16"
            or not 1.5 <= info.duration <= 6.0
        ):
            raise RuntimeError(f"Invalid audio: {path} {info}")
        if not row["text"].strip() or "/ufffd" in row["text"] or "|" in row["text"]:
            raise RuntimeError(f"Invalid text: {path}")
        digest = sha256(path)
        if digest in seen_hashes:
            raise RuntimeError(f"Duplicate waveform hash: {path}")
        seen_hashes.add(digest)
        validated.append(
            {
                **row,
                "audio": str(path),
                "duration": float(info.duration),
                "sha256": digest,
            }
        )

    OUTPUT.write_text(
        "/n".join(
            f"{row['audio']}|ireina_human_confirmed_broadcast_v3|JA|{row['text']}"
            for row in validated
        )
        + "/n",
        encoding="utf-8",
    )
    by_provenance = {}
    for provenance in sorted({row["provenance"] for row in validated}):
        selected = [row for row in validated if row["provenance"] == provenance]
        by_provenance[provenance] = {
            "count": len(selected),
            "duration_sec": sum(row["duration"] for row in selected),
        }
    report = {
        "version": "human_confirmed_broadcast_v3_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "count": len(validated),
        "duration_sec": sum(row["duration"] for row in validated),
        "by_provenance": by_provenance,
        "validation": {
            "all_paths_exist": True,
            "unique_paths": True,
            "unique_waveform_hashes": True,
            "format": "32kHz mono PCM16",
            "duration": "1.5-6.0 seconds",
            "human_heard_exact_waveforms": True,
        },
        "upstream_interval_dedupe": "v2 machine pool was source-interval deduped against the prior human pool before blind selection",
        "output": str(OUTPUT),
        "training_started": False,
        "existing_training_lists_modified": False,
        "score_90_claimed": False,
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "/n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
