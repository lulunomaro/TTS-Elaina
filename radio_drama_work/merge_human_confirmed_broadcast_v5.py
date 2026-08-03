# -*- coding: utf-8 -*-
"""Merge v4 strict radio clips with final B013/B024 edge rescues."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf


ROOT = Path(__file__).resolve().parent
INPUTS = (
    ROOT / "human_confirmed_broadcast_v4_train_1p5_6s.list",
    ROOT / "human_verified_full_semantic_v2_edge_final2_v1.list",
)
OUT = ROOT / "human_confirmed_broadcast_v5_train_1p5_6s.list"
REPORT = ROOT / "human_confirmed_broadcast_v5_report.json"


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
            if not path.is_file():
                raise FileNotFoundError(path)
            text = fields[3].strip()
            if not text or "/ufffd" in text:
                raise RuntimeError(f"Invalid text: {line!r}")
            normalized = str(path).lower()
            sha = digest(path)
            if normalized in seen_paths:
                raise RuntimeError(f"Duplicate path: {path}")
            if sha in seen_hashes:
                raise RuntimeError(f"Duplicate waveform: {path}")
            info = sf.info(str(path))
            if (
                info.samplerate != 32000
                or info.channels != 1
                or info.subtype != "PCM_16"
                or not 1.5 <= info.duration <= 6.0
            ):
                raise RuntimeError(f"Invalid train-safe audio: {path} {info}")
            seen_paths.add(normalized)
            seen_hashes.add(sha)
            rows.append(
                {
                    "path": str(path),
                    "language": fields[2],
                    "text": text,
                    "duration": float(info.duration),
                    "sha256": sha,
                    "source_list": source_list.name,
                }
            )
    if len(rows) != 89:
        raise RuntimeError(f"Expected 89 merged rows, got {len(rows)}")
    OUT.write_text(
        "/n".join(
            f"{row['path']}|ireina_human_confirmed_v5|{row['language']}|{row['text']}"
            for row in rows
        )
        + "/n",
        encoding="utf-8",
    )
    report = {
        "version": "human_confirmed_broadcast_v5_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": input_counts,
        "count": len(rows),
        "duration_sec": sum(row["duration"] for row in rows),
        "list": str(OUT.resolve()),
        "checks": {
            "paths_exist": True,
            "unique_paths": True,
            "unique_waveform_hashes": True,
            "format_32k_mono_pcm16": True,
            "duration_1p5_to_6s": True,
            "valid_text": True,
        },
        "training_started": False,
        "existing_lists_modified": False,
        "score_90_claimed": False,
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "/n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
