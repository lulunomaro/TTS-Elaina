# -*- coding: utf-8 -*-
"""Build a 48-clip stratified blind review from anime semantic candidates."""
from __future__ import annotations

import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_resource_policy import apply_cpu_friendly_policy


RESOURCE_POLICY = apply_cpu_friendly_policy(2)

import numpy as np
import soundfile as sf

import build_human_calibration_v1 as page_base


SOURCE_AUDIT = ROOT / "anime_semantic_quality_v1" / "audit_manifest.json"
OUT = ROOT / "anime_semantic_acceptance_v1"
AUDIO_DIR = OUT / "audio"
TARGET_COUNT = 48
PER_SOURCE_TARGET = 4
RANDOM_SEED = 202608022230


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8"
    )


def risk_key(row: dict) -> tuple:
    modes = row.get("boundary_modes", {})
    boundary_risk = int(
        "tight" in str(modes.get("left", ""))
        or "tight" in str(modes.get("right", ""))
    )
    quality = row.get("quality", {})
    return (
        boundary_risk,
        -float(row.get("minimum_anchor_margin", 1.0)),
        -float(row.get("minimum_target_center_delta", 1.0)),
        -float(row.get("minimum_asr_probability", 1.0)),
        float(quality.get("edge_energy_ratio", 0.0)),
        float(row.get("duration", 0.0)),
    )


def select_source(rows: list[dict], count: int, seed: int) -> list[dict]:
    if len(rows) <= count:
        return list(rows)
    chosen = []

    def add(row: dict) -> None:
        if row not in chosen:
            chosen.append(row)

    add(max(rows, key=risk_key))
    add(min(rows, key=lambda row: float(row.get("minimum_anchor_margin", 1.0))))
    add(min(rows, key=lambda row: float(row.get("minimum_target_center_delta", 1.0))))
    add(max(rows, key=lambda row: float(row.get("duration", 0.0))))
    generator = random.Random(seed)
    remainder = [row for row in rows if row not in chosen]
    generator.shuffle(remainder)
    for row in remainder:
        add(row)
        if len(chosen) >= count:
            break
    return chosen[:count]


def main() -> None:
    audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    rows = list(audit["samples"])
    if not rows:
        raise RuntimeError("No anime strict candidates available for acceptance")
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source_id"])].append(row)

    selected = []
    for index, source_id in enumerate(sorted(by_source)):
        selected.extend(
            select_source(
                by_source[source_id],
                PER_SOURCE_TARGET,
                RANDOM_SEED + 997 * index,
            )
        )
    if len(selected) < min(TARGET_COUNT, len(rows)):
        remainder = [row for row in rows if row not in selected]
        remainder.sort(key=risk_key, reverse=True)
        for row in remainder:
            selected.append(row)
            if len(selected) >= min(TARGET_COUNT, len(rows)):
                break
    selected = selected[: min(TARGET_COUNT, len(selected))]
    random.Random(RANDOM_SEED).shuffle(selected)

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    exported = []
    for index, row in enumerate(selected, 1):
        blind_id = f"S{index:03d}"
        source = Path(row["audio"])
        destination = AUDIO_DIR / f"{blind_id}.wav"
        shutil.copy2(source, destination)
        audio, sample_rate = sf.read(
            str(destination), dtype="float32", always_2d=True
        )
        audio = audio.mean(axis=1).astype(np.float32, copy=False)
        info = sf.info(str(destination))
        if (
            info.samplerate != 32000
            or info.channels != 1
            or info.subtype != "PCM_16"
            or not 1.5 <= info.duration <= 6.0
        ):
            raise RuntimeError(f"Invalid acceptance clip: {destination} {info}")
        exported.append(
            {
                **row,
                "origin_machine_id": str(row["blind_id"]),
                "blind_id": blind_id,
                "review_audio": str(destination.resolve()),
                "audio": str(destination.resolve()),
                "audio_rel": f"audio/{destination.name}",
                "output_duration": float(info.duration),
                "wave": page_base.waveform(audio),
            }
        )

    acceptance_audit = {
        "version": "anime_semantic_acceptance_v1_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "purpose": "blind human trainability review",
            "selection": "up to four risk-stratified candidates per episode, then global risk fill",
            "judge": "identity, overlap, complete boundaries, natural quality and noise",
            "source": "original anime BS-Roformer vocals",
            "status": "review only; not training data",
            "local_resources": RESOURCE_POLICY,
        },
        "sample_count": len(exported),
        "source_count": len({row["source_id"] for row in exported}),
        "samples": exported,
    }
    report = {
        "version": "anime_semantic_acceptance_v1_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_machine_candidates": len(rows),
        "selected_samples": len(exported),
        "selected_duration_sec": sum(float(row["output_duration"]) for row in exported),
        "per_source": dict(Counter(row["source_id"] for row in exported)),
        "training_started": False,
        "existing_training_lists_modified": False,
        "score_90_claimed": False,
    }
    write_json(OUT / "audit_manifest.json", acceptance_audit)
    write_json(OUT / "private_manifest.json", acceptance_audit)
    write_json(OUT / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
