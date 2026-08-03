# -*- coding: utf-8 -*-
"""Build a source-balanced 20-machine + 4-gold blind v2 acceptance pack."""
from __future__ import annotations

import csv
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf

import build_full_semantic_acceptance_v1 as acceptance_v1
import build_human_calibration_v1 as page_base


QUALITY_AUDIT = ROOT / "full_semantic_quality_v2" / "audit_manifest.json"
OUT = ROOT / "full_semantic_acceptance_v2"
AUDIO = OUT / "audio"
SEED = 2026080202
HIGH_RISK_SOURCES = {"1405372904-1-192", "306672160-1-192"}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8"
    )


def risk_score(row: dict) -> float:
    quality = row["quality_v2"]
    boundary = quality["boundary"]
    maximum_boundary = max(
        float(boundary["left_energy_ratio"]),
        float(boundary["right_energy_ratio"]),
    )
    boundary_risk = min(1.7, maximum_boundary * 2.0)
    delta_risk = max(0.0, 0.060 - float(row["minimum_target_center_delta"])) * 8.0
    anchor_risk = max(0.0, 0.130 - float(row["minimum_anchor_margin"])) * 6.0
    strong_risk = max(0.0, 0.80 - float(row["strong_word_fraction"])) * 1.8
    quality_risk = max(
        0.0,
        float(quality["source_p20"]) + 0.18 - float(quality["nisqa_tts_mos"]),
    )
    state_counts = row.get("word_state_counts", {})
    edge_soft_fraction = float(state_counts.get("edge_soft", 0)) / max(
        int(row.get("word_count", 0)), 1
    )
    high_risk_bonus = 1.4 if row["source_id"] in HIGH_RISK_SOURCES else 0.0
    return (
        boundary_risk
        + delta_risk
        + anchor_risk
        + strong_risk
        + quality_risk
        + edge_soft_fraction
        + high_risk_bonus
    )


def select_machine(rows: list[dict]) -> list[dict]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        item = dict(row)
        item["acceptance_risk_score"] = risk_score(item)
        by_source[item["source_id"]].append(item)

    selected: list[dict] = []
    selected_ids: set[str] = set()
    for source_id in sorted(by_source):
        row = max(
            by_source[source_id], key=lambda item: item["acceptance_risk_score"]
        )
        selected.append(row)
        selected_ids.add(row["blind_id"])

    caps = Counter(row["source_id"] for row in selected)
    for source_id in sorted(HIGH_RISK_SOURCES):
        if source_id not in by_source or len(by_source[source_id]) < 2:
            continue
        row = max(
            (
                item
                for item in by_source[source_id]
                if item["blind_id"] not in selected_ids
            ),
            key=lambda item: item["acceptance_risk_score"],
        )
        selected.append(row)
        selected_ids.add(row["blind_id"])
        caps[source_id] += 1

    remaining = sorted(
        (
            row
            for values in by_source.values()
            for row in values
            if row["blind_id"] not in selected_ids
        ),
        key=lambda item: item["acceptance_risk_score"],
        reverse=True,
    )
    for row in remaining:
        if len(selected) >= 20:
            break
        if caps[row["source_id"]] >= 3:
            continue
        selected.append(row)
        selected_ids.add(row["blind_id"])
        caps[row["source_id"]] += 1
    if len(selected) != 20:
        raise RuntimeError(f"Expected 20 machine samples, got {len(selected)}")
    return selected


def main() -> None:
    AUDIO.mkdir(parents=True, exist_ok=True)
    quality = json.loads(QUALITY_AUDIT.read_text(encoding="utf-8"))
    machine = select_machine(quality["samples"])
    gold = acceptance_v1.select_gold()
    mixed = [
        {"role": "machine_candidate_v2", "payload": row} for row in machine
    ] + [{"role": "known_gold_control", "payload": row} for row in gold]
    random.Random(SEED).shuffle(mixed)

    audit_rows = []
    private_rows = []
    for index, item in enumerate(mixed, start=1):
        blind_id = f"B{index:03d}"
        source = Path(item["payload"]["audio"])
        destination = AUDIO / f"{blind_id}.wav"
        shutil.copyfile(source, destination)
        audio, sample_rate = sf.read(
            str(destination), dtype="float32", always_2d=True
        )
        audio = np.asarray(audio.mean(axis=1), dtype=np.float32)
        duration = float(len(audio) / sample_rate)
        audit_rows.append(
            {
                "blind_id": blind_id,
                "audio": str(destination.resolve()),
                "audio_rel": f"audio/{destination.name}",
                "output_duration": duration,
                "wave": page_base.waveform(audio),
                "kind": "blind_acceptance_sample_v2",
            }
        )
        private_rows.append(
            {
                "blind_id": blind_id,
                "role": item["role"],
                "source_audio": str(source),
                "source_id": item["payload"]["source_id"],
                "payload": item["payload"],
            }
        )

    write_json(
        OUT / "audit_manifest.json",
        {
            "version": "full_semantic_acceptance_v2_blind",
            "sample_count": len(audit_rows),
            "samples": audit_rows,
        },
    )
    write_json(
        OUT / "private_manifest.json",
        {
            "version": "full_semantic_acceptance_v2_private",
            "policy": {
                "machine": "20 GPU-quality-passing candidates, all available sources represented and risk-biased",
                "high_risk": "306672 receives two probes; 140537 has only one passing clip and is always included",
                "control": "4 previously human-confirmed clean target clips",
                "blind": "the HTML exposes only B001-B024",
                "acceptance": "any clear non-target in the 20 machine clips blocks bulk promotion; boundary-only errors are rescued separately",
            },
            "machine_count": 20,
            "gold_control_count": 4,
            "samples": private_rows,
        },
    )
    with (OUT / "private_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["blind_id", "role", "source_id", "source_audio", "risk_score"]
        )
        for row in private_rows:
            writer.writerow(
                [
                    row["blind_id"],
                    row["role"],
                    row["source_id"],
                    row["source_audio"],
                    row["payload"].get("acceptance_risk_score", ""),
                ]
            )
    print(
        json.dumps(
            {
                "output": str(OUT),
                "sample_count": len(audit_rows),
                "machine_count": len(machine),
                "gold_count": len(gold),
                "machine_sources": sorted({row["source_id"] for row in machine}),
                "per_source": dict(Counter(row["source_id"] for row in machine)),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
