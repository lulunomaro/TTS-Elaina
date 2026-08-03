# -*- coding: utf-8 -*-
"""Build a 20-machine + 4-known-gold blind acceptance pack."""
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

import build_human_calibration_v1 as page_base


QUALITY_AUDIT = ROOT / "full_semantic_quality_v1" / "audit_manifest.json"
QUALITY_REPORT = ROOT / "full_semantic_quality_v1" / "report.json"
BASE_LABELS = ROOT / "human_calibration_combined_v3" / "labels_resolved.json"
WORD_REPORTS = ROOT / "full_source_local_word_scores_v1" / "source_reports"
OUT = ROOT / "full_semantic_acceptance_v1"
AUDIO = OUT / "audio"
SEED = 20260802


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8"
    )


def risk_score(row: dict) -> float:
    quality = row["quality_v1"]
    left = row["left_boundary"]
    right = row["right_boundary"]
    boundary_risk = min(1.5, float(left["energy_ratio"])) + min(
        1.5, float(right["energy_ratio"])
    )
    margin_risk = max(0.0, 0.09 - float(row["minimum_cluster_margin"])) * 8.0
    context_risk = max(
        0.0, 1.0 - float(row["minimum_context_target_fraction"])
    )
    quality_risk = max(
        0.0, float(quality["source_p20"]) + 0.12 - float(quality["nisqa_tts_mos"])
    )
    model_risk = 0.8 if not row["source_model_known_unsafe_rejected"] else 0.0
    return boundary_risk + margin_risk + context_risk + quality_risk + model_risk


def select_machine(rows: list[dict]) -> list[dict]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        item = dict(row)
        item["acceptance_risk_score"] = risk_score(item)
        by_source[item["source_id"]].append(item)
    selected = []
    selected_ids = set()
    for source_id in sorted(by_source):
        row = max(by_source[source_id], key=lambda item: item["acceptance_risk_score"])
        selected.append(row)
        selected_ids.add(row["blind_id"])

    caps = Counter(row["source_id"] for row in selected)
    remaining = sorted(
        [row for values in by_source.values() for row in values if row["blind_id"] not in selected_ids],
        key=lambda item: item["acceptance_risk_score"],
        reverse=True,
    )
    for row in remaining:
        if len(selected) >= 20:
            break
        if caps[row["source_id"]] >= 3:
            continue
        selected.append(row)
        caps[row["source_id"]] += 1
    if len(selected) != 20:
        raise RuntimeError(f"Expected 20 machine samples, got {len(selected)}")
    return selected


def select_gold() -> list[dict]:
    labels = json.loads(BASE_LABELS.read_text(encoding="utf-8"))["samples"]
    evaluations = {}
    for path in WORD_REPORTS.glob("*.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        for row in document["human_evaluations"]:
            evaluations[row["human_key"]] = row
    eligible = []
    for row in labels:
        if row["final_class"] != "target" or not Path(row["audio"]).is_file():
            continue
        duration = float(row["duration"])
        evaluation = evaluations.get(row["human_key"])
        if not evaluation or evaluation["strict_pass_fraction"] < 0.90:
            continue
        if 2.0 <= duration <= 6.0:
            eligible.append(
                {
                    "source_id": str(row["source_id"]),
                    "audio": str(row["audio"]),
                    "duration": duration,
                    "human_key": str(row["human_key"]),
                    "strict_pass_fraction": evaluation["strict_pass_fraction"],
                }
            )
    selected = []
    seen_sources = set()
    for row in sorted(
        eligible,
        key=lambda item: (-item["strict_pass_fraction"], item["source_id"], item["human_key"]),
    ):
        if row["source_id"] in seen_sources:
            continue
        selected.append(row)
        seen_sources.add(row["source_id"])
        if len(selected) == 4:
            break
    if len(selected) != 4:
        raise RuntimeError(f"Expected 4 gold controls, got {len(selected)}")
    return selected


def main() -> None:
    AUDIO.mkdir(parents=True, exist_ok=True)
    quality = json.loads(QUALITY_AUDIT.read_text(encoding="utf-8"))
    machine = select_machine(quality["samples"])
    gold = select_gold()
    mixed = [
        {"role": "machine_candidate", "payload": row} for row in machine
    ] + [{"role": "known_gold_control", "payload": row} for row in gold]
    random.Random(SEED).shuffle(mixed)

    audit_rows = []
    private_rows = []
    for index, item in enumerate(mixed, start=1):
        blind_id = f"A{index:03d}"
        source = Path(item["payload"]["audio"])
        destination = AUDIO / f"{blind_id}.wav"
        shutil.copyfile(source, destination)
        audio, sample_rate = sf.read(
            str(destination), dtype="float32", always_2d=False
        )
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        duration = float(len(audio) / sample_rate)
        audit_rows.append(
            {
                "blind_id": blind_id,
                "audio": str(destination),
                "audio_rel": f"audio/{destination.name}",
                "output_duration": duration,
                "wave": page_base.waveform(np.asarray(audio, dtype=np.float32)),
                "kind": "blind_acceptance_sample",
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
            "version": "full_semantic_acceptance_v1_blind",
            "sample_count": len(audit_rows),
            "samples": audit_rows,
        },
    )
    write_json(
        OUT / "private_manifest.json",
        {
            "version": "full_semantic_acceptance_v1_private",
            "policy": {
                "machine": "20 quality-strong candidates, source-balanced and biased toward risky margins/boundaries",
                "control": "4 previously human-confirmed clean target clips",
                "blind": "the HTML exposes only A001-A024",
                "acceptance": "any clear non-target in the 20 machine clips blocks automatic promotion",
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
        writer.writerow(["blind_id", "role", "source_id", "source_audio"])
        for row in private_rows:
            writer.writerow(
                [row["blind_id"], row["role"], row["source_id"], row["source_audio"]]
            )
    print(
        json.dumps(
            {
                "output": str(OUT),
                "sample_count": len(audit_rows),
                "machine_count": 20,
                "gold_count": 4,
                "machine_sources": sorted({row["source_id"] for row in machine}),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
