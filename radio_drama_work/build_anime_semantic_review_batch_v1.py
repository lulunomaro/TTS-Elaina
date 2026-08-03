# -*- coding: utf-8 -*-
"""Select the first stratified human-review batch from the anime review pool."""
from __future__ import annotations

import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import soundfile as sf

import build_human_calibration_v1 as page_base

POOL = ROOT / "anime_semantic_mining_v1" / "semantic_review_manifest.json"
ACCEPTED = ROOT / "human_anime_semantic_acceptance_v1" / "labels_resolved.json"
OUT = ROOT / "anime_semantic_review_batch_v1"
AUDIO_DIR = OUT / "audio"
REVIEW_VERSION = "anime_semantic_review_batch_v1"
EXCLUDED_ORIGIN_IDS_PATH: Path | None = None
MIN_PER_SOURCE = 3
MAX_PER_SOURCE = 4
TARGET_TOTAL = 48
SOURCE_COUNT = 12


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8"
    )


def overlaps(left: dict, right: dict, padding: float = 0.75) -> bool:
    if str(left["source_id"]) != str(right["source_id"]):
        return False
    return max(float(left["start"]), float(right["start"]) - padding) < min(
        float(left["end"]), float(right["end"]) + padding
    )


def ranking(row: dict) -> tuple:
    quality = row.get("quality") or {}
    eres = row.get("eres2net") or {}
    boundary_modes = row.get("boundary_modes") or []
    return (
        bool(quality.get("pass")),
        bool(eres.get("support")),
        not bool(eres.get("hard_negative_warning")),
        float(row.get("direct_target_word_fraction", 0.0)),
        float(row.get("mean_target_center_delta", -1.0)),
        float(row.get("mean_anchor_margin", -1.0)),
        not bool(row.get("seamless_hard_transition")),
        not bool(boundary_modes),
        float(row.get("duration", 0.0)),
    )


def main() -> None:
    pool = json.loads(POOL.read_text(encoding="utf-8"))
    accepted_rows = json.loads(ACCEPTED.read_text(encoding="utf-8"))["samples"]
    accepted = [row for row in accepted_rows if row.get("final_class") == "target"]
    excluded_origin_ids: set[str] = set()
    if EXCLUDED_ORIGIN_IDS_PATH is not None and EXCLUDED_ORIGIN_IDS_PATH.exists():
        previous = json.loads(EXCLUDED_ORIGIN_IDS_PATH.read_text(encoding="utf-8"))
        excluded_origin_ids = {
            str(row.get("origin_blind_id") or row.get("blind_id"))
            for row in previous.get("samples", [])
            if row.get("origin_blind_id") or row.get("blind_id")
        }

    candidates = []
    for row in pool["samples"]:
        if str(row["blind_id"]) in excluded_origin_ids:
            continue
        if not 1.5 <= float(row["duration"]) <= 6.0:
            continue
        if any(overlaps(row, old, 0.80) for old in accepted):
            continue
        candidates.append(dict(row))
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        by_source[str(row["source_id"])].append(row)
    expected_sources = {f"majo_ep{index:02d}" for index in range(1, 13)}
    if set(by_source) != expected_sources:
        raise RuntimeError(
            f"Review pool source coverage mismatch: {sorted(set(by_source))}"
        )

    selected = []
    selection_report = {}
    for source_id in sorted(by_source):
        rows = sorted(by_source[source_id], key=ranking, reverse=True)
        chosen = []
        for padding in (0.75, 0.20, 0.0):
            for row in rows:
                if row in chosen:
                    continue
                if any(overlaps(row, old, padding) for old in chosen):
                    continue
                chosen.append(row)
                if len(chosen) >= MIN_PER_SOURCE:
                    break
            if len(chosen) >= MIN_PER_SOURCE:
                break
        if len(chosen) != MIN_PER_SOURCE:
            raise RuntimeError(
                f"{source_id}: selected {len(chosen)} instead of {MIN_PER_SOURCE}"
            )
        selected.extend(chosen)
        selection_report[source_id] = {
            "pool_count": len(rows),
            "selected_count": len(chosen),
            "selected_origin_ids": [row["blind_id"] for row in chosen],
        }

    # Fill one extra slot per source in round-robin order, keeping the first
    # batch balanced instead of letting one high-yield episode dominate it.
    selected_ids = {str(row["blind_id"]) for row in selected}
    remaining_by_source = {
        source_id: sorted(
            [row for row in rows if str(row["blind_id"]) not in selected_ids],
            key=ranking,
            reverse=True,
        )
        for source_id, rows in by_source.items()
    }
    for padding in (0.75, 0.20, 0.0):
        progress = True
        while progress and len(selected) < TARGET_TOTAL:
            progress = False
            for source_id in sorted(remaining_by_source):
                if selection_report[source_id]["selected_count"] >= MAX_PER_SOURCE:
                    continue
                candidate = next(
                    (
                        row
                        for row in remaining_by_source[source_id]
                        if str(row["blind_id"]) not in selected_ids
                        and not any(overlaps(row, old, padding) for old in selected)
                    ),
                    None,
                )
                if candidate is None:
                    continue
                selected.append(candidate)
                selected_ids.add(str(candidate["blind_id"]))
                selection_report[source_id]["selected_count"] += 1
                selection_report[source_id]["selected_origin_ids"].append(
                    candidate["blind_id"]
                )
                progress = True
                if len(selected) >= TARGET_TOTAL:
                    break
        if len(selected) >= TARGET_TOTAL:
            break
    # If one slot is still missing because a source has too few separated
    # candidates, allow exactly one source to receive a fifth candidate.
    if len(selected) < TARGET_TOTAL:
        remaining = sorted(
            [row for row in candidates if str(row["blind_id"]) not in selected_ids],
            key=ranking,
            reverse=True,
        )
        for row in remaining:
            if any(overlaps(row, old, 0.0) for old in selected):
                continue
            source_id = str(row["source_id"])
            if selection_report[source_id]["selected_count"] >= MAX_PER_SOURCE + 1:
                continue
            selected.append(row)
            selected_ids.add(str(row["blind_id"]))
            selection_report[source_id]["selected_count"] += 1
            selection_report[source_id]["selected_origin_ids"].append(row["blind_id"])
            break
    if len(selected) != TARGET_TOTAL:
        raise RuntimeError(f"Expected {TARGET_TOTAL} selected rows, got {len(selected)}")

    OUT.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    audit_samples = []
    for index, source_row in enumerate(selected, 1):
        blind_id = f"B{index:03d}"
        source_audio = Path(source_row["audio"]).resolve()
        output = AUDIO_DIR / f"{blind_id}.wav"
        shutil.copy2(source_audio, output)
        info = sf.info(str(output))
        if (
            info.samplerate != 32000
            or info.channels != 1
            or info.subtype != "PCM_16"
            or not 1.5 <= info.duration <= 6.0
        ):
            raise RuntimeError(f"Invalid review batch audio: {output} {info}")
        audit_samples.append(
            {
                **source_row,
                "batch_blind_id": blind_id,
                "origin_blind_id": str(source_row["blind_id"]),
                "blind_id": blind_id,
                "audio": str(output.resolve()),
                "audio_rel": f"audio/{output.name}",
                "output_duration": float(info.duration),
                "wave": source_row["wave"],
                "review_policy": "human identity, mixed speech, boundary, quality and noise review",
            }
        )

    report = {
        "version": f"{REVIEW_VERSION}_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pool": str(POOL.resolve()),
        "pool_count": len(pool["samples"]),
        "excluded_already_human_accepted_count": len(accepted),
        "excluded_previous_batch_count": len(excluded_origin_ids),
        "selected_count": len(audit_samples),
        "selected_duration_sec": sum(
            float(row["output_duration"]) for row in audit_samples
        ),
        "per_source": selection_report,
        "machine_scores_hidden_in_html": True,
        "training_started": False,
        "existing_training_lists_modified": False,
        "score_90_claimed": False,
    }
    audit = {
        "version": f"{REVIEW_VERSION}_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "purpose": "stratified human review batch from the remaining semantic candidates",
            "coverage": "four candidates per episode",
            "input": "original BS-Roformer vocals-derived review clips",
            "machine_scores_hidden": True,
            "not_training_until_human_pass": True,
        },
        "sample_count": len(audit_samples),
        "samples": audit_samples,
    }
    write_json(OUT / "audit_manifest.json", audit)
    write_json(OUT / "private_manifest.json", audit)
    write_json(OUT / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
