# -*- coding: utf-8 -*-
"""Merge episode-12 stable-window labels into final anime calibration v3."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "anime_speaker_calibration_round3_v1" / "audit_manifest.json"
V2_LABELS = ROOT / "human_anime_speaker_calibration_v2" / "labels_resolved.json"
MANIFEST = ROOT / "anime_batch_v1" / "manifest.json"
OUT = ROOT / "human_anime_speaker_calibration_v3"

CLASS_MAP = {
    "target": "target",
    "non_target": "pure_non_target",
    "mixed": "mixed",
    "uncertain": "uncertain",
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback", type=Path, required=True)
    args = parser.parse_args()
    feedback_path = args.feedback.resolve()
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    if not bool(feedback.get("complete")):
        raise RuntimeError("Anime speaker calibration round3 feedback is incomplete")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    audit_by_id = {str(row["blind_id"]): row for row in audit["samples"]}
    human_by_id = {str(row["blind_id"]): row for row in feedback["samples"]}
    if set(audit_by_id) != set(human_by_id):
        raise RuntimeError("Round3 feedback IDs do not match audit")

    labels = list(json.loads(V2_LABELS.read_text(encoding="utf-8"))["samples"])
    for blind_id in sorted(audit_by_id):
        source = audit_by_id[blind_id]
        human = human_by_id[blind_id]
        decision = str(human.get("decision") or "")
        if decision not in CLASS_MAP:
            raise RuntimeError(f"{blind_id}: invalid decision {decision!r}")
        final_class = CLASS_MAP[decision]
        labels.append(
            {
                "human_key": f"anime_speaker_calibration_round3_v1:{blind_id}",
                "blind_id": blind_id,
                "calibration_round": 3,
                "source_id": str(source["source_id"]),
                "source": str(source["source"]),
                "start": float(source["start"]),
                "end": float(source["end"]),
                "duration": float(source["output_duration"]),
                "text": "",
                "word_start_index": -1,
                "word_end_index": -1,
                "audio": str(source["audio"]),
                "machine_prediction": str(source["predicted_tier"]),
                "primary_cluster": int(source["primary_cluster"]),
                "cam_window_index": int(source["cam_window_index"]),
                "final_class": final_class,
                "decision": decision,
                "human_note": str(human.get("note") or ""),
                "play_count": int(human.get("play_count") or 0),
                "max_listened_sec": float(human.get("max_listened_sec") or 0.0),
                "speaker_training_role": (
                    "positive"
                    if final_class == "target"
                    else "negative"
                    if final_class == "pure_non_target"
                    else "validation_only"
                ),
            }
        )

    source_ids = [
        str(row["id"])
        for row in json.loads(MANIFEST.read_text(encoding="utf-8"))["tracks"]
    ]
    source_counts: dict[str, Counter] = defaultdict(Counter)
    masks = []
    for row in labels:
        source_counts[str(row["source_id"])][str(row["final_class"])] += 1
        if row["final_class"] in {"pure_non_target", "mixed"}:
            masks.append(
                {
                    "source_id": str(row["source_id"]),
                    "start": float(row["start"]),
                    "end": float(row["end"]),
                    "reason": (
                        f"human_anime_calibration:{row['human_key']}:{row['final_class']}"
                    ),
                }
            )
    missing_target = [sid for sid in source_ids if not source_counts[sid]["target"]]
    missing_negative = [
        sid for sid in source_ids if not source_counts[sid]["pure_non_target"]
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / f"feedback_received_{feedback_path.name}"
    shutil.copy2(feedback_path, archive)
    write_json(
        OUT / "labels_resolved.json",
        {
            "version": "human_anime_speaker_calibration_v3_labels",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "feedback_source_round3": str(feedback_path),
            "samples": labels,
        },
    )
    write_json(
        OUT / "hard_exclusion_intervals.json",
        {
            "version": "human_anime_speaker_calibration_v3_hard_exclusions",
            "policy": "pure non-target and mixed calibration intervals are never training output",
            "intervals": masks,
        },
    )
    report = {
        "version": "human_anime_speaker_calibration_v3_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feedback_source_round3": str(feedback_path),
        "feedback_archive_round3": str(archive),
        "feedback_sha256_round3": sha256(feedback_path),
        "sample_count": len(labels),
        "class_counts": dict(Counter(str(row["final_class"]) for row in labels)),
        "per_source": {
            sid: dict(sorted(source_counts[sid].items())) for sid in source_ids
        },
        "missing_target_sources": missing_target,
        "missing_pure_negative_sources": missing_negative,
        "calibration_complete": not missing_target and not missing_negative,
        "hard_exclusion_count": len(masks),
        "training_started": False,
        "existing_training_lists_modified": False,
    }
    write_json(OUT / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
