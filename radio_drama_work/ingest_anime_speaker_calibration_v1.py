# -*- coding: utf-8 -*-
"""Resolve the user's anime identity labels without promoting any audio."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "anime_speaker_calibration_v1" / "audit_manifest.json"
OUT = ROOT / "human_anime_speaker_calibration_v1"

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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback", type=Path, required=True)
    args = parser.parse_args()
    feedback_path = args.feedback.resolve()
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if not bool(feedback.get("complete")):
        raise RuntimeError("Anime speaker calibration feedback is incomplete")

    audit_by_id = {str(row["blind_id"]): row for row in audit["samples"]}
    human_by_id = {str(row["blind_id"]): row for row in feedback["samples"]}
    if set(audit_by_id) != set(human_by_id):
        raise RuntimeError("Feedback IDs do not match the calibration audit")

    labels = []
    masks = []
    source_counts: dict[str, Counter] = defaultdict(Counter)
    for blind_id in sorted(audit_by_id):
        source = audit_by_id[blind_id]
        human = human_by_id[blind_id]
        decision = str(human.get("decision") or "")
        if decision not in CLASS_MAP:
            raise RuntimeError(f"{blind_id}: invalid decision {decision!r}")
        final_class = CLASS_MAP[decision]
        row = {
            "human_key": f"anime_speaker_calibration_v1:{blind_id}",
            "blind_id": blind_id,
            "source_id": str(source["source_id"]),
            "source": str(source["source"]),
            "start": float(source["start"]),
            "end": float(source["end"]),
            "duration": float(source["output_duration"]),
            "text": str(source["text"]),
            "word_start_index": int(source["word_start_index"]),
            "word_end_index": int(source["word_end_index"]),
            "audio": str(source["audio"]),
            "machine_prediction": str(source["predicted_tier"]),
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
        labels.append(row)
        source_counts[row["source_id"]][final_class] += 1
        if final_class in {"pure_non_target", "mixed"}:
            masks.append(
                {
                    "source_id": row["source_id"],
                    "start": row["start"],
                    "end": row["end"],
                    "reason": f"human_anime_calibration:{blind_id}:{final_class}",
                }
            )

    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / f"feedback_received_{feedback_path.name}"
    shutil.copy2(feedback_path, archive)
    write_json(
        OUT / "labels_resolved.json",
        {
            "version": "human_anime_speaker_calibration_v1_labels",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "feedback_source": str(feedback_path),
            "samples": labels,
        },
    )
    write_json(
        OUT / "hard_exclusion_intervals.json",
        {
            "version": "human_anime_speaker_calibration_v1_hard_exclusions",
            "policy": "pure non-target and mixed calibration intervals are never training output",
            "intervals": masks,
        },
    )
    per_source = {
        source_id: dict(sorted(counts.items()))
        for source_id, counts in sorted(source_counts.items())
    }
    missing_target = [
        source_id for source_id, counts in source_counts.items() if not counts["target"]
    ]
    missing_negative = [
        source_id
        for source_id, counts in source_counts.items()
        if not counts["pure_non_target"]
    ]
    report = {
        "version": "human_anime_speaker_calibration_v1_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feedback_source": str(feedback_path),
        "feedback_archive": str(archive),
        "feedback_sha256": sha256(feedback_path),
        "sample_count": len(labels),
        "class_counts": dict(Counter(row["final_class"] for row in labels)),
        "per_source": per_source,
        "missing_target_sources": missing_target,
        "missing_pure_negative_sources": missing_negative,
        "needs_targeted_round2": bool(missing_target or missing_negative),
        "hard_exclusion_count": len(masks),
        "training_started": False,
        "existing_training_lists_modified": False,
    }
    write_json(OUT / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
