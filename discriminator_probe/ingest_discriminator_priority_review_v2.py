# -*- coding: utf-8 -*-
"""Ingest discriminator-priority review v2 feedback (isolated, no radio_drama_work writes).

v2 differs from v1: majority are boundary_cut (38/48). These are speaker-correct
positives for the discriminator, and they also seed a boundary-rescue queue so
the next pass re-cuts them from original vocals.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")
OUT = PROBE / "discriminator_priority_review_v2"
AUDIT = OUT / "audit_manifest.json"
HUMAN_OUT = OUT / "human_feedback"
RESCUE_OUT = PROBE / "boundary_rescue_queue_v2.json"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def classify(human: dict) -> str:
    if not bool(human.get("is_error")):
        return "target"
    text = f"{human.get('error_type', '')} {human.get('note', '')}"
    if any(t in text for t in ("非目标", "非伊蕾娜", "other voice")):
        return "pure_non_target"
    if any(t in text for t in ("混合", "无缝", "接话", "mixed")):
        return "mixed"
    if any(t in text for t in ("重叠", "overlap")):
        return "overlap"
    if any(t in text for t in ("开头", "结尾", "切头", "切尾", "boundary")):
        return "boundary_cut"
    if any(t in text for t in ("杂音", "音质", "环境", "bgm", "BGM", "quality")):
        return "quality_issue"
    if any(t in text for t in ("不确定", "无法确定", "uncertain")):
        return "uncertain"
    return "uncertain"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback", type=Path, required=True)
    args = parser.parse_args()
    feedback_path = args.feedback.resolve()
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    if not bool(feedback.get("complete")):
        raise RuntimeError("feedback is incomplete")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    audit_by_id = {str(row["blind_id"]): row for row in audit.get("samples", audit.get("rows", []))}
    human_by_id = {str(row["blind_id"]): row for row in feedback.get("samples", [])}
    if set(audit_by_id) != set(human_by_id):
        raise RuntimeError("feedback IDs do not match v2 audit")

    labels = []
    for blind_id in sorted(audit_by_id):
        source = audit_by_id[blind_id]
        human = human_by_id[blind_id]
        if not bool(human.get("reviewed")):
            raise RuntimeError(f"{blind_id}: not reviewed")
        labels.append({
            "blind_id": blind_id,
            "row_id": source.get("row_id", ""),
            "pool": source.get("pool", ""),
            "prob": float(source.get("prob", 0.0)),
            "tier": source.get("tier", ""),
            "source_id": source["source_id"],
            "start": float(source["start"]),
            "end": float(source["end"]),
            "duration": float(source.get("output_duration", source.get("duration", 0.0))),
            "text": source.get("text", ""),
            "audio": source.get("audio", ""),
            "final_class": classify(human),
            "error_type": str(human.get("error_type") or ""),
            "human_note": str(human.get("note") or ""),
        })

    training_intervals = []
    rescue_queue = []
    for row in labels:
        if row["final_class"] in {"target", "boundary_cut"}:
            training_intervals.append({**row, "label": 1})
        elif row["final_class"] == "pure_non_target":
            training_intervals.append({**row, "label": 0})
        if row["final_class"] == "boundary_cut":
            rescue_queue.append({
                "blind_id": row["blind_id"],
                "source_id": row["source_id"],
                "start": row["start"],
                "end": row["end"],
                "duration": row["duration"],
                "text": row["text"],
                "note": row["human_note"],
            })

    HUMAN_OUT.mkdir(parents=True, exist_ok=True)
    archive = HUMAN_OUT / f"feedback_received_{feedback_path.name}"
    shutil.copy2(feedback_path, archive)
    write_json(HUMAN_OUT / "labels_resolved.json",
               {"version": "discriminator_priority_review_v2_labels", "samples": labels})
    write_json(HUMAN_OUT / "training_intervals_for_next_discriminator.json",
               {"version": "discriminator_training_intervals_v2",
                "policy": "target and boundary_cut positive; pure_non_target negative; mixed/overlap/quality/uncertain excluded",
                "samples": training_intervals})
    write_json(RESCUE_OUT,
               {"version": "boundary_rescue_queue_v2",
                "policy": "speaker-correct but boundary-cut clips; re-cut from original BS-Roformer vocals",
                "samples": rescue_queue})
    report = {
        "version": "discriminator_priority_review_v2_ingest_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feedback_source": str(feedback_path),
        "feedback_archive": str(archive),
        "classes": dict(Counter(row["final_class"] for row in labels)),
        "training_interval_counts": dict(Counter(str(row["label"]) for row in training_intervals)),
        "rescue_queue_count": len(rescue_queue),
        "next_step": "isolated discriminator retrain/rerank with v2 intervals; then boundary rescue of queue",
        "training_started": False,
        "training_list_written": False,
        "existing_training_lists_modified": False,
    }
    write_json(HUMAN_OUT / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
