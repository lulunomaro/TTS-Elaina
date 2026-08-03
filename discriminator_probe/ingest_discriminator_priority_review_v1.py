# -*- coding: utf-8 -*-
"""Ingest the isolated discriminator-priority review without touching training data."""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")
OUT = PROBE / "discriminator_priority_review_v1"
AUDIT = OUT / "audit_manifest.json"
HUMAN_OUT = OUT / "human_feedback"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def classify(human: dict) -> str:
    if not bool(human.get("is_error")):
        return "target"
    text = f"{human.get('error_type', '')} {human.get('note', '')}"
    if any(token in text for token in ("非目标", "非伊蕾娜", "闈炰紛", "other voice")):
        return "pure_non_target"
    if any(token in text for token in ("混合", "无缝", "娣峰", "鎺ヨ瘽", "mixed")):
        return "mixed"
    if any(token in text for token in ("重叠", "閲嶅彔", "overlap")):
        return "overlap"
    if any(token in text for token in ("开头", "结尾", "切头", "切尾", "寮€澶", "缁撳熬", "boundary")):
        return "boundary_cut"
    if any(token in text for token in ("杂音", "音质", "鏉傞煶", "闊宠川", "quality")):
        return "quality_issue"
    if any(token in text for token in ("不确定", "无法确定", "鏃犳硶纭畾", "uncertain")):
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
    audit_by_id = {str(row["blind_id"]): row for row in audit["samples"]}
    human_by_id = {str(row["blind_id"]): row for row in feedback.get("samples", [])}
    if set(audit_by_id) != set(human_by_id):
        raise RuntimeError("feedback IDs do not match discriminator review audit")

    labels = []
    for blind_id in sorted(audit_by_id):
        source = audit_by_id[blind_id]
        human = human_by_id[blind_id]
        if not bool(human.get("reviewed")):
            raise RuntimeError(f"{blind_id}: not reviewed")
        final_class = classify(human)
        labels.append(
            {
                "blind_id": blind_id,
                "row_id": source["row_id"],
                "pool": source["pool"],
                "prob": float(source["prob"]),
                "tier": source["tier"],
                "source_id": source["source_id"],
                "start": float(source["start"]),
                "end": float(source["end"]),
                "duration": float(source["output_duration"]),
                "text": source["text"],
                "audio": source["audio"],
                "final_class": final_class,
                "error_type": str(human.get("error_type") or ""),
                "human_note": str(human.get("note") or ""),
                "play_count": int(human.get("play_count") or 0),
                "max_listened_sec": float(human.get("max_listened_sec") or 0.0),
            }
        )

    # Only these labels are safe to feed back into a speaker classifier:
    # target/boundary target are positive; pure non-target is negative. Mixed,
    # overlap, quality and uncertain remain review evidence, not classifier labels.
    training_intervals = []
    for row in labels:
        if row["final_class"] in {"target", "boundary_cut"}:
            training_intervals.append({**row, "label": 1})
        elif row["final_class"] == "pure_non_target":
            training_intervals.append({**row, "label": 0})

    HUMAN_OUT.mkdir(parents=True, exist_ok=True)
    archive = HUMAN_OUT / f"feedback_received_{feedback_path.name}"
    shutil.copy2(feedback_path, archive)
    write_json(HUMAN_OUT / "labels_resolved.json", {"version": "discriminator_priority_review_v1_labels", "samples": labels})
    write_json(
        HUMAN_OUT / "training_intervals_for_next_discriminator.json",
        {
            "version": "discriminator_training_intervals_v1",
            "policy": "target and boundary_cut positive; pure_non_target negative; mixed/overlap/quality/uncertain excluded",
            "samples": training_intervals,
        },
    )
    report = {
        "version": "discriminator_priority_review_v1_ingest_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feedback_source": str(feedback_path),
        "feedback_archive": str(archive),
        "classes": dict(Counter(row["final_class"] for row in labels)),
        "training_interval_counts": dict(Counter(str(row["label"]) for row in training_intervals)),
        "next_step": "run an isolated discriminator retrain/rerank using training_intervals_for_next_discriminator.json; do not modify radio_drama_work",
        "training_started": False,
        "training_list_written": False,
        "existing_training_lists_modified": False,
    }
    write_json(HUMAN_OUT / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
