# -*- coding: utf-8 -*-
"""Resolve the blind human-calibration feedback into auditable label sets.

The current blind judgement supersedes the historical control subtype.  A
control still counts as a safety match when both judgements say "unsafe", even
if the newer pass refines an old "pure non-target" label to "mixed".
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_AUDIT = SCRIPT_DIR / "human_calibration_v1" / "audit_manifest.json"
DEFAULT_OUT = SCRIPT_DIR / "human_calibration_v1"

ERROR_TO_CLASS = {
    "非伊蕾娜": "pure_non_target",
    "混入/无缝接话": "mixed",
    "重叠说话": "overlap",
    "切头/切尾": "boundary_cut",
    "杂音/音质问题": "quality_issue",
    "无法确定": "uncertain",
    "其他": "other_unsafe",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8")


def write_paths(path: Path, values: list[str]) -> None:
    path.write_text("".join(f"{value}/n" for value in values), encoding="utf-8")


def final_class(feedback: dict) -> str:
    if not feedback.get("is_error"):
        return "target"
    error_type = str(feedback.get("error_type") or "").strip()
    note = str(feedback.get("note") or "").strip()
    # This is the same character in a deliberately different child voice.  It
    # is unsafe for the requested adult target, but is not a pure other-speaker
    # negative and must not distort a speaker classifier.
    if error_type == "非伊蕾娜" and "幼年伊蕾娜" in note:
        return "target_variant_exclude"
    if error_type != "非伊蕾娜" and any(token in note for token in ("非伊蕾娜", "不是伊蕾娜", "非目标", "其他角色")):
        return "mixed"
    if not error_type and note.startswith("还是") and any(token in note for token in ("开头", "结尾", "一点点")):
        return "mixed"
    if not error_type and any(token in note for token in ("环境声", "环境声音", "杂音", "噪声", "撞击", "咚")):
        return "quality_issue"
    return ERROR_TO_CLASS.get(error_type, "other_unsafe")


def interval_for(audit: dict) -> tuple[str | None, float | None, float | None]:
    if audit.get("source_id") is not None and audit.get("start") is not None and audit.get("end") is not None:
        return str(audit["source_id"]), float(audit["start"]), float(audit["end"])
    source_interval = audit.get("source_interval")
    if isinstance(source_interval, list) and len(source_interval) == 3:
        return str(source_interval[0]), float(source_interval[1]), float(source_interval[2])
    return None, None, None


def expected_safety(expected_label: str | None) -> bool | None:
    if expected_label == "A":
        return True
    if expected_label in {"C", "D"}:
        return False
    return None


def expected_subtype(expected_label: str | None) -> str | None:
    return {"A": "target", "C": "pure_non_target", "D": "mixed"}.get(str(expected_label))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    feedback_path = args.feedback.resolve()
    audit_path = args.audit.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    feedback_doc = read_json(feedback_path)
    audit_doc = read_json(audit_path)
    feedback_rows = feedback_doc.get("samples", [])
    audit_rows = audit_doc.get("samples", [])
    if not feedback_doc.get("complete"):
        raise RuntimeError("Feedback export is not marked complete")
    if not feedback_rows or len(feedback_rows) != len(audit_rows):
        raise RuntimeError(f"Feedback/audit row mismatch: {len(feedback_rows)} versus {len(audit_rows)}")

    feedback_by_id = {str(row["blind_id"]): row for row in feedback_rows}
    audit_by_id = {str(row["blind_id"]): row for row in audit_rows}
    if len(feedback_by_id) != len(feedback_rows) or set(feedback_by_id) != set(audit_by_id):
        raise RuntimeError("Blind IDs are missing, duplicated, or do not match the audit manifest")
    if any(not row.get("reviewed") for row in feedback_rows):
        raise RuntimeError("At least one sample is not reviewed")

    resolved: list[dict] = []
    controls: list[dict] = []
    hard_intervals: list[dict] = []
    for blind_id in sorted(audit_by_id):
        audit = audit_by_id[blind_id]
        feedback = feedback_by_id[blind_id]
        label = final_class(feedback)
        source_id, start, end = interval_for(audit)
        audio_path = Path(str(audit["audio"])).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        row = {
            "blind_id": blind_id,
            "kind": audit.get("kind"),
            "source_id": source_id,
            "start": start,
            "end": end,
            "duration": float(audit.get("output_duration") or audit.get("duration") or feedback.get("duration_sec") or 0.0),
            "audio": str(audio_path),
            "selection_profile": audit.get("selection_profile"),
            "private_text": audit.get("private_text"),
            "human_is_error": bool(feedback.get("is_error")),
            "human_error_type": str(feedback.get("error_type") or ""),
            "human_note": str(feedback.get("note") or ""),
            "play_count": int(feedback.get("play_count") or 0),
            "max_listened_sec": float(feedback.get("max_listened_sec") or 0.0),
            "final_class": label,
            "speaker_training_role": (
                "positive"
                if label in {"target", "boundary_cut"}
                else "negative"
                if label == "pure_non_target"
                else "validation_only"
            ),
            "hard_exclusion": label != "target",
        }
        resolved.append(row)
        if row["hard_exclusion"] and source_id is not None and start is not None and end is not None:
            hard_intervals.append(
                {
                    "blind_id": blind_id,
                    "source_id": source_id,
                    "start": float(start),
                    "end": float(end),
                    "reason": label,
                    "human_error_type": row["human_error_type"],
                    "human_note": row["human_note"],
                }
            )
        if audit.get("kind") == "control":
            expected_safe = expected_safety(audit.get("expected_label"))
            current_safe = label == "target"
            expected_type = expected_subtype(audit.get("expected_label"))
            control = {
                "blind_id": blind_id,
                "control_type": audit.get("control_type"),
                "source_id": source_id,
                "start": start,
                "end": end,
                "historical_expected_label": audit.get("expected_label"),
                "historical_expected_class": expected_type,
                "current_human_class": label,
                "safety_match": expected_safe is None or expected_safe == current_safe,
                "subtype_match": expected_type == label,
            }
            controls.append(control)

    # Every non-control blind sample can calibrate speaker identity.  Later
    # acceptance rounds use names such as ``fresh_cluster_candidate`` rather
    # than the original ``probe`` kind.
    probes = [row for row in resolved if row["kind"] != "control"]
    positive = [row["audio"] for row in probes if row["speaker_training_role"] == "positive"]
    negative = [row["audio"] for row in probes if row["speaker_training_role"] == "negative"]
    unsafe = [row["audio"] for row in resolved if row["hard_exclusion"]]
    if len(set(positive + negative)) != len(positive) + len(negative):
        raise RuntimeError("A path appears in both human speaker-training classes")

    class_counts = collections.Counter(row["final_class"] for row in resolved)
    probe_class_counts = collections.Counter(row["final_class"] for row in probes)
    by_source: dict[str, collections.Counter] = {}
    by_profile: dict[str, collections.Counter] = {}
    for row in probes:
        by_source.setdefault(str(row["source_id"]), collections.Counter())[row["final_class"]] += 1
        by_profile.setdefault(str(row["selection_profile"]), collections.Counter())[row["final_class"]] += 1

    raw_bytes = feedback_path.read_bytes()
    archive_path = output_dir / f"feedback_received_{feedback_path.stem}.json"
    shutil.copy2(feedback_path, archive_path)
    write_json(output_dir / "labels_resolved.json", {"version": "human_calibration_labels_resolved_v1", "samples": resolved})
    write_json(output_dir / "hard_exclusion_intervals.json", {"version": "human_hard_exclusions_v1", "intervals": hard_intervals})
    write_paths(output_dir / "human_probe_positive.paths.txt", positive)
    write_paths(output_dir / "human_probe_pure_negative.paths.txt", negative)
    write_paths(output_dir / "human_unsafe_all.paths.txt", unsafe)

    listened_ratios = [
        min(1.0, row["max_listened_sec"] / row["duration"]) if row["duration"] > 0 else 0.0 for row in resolved
    ]
    report = {
        "version": "human_calibration_feedback_ingest_v1",
        "feedback_source": str(feedback_path),
        "feedback_archive": str(archive_path),
        "feedback_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "complete": True,
        "counts": {
            "all": len(resolved),
            "probes": len(probes),
            "controls": len(controls),
            "all_classes": dict(sorted(class_counts.items())),
            "probe_classes": dict(sorted(probe_class_counts.items())),
            "speaker_positive_probes": len(positive),
            "speaker_pure_negative_probes": len(negative),
            "hard_exclusion_samples": len(unsafe),
            "hard_exclusion_intervals": len(hard_intervals),
        },
        "listening": {
            "all_reviewed": all(row["play_count"] > 0 for row in resolved),
            "minimum_play_count": min(row["play_count"] for row in resolved),
            "minimum_listened_ratio": min(listened_ratios),
            "at_least_90_percent_listened": sum(value >= 0.90 for value in listened_ratios),
        },
        "hidden_controls": {
            "safety_matches": sum(bool(row["safety_match"]) for row in controls),
            "subtype_matches": sum(bool(row["subtype_match"]) for row in controls),
            "total": len(controls),
            "details": controls,
        },
        "probe_by_source": {key: dict(sorted(value.items())) for key, value in sorted(by_source.items())},
        "probe_by_selection_profile": {key: dict(sorted(value.items())) for key, value in sorted(by_profile.items())},
        "policy": {
            "current_blind_label_supersedes_historical_control_subtype": True,
            "mixed_overlap_quality_uncertain_are_validation_only": True,
            "boundary_cut_is_positive_for_speaker_identity_but_hard_excluded_from_training_audio": True,
            "child_target_variant_is_validation_only_not_pure_non_target": True,
            "controls_are_not_added_to_probe_training_sets": True,
        },
    }
    write_json(output_dir / "feedback_ingest_report.json", report)
    print(json.dumps(report["counts"] | {"control_safety": report["hidden_controls"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
