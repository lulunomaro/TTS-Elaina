# -*- coding: utf-8 -*-
"""Resolve the first full-semantic blind acceptance feedback safely.

Boundary failures are positive evidence for speaker identity but are never
used as training audio until re-cut and heard again.  Mixed/non-target clips
become source-time hard exclusions.  Quality failures stay out of all
automatic training lists without being turned into speaker negatives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_FEEDBACK = Path(
    r"C:/Users/lulu/Downloads/full_semantic_acceptance_feedback_v1_2026-08-02T09-13-25-238Z.json"
)
PRIVATE = ROOT / "full_semantic_acceptance_v1" / "private_manifest.json"
OUT = ROOT / "human_full_semantic_acceptance_v1"

ROUND_AUDITS = {
    "round1": ROOT / "human_calibration_v1" / "audit_manifest.json",
    "round2": ROOT / "human_calibration_round2_v1" / "audit_manifest.json",
    "round3": ROOT / "cluster_acceptance_round3_v1" / "audit_manifest.json",
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


def error_class(error_type: str) -> str:
    if error_type == "切头/切尾":
        return "boundary_cut"
    if error_type == "杂音/音质问题":
        return "quality_issue"
    if error_type == "重叠说话":
        return "overlap"
    if error_type == "混入/无缝接话":
        return "mixed"
    if error_type == "非伊蕾娜":
        return "pure_non_target"
    if error_type in {"无法确定", "其他"}:
        return "uncertain"
    return "unknown_error"


def control_lookup() -> dict[str, dict]:
    result: dict[str, dict] = {}
    for round_name, path in ROUND_AUDITS.items():
        document = json.loads(path.read_text(encoding="utf-8"))
        for row in document["samples"]:
            key = f"{round_name}:{row['blind_id']}"
            result[key] = row
    return result


def resolve_origin(private_row: dict, controls: dict[str, dict]) -> dict:
    payload = private_row["payload"]
    if private_row["role"] == "machine_candidate":
        return {
            "source_id": str(private_row["source_id"]),
            "source": str(payload["source"]),
            "start": float(payload["start"]),
            "end": float(payload["end"]),
            "text": str(payload.get("text") or ""),
            "word_start_index": int(payload["word_start_index"]),
            "word_end_index": int(payload["word_end_index"]),
            "machine_id": str(payload["blind_id"]),
        }

    key = str(payload["human_key"])
    source = controls.get(key)
    if source is None:
        raise KeyError(f"Cannot resolve gold control {key}")
    indices = [int(value) for value in source.get("included_word_indices", [])]
    return {
        "source_id": str(source["source_id"]),
        "source": str(source["source"]),
        "start": float(source["start"]),
        "end": float(source["end"]),
        "text": str(source.get("private_text") or ""),
        "word_start_index": min(indices) if indices else None,
        "word_end_index": max(indices) if indices else None,
        "machine_id": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    feedback_path = args.feedback.resolve()
    out = args.out.resolve()
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    private = json.loads(PRIVATE.read_text(encoding="utf-8"))
    if not bool(feedback.get("complete")):
        raise RuntimeError("Feedback is not complete")
    if int(feedback["summary"]["reviewed"]) != len(private["samples"]):
        raise RuntimeError("Feedback/private-manifest sample count mismatch")

    controls = control_lookup()
    private_by_id = {str(row["blind_id"]): row for row in private["samples"]}
    resolved = []
    hard_speaker_exclusions = []
    training_exclusions = []

    for human in feedback["samples"]:
        blind_id = str(human["blind_id"])
        private_row = private_by_id[blind_id]
        origin = resolve_origin(private_row, controls)
        is_error = bool(human["is_error"])
        final_class = "target" if not is_error else error_class(str(human["error_type"]))
        note = str(human.get("note") or "")
        secondary_flags = []
        if "音质" in note or "音質" in note:
            secondary_flags.append("quality_issue")
        row = {
            "human_key": f"full_semantic_acceptance_v1:{blind_id}",
            "blind_id": blind_id,
            "role": str(private_row["role"]),
            **origin,
            "duration": float(origin["end"] - origin["start"]),
            "reviewed": bool(human["reviewed"]),
            "is_error": is_error,
            "error_type": str(human["error_type"]),
            "human_note": note,
            "final_class": final_class,
            "secondary_flags": secondary_flags,
            "play_count": int(human.get("play_count") or 0),
            "max_listened_sec": float(human.get("max_listened_sec") or 0.0),
            "speaker_learning_policy": (
                "positive_identity_only_recut_required"
                if final_class == "boundary_cut"
                else "speaker_negative"
                if final_class == "pure_non_target"
                else "validation_only"
                if final_class in {"mixed", "overlap", "quality_issue", "uncertain"}
                else "human_target"
            ),
        }
        resolved.append(row)

        if final_class != "target":
            training_exclusions.append(
                {
                    "source_id": row["source_id"],
                    "start": row["start"],
                    "end": row["end"],
                    "reason": f"{blind_id}:{final_class}",
                }
            )
        if final_class in {"pure_non_target", "mixed", "overlap"}:
            hard_speaker_exclusions.append(
                {
                    "source_id": row["source_id"],
                    "start": max(0.0, row["start"] - 0.10),
                    "end": row["end"] + 0.10,
                    "reason": f"human_confirmed_{final_class}:{blind_id}",
                    "note": note,
                }
            )

    out.mkdir(parents=True, exist_ok=True)
    archive = out / f"feedback_received_{feedback_path.name}"
    shutil.copy2(feedback_path, archive)
    write_json(
        out / "labels_resolved.json",
        {
            "version": "human_full_semantic_acceptance_v1_labels",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "feedback_source": str(feedback_path),
            "samples": resolved,
        },
    )
    write_json(
        out / "hard_exclusion_intervals.json",
        {
            "version": "human_full_semantic_acceptance_v1_hard_exclusions",
            "policy": "mixed/non-target/overlap only; boundary and quality errors are not speaker negatives",
            "intervals": hard_speaker_exclusions,
        },
    )
    write_json(
        out / "training_exclusion_intervals.json",
        {
            "version": "human_full_semantic_acceptance_v1_training_exclusions",
            "policy": "all failed clips remain excluded until an independently re-cut version passes human review",
            "intervals": training_exclusions,
        },
    )

    classes = Counter(row["final_class"] for row in resolved)
    roles: dict[str, Counter] = defaultdict(Counter)
    for row in resolved:
        roles[row["role"]][row["final_class"]] += 1
    report = {
        "version": "human_full_semantic_acceptance_v1_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feedback_source": str(feedback_path),
        "feedback_archive": str(archive),
        "feedback_sha256": sha256(feedback_path),
        "complete": True,
        "counts": {
            "all": len(resolved),
            "classes": dict(sorted(classes.items())),
            "by_role": {
                role: dict(sorted(counts.items())) for role, counts in sorted(roles.items())
            },
            "speaker_hard_exclusions": len(hard_speaker_exclusions),
            "training_exclusions": len(training_exclusions),
        },
        "diagnosis": {
            "boundary_failures_are_speaker_positive": classes["boundary_cut"],
            "mixed_intervals_are_not_recoverable_by_padding": classes["mixed"]
            + classes["overlap"],
            "quality_failures_are_not_speaker_negatives": classes["quality_issue"],
            "gold_controls_failed_boundary": sum(
                1
                for row in resolved
                if row["role"] == "known_gold_control"
                and row["final_class"] == "boundary_cut"
            ),
            "gold_controls_failed_quality": sum(
                1
                for row in resolved
                if row["role"] == "known_gold_control"
                and row["final_class"] == "quality_issue"
            ),
        },
        "training_started": False,
        "existing_training_lists_modified": False,
        "score_90_claimed": False,
    }
    write_json(out / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
