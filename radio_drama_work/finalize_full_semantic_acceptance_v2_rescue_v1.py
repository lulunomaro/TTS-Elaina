# -*- coding: utf-8 -*-
"""Finalize passed v2 rescues and localize the fully non-target B009 clip."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf


ROOT = Path(__file__).resolve().parent
DEFAULT_FEEDBACK = Path(
    r"C:/Users/lulu/Downloads/full_semantic_acceptance_v2_rescue_feedback_v1_2026-08-02T10-55-34-765Z.json"
)
AUDIT = ROOT / "full_semantic_acceptance_v2_rescue_v1" / "audit_manifest.json"
ROOT_LABELS = ROOT / "human_full_semantic_acceptance_v2" / "labels_resolved.json"
OUT = ROOT / "human_full_semantic_acceptance_v2_rescue_v1"
FROZEN = ROOT / "human_verified_full_semantic_v2_rescues_v1"
FROZEN_LIST = ROOT / "human_verified_full_semantic_v2_rescues_v1.list"

FINAL_SELECTION = {
    "B005": {"complete_clause_1", "complete_clause_2"},
    "B007": {"restore_missing_toku"},
    "B022": {"restore_question_ka"},
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


def classify(human: dict, source: dict) -> str:
    if not bool(human.get("is_error")):
        return "target"
    error_type = str(human.get("error_type") or "")
    note = str(human.get("note") or "")
    if error_type == "非伊蕾娜":
        return "pure_non_target"
    if error_type in {"混入/无缝接话", "重叠说话"}:
        return "mixed_edge"
    if any(token in note for token in ("还是有", "一样的问题", "开头")):
        return "mixed_edge"
    if error_type in {"开头被切", "结尾被切", "开头和结尾都被切", "切头/切尾"}:
        return "boundary_cut"
    return "uncertain"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    args = parser.parse_args()
    feedback_path = args.feedback.resolve()
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if not bool(feedback.get("complete")):
        raise RuntimeError("Feedback is incomplete")
    audit_by_id = {str(row["blind_id"]): row for row in audit["samples"]}
    if set(audit_by_id) != {str(row["blind_id"]) for row in feedback["samples"]}:
        raise RuntimeError("Feedback/audit IDs differ")

    labels = []
    for human in feedback["samples"]:
        source = audit_by_id[str(human["blind_id"])]
        labels.append(
            {
                "human_key": f"full_semantic_acceptance_v2_rescue_v1:{human['blind_id']}",
                "blind_id": str(human["blind_id"]),
                "origin_blind_id": str(source["origin_blind_id"]),
                "source_id": str(source["source_id"]),
                "source": str(source["source"]),
                "variant": str(source["variant"]),
                "start": float(source["start"]),
                "end": float(source["end"]),
                "duration": float(source["output_duration"]),
                "word_start_index": int(source["word_start_index"]),
                "word_end_index": int(source["word_end_index"]),
                "text": str(source["text"]),
                "audio": str(source["audio"]),
                "final_class": classify(human, source),
                "error_type": str(human.get("error_type") or ""),
                "human_note": str(human.get("note") or ""),
                "play_count": int(human.get("play_count") or 0),
                "max_listened_sec": float(human.get("max_listened_sec") or 0.0),
            }
        )

    selected = []
    for row in labels:
        wanted = FINAL_SELECTION.get(row["origin_blind_id"], set())
        if row["variant"] in wanted:
            if row["final_class"] != "target":
                raise RuntimeError(f"Selected rescue did not pass: {row['blind_id']}")
            selected.append(row)
    expected_selected = sum(len(values) for values in FINAL_SELECTION.values())
    if len(selected) != expected_selected:
        raise RuntimeError(f"Expected {expected_selected} selected rescues, got {len(selected)}")

    frozen_audio = FROZEN / "audio"
    frozen_audio.mkdir(parents=True, exist_ok=True)
    frozen = []
    for index, row in enumerate(
        sorted(selected, key=lambda item: (item["source_id"], item["start"])), 1
    ):
        source_audio = Path(row["audio"])
        output = frozen_audio / f"HFV2R{index:03d}_{row['source_id']}.wav"
        shutil.copy2(source_audio, output)
        if sha256(source_audio) != sha256(output):
            raise RuntimeError(f"Copy hash mismatch: {output}")
        info = sf.info(str(output))
        if (
            info.samplerate != 32000
            or info.channels != 1
            or info.subtype != "PCM_16"
            or not 1.5 <= info.duration <= 6.0
        ):
            raise RuntimeError(f"Invalid frozen rescue: {output} {info}")
        frozen.append(
            {
                **row,
                "final_id": f"HFV2R{index:03d}",
                "source_review_audio": str(source_audio),
                "audio": str(output.resolve()),
                "duration": float(info.duration),
                "sha256": sha256(output),
                "provenance": "human_verified_full_semantic_v2_rescue_v1",
            }
        )

    root_labels = {
        str(row["blind_id"]): row
        for row in json.loads(ROOT_LABELS.read_text(encoding="utf-8"))["samples"]
    }
    b009 = root_labels["B009"]
    if not all(
        row["final_class"] == "pure_non_target"
        for row in labels
        if row["origin_blind_id"] == "B009"
    ):
        raise RuntimeError("B009 was not unanimously pure non-target")
    hard_exclusions = [
        {
            "source_id": str(b009["source_id"]),
            "start": max(0.0, float(b009["start"]) - 0.10),
            "end": float(b009["end"]) + 0.10,
            "reason": "human_confirmed_pure_non_target:v2_rescue_B009",
            "note": "original candidate and all three isolated second-reply variants were non-target",
        }
    ]
    pending = [
        {
            "origin_blind_id": origin,
            "reason": "all millisecond variants retained the previous speaker edge",
            "next_strategy": strategy,
        }
        for origin, strategy in (
            ("B013", "drop ご安心ください and restart at 先生 (word 1483)"),
            ("B024", "drop そうですね and restart at クリソツですね (word 3293)"),
        )
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / f"feedback_received_{feedback_path.name}"
    shutil.copy2(feedback_path, archive)
    write_json(
        OUT / "labels_resolved.json",
        {
            "version": "human_full_semantic_acceptance_v2_rescue_v1_labels",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "feedback_source": str(feedback_path),
            "samples": labels,
        },
    )
    write_json(
        OUT / "hard_exclusion_intervals.json",
        {
            "version": "human_full_semantic_acceptance_v2_rescue_v1_hard_exclusions",
            "policy": "only unanimously confirmed pure non-target; mixed edges remain pending",
            "intervals": hard_exclusions,
        },
    )
    write_json(
        OUT / "pending_final_rescues.json",
        {
            "version": "human_full_semantic_acceptance_v2_rescue_v1_pending",
            "samples": pending,
        },
    )
    write_json(
        FROZEN / "audit_manifest.json",
        {
            "version": "human_verified_full_semantic_v2_rescues_v1_audit",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "selection": "B005 train-length clauses plus passed B007/B022 complete-word rescues",
            "excluded_clean_reference": "B005 6.335s complete sentence passed but is not train-length",
            "samples": frozen,
        },
    )
    FROZEN_LIST.write_text(
        "/n".join(
            f"{row['audio']}|ireina_human_verified_full_semantic_v2_rescue|JA|{row['text']}"
            for row in frozen
        )
        + "/n",
        encoding="utf-8",
    )

    classes = Counter(row["final_class"] for row in labels)
    report = {
        "version": "human_verified_full_semantic_v2_rescues_v1_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feedback_source": str(feedback_path),
        "feedback_archive": str(archive),
        "feedback_sha256": sha256(feedback_path),
        "review": {
            "samples": len(labels),
            "classes": dict(sorted(classes.items())),
            "passed_samples": classes["target"],
        },
        "frozen": {
            "count": len(frozen),
            "duration_sec": sum(row["duration"] for row in frozen),
            "list": str(FROZEN_LIST),
        },
        "clean_over_6s_reference_not_frozen": 1,
        "hard_exclusions": hard_exclusions,
        "pending_final_rescues": pending,
        "training_started": False,
        "existing_training_lists_modified": False,
        "score_90_claimed": False,
    }
    write_json(OUT / "report.json", report)
    write_json(FROZEN / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
