# -*- coding: utf-8 -*-
"""Ingest boundary-rescue feedback and freeze one heard pass per origin."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf


ROOT = Path(__file__).resolve().parent
DEFAULT_FEEDBACK = Path(
    r"C:/Users/lulu/Downloads/full_semantic_boundary_rescue_feedback_v1_2026-08-02T09-33-03-592Z.json"
)
AUDIT = ROOT / "full_semantic_acceptance_boundary_rescue_v1" / "audit_manifest.json"
OUT = ROOT / "human_full_semantic_boundary_rescue_v1"
FINAL = ROOT / "human_verified_semantic_rescues_v1"
MASTER_LIST = ROOT / "human_verified_semantic_rescues_v1.list"
TRAIN_SAFE_LIST = ROOT / "human_verified_semantic_rescues_v1_train_1p5_6s.list"


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


def classify(human: dict) -> str:
    if not bool(human["is_error"]):
        return "target"
    note = str(human.get("note") or "")
    error_type = str(human.get("error_type") or "")
    if "环境音" in note or "環境音" in note or error_type == "杂音/音质问题":
        return "quality_issue"
    if any(token in note for token in ("不是伊蕾娜", "非伊蕾娜", "别人", "其他人")):
        return "mixed_edge"
    if error_type == "切头/切尾":
        return "boundary_cut"
    if error_type == "重叠说话":
        return "overlap"
    if error_type in {"混入/无缝接话", "非伊蕾娜"}:
        return "mixed"
    return "uncertain"


def selection_score(row: dict) -> tuple:
    variant = str(row["variant"])
    priority = {
        "sentence_complete": 8,
        "sentence_complete_gold_recut": 8,
        "sentence_complete_long_quality_recheck": 8,
        "restore_sentence_head": 7,
        "restore_split_run_tail": 7,
        "restore_terminal_word": 7,
        "word_complete": 5,
        "eroded_words_restored": 4,
    }.get(variant, 0)
    return (priority, len(str(row["text"])), float(row["duration"]))


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
        raise RuntimeError("Feedback and audit IDs do not match")

    labels = []
    for human in feedback["samples"]:
        source = audit_by_id[str(human["blind_id"])]
        final_class = classify(human)
        labels.append(
            {
                "human_key": f"full_semantic_boundary_rescue_v1:{human['blind_id']}",
                "blind_id": str(human["blind_id"]),
                "origin_blind_id": str(source["origin_blind_id"]),
                "origin_human_key": str(source["origin_human_key"]),
                "source_id": str(source["source_id"]),
                "source": str(source["source"]),
                "start": float(source["start"]),
                "end": float(source["end"]),
                "duration": float(source["duration"]),
                "word_start_index": int(source["word_start_index"]),
                "word_end_index": int(source["word_end_index"]),
                "text": str(source["text"]),
                "variant": str(source["variant"]),
                "audio": str(source["audio"]),
                "final_class": final_class,
                "error_type": str(human.get("error_type") or ""),
                "human_note": str(human.get("note") or ""),
                "play_count": int(human.get("play_count") or 0),
                "max_listened_sec": float(human.get("max_listened_sec") or 0.0),
                "speaker_learning_policy": (
                    "human_target"
                    if final_class == "target"
                    else "positive_identity_recut_required"
                    if final_class == "boundary_cut"
                    else "validation_only_not_pure_negative"
                ),
            }
        )

    # The failed long A001 expansion and the passed conservative sibling give
    # a precise edge mask: only the extra left context is unsafe.  Do not mask
    # the passed R013 interval itself.
    passed_by_origin: dict[str, list[dict]] = defaultdict(list)
    for row in labels:
        if row["final_class"] == "target":
            passed_by_origin[row["origin_blind_id"]].append(row)
    hard_exclusions = []
    for row in labels:
        if row["final_class"] not in {"mixed", "mixed_edge", "overlap"}:
            continue
        siblings = passed_by_origin.get(row["origin_blind_id"], [])
        if row["final_class"] == "mixed_edge" and siblings:
            safe_start = min(float(item["start"]) for item in siblings)
            end = min(float(row["end"]), safe_start - 0.005)
            start = max(0.0, float(row["start"]) - 0.10)
        else:
            start = max(0.0, float(row["start"]) - 0.10)
            end = float(row["end"]) + 0.10
        if end > start:
            hard_exclusions.append(
                {
                    "source_id": row["source_id"],
                    "start": start,
                    "end": end,
                    "reason": f"human_confirmed_{row['final_class']}:{row['blind_id']}",
                    "note": row["human_note"],
                }
            )

    selected = []
    for origin_id, rows in sorted(passed_by_origin.items()):
        selected.append(max(rows, key=selection_score))
    if len(selected) != 11:
        raise RuntimeError(f"Expected 11 rescued origins, got {len(selected)}")

    FINAL_AUDIO = FINAL / "audio"
    FINAL_AUDIO.mkdir(parents=True, exist_ok=True)
    frozen = []
    for index, row in enumerate(sorted(selected, key=lambda item: (item["source_id"], item["start"])), 1):
        source_audio = Path(row["audio"])
        output = FINAL_AUDIO / f"HFSR{index:03d}_{row['source_id']}.wav"
        shutil.copy2(source_audio, output)
        if sha256(source_audio) != sha256(output):
            raise RuntimeError(f"Copy hash mismatch: {output}")
        info = sf.info(str(output))
        if info.samplerate != 32000 or info.channels != 1 or info.subtype != "PCM_16":
            raise RuntimeError(f"Invalid frozen audio format: {output}")
        frozen.append(
            {
                **row,
                "final_id": f"HFSR{index:03d}",
                "source_review_audio": str(source_audio),
                "audio": str(output.resolve()),
                "duration": float(info.duration),
                "sha256": sha256(output),
                "provenance": "human_verified_full_semantic_boundary_rescue_v1",
                "training_duration_safe": 1.5 <= float(info.duration) <= 6.0,
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / f"feedback_received_{feedback_path.name}"
    shutil.copy2(feedback_path, archive)
    write_json(
        OUT / "labels_resolved.json",
        {
            "version": "human_full_semantic_boundary_rescue_v1_labels",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "feedback_source": str(feedback_path),
            "samples": labels,
        },
    )
    write_json(
        OUT / "hard_exclusion_intervals.json",
        {
            "version": "human_full_semantic_boundary_rescue_v1_hard_exclusions",
            "policy": "only localized non-target/mixed edges; boundary and environment-noise failures are not speaker negatives",
            "intervals": hard_exclusions,
        },
    )
    write_json(
        FINAL / "audit_manifest.json",
        {
            "version": "human_verified_semantic_rescues_v1_audit",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "selection": "one highest-priority human-passed version per original boundary problem",
            "samples": frozen,
        },
    )

    def list_line(row: dict) -> str:
        return f"{row['audio']}|ireina_human_verified_semantic_rescue|JA|{row['text']}"

    MASTER_LIST.write_text(
        "/n".join(list_line(row) for row in frozen) + "/n", encoding="utf-8"
    )
    train_safe = [row for row in frozen if row["training_duration_safe"]]
    TRAIN_SAFE_LIST.write_text(
        "/n".join(list_line(row) for row in train_safe) + "/n", encoding="utf-8"
    )

    class_counts = Counter(row["final_class"] for row in labels)
    source_stats: dict[str, dict] = defaultdict(lambda: {"segments": 0, "duration": 0.0})
    for row in frozen:
        source_stats[row["source_id"]]["segments"] += 1
        source_stats[row["source_id"]]["duration"] += row["duration"]
    report = {
        "version": "human_verified_semantic_rescues_v1_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feedback_source": str(feedback_path),
        "feedback_archive": str(archive),
        "feedback_sha256": sha256(feedback_path),
        "review_counts": {
            "samples": len(labels),
            "classes": dict(sorted(class_counts.items())),
            "passed_samples": class_counts["target"],
            "passed_unique_origins": len(frozen),
        },
        "frozen": {
            "segments": len(frozen),
            "duration_sec": sum(row["duration"] for row in frozen),
            "train_1p5_6s_segments": len(train_safe),
            "train_1p5_6s_duration_sec": sum(row["duration"] for row in train_safe),
            "source_stats": dict(sorted(source_stats.items())),
        },
        "remaining": {
            "boundary_origins": sorted(
                {row["origin_blind_id"] for row in labels if row["final_class"] == "boundary_cut"}
            ),
            "quality_origins": sorted(
                {row["origin_blind_id"] for row in labels if row["final_class"] == "quality_issue"}
            ),
            "resolved_by_sibling_pass": sorted(
                {
                    row["origin_blind_id"]
                    for row in labels
                    if row["final_class"] == "mixed_edge"
                    and passed_by_origin.get(row["origin_blind_id"])
                }
            ),
        },
        "hard_exclusion_intervals": hard_exclusions,
        "outputs": {
            "master_list": str(MASTER_LIST),
            "train_safe_list": str(TRAIN_SAFE_LIST),
            "audit": str(FINAL / "audit_manifest.json"),
        },
        "training_started": False,
        "existing_training_lists_modified": False,
        "score_90_claimed": False,
    }
    write_json(FINAL / "report.json", report)
    write_json(OUT / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
