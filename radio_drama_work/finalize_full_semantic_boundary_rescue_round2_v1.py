# -*- coding: utf-8 -*-
"""Merge heard round-2 passes into a duration-safe semantic rescue set."""
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
    r"C:/Users/lulu/Downloads/full_semantic_boundary_rescue_round2_feedback_v1_2026-08-02T09-41-12-047Z.json"
)
ROUND2_AUDIT = ROOT / "full_semantic_boundary_rescue_round2_v1" / "audit_manifest.json"
V1_AUDIT = ROOT / "human_verified_semantic_rescues_v1" / "audit_manifest.json"
V1_HARD = ROOT / "human_full_semantic_boundary_rescue_v1" / "hard_exclusion_intervals.json"
OUT = ROOT / "human_full_semantic_boundary_rescue_round2_v1"
FINAL = ROOT / "human_verified_semantic_rescues_v2"
FINAL_LIST = ROOT / "human_verified_semantic_rescues_v2.list"


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


def classify(row: dict) -> str:
    if not bool(row["is_error"]):
        return "target"
    note = str(row.get("note") or "")
    error_type = str(row.get("error_type") or "")
    if "环境音" in note or "環境音" in note or error_type == "杂音/音质问题":
        return "quality_issue"
    if any(token in note for token in ("最开始", "开头", "開頭", "一点点")):
        return "dirty_boundary"
    if error_type == "切头/切尾":
        return "boundary_cut"
    return "uncertain"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    args = parser.parse_args()
    feedback_path = args.feedback.resolve()
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    audit = json.loads(ROUND2_AUDIT.read_text(encoding="utf-8"))
    if not bool(feedback.get("complete")):
        raise RuntimeError("Feedback is incomplete")
    audit_by_id = {str(row["blind_id"]): row for row in audit["samples"]}
    if set(audit_by_id) != {str(row["blind_id"]) for row in feedback["samples"]}:
        raise RuntimeError("Feedback/audit IDs do not match")

    labels = []
    for human in feedback["samples"]:
        source = audit_by_id[str(human["blind_id"])]
        labels.append(
            {
                "human_key": f"full_semantic_boundary_rescue_round2_v1:{human['blind_id']}",
                "blind_id": str(human["blind_id"]),
                "parent_blind_id": str(source["parent_blind_id"]),
                "origin_blind_id": str(source["root_origin_blind_id"]),
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
                "final_class": classify(human),
                "error_type": str(human.get("error_type") or ""),
                "human_note": str(human.get("note") or ""),
                "play_count": int(human.get("play_count") or 0),
                "max_listened_sec": float(human.get("max_listened_sec") or 0.0),
            }
        )

    passed = [row for row in labels if row["final_class"] == "target"]
    if {row["blind_id"] for row in passed} != {"T001", "T004", "T005", "T006"}:
        raise RuntimeError("Unexpected round-2 pass set")

    old = json.loads(V1_AUDIT.read_text(encoding="utf-8"))["samples"]
    replaced_origins = {"A004", "A011"}
    merged = [dict(row) for row in old if str(row["origin_blind_id"]) not in replaced_origins]
    for row in passed:
        merged.append(
            {
                **row,
                "origin_human_key": f"full_semantic_acceptance_v1:{row['origin_blind_id']}",
                "speaker_learning_policy": "human_target",
                "provenance": "human_verified_full_semantic_boundary_rescue_round2_v1",
                "training_duration_safe": True,
            }
        )
    if len(merged) != 13:
        raise RuntimeError(f"Expected 13 merged segments, got {len(merged)}")

    FINAL_AUDIO = FINAL / "audio"
    FINAL_AUDIO.mkdir(parents=True, exist_ok=True)
    frozen = []
    for index, row in enumerate(sorted(merged, key=lambda item: (item["source_id"], float(item["start"]), float(item["end"]))), 1):
        source_audio = Path(row["audio"])
        output = FINAL_AUDIO / f"HVSR{index:03d}_{row['source_id']}.wav"
        shutil.copy2(source_audio, output)
        if sha256(source_audio) != sha256(output):
            raise RuntimeError(f"Copy mismatch: {output}")
        info = sf.info(str(output))
        if info.samplerate != 32000 or info.channels != 1 or info.subtype != "PCM_16":
            raise RuntimeError(f"Invalid audio format: {output}")
        if not (1.5 <= info.duration <= 6.0005):
            raise RuntimeError(f"Duration is not training-safe: {output} {info.duration}")
        clean = {
            key: value
            for key, value in row.items()
            if key not in {"final_id", "sha256", "source_review_audio"}
        }
        frozen.append(
            {
                **clean,
                "final_id": f"HVSR{index:03d}",
                "source_review_audio": str(source_audio),
                "audio": str(output.resolve()),
                "duration": float(info.duration),
                "sha256": sha256(output),
                "training_duration_safe": True,
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / f"feedback_received_{feedback_path.name}"
    shutil.copy2(feedback_path, archive)
    write_json(
        OUT / "labels_resolved.json",
        {
            "version": "human_full_semantic_boundary_rescue_round2_v1_labels",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "samples": labels,
        },
    )
    dirty_boundaries = [
        {
            "source_id": row["source_id"],
            "start": row["start"],
            "end": min(row["end"], row["start"] + 0.12),
            "reason": f"human_dirty_boundary:{row['blind_id']}",
            "note": row["human_note"],
        }
        for row in labels
        if row["final_class"] == "dirty_boundary"
    ]
    write_json(
        OUT / "dirty_boundary_intervals.json",
        {
            "version": "human_full_semantic_boundary_rescue_round2_v1_dirty_boundaries",
            "policy": "validation/recut boundary only; not a pure speaker-negative interval",
            "intervals": dirty_boundaries,
        },
    )
    hard = json.loads(V1_HARD.read_text(encoding="utf-8"))
    write_json(FINAL / "hard_exclusion_intervals.json", hard)
    write_json(
        FINAL / "audit_manifest.json",
        {
            "version": "human_verified_semantic_rescues_v2_audit",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "selection": "v1 human passes with long/quality items replaced by independently heard <=6s round-2 passes",
            "samples": frozen,
        },
    )
    FINAL_LIST.write_text(
        "/n".join(
            f"{row['audio']}|ireina_human_verified_semantic_rescue|JA|{row['text']}"
            for row in frozen
        )
        + "/n",
        encoding="utf-8",
    )

    counts = Counter(row["final_class"] for row in labels)
    source_stats: dict[str, dict] = defaultdict(lambda: {"segments": 0, "duration": 0.0})
    for row in frozen:
        source_stats[row["source_id"]]["segments"] += 1
        source_stats[row["source_id"]]["duration"] += row["duration"]
    report = {
        "version": "human_verified_semantic_rescues_v2_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feedback_source": str(feedback_path),
        "feedback_archive": str(archive),
        "feedback_sha256": sha256(feedback_path),
        "round2": {
            "samples": len(labels),
            "classes": dict(sorted(counts.items())),
            "passed": len(passed),
        },
        "frozen": {
            "segments": len(frozen),
            "duration_sec": sum(row["duration"] for row in frozen),
            "all_1p5_6s": all(row["training_duration_safe"] for row in frozen),
            "source_stats": dict(sorted(source_stats.items())),
        },
        "remaining_edge_origin": "A021",
        "quality_partial_rescue": {
            "origin": "A017",
            "accepted": "T004",
            "rejected_environment_sound": "T003",
        },
        "dirty_boundary_intervals": dirty_boundaries,
        "output_list": str(FINAL_LIST),
        "training_started": False,
        "existing_training_lists_modified": False,
        "score_90_claimed": False,
    }
    write_json(FINAL / "report.json", report)
    write_json(OUT / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
