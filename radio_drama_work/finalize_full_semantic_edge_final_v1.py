# -*- coding: utf-8 -*-
"""Freeze the earliest human-passed final edge and produce rescue set v3."""
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
    r"C:/Users/lulu/Downloads/full_semantic_edge_final_feedback_v1_2026-08-02T09-47-17-672Z.json"
)
EDGE_AUDIT = ROOT / "full_semantic_edge_final_review_v1" / "audit_manifest.json"
V2_AUDIT = ROOT / "human_verified_semantic_rescues_v2" / "audit_manifest.json"
V2_HARD = ROOT / "human_verified_semantic_rescues_v2" / "hard_exclusion_intervals.json"
ROUND2_LABELS = ROOT / "human_full_semantic_boundary_rescue_round2_v1" / "labels_resolved.json"
OUT = ROOT / "human_full_semantic_edge_final_v1"
FINAL = ROOT / "human_verified_semantic_rescues_v3"
FINAL_LIST = ROOT / "human_verified_semantic_rescues_v3.list"


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
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    args = parser.parse_args()
    feedback_path = args.feedback.resolve()
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    audit = json.loads(EDGE_AUDIT.read_text(encoding="utf-8"))
    if not bool(feedback.get("complete")):
        raise RuntimeError("Feedback is incomplete")
    audit_by_id = {str(row["blind_id"]): row for row in audit["samples"]}
    labels = []
    for human in feedback["samples"]:
        source = audit_by_id[str(human["blind_id"])]
        labels.append(
            {
                "blind_id": str(human["blind_id"]),
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
                "final_class": "target" if not bool(human["is_error"]) else "dirty_boundary",
                "error_type": str(human.get("error_type") or ""),
                "human_note": str(human.get("note") or ""),
                "play_count": int(human.get("play_count") or 0),
                "max_listened_sec": float(human.get("max_listened_sec") or 0.0),
            }
        )
    passed = [row for row in labels if row["final_class"] == "target"]
    if {row["blind_id"] for row in passed} != {"U001", "U002"}:
        raise RuntimeError("Unexpected final-edge pass set")
    chosen = min(passed, key=lambda row: float(row["start"]))
    if chosen["blind_id"] != "U001":
        raise RuntimeError("Expected U001 to be the earliest passing edge")

    old = json.loads(V2_AUDIT.read_text(encoding="utf-8"))["samples"]
    merged = [dict(row) for row in old]
    merged.append(
        {
            **chosen,
            "human_key": "full_semantic_edge_final_v1:U001",
            "origin_blind_id": "A021",
            "origin_human_key": "full_semantic_acceptance_v1:A021",
            "speaker_learning_policy": "human_target",
            "provenance": "human_verified_full_semantic_edge_final_v1",
            "training_duration_safe": True,
        }
    )
    if len(merged) != 14:
        raise RuntimeError(f"Expected 14 merged rows, got {len(merged)}")

    FINAL_AUDIO = FINAL / "audio"
    FINAL_AUDIO.mkdir(parents=True, exist_ok=True)
    frozen = []
    for index, row in enumerate(sorted(merged, key=lambda item: (item["source_id"], float(item["start"]), float(item["end"]))), 1):
        source_audio = Path(row["audio"])
        output = FINAL_AUDIO / f"HVSR3_{index:03d}_{row['source_id']}.wav"
        shutil.copy2(source_audio, output)
        if sha256(source_audio) != sha256(output):
            raise RuntimeError(f"Copy mismatch: {output}")
        info = sf.info(str(output))
        if info.samplerate != 32000 or info.channels != 1 or info.subtype != "PCM_16":
            raise RuntimeError(f"Invalid audio format: {output}")
        if not (1.5 <= info.duration <= 6.0005):
            raise RuntimeError(f"Invalid duration: {output} {info.duration}")
        clean = {
            key: value
            for key, value in row.items()
            if key not in {"final_id", "sha256", "source_review_audio"}
        }
        frozen.append(
            {
                **clean,
                "final_id": f"HVSR3_{index:03d}",
                "source_review_audio": str(source_audio),
                "audio": str(output.resolve()),
                "duration": float(info.duration),
                "sha256": sha256(output),
                "training_duration_safe": True,
            }
        )

    # U003 starts at 475.590 and still contains the preceding vowel; U001 at
    # 475.610 passes.  The half-open dirty interval ends exactly at U001.
    boundary_exclusion = {
        "source_id": chosen["source_id"],
        "start": 475.565,
        "end": float(chosen["start"]),
        "reason": "human_confirmed_dirty_transition_before_target:A021",
        "note": "475.590 still contains residual e; 475.610 is the earliest human-passed start",
    }
    round2 = json.loads(ROUND2_LABELS.read_text(encoding="utf-8"))["samples"]
    quality_row = next(row for row in round2 if row["blind_id"] == "T003")
    safe_quality_row = next(row for row in round2 if row["blind_id"] == "T004")
    quality_exclusion = {
        "source_id": quality_row["source_id"],
        "start": max(float(quality_row["start"]), float(safe_quality_row["end"])),
        "end": float(quality_row["end"]),
        "reason": "human_confirmed_page_turn_and_book_close:T003",
        "note": quality_row["human_note"],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / f"feedback_received_{feedback_path.name}"
    shutil.copy2(feedback_path, archive)
    write_json(
        OUT / "labels_resolved.json",
        {
            "version": "human_full_semantic_edge_final_v1_labels",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "selected": chosen["blind_id"],
            "samples": labels,
        },
    )
    write_json(
        FINAL / "boundary_exclusion_intervals.json",
        {
            "version": "human_verified_semantic_rescues_v3_boundary_exclusions",
            "policy": "hard output boundary; not a pure speaker-negative reference",
            "intervals": [boundary_exclusion],
        },
    )
    write_json(
        FINAL / "quality_exclusion_intervals.json",
        {
            "version": "human_verified_semantic_rescues_v3_quality_exclusions",
            "policy": "audible non-speech effects; not a speaker-negative reference",
            "intervals": [quality_exclusion],
        },
    )
    prior_hard = json.loads(V2_HARD.read_text(encoding="utf-8"))
    write_json(FINAL / "hard_exclusion_intervals.json", prior_hard)
    write_json(
        FINAL / "audit_manifest.json",
        {
            "version": "human_verified_semantic_rescues_v3_audit",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "selection": "v2 duration-safe passes plus earliest human-passed A021 millisecond edge",
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

    source_stats: dict[str, dict] = defaultdict(lambda: {"segments": 0, "duration": 0.0})
    for row in frozen:
        source_stats[row["source_id"]]["segments"] += 1
        source_stats[row["source_id"]]["duration"] += row["duration"]
    report = {
        "version": "human_verified_semantic_rescues_v3_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feedback_source": str(feedback_path),
        "feedback_archive": str(archive),
        "feedback_sha256": sha256(feedback_path),
        "feedback_classes": dict(sorted(Counter(row["final_class"] for row in labels).items())),
        "selected_edge": {
            "blind_id": chosen["blind_id"],
            "start": chosen["start"],
            "reason": "earliest of the two human-passed starts",
        },
        "frozen": {
            "segments": len(frozen),
            "duration_sec": sum(row["duration"] for row in frozen),
            "all_1p5_6s": True,
            "source_stats": dict(sorted(source_stats.items())),
        },
        "boundary_exclusion": boundary_exclusion,
        "quality_exclusion": quality_exclusion,
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
