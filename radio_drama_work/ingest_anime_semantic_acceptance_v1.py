# -*- coding: utf-8 -*-
"""Resolve anime trainability feedback and freeze only direct human passes."""
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
AUDIT = ROOT / "anime_semantic_acceptance_v1" / "audit_manifest.json"
OUT = ROOT / "human_anime_semantic_acceptance_v1"
FROZEN = ROOT / "human_verified_anime_semantic_v1_direct"
FROZEN_LIST = ROOT / "human_verified_anime_semantic_v1_direct.list"


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


def classify(row: dict) -> str:
    if not bool(row.get("is_error")):
        return "target"
    error = str(row.get("error_type") or "")
    if error == "非伊蕾娜":
        return "pure_non_target"
    if error == "混入/无缝接话":
        return "mixed"
    if error == "重叠说话":
        return "overlap"
    if error in {
        "开头被切",
        "结尾被切",
        "开头和结尾都被切",
        "切头/切尾",
    }:
        return "boundary_cut"
    if error in {"杂音/音质问题", "环境音/道具声"}:
        return "quality_issue"
    if error == "文本不对应":
        return "text_issue"
    return "uncertain"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback", type=Path, required=True)
    args = parser.parse_args()
    feedback_path = args.feedback.resolve()
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if not bool(feedback.get("complete")):
        raise RuntimeError("Anime semantic feedback is incomplete")
    audit_by_id = {str(row["blind_id"]): row for row in audit["samples"]}
    human_by_id = {str(row["blind_id"]): row for row in feedback["samples"]}
    if set(audit_by_id) != set(human_by_id):
        raise RuntimeError("Feedback IDs do not match anime acceptance audit")

    labels = []
    for blind_id in sorted(audit_by_id):
        source = audit_by_id[blind_id]
        human = human_by_id[blind_id]
        labels.append(
            {
                "human_key": f"anime_semantic_acceptance_v1:{blind_id}",
                "blind_id": blind_id,
                "source_id": str(source["source_id"]),
                "source": str(source["source"]),
                "start": float(source["start"]),
                "end": float(source["end"]),
                "duration": float(source["output_duration"]),
                "text": str(source["text"]),
                "word_start_index": int(source["word_start_index"]),
                "word_end_index": int(source["word_end_index"]),
                "review_audio": str(source["audio"]),
                "origin_machine_id": str(source["origin_machine_id"]),
                "final_class": classify(human),
                "error_type": str(human.get("error_type") or ""),
                "human_note": str(human.get("note") or ""),
                "play_count": int(human.get("play_count") or 0),
                "max_listened_sec": float(human.get("max_listened_sec") or 0.0),
            }
        )

    direct = [row for row in labels if row["final_class"] == "target"]
    frozen_audio = FROZEN / "audio"
    frozen_audio.mkdir(parents=True, exist_ok=True)
    frozen_rows = []
    for index, row in enumerate(
        sorted(direct, key=lambda item: (item["source_id"], item["start"])), 1
    ):
        source_audio = Path(row["review_audio"])
        output = frozen_audio / f"HAV1{index:04d}_{row['source_id']}.wav"
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
            raise RuntimeError(f"Invalid frozen anime clip: {output} {info}")
        frozen_rows.append(
            {
                **row,
                "final_id": f"HAV1{index:04d}",
                "audio": str(output.resolve()),
                "duration": float(info.duration),
                "sha256": sha256(output),
                "provenance": "human_verified_anime_semantic_v1_direct",
            }
        )

    FROZEN_LIST.write_text(
        "/n".join(
            f"{row['audio']}|ireina_human_verified_anime_v1|JA|{row['text']}"
            for row in frozen_rows
        )
        + ("/n" if frozen_rows else ""),
        encoding="utf-8",
    )
    hard_masks = [
        {
            "source_id": row["source_id"],
            "start": row["start"],
            "end": row["end"],
            "reason": f"human_anime_acceptance:{row['blind_id']}:{row['final_class']}",
        }
        for row in labels
        if row["final_class"] in {"pure_non_target", "overlap"}
    ]
    mixed_queue = [
        row for row in labels if row["final_class"] == "mixed"
    ]
    boundary_queue = [
        row for row in labels if row["final_class"] == "boundary_cut"
    ]
    quality_exclusions = [
        row for row in labels if row["final_class"] == "quality_issue"
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / f"feedback_received_{feedback_path.name}"
    shutil.copy2(feedback_path, archive)
    write_json(
        OUT / "labels_resolved.json",
        {
            "version": "human_anime_semantic_acceptance_v1_labels",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "feedback_source": str(feedback_path),
            "samples": labels,
        },
    )
    write_json(
        OUT / "hard_exclusion_intervals.json",
        {
            "version": "human_anime_semantic_acceptance_v1_hard_exclusions",
            "policy": "whole-clip pure non-target and overlap only; mixed requires localization",
            "intervals": hard_masks,
        },
    )
    write_json(OUT / "contamination_localization_queue.json", {"samples": mixed_queue})
    write_json(OUT / "boundary_rescue_queue.json", {"samples": boundary_queue})
    write_json(OUT / "quality_exclusion_intervals.json", {"samples": quality_exclusions})
    classes = Counter(row["final_class"] for row in labels)
    report = {
        "version": "human_anime_semantic_acceptance_v1_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feedback_source": str(feedback_path),
        "feedback_archive": str(archive),
        "feedback_sha256": sha256(feedback_path),
        "review": {"samples": len(labels), "classes": dict(sorted(classes.items()))},
        "frozen": {
            "count": len(frozen_rows),
            "duration_sec": sum(row["duration"] for row in frozen_rows),
            "list": str(FROZEN_LIST.resolve()),
        },
        "hard_exclusion_count": len(hard_masks),
        "mixed_localization_count": len(mixed_queue),
        "boundary_rescue_count": len(boundary_queue),
        "quality_exclusion_count": len(quality_exclusions),
        "training_started": False,
        "existing_training_lists_modified": False,
        "score_90_claimed": False,
    }
    write_json(OUT / "report.json", report)
    write_json(FROZEN / "audit_manifest.json", {"samples": frozen_rows})
    write_json(FROZEN / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
