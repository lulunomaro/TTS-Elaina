# -*- coding: utf-8 -*-
"""Resolve boundary rescue feedback and freeze one best pass per origin."""
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
AUDIT = ROOT / "anime_boundary_rescue_batch_v1" / "audit_manifest.json"
OUT = ROOT / "human_anime_boundary_rescue_batch_v1"
FROZEN = ROOT / "human_verified_anime_boundary_rescue_v1"
LIST = ROOT / "human_verified_anime_boundary_rescue_v1.list"
BATCH_VERSION = "anime_boundary_rescue_batch_v1"
FINAL_PREFIX = "HABR1"


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
    if error in {"闈炰紛钑惧", "非伊蕾娜"}:
        return "pure_non_target"
    if error in {"娣峰叆/鏃犵紳鎺ヨ瘽", "混入/无缝接话"}:
        return "mixed"
    if error in {"閲嶅彔璇磋瘽", "重叠说话"}:
        return "overlap"
    if any(token in error for token in ("寮€澶", "缁撳熬", "鍒囧ご", "开头", "结尾", "切头")):
        return "boundary_cut"
    if any(token in error for token in ("鏉傞煶", "闊宠川", "杂音", "音质")):
        return "quality_issue"
    if any(token in error for token in ("鏂囨湰", "文本")):
        return "text_issue"
    return "uncertain"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback", type=Path, required=True)
    args = parser.parse_args()
    feedback_path = args.feedback.resolve()
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    if not bool(feedback.get("complete")):
        raise RuntimeError("Boundary rescue feedback is incomplete")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    audit_by_id = {str(row["blind_id"]): row for row in audit["samples"]}
    human_by_id = {str(row["blind_id"]): row for row in feedback["samples"]}
    if set(audit_by_id) != set(human_by_id):
        raise RuntimeError("Boundary rescue feedback IDs do not match audit")

    labels = []
    for blind_id in sorted(audit_by_id):
        source = audit_by_id[blind_id]
        human = human_by_id[blind_id]
        if not bool(human.get("reviewed")):
            raise RuntimeError(f"{blind_id}: not reviewed")
        labels.append(
            {
                "human_key": f"{BATCH_VERSION}:{blind_id}",
                "blind_id": blind_id,
                "origin_batch_blind_id": str(source["origin_batch_blind_id"]),
                "origin_pool_blind_id": str(source["origin_pool_blind_id"]),
                "source_id": str(source["source_id"]),
                "source": str(source["source"]),
                "variant": str(source["variant"]),
                "start": float(source["start"]),
                "end": float(source["end"]),
                "duration": float(source["output_duration"]),
                "text": str(source["text"]),
                "word_start_index": int(source["word_start_index"]),
                "word_end_index": int(source["word_end_index"]),
                "review_audio": str(source["audio"]),
                "final_class": classify(human),
                "error_type": str(human.get("error_type") or ""),
                "human_note": str(human.get("note") or ""),
                "play_count": int(human.get("play_count") or 0),
                "max_listened_sec": float(human.get("max_listened_sec") or 0.0),
            }
        )

    by_origin: dict[str, list[dict]] = defaultdict(list)
    for row in labels:
        if row["final_class"] == "target":
            by_origin[row["origin_batch_blind_id"]].append(row)
    selected = []
    for origin, rows in sorted(by_origin.items()):
        rows.sort(
            key=lambda row: (
                row["variant"] != "word_expand",
                -float(row["duration"]),
                row["blind_id"],
            )
        )
        selected.append(rows[0])

    frozen_audio = FROZEN / "audio"
    frozen_audio.mkdir(parents=True, exist_ok=True)
    frozen_rows = []
    seen_hashes = set()
    for index, row in enumerate(
        sorted(selected, key=lambda item: (item["source_id"], item["start"])), 1
    ):
        source_audio = Path(row["review_audio"])
        output = frozen_audio / f"{FINAL_PREFIX}{index:04d}_{row['source_id']}.wav"
        shutil.copy2(source_audio, output)
        info = sf.info(str(output))
        wave_hash = sha256(output)
        if wave_hash in seen_hashes:
            raise RuntimeError(f"Duplicate frozen rescue waveform: {output}")
        seen_hashes.add(wave_hash)
        if (
            info.samplerate != 32000
            or info.channels != 1
            or info.subtype != "PCM_16"
            or not 1.5 <= info.duration <= 6.0
        ):
            raise RuntimeError(f"Invalid frozen rescue audio: {output} {info}")
        frozen_rows.append(
            {
                **row,
                "final_id": f"{FINAL_PREFIX}{index:04d}",
                "audio": str(output.resolve()),
                "duration": float(info.duration),
                "sha256": wave_hash,
                "provenance": f"human_verified_{BATCH_VERSION}",
            }
        )
    LIST.write_text(
        "/n".join(
            f"{row['audio']}|ireina_human_verified_{BATCH_VERSION}|JA|{row['text']}"
            for row in frozen_rows
        )
        + ("/n" if frozen_rows else ""),
        encoding="utf-8",
    )

    queues = {
        "boundary": [row for row in labels if row["final_class"] == "boundary_cut"],
        "mixed": [row for row in labels if row["final_class"] == "mixed"],
        "quality": [row for row in labels if row["final_class"] == "quality_issue"],
        "non_target_or_overlap": [
            row
            for row in labels
            if row["final_class"] in {"pure_non_target", "overlap"}
        ],
        "uncertain": [row for row in labels if row["final_class"] == "uncertain"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / f"feedback_received_{feedback_path.name}"
    shutil.copy2(feedback_path, archive)
    write_json(OUT / "labels_resolved.json", {"version": f"human_{BATCH_VERSION}_labels", "samples": labels})
    write_json(OUT / "selected_best_passes.json", {"samples": frozen_rows})
    for name, rows in queues.items():
        write_json(OUT / f"{name}_queue.json", {"samples": rows})
    report = {
        "version": f"human_{BATCH_VERSION}_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feedback_source": str(feedback_path),
        "feedback_archive": str(archive),
        "review": {"samples": len(labels), "classes": dict(Counter(row["final_class"] for row in labels))},
        "frozen": {"count": len(frozen_rows), "duration_sec": sum(row["duration"] for row in frozen_rows), "list": str(LIST.resolve())},
        "unique_origin_passes": len(selected),
        "queues": {name: len(rows) for name, rows in queues.items()},
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
