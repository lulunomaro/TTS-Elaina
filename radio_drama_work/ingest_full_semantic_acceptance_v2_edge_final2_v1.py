# -*- coding: utf-8 -*-
"""Freeze passed final B013/B024 edge rescues, deduplicated by origin."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf


ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "full_semantic_acceptance_v2_edge_final2_v1" / "audit_manifest.json"
OUT = ROOT / "human_verified_full_semantic_v2_edge_final2_v1"
LIST = ROOT / "human_verified_full_semantic_v2_edge_final2_v1.list"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback", type=Path, required=True)
    args = parser.parse_args()
    feedback_path = args.feedback.resolve()
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if not bool(feedback.get("complete")):
        raise RuntimeError("Final edge feedback is incomplete")

    audit_by_id = {str(row["blind_id"]): row for row in audit["samples"]}
    feedback_by_id = {str(row["blind_id"]): row for row in feedback["samples"]}
    if set(audit_by_id) != set(feedback_by_id):
        raise RuntimeError("Feedback IDs do not match final edge audit")

    passed_by_origin: dict[str, list[dict]] = defaultdict(list)
    rejected = []
    for blind_id, source in audit_by_id.items():
        human = feedback_by_id[blind_id]
        if not bool(human.get("reviewed")):
            raise RuntimeError(f"{blind_id}: not reviewed")
        resolved = {
            **source,
            "human_is_error": bool(human.get("is_error")),
            "human_error_type": str(human.get("error_type") or ""),
            "human_note": str(human.get("note") or ""),
            "play_count": int(human.get("play_count") or 0),
            "max_listened_sec": float(human.get("max_listened_sec") or 0.0),
        }
        if resolved["human_is_error"]:
            rejected.append(resolved)
        else:
            passed_by_origin[str(source["origin_blind_id"])].append(resolved)

    # Multiple millisecond variants can pass for the same utterance. Keep the
    # one that begins closest to the ASR word boundary and preserves most audio.
    selected = []
    passed_alternates = []
    for origin, rows in sorted(passed_by_origin.items()):
        rows.sort(
            key=lambda row: (
                float(row.get("boundary", {}).get("start_inside_first_word_sec", 9.0)),
                -float(row["output_duration"]),
                str(row["blind_id"]),
            )
        )
        selected.append(rows[0])
        passed_alternates.extend(rows[1:])

    OUT.mkdir(parents=True, exist_ok=True)
    audio_dir = OUT / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    frozen = []
    seen_hashes = set()
    list_lines = []
    for index, row in enumerate(selected, 1):
        source_audio = Path(row["audio"]).resolve()
        output = audio_dir / f"HFV2E{index:03d}_{row['source_id']}.wav"
        shutil.copy2(source_audio, output)
        info = sf.info(str(output))
        if (
            info.samplerate != 32000
            or info.channels != 1
            or info.subtype != "PCM_16"
            or not 1.5 <= info.duration <= 6.0
        ):
            raise RuntimeError(f"Invalid frozen edge audio: {output} {info}")
        digest = sha256(output)
        if digest in seen_hashes:
            raise RuntimeError(f"Duplicate frozen waveform: {output}")
        seen_hashes.add(digest)
        text = str(row.get("text") or "").strip()
        if not text or "/ufffd" in text or "|" in text:
            raise RuntimeError(f"Invalid frozen text: {text!r}")
        item = {
            **row,
            "frozen_audio": str(output.resolve()),
            "sha256": digest,
            "duration": float(info.duration),
        }
        frozen.append(item)
        list_lines.append(
            f"{output.resolve()}|ireina_human_verified_v2_edge_final2|JA|{text}"
        )

    LIST.write_text("/n".join(list_lines) + "/n", encoding="utf-8")
    archive = OUT / f"feedback_received_{feedback_path.name}"
    shutil.copy2(feedback_path, archive)
    write_json(
        OUT / "audit_manifest.json",
        {
            "version": "human_verified_full_semantic_v2_edge_final2_v1_audit",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "feedback_source": str(feedback_path),
            "feedback_archive": str(archive),
            "feedback_sha256": sha256(feedback_path),
            "frozen": frozen,
            "passed_alternates_not_frozen": passed_alternates,
            "rejected": rejected,
        },
    )
    report = {
        "version": "human_verified_full_semantic_v2_edge_final2_v1_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reviewed_count": len(feedback_by_id),
        "human_pass_count": sum(len(rows) for rows in passed_by_origin.values()),
        "frozen_count": len(frozen),
        "frozen_duration_sec": sum(float(row["duration"]) for row in frozen),
        "passed_duplicate_alternates_not_frozen": len(passed_alternates),
        "rejected_count": len(rejected),
        "list": str(LIST.resolve()),
        "policy": "one complete human-passed variant per original utterance",
        "training_started": False,
        "existing_training_lists_modified": False,
        "score_90_claimed": False,
    }
    write_json(OUT / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
