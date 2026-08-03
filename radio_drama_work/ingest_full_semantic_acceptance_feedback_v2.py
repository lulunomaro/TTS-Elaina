# -*- coding: utf-8 -*-
"""Ingest the v2 blind acceptance feedback and freeze exact heard passes.

Mixed clips are queued for source-time localization instead of masking the
whole candidate. Boundary failures remain positive identity evidence, while
quality failures never become speaker negatives.
"""
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
    r"C:/Users/lulu/Downloads/full_semantic_acceptance_feedback_v2_2026-08-02T10-38-49-359Z.json"
)
PRIVATE = ROOT / "full_semantic_acceptance_v2" / "private_manifest.json"
OUT = ROOT / "human_full_semantic_acceptance_v2"
FROZEN = ROOT / "human_verified_full_semantic_v2_direct"
FROZEN_LIST = ROOT / "human_verified_full_semantic_v2_direct.list"

ROUND_AUDITS = {
    "round1": ROOT / "human_calibration_v1" / "audit_manifest.json",
    "round2": ROOT / "human_calibration_round2_v1" / "audit_manifest.json",
    "round3": ROOT / "cluster_acceptance_round3_v1" / "audit_manifest.json",
}

BOUNDARY_TYPES = {
    "切头/切尾",
    "开头被切",
    "结尾被切",
    "开头和结尾都被切",
}
QUALITY_NOTE_TOKENS = (
    "脚步",
    "走动",
    "翻书",
    "合书",
    "杂音",
    "雜音",
    "音质",
    "音質",
    "环境音",
    "環境音",
    "噪",
    "点击",
    "道具",
)
NON_TARGET_NOTE_TOKENS = (
    "非伊蕾娜",
    "不是伊蕾娜",
    "别人",
    "其他角色",
    "其他人",
)


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
    if not bool(human.get("is_error")):
        return "target"
    error_type = str(human.get("error_type") or "")
    note = str(human.get("note") or "")
    if error_type in BOUNDARY_TYPES:
        return "boundary_cut"
    if error_type in {"杂音/音质问题", "环境音/道具声"}:
        return "quality_issue"
    if error_type == "重叠说话":
        return "overlap"
    if error_type == "混入/无缝接话":
        return "mixed"
    if error_type == "非伊蕾娜":
        return "pure_non_target"
    if any(token in note for token in QUALITY_NOTE_TOKENS):
        return "quality_issue"
    if any(token in note for token in NON_TARGET_NOTE_TOKENS):
        return "mixed"
    if any(token in note for token in ("开头", "结尾", "切头", "切尾")):
        return "boundary_cut"
    return "uncertain"


def control_lookup() -> dict[str, dict]:
    result: dict[str, dict] = {}
    for round_name, path in ROUND_AUDITS.items():
        document = json.loads(path.read_text(encoding="utf-8"))
        for row in document["samples"]:
            result[f"{round_name}:{row['blind_id']}"] = row
    return result


def resolve_origin(private_row: dict, controls: dict[str, dict]) -> dict:
    payload = private_row["payload"]
    if private_row["role"] == "machine_candidate_v2":
        return {
            "source_id": str(private_row["source_id"]),
            "source": str(payload["source"]),
            "start": float(payload["start"]),
            "end": float(payload["end"]),
            "text": str(payload.get("text") or ""),
            "word_start_index": int(payload["word_start_index"]),
            "word_end_index": int(payload["word_end_index"]),
            "machine_id": str(payload["blind_id"]),
            "review_audio": str(private_row["source_audio"]),
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
        "review_audio": str(private_row["source_audio"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    args = parser.parse_args()

    feedback_path = args.feedback.resolve()
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    private = json.loads(PRIVATE.read_text(encoding="utf-8"))
    if not bool(feedback.get("complete")):
        raise RuntimeError("Feedback is incomplete")
    if int(feedback["summary"]["reviewed"]) != len(private["samples"]):
        raise RuntimeError("Feedback/private-manifest count mismatch")

    private_by_id = {str(row["blind_id"]): row for row in private["samples"]}
    if set(private_by_id) != {str(row["blind_id"]) for row in feedback["samples"]}:
        raise RuntimeError("Feedback/private-manifest IDs differ")
    controls = control_lookup()
    resolved = []
    training_exclusions = []
    hard_speaker_exclusions = []
    localization_queue = []
    quality_exclusions = []

    for human in feedback["samples"]:
        blind_id = str(human["blind_id"])
        private_row = private_by_id[blind_id]
        origin = resolve_origin(private_row, controls)
        final_class = classify(human)
        note = str(human.get("note") or "")
        row = {
            "human_key": f"full_semantic_acceptance_v2:{blind_id}",
            "blind_id": blind_id,
            "role": str(private_row["role"]),
            **origin,
            "duration": float(origin["end"] - origin["start"]),
            "reviewed": bool(human["reviewed"]),
            "is_error": bool(human["is_error"]),
            "error_type": str(human.get("error_type") or ""),
            "human_note": note,
            "final_class": final_class,
            "play_count": int(human.get("play_count") or 0),
            "max_listened_sec": float(human.get("max_listened_sec") or 0.0),
            "speaker_learning_policy": (
                "human_target"
                if final_class == "target"
                else "positive_identity_recut_required"
                if final_class == "boundary_cut"
                else "speaker_negative"
                if final_class == "pure_non_target"
                else "localize_contamination_before_masking"
                if final_class in {"mixed", "overlap"}
                else "quality_only_not_speaker_negative"
                if final_class == "quality_issue"
                else "validation_only"
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
        if final_class in {"pure_non_target", "overlap"}:
            hard_speaker_exclusions.append(
                {
                    "source_id": row["source_id"],
                    "start": max(0.0, row["start"] - 0.10),
                    "end": row["end"] + 0.10,
                    "reason": f"human_confirmed_{final_class}:{blind_id}",
                    "note": note,
                }
            )
        if final_class == "mixed":
            localization_queue.append(
                {
                    "blind_id": blind_id,
                    "source_id": row["source_id"],
                    "source": row["source"],
                    "start": row["start"],
                    "end": row["end"],
                    "word_start_index": row["word_start_index"],
                    "word_end_index": row["word_end_index"],
                    "note": note,
                    "policy": "do not hard-mask the whole clip; localize the contaminated sub-interval first",
                }
            )
        if final_class == "quality_issue":
            quality_exclusions.append(
                {
                    "blind_id": blind_id,
                    "source_id": row["source_id"],
                    "start": row["start"],
                    "end": row["end"],
                    "reason": f"human_quality_issue:{blind_id}",
                    "note": note,
                }
            )

    FROZEN_AUDIO = FROZEN / "audio"
    FROZEN_AUDIO.mkdir(parents=True, exist_ok=True)
    direct_passes = [
        row
        for row in resolved
        if row["role"] == "machine_candidate_v2" and row["final_class"] == "target"
    ]
    frozen = []
    for index, row in enumerate(
        sorted(direct_passes, key=lambda item: (item["source_id"], item["start"])), 1
    ):
        source_audio = Path(row["review_audio"])
        output = FROZEN_AUDIO / f"HFV2D{index:03d}_{row['source_id']}.wav"
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
            raise RuntimeError(f"Invalid frozen audio: {output} {info}")
        frozen.append(
            {
                **row,
                "final_id": f"HFV2D{index:03d}",
                "source_review_audio": str(source_audio),
                "audio": str(output.resolve()),
                "duration": float(info.duration),
                "sha256": sha256(output),
                "provenance": "human_verified_full_semantic_acceptance_v2_direct",
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / f"feedback_received_{feedback_path.name}"
    shutil.copy2(feedback_path, archive)
    write_json(
        OUT / "labels_resolved.json",
        {
            "version": "human_full_semantic_acceptance_v2_labels",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "feedback_source": str(feedback_path),
            "samples": resolved,
        },
    )
    write_json(
        OUT / "training_exclusion_intervals.json",
        {
            "version": "human_full_semantic_acceptance_v2_training_exclusions",
            "policy": "failed heard waveforms stay excluded until an independently re-cut version passes",
            "intervals": training_exclusions,
        },
    )
    write_json(
        OUT / "hard_exclusion_intervals.json",
        {
            "version": "human_full_semantic_acceptance_v2_hard_exclusions",
            "policy": "only whole-clip pure non-target or overlap; mixed clips require localization first",
            "intervals": hard_speaker_exclusions,
        },
    )
    write_json(
        OUT / "contamination_localization_queue.json",
        {
            "version": "human_full_semantic_acceptance_v2_localization_queue",
            "samples": localization_queue,
        },
    )
    write_json(
        OUT / "quality_exclusion_intervals.json",
        {
            "version": "human_full_semantic_acceptance_v2_quality_exclusions",
            "policy": "quality/environment events are never speaker negatives",
            "intervals": quality_exclusions,
        },
    )
    write_json(
        FROZEN / "audit_manifest.json",
        {
            "version": "human_verified_full_semantic_v2_direct_audit",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "selection": "exact machine waveforms heard and passed in the v2 blind acceptance",
            "samples": frozen,
        },
    )
    FROZEN_LIST.write_text(
        "/n".join(
            f"{row['audio']}|ireina_human_verified_full_semantic_v2_direct|JA|{row['text']}"
            for row in frozen
        )
        + ("/n" if frozen else ""),
        encoding="utf-8",
    )

    classes = Counter(row["final_class"] for row in resolved)
    roles: dict[str, Counter] = defaultdict(Counter)
    for row in resolved:
        roles[row["role"]][row["final_class"]] += 1
    report = {
        "version": "human_full_semantic_acceptance_v2_report",
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
            "machine_direct_passes": len(frozen),
            "machine_direct_pass_duration_sec": sum(row["duration"] for row in frozen),
            "gold_controls_passed": sum(
                row["role"] == "known_gold_control" and row["final_class"] == "target"
                for row in resolved
            ),
            "training_exclusions": len(training_exclusions),
            "hard_speaker_exclusions": len(hard_speaker_exclusions),
            "mixed_localization_queue": len(localization_queue),
            "quality_exclusions": len(quality_exclusions),
        },
        "decision": {
            "bulk_machine_promotion_allowed": False,
            "reason": "13/20 machine candidates passed; three machine clips contained non-target speech",
            "direct_human_passes_frozen": True,
            "boundary_and_mixed_edge_rescue_next": True,
        },
        "outputs": {
            "direct_pass_list": str(FROZEN_LIST),
            "direct_pass_audit": str(FROZEN / "audit_manifest.json"),
            "localization_queue": str(OUT / "contamination_localization_queue.json"),
        },
        "training_started": False,
        "existing_training_lists_modified": False,
        "score_90_claimed": False,
    }
    write_json(OUT / "report.json", report)
    write_json(FROZEN / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
