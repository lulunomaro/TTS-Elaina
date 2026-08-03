from __future__ import annotations

import csv
import hashlib
import json
import shutil
import wave
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


WORK = Path(r"radio_drama_work/radio_drama_work")

BOUNDARY_AUDIT = WORK / "boundary_rescue_review_v1" / "audit_manifest.json"
BOUNDARY_LABELS = WORK / "human_boundary_rescue_v1" / "labels_resolved.json"
EDGE_V2_AUDIT = WORK / "edge_trim_rescue_v2" / "audit_manifest.json"
EDGE_V2_LABELS = WORK / "human_edge_trim_rescue_v2" / "labels_resolved.json"
EDGE_V3_AUDIT = WORK / "edge_trim_rescue_v3" / "audit_manifest.json"
EDGE_V3_CONFIRMATION_DIR = WORK / "human_edge_trim_rescue_v3"

OUTPUT_DIR = WORK / "human_verified_boundary_rescues_v1"
OUTPUT_AUDIO_DIR = OUTPUT_DIR / "audio"
OUTPUT_LIST = WORK / "human_verified_boundary_rescues_v1.list"
OUTPUT_DURATION_SAFE_LIST = WORK / "human_verified_boundary_rescues_v1_train_1p5_6s.list"

SPEAKER_NAME = "ireina_human_verified"
LANGUAGE = "JA"
AUTOMATIC_TRAIN_MIN_SEC = 1.5
AUTOMATIC_TRAIN_MAX_SEC = 6.0


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "/n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_wav(path: Path) -> dict:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        compression = wav.getcomptype()
    duration = frames / sample_rate
    return {
        "channels": channels,
        "sample_width_bytes": sample_width,
        "sample_rate": sample_rate,
        "frames": frames,
        "compression": compression,
        "duration": duration,
    }


def validate_text(text: object, label: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{label}: empty ASR text")
    text = text.strip()
    if "/ufffd" in text:
        raise ValueError(f"{label}: ASR text contains U+FFFD")
    if any(token in text for token in ("|", "/r", "/n")):
        raise ValueError(f"{label}: ASR text contains a list delimiter or newline")
    return text


def labels_by_id(path: Path) -> dict[str, dict]:
    rows = read_json(path)["samples"]
    return {row["blind_id"]: row for row in rows}


def select_labeled_targets(
    audit_path: Path,
    labels_path: Path,
    stage: str,
) -> list[dict]:
    audit_rows = read_json(audit_path)["samples"]
    labels = labels_by_id(labels_path)
    selected = []
    for audit in audit_rows:
        blind_id = audit["blind_id"]
        label = labels.get(blind_id)
        if label is None:
            raise ValueError(f"{stage}:{blind_id}: missing human label")
        if label["final_class"] != "target":
            continue
        if label.get("hard_exclusion"):
            raise ValueError(f"{stage}:{blind_id}: target is also hard-excluded")
        row = dict(audit)
        row["selection_stage"] = stage
        row["human_label_source"] = str(labels_path)
        row["root_origin_human_key"] = audit.get("root_origin_human_key") or audit.get(
            "origin_human_key"
        )
        row["lineage"] = [blind_id]
        selected.append(row)
    return selected


def create_v3_confirmation() -> list[dict]:
    audit = read_json(EDGE_V3_AUDIT)
    now = datetime.now(timezone.utc).isoformat()
    resolved = []
    for row in audit["samples"]:
        resolved.append(
            {
                "blind_id": row["blind_id"],
                "kind": row.get("kind"),
                "source_id": row["source_id"],
                "start": row["start"],
                "end": row["end"],
                "duration": row["output_duration"],
                "audio": row["audio"],
                "human_is_error": False,
                "human_error_type": "",
                "human_note": "用户在本任务中明确回复：这个通过了",
                "final_class": "target",
                "speaker_training_role": "positive",
                "hard_exclusion": False,
                "confirmation_method": "explicit_user_message_in_codex_task",
            }
        )

    confirmation = {
        "version": "human_edge_trim_rescue_v3_manual_confirmation_v1",
        "created_at": now,
        "source_audit_manifest": str(EDGE_V3_AUDIT),
        "user_message": "这个通过了",
        "interpretation": "The review page contained exactly E001-E003; all three were accepted.",
        "sample_count": len(resolved),
        "blind_ids": [row["blind_id"] for row in resolved],
        "status": "human-confirmed target voice and acceptable boundaries",
    }
    write_json(EDGE_V3_CONFIRMATION_DIR / "manual_confirmation.json", confirmation)
    write_json(
        EDGE_V3_CONFIRMATION_DIR / "labels_resolved.json",
        {"version": "human_calibration_labels_resolved_v1", "samples": resolved},
    )
    return resolved


def select_v3_targets() -> list[dict]:
    create_v3_confirmation()
    audit_rows = read_json(EDGE_V3_AUDIT)["samples"]
    labels = labels_by_id(EDGE_V3_CONFIRMATION_DIR / "labels_resolved.json")
    selected = []
    for audit in audit_rows:
        blind_id = audit["blind_id"]
        label = labels[blind_id]
        if label["final_class"] != "target" or label.get("hard_exclusion"):
            continue
        row = dict(audit)
        row["selection_stage"] = "edge_trim_rescue_v3"
        row["human_label_source"] = str(
            EDGE_V3_CONFIRMATION_DIR / "labels_resolved.json"
        )
        row["root_origin_human_key"] = audit["root_origin_human_key"]
        row["lineage"] = [audit.get("parent_blind_id"), blind_id]
        selected.append(row)
    return selected


def interval_overlap_ratio(left: dict, right: dict) -> float:
    if left["source_id"] != right["source_id"]:
        return 0.0
    overlap = max(0.0, min(left["end"], right["end"]) - max(left["start"], right["start"]))
    shorter = min(left["end"] - left["start"], right["end"] - right["start"])
    return overlap / shorter if shorter > 0 else 0.0


def main() -> None:
    now = datetime.now(timezone.utc).isoformat()

    first_pass = select_labeled_targets(
        BOUNDARY_AUDIT, BOUNDARY_LABELS, "boundary_rescue_review_v1"
    )
    second_pass = select_labeled_targets(
        EDGE_V2_AUDIT, EDGE_V2_LABELS, "edge_trim_rescue_v2"
    )
    third_pass = select_v3_targets()

    if len(first_pass) != 7 or len(second_pass) != 2 or len(third_pass) != 3:
        raise ValueError(
            "Unexpected stage counts: "
            f"{len(first_pass)} + {len(second_pass)} + {len(third_pass)}"
        )

    selected = first_pass + second_pass + third_pass
    roots = [row["root_origin_human_key"] for row in selected]
    if None in roots or len(roots) != len(set(roots)):
        duplicates = [key for key, count in Counter(roots).items() if count > 1]
        raise ValueError(f"Root-origin dedupe failed: {duplicates}")
    if len(selected) != 12:
        raise ValueError(f"Expected 12 final rescues, got {len(selected)}")

    selected.sort(key=lambda row: (row["source_id"], row["start"], row["end"]))
    for index, left in enumerate(selected):
        for right in selected[index + 1 :]:
            ratio = interval_overlap_ratio(left, right)
            if ratio >= 0.80:
                raise ValueError(
                    "Source interval dedupe failed: "
                    f"{left['blind_id']} vs {right['blind_id']} ({ratio:.3f})"
                )

    OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    final_rows = []
    for index, row in enumerate(selected, start=1):
        final_id = f"HBR{index:03d}"
        source_clip = Path(row["audio"])
        original_vocals = Path(row["source"])
        if not source_clip.is_file():
            raise FileNotFoundError(source_clip)
        if not original_vocals.is_file():
            raise FileNotFoundError(original_vocals)

        identity_blob = " ".join(
            [str(source_clip), str(original_vocals), row["source_id"], row.get("text", "")]
        ).lower()
        if "safe_word" in identity_blob:
            raise ValueError(f"{final_id}: forbidden safe_word material")
        if "vocals" not in original_vocals.name.lower():
            raise ValueError(f"{final_id}: source is not a BS-Roformer vocals track")

        text = validate_text(row.get("text"), final_id)
        wav_info = inspect_wav(source_clip)
        if (
            wav_info["channels"] != 1
            or wav_info["sample_rate"] != 32000
            or wav_info["sample_width_bytes"] != 2
            or wav_info["compression"] != "NONE"
        ):
            raise ValueError(f"{final_id}: audio is not 32 kHz mono PCM16: {wav_info}")
        if abs(wav_info["duration"] - float(row["output_duration"])) > 0.03:
            raise ValueError(f"{final_id}: manifest/audio duration mismatch")

        destination = OUTPUT_AUDIO_DIR / f"{final_id}_{row['source_id']}.wav"
        shutil.copyfile(source_clip, destination)
        source_hash = sha256(source_clip)
        destination_hash = sha256(destination)
        if source_hash != destination_hash:
            raise ValueError(f"{final_id}: copied WAV hash mismatch")

        within_duration_policy = (
            AUTOMATIC_TRAIN_MIN_SEC
            <= wav_info["duration"]
            <= AUTOMATIC_TRAIN_MAX_SEC
        )
        final_rows.append(
            {
                "final_id": final_id,
                "root_origin_human_key": row["root_origin_human_key"],
                "review_blind_id": row["blind_id"],
                "selection_stage": row["selection_stage"],
                "lineage": [item for item in row.get("lineage", []) if item],
                "source_id": row["source_id"],
                "source_vocals": str(original_vocals),
                "source_start": row["start"],
                "source_end": row["end"],
                "duration": wav_info["duration"],
                "text": text,
                "audio": str(destination),
                "review_audio": str(source_clip),
                "sha256": destination_hash,
                "format": {
                    "sample_rate": wav_info["sample_rate"],
                    "channels": wav_info["channels"],
                    "sample_width_bytes": wav_info["sample_width_bytes"],
                    "compression": wav_info["compression"],
                },
                "human_confirmation": "target voice, clean enough, and boundary accepted",
                "human_label_source": row["human_label_source"],
                "within_automatic_1p5_6s_policy": within_duration_policy,
            }
        )

    list_lines = [
        f"{row['audio']}|{SPEAKER_NAME}|{LANGUAGE}|{row['text']}" for row in final_rows
    ]
    OUTPUT_LIST.write_text("/n".join(list_lines) + "/n", encoding="utf-8")

    duration_safe_rows = [
        row for row in final_rows if row["within_automatic_1p5_6s_policy"]
    ]
    duration_safe_lines = [
        f"{row['audio']}|{SPEAKER_NAME}|{LANGUAGE}|{row['text']}"
        for row in duration_safe_rows
    ]
    OUTPUT_DURATION_SAFE_LIST.write_text(
        "/n".join(duration_safe_lines) + ("/n" if duration_safe_lines else ""),
        encoding="utf-8",
    )

    tsv_path = OUTPUT_DIR / "human_verified_boundary_rescues_v1.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="/t", lineterminator="/n")
        writer.writerow(
            [
                "final_id",
                "audio",
                "speaker",
                "language",
                "text",
                "source_id",
                "source_start",
                "source_end",
                "duration",
                "root_origin_human_key",
                "selection_stage",
                "within_automatic_1p5_6s_policy",
            ]
        )
        for row in final_rows:
            writer.writerow(
                [
                    row["final_id"],
                    row["audio"],
                    SPEAKER_NAME,
                    LANGUAGE,
                    row["text"],
                    row["source_id"],
                    f"{row['source_start']:.6f}",
                    f"{row['source_end']:.6f}",
                    f"{row['duration']:.6f}",
                    row["root_origin_human_key"],
                    row["selection_stage"],
                    row["within_automatic_1p5_6s_policy"],
                ]
            )

    source_stats = defaultdict(lambda: {"segments": 0, "duration": 0.0})
    for row in final_rows:
        source_stats[row["source_id"]]["segments"] += 1
        source_stats[row["source_id"]]["duration"] += row["duration"]
    source_stats = {
        source_id: {
            "segments": values["segments"],
            "duration": round(values["duration"], 6),
        }
        for source_id, values in sorted(source_stats.items())
    }

    audit_manifest = {
        "version": "human_verified_boundary_rescues_v1",
        "created_at": now,
        "status": "human-verified rescue pool; not automatically added to training",
        "policy": {
            "source_audio": "original BS-Roformer vocals only",
            "audio_reconstruction": "none (no ClearVoice or VASR output)",
            "dedupe": "exactly one final accepted version per root human-review origin",
            "speaker": SPEAKER_NAME,
            "language": LANGUAGE,
            "automatic_train_duration_sec": [
                AUTOMATIC_TRAIN_MIN_SEC,
                AUTOMATIC_TRAIN_MAX_SEC,
            ],
            "duration_note": (
                "The complete 12-item pool preserves every human-approved rescue. "
                "A separate list enforces the automatic 1.5-6.0 second policy."
            ),
        },
        "sample_count": len(final_rows),
        "samples": final_rows,
    }
    write_json(OUTPUT_DIR / "audit_manifest.json", audit_manifest)

    report = {
        "version": "human_verified_boundary_rescues_v1_report",
        "created_at": now,
        "result": "pass",
        "rescue_origins": 12,
        "human_verified_segments": len(final_rows),
        "total_duration_sec": round(sum(row["duration"] for row in final_rows), 6),
        "total_duration_min": round(
            sum(row["duration"] for row in final_rows) / 60.0, 6
        ),
        "stage_counts": dict(Counter(row["selection_stage"] for row in final_rows)),
        "source_stats": source_stats,
        "automatic_1p5_6s_count": len(duration_safe_rows),
        "automatic_1p5_6s_duration_sec": round(
            sum(row["duration"] for row in duration_safe_rows), 6
        ),
        "over_6s_human_verified": [
            {
                "final_id": row["final_id"],
                "duration": round(row["duration"], 6),
                "source_id": row["source_id"],
            }
            for row in final_rows
            if row["duration"] > AUTOMATIC_TRAIN_MAX_SEC
        ],
        "validation": {
            "unique_root_origins": len(set(roots)) == 12,
            "unique_output_paths": len({row["audio"] for row in final_rows}) == 12,
            "all_files_exist": all(Path(row["audio"]).is_file() for row in final_rows),
            "all_32khz_mono_pcm16": True,
            "all_text_nonempty_utf8_without_replacement": True,
            "safe_word_count": 0,
            "source_interval_overlap_ge_80pct_count": 0,
            "copied_hash_mismatch_count": 0,
        },
        "outputs": {
            "complete_human_verified_list": str(OUTPUT_LIST),
            "automatic_duration_safe_list": str(OUTPUT_DURATION_SAFE_LIST),
            "audit_manifest": str(OUTPUT_DIR / "audit_manifest.json"),
            "tsv": str(tsv_path),
            "audio_dir": str(OUTPUT_AUDIO_DIR),
        },
        "training_started": False,
        "existing_training_lists_modified": False,
        "score_90_claimed": False,
    }
    write_json(OUTPUT_DIR / "report.json", report)

    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
