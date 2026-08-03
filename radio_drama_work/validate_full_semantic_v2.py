# -*- coding: utf-8 -*-
"""Validate the v2 semantic mining outputs before GPU quality scoring."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "full_track_manifest_v1.json"
MINING = ROOT / "full_semantic_mining_v2"
AUDIT = MINING / "audit_manifest.json"
MASKS = MINING / "hard_exclusion_masks.json"
REPORT = MINING / "report.json"
MACHINE_LIST = ROOT / "full_semantic_machine_strict_review_v2.list"
HUMAN_LIST = ROOT / "human_confirmed_broadcast_v2_train_1p5_6s.list"
VALIDATION = MINING / "validation_report.json"

FORBIDDEN = (
    "28055832533-1-192_vocals_0013268480_0013539200_safe_word_000.wav",
    "28055832533-1-192_vocals_0013268480_0013539200_safe_word_001.wav",
)


def overlap_seconds(start: float, end: float, other_start: float, other_end: float) -> float:
    return max(0.0, min(end, other_end) - max(start, other_start))


def interval_overlap_ratio(a: dict, b: dict) -> float:
    overlap = overlap_seconds(
        float(a["start"]), float(a["end"]), float(b["start"]), float(b["end"])
    )
    shorter = min(float(a["end"]) - float(a["start"]), float(b["end"]) - float(b["start"]))
    return overlap / max(shorter, 1e-9)


def read_list(path: Path) -> list[dict]:
    rows = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        parts = raw.split("|", 3)
        if len(parts) != 4:
            rows.append({"line": line_number, "malformed": raw})
            continue
        rows.append(
            {
                "line": line_number,
                "audio": parts[0],
                "speaker": parts[1],
                "language": parts[2],
                "text": parts[3],
            }
        )
    return rows


def inspect_audio(path: Path) -> dict:
    info = sf.info(str(path))
    return {
        "samplerate": int(info.samplerate),
        "channels": int(info.channels),
        "subtype": str(info.subtype),
        "frames": int(info.frames),
        "duration": float(info.duration),
    }


def main() -> None:
    errors: list[dict] = []
    warnings: list[dict] = []

    def fail(code: str, **details: object) -> None:
        errors.append({"code": code, **details})

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected_sources = {str(row["id"]) for row in manifest["tracks"]}
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    masks = json.loads(MASKS.read_text(encoding="utf-8"))
    mining_report = json.loads(REPORT.read_text(encoding="utf-8"))
    rows = audit.get("samples", [])

    if int(audit.get("sample_count", -1)) != len(rows):
        fail("audit_count_mismatch", declared=audit.get("sample_count"), actual=len(rows))
    if mining_report.get("coverage") != {"expected_sources": 12, "completed_sources": 12}:
        fail("source_coverage_incomplete", coverage=mining_report.get("coverage"))
    report_sources = {str(row["source_id"]) for row in mining_report.get("source_stats", [])}
    if report_sources != expected_sources:
        fail(
            "source_set_mismatch",
            missing=sorted(expected_sources - report_sources),
            unexpected=sorted(report_sources - expected_sources),
        )

    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    by_source: dict[str, list[dict]] = defaultdict(list)
    formats = Counter()
    actual_duration = 0.0
    audit_by_audio: dict[str, dict] = {}

    for row in rows:
        blind_id = str(row.get("blind_id") or "")
        if not blind_id or blind_id in seen_ids:
            fail("duplicate_or_empty_blind_id", blind_id=blind_id)
        seen_ids.add(blind_id)

        source_id = str(row.get("source_id") or "")
        if source_id not in expected_sources:
            fail("unexpected_source", blind_id=blind_id, source_id=source_id)
        start = float(row.get("start", -1.0))
        end = float(row.get("end", -1.0))
        if start < 0.0 or end <= start:
            fail("invalid_source_interval", blind_id=blind_id, start=start, end=end)
        duration = end - start
        if not 1.5 <= duration <= 6.0:
            fail("duration_out_of_range", blind_id=blind_id, duration=duration)
        text = str(row.get("text") or "")
        if not text.strip() or "/ufffd" in text or "|" in text:
            fail("invalid_text", blind_id=blind_id, text=text)
        if row.get("hard_mask_hits"):
            fail("declared_hard_mask_hit", blind_id=blind_id, hits=row["hard_mask_hits"])
        for mask in masks.get(source_id, []):
            overlap = overlap_seconds(start, end, float(mask["start"]), float(mask["end"]))
            if overlap > 1e-6:
                fail(
                    "hard_mask_overlap",
                    blind_id=blind_id,
                    overlap_sec=overlap,
                    mask=mask,
                )

        audio = Path(str(row.get("audio") or ""))
        normalized = str(audio.resolve()).casefold()
        if normalized in seen_paths:
            fail("duplicate_audio_path", blind_id=blind_id, audio=str(audio))
        seen_paths.add(normalized)
        if not audio.is_file():
            fail("missing_audio", blind_id=blind_id, audio=str(audio))
            continue
        if any(name.casefold() in normalized for name in FORBIDDEN):
            fail("forbidden_safe_word", blind_id=blind_id, audio=str(audio))
        relative = MINING / str(row.get("audio_rel") or "")
        if relative.resolve() != audio.resolve():
            fail(
                "audio_relative_path_mismatch",
                blind_id=blind_id,
                audio=str(audio),
                resolved_relative=str(relative.resolve()),
            )
        info = inspect_audio(audio)
        formats[(info["samplerate"], info["channels"], info["subtype"])] += 1
        if (info["samplerate"], info["channels"], info["subtype"]) != (32000, 1, "PCM_16"):
            fail("invalid_audio_format", blind_id=blind_id, audio_info=info)
        if abs(info["duration"] - duration) > 0.003:
            fail(
                "audio_duration_mismatch",
                blind_id=blind_id,
                interval_duration=duration,
                audio_duration=info["duration"],
            )
        actual_duration += info["duration"]
        by_source[source_id].append(row)
        audit_by_audio[normalized] = row

    for source_id, source_rows in by_source.items():
        ordered = sorted(source_rows, key=lambda row: (float(row["start"]), float(row["end"])))
        for index, row in enumerate(ordered):
            for other in ordered[index + 1 :]:
                if float(other["start"]) >= float(row["end"]):
                    break
                ratio = interval_overlap_ratio(row, other)
                if ratio >= 0.80:
                    fail(
                        "duplicate_source_interval",
                        source_id=source_id,
                        first=row["blind_id"],
                        second=other["blind_id"],
                        overlap_ratio=ratio,
                    )

    machine_rows = read_list(MACHINE_LIST)
    if len(machine_rows) != len(rows):
        fail("machine_list_count_mismatch", list_count=len(machine_rows), audit_count=len(rows))
    list_paths: set[str] = set()
    for list_row in machine_rows:
        if "malformed" in list_row:
            fail("malformed_machine_list_row", **list_row)
            continue
        path = Path(list_row["audio"])
        normalized = str(path.resolve()).casefold()
        if normalized in list_paths:
            fail("duplicate_machine_list_path", audio=str(path))
        list_paths.add(normalized)
        audit_row = audit_by_audio.get(normalized)
        if audit_row is None:
            fail("machine_list_path_not_in_audit", audio=str(path))
        elif list_row["text"] != audit_row["text"]:
            fail("machine_list_text_mismatch", audio=str(path))

    human_rows = read_list(HUMAN_LIST)
    human_formats = Counter()
    human_duration = 0.0
    human_paths: set[str] = set()
    for row in human_rows:
        if "malformed" in row:
            fail("malformed_human_list_row", **row)
            continue
        path = Path(row["audio"])
        normalized = str(path.resolve()).casefold()
        if normalized in human_paths:
            fail("duplicate_human_list_path", audio=str(path))
        human_paths.add(normalized)
        if not row["text"].strip() or "/ufffd" in row["text"] or "|" in row["text"]:
            fail("invalid_human_text", audio=str(path), text=row["text"])
        if not path.is_file():
            fail("missing_human_audio", audio=str(path))
            continue
        info = inspect_audio(path)
        human_formats[(info["samplerate"], info["channels"], info["subtype"])] += 1
        if not 1.5 <= info["duration"] <= 6.0:
            fail("human_duration_out_of_range", audio=str(path), duration=info["duration"])
        if (info["samplerate"], info["channels"], info["subtype"]) != (32000, 1, "PCM_16"):
            fail("invalid_human_audio_format", audio=str(path), audio_info=info)
        if any(name.casefold() in normalized for name in FORBIDDEN):
            fail("forbidden_human_safe_word", audio=str(path))
        human_duration += info["duration"]

    if len(human_rows) != int(mining_report["human_confirmed"]["train_1p5_6s_count"]):
        fail(
            "human_list_count_mismatch",
            list_count=len(human_rows),
            report_count=mining_report["human_confirmed"]["train_1p5_6s_count"],
        )

    if set(masks) != expected_sources:
        warnings.append(
            {
                "code": "mask_source_set_differs",
                "without_masks": sorted(expected_sources - set(masks)),
                "unexpected": sorted(set(masks) - expected_sources),
            }
        )

    result = {
        "version": "full_semantic_mining_v2_validation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pass": not errors,
        "machine": {
            "count": len(rows),
            "duration_sec": actual_duration,
            "sources_with_candidates": sorted(by_source),
            "format_counts": {str(key): value for key, value in formats.items()},
        },
        "human_confirmed_train_safe": {
            "count": len(human_rows),
            "duration_sec": human_duration,
            "format_counts": {str(key): value for key, value in human_formats.items()},
        },
        "errors": errors,
        "warnings": warnings,
    }
    VALIDATION.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "/n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
