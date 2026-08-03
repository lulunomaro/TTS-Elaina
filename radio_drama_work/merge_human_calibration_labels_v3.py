# -*- coding: utf-8 -*-
"""Merge all three completed blind human-review rounds."""
from __future__ import annotations

import collections
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUTS = [
    ("round1", ROOT / "human_calibration_v1" / "labels_resolved.json"),
    ("round2", ROOT / "human_calibration_round2_v1" / "labels_resolved.json"),
    ("round3", ROOT / "human_calibration_round3_v1" / "labels_resolved.json"),
]
OUT = ROOT / "human_calibration_combined_v3"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8")


def main() -> None:
    merged: list[dict] = []
    by_round: dict[str, dict[str, int]] = {}
    for round_name, path in INPUTS:
        rows = json.loads(path.read_text(encoding="utf-8-sig"))["samples"]
        counts = collections.Counter()
        for row in rows:
            item = dict(row)
            item["human_round"] = round_name
            item["human_key"] = f"{round_name}:{item['blind_id']}"
            item["speaker_training_role"] = (
                "positive"
                if item["final_class"] == "target" or (round_name == "round3" and item["final_class"] == "boundary_cut")
                else "negative"
                if item["final_class"] == "pure_non_target"
                else "validation_only"
            )
            merged.append(item)
            counts[item["final_class"]] += 1
        by_round[round_name] = dict(sorted(counts.items()))

    keys = [str(row["human_key"]) for row in merged]
    paths = [str(Path(row["audio"]).resolve()).lower() for row in merged]
    if len(merged) != 141 or len(set(keys)) != len(keys) or len(set(paths)) != len(paths):
        raise RuntimeError("Expected 141 unique samples after merging three rounds")
    missing = [row["audio"] for row in merged if not Path(row["audio"]).is_file()]
    if missing:
        raise FileNotFoundError(missing)

    counts = collections.Counter(row["final_class"] for row in merged)
    by_source: dict[str, collections.Counter] = {}
    for row in merged:
        by_source.setdefault(str(row["source_id"]), collections.Counter())[row["final_class"]] += 1
    hard_intervals = [
        {
            "human_key": row["human_key"],
            "blind_id": row["blind_id"],
            "human_round": row["human_round"],
            "source_id": row["source_id"],
            "start": row["start"],
            "end": row["end"],
            "reason": row["final_class"],
            "human_error_type": row["human_error_type"],
            "human_note": row["human_note"],
        }
        for row in merged
        if row["hard_exclusion"] and row.get("source_id") is not None
    ]
    write_json(OUT / "labels_resolved.json", {"version": "human_calibration_combined_v3", "samples": merged})
    write_json(
        OUT / "hard_exclusion_intervals.json",
        {"version": "human_hard_exclusions_combined_v3", "intervals": hard_intervals},
    )
    report = {
        "version": "human_calibration_combined_v3_report",
        "total": len(merged),
        "class_counts": dict(sorted(counts.items())),
        "by_round": by_round,
        "by_source": {source: dict(sorted(values.items())) for source, values in sorted(by_source.items())},
        "speaker_training": {
            "positive": sum(row.get("speaker_training_role") == "positive" for row in merged),
            "pure_non_target_negative": sum(row.get("speaker_training_role") == "negative" for row in merged),
            "validation_only": sum(row.get("speaker_training_role") == "validation_only" for row in merged),
            "explicit_round3_boundary_positive_but_training_audio_excluded": sum(
                row["human_round"] == "round3" and row["final_class"] == "boundary_cut" for row in merged
            ),
        },
        "hard_exclusion_intervals": len(hard_intervals),
    }
    write_json(OUT / "merge_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
