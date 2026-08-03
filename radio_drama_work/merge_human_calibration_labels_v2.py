# -*- coding: utf-8 -*-
"""Merge the two blind human-review rounds without changing label semantics."""
from __future__ import annotations

import collections
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUTS = [
    ("round1", ROOT / "human_calibration_v1" / "labels_resolved.json"),
    ("round2", ROOT / "human_calibration_round2_v1" / "labels_resolved.json"),
]
OUT = ROOT / "human_calibration_combined_v2"


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
            merged.append(item)
            counts[item["final_class"]] += 1
        by_round[round_name] = dict(sorted(counts.items()))
    ids = [str(row["blind_id"]) for row in merged]
    paths = [str(Path(row["audio"]).resolve()).lower() for row in merged]
    if len(ids) != 120 or len(set(ids)) != 120 or len(set(paths)) != 120:
        raise RuntimeError("Expected 120 unique samples after merging two rounds")
    missing = [row["audio"] for row in merged if not Path(row["audio"]).is_file()]
    if missing:
        raise FileNotFoundError(missing)
    counts = collections.Counter(row["final_class"] for row in merged)
    by_source: dict[str, collections.Counter] = {}
    for row in merged:
        by_source.setdefault(str(row["source_id"]), collections.Counter())[row["final_class"]] += 1
    hard_intervals = [
        {
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
        if row["hard_exclusion"]
    ]
    write_json(OUT / "labels_resolved.json", {"version": "human_calibration_combined_v2", "samples": merged})
    write_json(OUT / "hard_exclusion_intervals.json", {"version": "human_hard_exclusions_combined_v2", "intervals": hard_intervals})
    report = {
        "version": "human_calibration_combined_v2_report",
        "total": len(merged),
        "class_counts": dict(sorted(counts.items())),
        "by_round": by_round,
        "by_source": {source: dict(sorted(values.items())) for source, values in sorted(by_source.items())},
        "speaker_training": {
            "target_probe": sum(row["kind"] == "probe" and row["final_class"] == "target" for row in merged),
            "pure_non_target_probe": sum(
                row["kind"] == "probe" and row["final_class"] == "pure_non_target" for row in merged
            ),
            "validation_or_hard_exclusion": sum(row["final_class"] not in {"target", "pure_non_target"} for row in merged),
        },
        "hard_exclusion_intervals": len(hard_intervals),
    }
    write_json(OUT / "merge_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
