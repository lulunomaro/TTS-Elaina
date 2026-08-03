# -*- coding: utf-8 -*-
"""Inspect alternate continuity thresholds from an existing speaker-aware report."""
from __future__ import annotations

import json
from pathlib import Path


REPORT = Path(r"radio_drama_work/new_radio_speaker_aware_v1/speaker_aware_report.json")
CONFIGS = [
    ("strict", .79, .64, .56, .08),
    ("balanced", .76, .62, .52, .05),
    ("recall", .74, .60, .50, .03),
    ("wide", .72, .58, .48, .02),
]


def runs(rows: list[dict], config: tuple) -> list[dict]:
    name, center_min, median_min, p25_min, margin_min = config
    active = [
        row["center"] >= center_min
        and row["median"] >= median_min
        and row["p25"] >= p25_min
        and row.get("target_negative_margin", 1.0) >= margin_min
        for row in rows
    ]
    output = []
    start = None
    for index, yes in enumerate(active + [False]):
        if yes and start is None:
            start = index
        elif not yes and start is not None:
            selected = rows[start:index]
            output.append({
                "start": selected[0]["start"],
                "end": selected[-1]["end"],
                "duration": selected[-1]["end"] - selected[0]["start"],
                "windows": len(selected),
                "min_center": min(row["center"] for row in selected),
                "min_median": min(row["median"] for row in selected),
                "min_p25": min(row["p25"] for row in selected),
                "min_margin": min(row.get("target_negative_margin", 1.0) for row in selected),
            })
            start = None
    return output


def main() -> None:
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    for config in CONFIGS:
        print("CONFIG", config[0])
        total = 0
        stable = 0
        seconds = 0.0
        for item in data["files"]:
            candidates = runs(item["windows"], config)
            stable_candidates = [row for row in candidates if row["windows"] >= 2 and row["duration"] >= 1.5]
            total += len(candidates)
            stable += len(stable_candidates)
            seconds += sum(row["duration"] for row in stable_candidates)
            if stable_candidates:
                print(Path(item["file"]).name, "stable", stable_candidates[:8])
        print("SUMMARY", "runs", total, "stable", stable, "seconds", round(seconds, 1))


if __name__ == "__main__":
    main()
