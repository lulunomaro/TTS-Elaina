# -*- coding: utf-8 -*-
"""Audit candidate segments against their recorded word-level evidence."""
from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(r"radio_drama_work/radio_drama_work")
REPORT = ROOT / "new_radio_quality_v1" / "report.json"
TARGETS = {
    "newradio_00001_243358894-1-192_vocals",
    "newradio_00003_243358894-1-192_vocals",
    "newradio_00004_243358894-1-192_vocals",
}


def show_segment(file_report: dict, segment: dict) -> None:
    name = Path(segment["file"]).name
    if not any(name.startswith(prefix) for prefix in TARGETS):
        return
    print("FILE", name)
    print("SEGMENT", {key: segment.get(key) for key in ("start", "end", "duration", "text", "other_ratio", "anchor_best_center", "anchor_min_p25")})
    print("WORDS")
    for word in file_report["word_scores"]:
        if word["end"] > segment["start"] and word["start"] < segment["end"]:
            print({key: word.get(key) for key in ("text", "start", "end", "center", "p25", "negative_max", "margin", "clear_non_target")})


def main() -> None:
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    print("REFERENCE_COUNT", data["reference_count"], "NEGATIVE_COUNT", data["negative_count"])
    print("SUSPECTS")
    for file_report in data["files"]:
        for segment in file_report["segments"]:
            show_segment(file_report, segment)

    print("STRICT_WORD_AUDIT")
    # This is intentionally stricter than the first mining pass.  A final
    # clip must not contain a word that is weak on all target-reference
    # dimensions, even if a nearby 2-second window was strong.
    for file_report in data["files"]:
        weak_segments = 0
        for segment in file_report["segments"]:
            words = [
                word for word in file_report["word_scores"]
                if word["end"] > segment["start"] and word["start"] < segment["end"]
            ]
            strict_bad = [
                word for word in words
                if not (
                    word.get("center", 0.0) >= 0.72
                    and word.get("p25", 0.0) >= 0.50
                    and word.get("margin", 0.0) >= 0.02
                )
            ]
            if strict_bad:
                weak_segments += 1
                print(
                    Path(segment["file"]).name,
                    "bad_words=", len(strict_bad),
                    "of=", len(words),
                    "min_center=", round(min((word.get("center", 0.0) for word in words), default=0.0), 3),
                    "min_p25=", round(min((word.get("p25", 0.0) for word in words), default=0.0), 3),
                )
        print(Path(file_report["source"]).name, "weak_segments=", weak_segments, "total=", len(file_report["segments"]))


if __name__ == "__main__":
    main()
