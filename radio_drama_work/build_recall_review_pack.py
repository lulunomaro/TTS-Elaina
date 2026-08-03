"""Export a ranked, non-overlapping audition pack from speaker-aware windows.

This deliberately creates review audio only.  It never writes a training list.
The ranking is for recall, not proof of target-speaker identity.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torchaudio


REPORT = Path(sys.argv[1])
INPUT_DIR = Path(sys.argv[2])
OUTPUT_DIR = Path(sys.argv[3])
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else 24
MIN_MARGIN = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
SR = 32000

# These source slices contain user-confirmed non-target audio.  Do not let a
# broad recall pack silently re-present the exact clip as a fresh candidate.
SKIP_SOURCES = {
    "28055832533-1-192_vocals_0017516160_0017573120.wav",
}
SKIP_INTERVALS = {
    "1405440540-1-192_vocals_0012969600_0013081600.wav": [(1.50, 3.00)],
    "28055832533-1-192_vocals_0018346880_0018502400.wav": [(2.00, 3.50)],
    "28055832533-1-192_vocals_0013268480_0013539200.wav": [(5.50, 7.00)],
    "1311215796_u1-1-192_vocals_0006943360_0007132160.wav": [(4.40, 5.90)],
}


def rank(row: dict) -> float:
    # Balanced recall score; the user must still listen to every exported clip.
    return (
        0.50 * float(row["p25"])
        + 0.25 * float(row["median"])
        + 0.15 * float(row["center"])
        + 0.10 * float(row["target_negative_margin"])
    )


def overlaps(a: dict, b: dict) -> bool:
    if a["source"] != b["source"]:
        return False
    # Keep a little distance between exported windows from one source slice.
    return not (a["end"] + 0.50 <= b["start"] or b["end"] + 0.50 <= a["start"])


def is_confirmed_bad(row: dict) -> bool:
    if row["source"] in SKIP_SOURCES:
        return True
    return any(
        not (row["end"] <= start or row["start"] >= end)
        for start, end in SKIP_INTERVALS.get(row["source"], [])
    )


def main() -> None:
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    rows = []
    for file_row in data["files"]:
        source = file_row["file"]
        for window in file_row["windows"]:
            row = {"source": source, **window}
            if is_confirmed_bad(row):
                continue
            if float(row.get("target_negative_margin", 0.0)) < MIN_MARGIN:
                continue
            row["rank"] = rank(row)
            rows.append(row)

    chosen = []
    for row in sorted(rows, key=lambda x: x["rank"], reverse=True):
        if any(overlaps(row, old) for old in chosen):
            continue
        chosen.append(row)
        if len(chosen) >= LIMIT:
            break

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, row in enumerate(chosen, 1):
        source_path = INPUT_DIR / row["source"]
        wav, sr = torchaudio.load(str(source_path))
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
            sr = SR
        start = max(0, int(round(float(row["start"]) * sr)))
        end = min(wav.shape[1], int(round(float(row["end"]) * sr)))
        clip = wav[:, start:end]
        name = (
            f"recall_{index:03d}_rank{row['rank']:.3f}_p25{row['p25']:.3f}_"
            f"{Path(row['source']).stem}_{start:010d}_{end:010d}.wav"
        )
        out = OUTPUT_DIR / name
        torchaudio.save(str(out), clip, sr)
        manifest.append({**row, "file": str(out)})

    (OUTPUT_DIR / "recall_review_manifest.json").write_text(
        json.dumps(
            {
                "note": "ranked audition pack only; not a training list",
                "report": str(REPORT),
                "limit": LIMIT,
                "min_margin": MIN_MARGIN,
                "selected": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"ranked_rows={len(rows)} selected={len(manifest)} output={OUTPUT_DIR}")
    for row in manifest:
        print(
            f"{Path(row['file']).name}/t{row['source']}"
            f"/t{row['start']:.2f}-{row['end']:.2f}"
            f"/trank={row['rank']:.3f} p25={row['p25']:.3f}"
            f" center={row['center']:.3f} margin={row['target_negative_margin']:.3f}"
        )


if __name__ == "__main__":
    main()
