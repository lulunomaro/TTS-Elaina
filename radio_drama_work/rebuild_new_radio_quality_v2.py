# -*- coding: utf-8 -*-
"""Rebuild new-radio candidates with a strict word-level acceptance gate."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import soundfile as sf
import torch
import torchaudio


ROOT = Path(r"radio_drama_work/radio_drama_work")
V1 = ROOT / "new_radio_quality_v1"
REPORT_IN = V1 / "report.json"
V2 = ROOT / "new_radio_quality_v2"
ACCEPTED = V2 / "accepted_segments"
REVIEW = V2 / "review"
REJECTED = V2 / "rejected"
REPORT_OUT = V2 / "report.json"
VALIDATION_OUT = V2 / "validation.json"
LIST_OUT = ROOT / "new_radio_quality_train_candidate_v2.list"

SAMPLE_RATE = 32000
MIN_SEC = 1.50
MAX_SEC = 8.00
WORD_GAP = 0.28
EDGE = 0.055
# Calibrated middle gate: stricter than the v1 anchor-only rule, while
# allowing a short ASR word's 1.2 s context to be slightly below the pure
# reference median.  Every retained word must pass all three dimensions.
WORD_CENTER_MIN = 0.70
WORD_P25_MIN = 0.48
WORD_MARGIN_MIN = 0.04
OTHER_RATIO_MAX = 0.80

# User-confirmed false positives from v1.  They stay excluded even if a
# future threshold or report change would otherwise admit them.
HARD_EXCLUDE = {
    "newradio_00001_243358894-1-192_vocals_0003417760_0003505760.wav",
    "newradio_00003_243358894-1-192_vocals_0007518240_0007613280.wav",
    "newradio_00004_243358894-1-192_vocals_0007968000_0008032000.wav",
}


def load_audio(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav.contiguous()


def rms(wav: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(wav.float() ** 2) + 1e-12)) if wav.numel() else 0.0


def other_ratio(vocal: torch.Tensor, other: torch.Tensor | None, start: float, end: float) -> float:
    if other is None:
        return 0.0
    a = max(0, int(round(start * SAMPLE_RATE)))
    b = min(vocal.shape[1], other.shape[1], int(round(end * SAMPLE_RATE)))
    if b <= a:
        return 0.0
    return rms(other[:, a:b]) / max(rms(vocal[:, a:b]), 1e-6)


def strict_pass(word: dict) -> bool:
    return bool(
        word.get("center", 0.0) >= WORD_CENTER_MIN
        and word.get("p25", 0.0) >= WORD_P25_MIN
        and word.get("margin", 0.0) >= WORD_MARGIN_MIN
        and not word.get("clear_non_target", False)
    )


def group_strict_words(words: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    current: list[dict] = []
    for word in sorted(words, key=lambda row: row["start"]):
        if not strict_pass(word):
            if current:
                groups.append(current)
                current = []
            continue
        if not current:
            current = [word]
            continue
        gap = word["start"] - current[-1]["end"]
        duration = word["end"] - current[0]["start"]
        if gap <= WORD_GAP and duration <= MAX_SEC:
            current.append(word)
        else:
            groups.append(current)
            current = [word]
    if current:
        groups.append(current)
    return groups


def validate(lines: list[str]) -> dict:
    paths = [Path(line.split("|", 1)[0]) for line in lines]
    missing = [str(path) for path in paths if not path.exists()]
    duplicates = [name for name, count in Counter(str(path) for path in paths).items() if count > 1]
    short = []
    for path in paths:
        if path.exists():
            duration = sf.info(str(path)).duration
            if duration < MIN_SEC:
                short.append({"path": str(path), "duration": duration})
    result = {
        "candidate_count": len(lines),
        "missing_paths": missing,
        "duplicate_paths": duplicates,
        "hard_exclude_hits": [line for line in lines if any(name in line for name in HARD_EXCLUDE)],
        "short_outputs": short,
        "passed": not missing and not duplicates and not short and not any(name in line for line in lines for name in HARD_EXCLUDE),
    }
    VALIDATION_OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    data = json.loads(REPORT_IN.read_text(encoding="utf-8"))
    for path in (ACCEPTED, REVIEW, REJECTED):
        path.mkdir(parents=True, exist_ok=True)

    report = {
        "version": "new_radio_quality_v2",
        "method": "strict word-level target gate after BS-Roformer separation; no reconstruction",
        "source_report": str(REPORT_IN),
        "parameters": {
            "word_center_min": WORD_CENTER_MIN,
            "word_p25_min": WORD_P25_MIN,
            "word_margin_min": WORD_MARGIN_MIN,
            "word_gap": WORD_GAP,
            "min_sec": MIN_SEC,
            "other_ratio_max": OTHER_RATIO_MAX,
        },
        "hard_excluded_v1_files": sorted(HARD_EXCLUDE),
        "files": [],
    }
    list_lines: list[str] = []
    output_index = 0

    for file_report in data.get("files", []):
        source = Path(file_report["source"])
        other_path = Path(file_report["other_source"]) if file_report.get("other_source") else None
        vocal = load_audio(source)
        other = load_audio(other_path) if other_path and other_path.exists() else None
        words = file_report.get("word_scores", [])
        passed_words = [word for word in words if strict_pass(word)]
        groups = group_strict_words(words)
        segments = []
        counters = Counter()
        for group in groups:
            start = max(0.0, group[0]["start"] - EDGE)
            end = min(vocal.shape[1] / SAMPLE_RATE, group[-1]["end"] + EDGE)
            if end - start < MIN_SEC:
                counters["strict_group_too_short"] += 1
                continue
            text = "".join(word["text"] for word in group)
            ratio = other_ratio(vocal, other, start, end)
            if ratio > OTHER_RATIO_MAX:
                counters["other_stem_dominant"] += 1
                continue
            output = ACCEPTED / f"strict_{output_index:05d}_{source.stem}_{int(round(start * SAMPLE_RATE)):010d}_{int(round(end * SAMPLE_RATE)):010d}.wav"
            if output.name in HARD_EXCLUDE:
                counters["hard_excluded"] += 1
                continue
            a = int(round(start * SAMPLE_RATE))
            b = int(round(end * SAMPLE_RATE))
            sf.write(str(output), vocal[:, a:b].numpy().T, SAMPLE_RATE, subtype="PCM_16")
            segment = {
                "file": str(output),
                "start": start,
                "end": end,
                "duration": end - start,
                "text": text,
                "word_count": len(group),
                "other_ratio": ratio,
                "min_word_center": min(word["center"] for word in group),
                "min_word_p25": min(word["p25"] for word in group),
                "min_word_margin": min(word["margin"] for word in group),
                "words": group,
            }
            segments.append(segment)
            list_lines.append(f"{output.resolve()}|new_radio_quality_v2|JA|{text}")
            output_index += 1

        report["files"].append({
            "source": str(source),
            "duration": vocal.shape[1] / SAMPLE_RATE,
            "asr_word_count": file_report.get("asr_word_count", len(words)),
            "scored_word_count": len(words),
            "strict_pass_word_count": len(passed_words),
            "strict_pass_word_duration": sum(word["end"] - word["start"] for word in passed_words),
            "candidate_count": len(segments),
            "candidate_duration": sum(segment["duration"] for segment in segments),
            "reject_counts": dict(counters),
            "segments": segments,
        })
        print(
            f"{source.name} words={len(words)} strict_words={len(passed_words)} candidates={len(segments)} sec={sum(segment['duration'] for segment in segments):.1f}",
            flush=True,
        )

    LIST_OUT.write_text("/n".join(list_lines) + ("/n" if list_lines else ""), encoding="utf-8")
    report["candidate_count"] = len(list_lines)
    report["candidate_duration"] = sum(item["candidate_duration"] for item in report["files"])
    REPORT_OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    validation = validate(list_lines)
    print(f"saved_report={REPORT_OUT}", flush=True)
    print(f"saved_list={LIST_OUT} count={len(list_lines)} sec={report['candidate_duration']:.1f}", flush=True)
    print(f"validation_passed={validation['passed']}", flush=True)
    if not validation["passed"]:
        raise RuntimeError(f"Validation failed: {validation}")


if __name__ == "__main__":
    main()
