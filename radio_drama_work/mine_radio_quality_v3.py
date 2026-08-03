# -*- coding: utf-8 -*-
"""Mine high-quality radio-drama training clips from existing score reports.

The previous full-word pass used word acceptance as the only way to create a
training segment.  That discarded long target-speaker turns because ASR words
are short and their surrounding 1.2 s windows are not independently stable.
This pass uses already-scored high-confidence speech windows as the segment
anchor, then uses word-level scores only to cut clear non-target words and
known contaminated intervals.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

ROOT = Path(r"radio_drama_work/radio_drama_work")
OUT = ROOT / "radio_quality_v6"
ACCEPTED_DIR = OUT / "accepted_segments"
REVIEW_DIR = OUT / "review"
REPORT_PATH = OUT / "report.json"
VALIDATION_PATH = OUT / "validation.json"
CANDIDATE_LIST = ROOT / "radio_quality_train_candidate_v6.list"

SAMPLE_RATE = 32000
INDEX_RATE = 32000
MIN_SEC = 1.50
MAX_SEC = 8.00
EDGE_PAD = 0.055

SOURCE_PATHS = {
    "1311215796_u1-1-192_vocals.wav": ROOT / "new_vocals" / "1311215796_u1-1-192_vocals.wav",
    "1405440540-1-192_vocals.wav": ROOT / "new_vocals" / "1405440540-1-192_vocals.wav",
    "28055832533-1-192_vocals.wav": ROOT / "new_vocals" / "28055832533-1-192_vocals.wav",
    "280479210-1-192_vocals.wav": ROOT / "vocals" / "280479210-1-192_vocals.wav",
    "113814583-1-160_vocals.wav": ROOT / "vocals" / "113814583-1-160_vocals.wav",
}

REPORT_INPUTS = [
    {
        "report": ROOT / "new_multiref_scores.json",
        "clip_dir": ROOT / "new_candidates_sp75",
        "tier": "A",
        "center_min": 0.75,
        "p25_min": 0.52,
        "margin_min": 0.02,
    },
    {
        "report": ROOT / "radio_drama_fine_multiref_scores.json",
        "clip_dir": ROOT / "radio_drama_fine",
        "tier": "A",
        "center_min": 0.78,
        "p25_min": 0.54,
        "margin_min": -1.0,
    },
]

WORD_REPORT = ROOT / "full_word_target_v1" / "report.json"
EXCLUDED_OUTPUT_NAMES = {
    "28055832533-1-192_vocals_0013268480_0013539200_safe_word_000.wav",
    "28055832533-1-192_vocals_0013268480_0013539200_safe_word_001.wav",
}


def load_audio(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav.contiguous()


def safe_text(text: str) -> str:
    text = re.sub(r"[^0-9A-Za-zぁ-んァ-ヶ一-龯々ー]", "_", text)
    return text[:24] or "speech"


def overlaps(start: float, end: float, intervals: list[tuple[float, float]]) -> bool:
    return any(max(start, a) < min(end, b) for a, b in intervals)


def parse_source_interval(name: str, source_name: str) -> tuple[float, float] | None:
    for prefix in (source_name, source_name[:-4]):
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix) :]
        match = re.fullmatch(r"_(/d{10})_(/d{10})/.wav", rest)
        if match:
            return int(match.group(1)) / INDEX_RATE, int(match.group(2)) / INDEX_RATE
    return None


def forbidden_intervals() -> dict[str, list[tuple[float, float]]]:
    result = defaultdict(list)
    negative = ROOT / "confirmed_non_target_or_mixed.txt"
    if negative.exists():
        for line in negative.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "|" not in line:
                continue
            raw_path = line.split("|", 1)[0].strip()
            name = Path(raw_path).name
            source_name = next((source for source in SOURCE_PATHS if source in name), None)
            if source_name is None:
                continue
            numbers = [int(x) for x in re.findall(r"/d{10}", name)]
            if len(numbers) >= 4:
                start = (numbers[-4] + numbers[-2]) / INDEX_RATE
                end = (numbers[-4] + numbers[-1]) / INDEX_RATE
            elif len(numbers) >= 2:
                start = numbers[-2] / INDEX_RATE
                end = numbers[-1] / INDEX_RATE
            else:
                continue
            result[source_name].append((start, end))

    # The two experimental safe_word files are excluded as exact outputs.
    # Their source-level time spans are also blocked, so a new candidate
    # cannot silently contain those two words under a different filename.
    result["28055832533-1-192_vocals.wav"].extend([(421.90, 422.12), (422.42, 422.70)])
    return result


def load_word_rows() -> dict[str, list[dict]]:
    data = json.loads(WORD_REPORT.read_text(encoding="utf-8"))
    result = {}
    for item in data.get("files", []):
        source_name = Path(item["source"]).name
        result[source_name] = item.get("scored_words", [])
    return result


def clear_non_target_words(words: list[dict], start: float, end: float, forbidden: list[tuple[float, float]]) -> list[dict]:
    kept = []
    for word in words:
        if word["end"] <= start or word["start"] >= end:
            continue
        if overlaps(word["start"], word["end"], forbidden):
            continue
        # Only cut words with clear non-target evidence.  Borderline words
        # remain inside an already high-scoring candidate window and are
        # recorded for audit rather than silently treated as negatives.
        clear_non_target = word.get("target_center", 0.0) < 0.55 and word.get("margin", 0.0) < 0.0
        extremely_low = word.get("target_center", 0.0) < 0.48 and word.get("target_p25", 0.0) < 0.38
        if clear_non_target or extremely_low:
            continue
        kept.append(word)
    return kept


def split_long_candidate(words: list[dict], start: float, end: float, preserve_window: bool = False) -> list[tuple[float, float, str]]:
    if end - start <= MAX_SEC and preserve_window:
        return [(start, end, "".join(word["text"] for word in words))]
    groups = []
    current = []
    for word in words:
        if not current:
            current = [word]
            continue
        gap = word["start"] - current[-1]["end"]
        proposed_end = word["end"]
        if gap > 0.45 or proposed_end - current[0]["start"] > MAX_SEC:
            groups.append(current)
            current = [word]
        else:
            current.append(word)
    if current:
        groups.append(current)
    result = []
    for group in groups:
        a = max(start, group[0]["start"] - EDGE_PAD)
        b = min(end, group[-1]["end"] + EDGE_PAD)
        if b - a >= MIN_SEC:
            result.append((a, b, "".join(word["text"] for word in group)))
    return result


def write_clip(wav: torch.Tensor, start: float, end: float, path: Path) -> None:
    st = max(0, int(round(start * SAMPLE_RATE)))
    en = min(wav.shape[1], int(round(end * SAMPLE_RATE)))
    if en <= st:
        return
    sf.write(str(path), wav[:, st:en].numpy().T, SAMPLE_RATE, subtype="PCM_16")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ACCEPTED_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    forbidden = forbidden_intervals()
    words_by_source = load_word_rows()
    waves = {name: load_audio(path) for name, path in SOURCE_PATHS.items()}
    candidates = []
    for cfg in REPORT_INPUTS:
        data = json.loads(cfg["report"].read_text(encoding="utf-8"))
        for row in data.get("rows", []):
            if row.get("center", 0.0) < cfg["center_min"] or row.get("p25", 0.0) < cfg["p25_min"]:
                continue
            if row.get("target_negative_margin", 1.0) < cfg["margin_min"]:
                continue
            clip_path = cfg["clip_dir"] / row["name"]
            if not clip_path.exists():
                continue
            source_name = next(
                (name for name in SOURCE_PATHS if name in row["name"] or name[:-4] in row["name"]),
                None,
            )
            if source_name is None:
                continue
            interval = parse_source_interval(row["name"], source_name)
            if interval is None:
                continue
            start, end = interval
            if overlaps(start, end, forbidden[source_name]):
                continue
            candidates.append({
                "source": source_name,
                "start": start,
                "end": end,
                "center": float(row.get("center", 0.0)),
                "p25": float(row.get("p25", 0.0)),
                "margin": float(row.get("target_negative_margin", -1.0)),
                "origin": str(clip_path),
                "tier": cfg["tier"],
            })

    # Keep overlapping report rows until after they have been split.  A long
    # high-score row can cover several distinct 1.5-8 s training clips; doing
    # row-level deduplication here would throw those clips away.
    candidates.sort(key=lambda item: (item["source"], item["start"], -item["center"], -item["p25"]))
    unique = []
    exact_seen = set()
    for candidate in candidates:
        key = (candidate["source"], round(candidate["start"], 4), round(candidate["end"], 4))
        if key in exact_seen:
            continue
        exact_seen.add(key)
        unique.append(candidate)

    output_rows = []
    list_lines = []
    counters = defaultdict(int)
    for candidate in unique:
        source = candidate["source"]
        start, end = candidate["start"], candidate["end"]
        source_words = words_by_source.get(source, [])
        kept_words = clear_non_target_words(source_words, start, end, forbidden[source])
        in_window = [word for word in source_words if word["end"] > start and word["start"] < end]
        if not kept_words or not in_window:
            counters["no_target_word_anchor"] += 1
            continue
        bad_count = len(in_window) - len(kept_words)
        if bad_count > max(2, len(in_window) // 2):
            counters["too_many_clear_non_target_words"] += 1
            continue
        spans = split_long_candidate(kept_words, start, end, preserve_window=(bad_count == 0))
        if not spans:
            counters["too_short_after_word_cut"] += 1
            continue
        wav = waves[source]
        for span_start, span_end, text in spans:
            if overlaps(span_start, span_end, forbidden[source]):
                counters["forbidden_overlap"] += 1
                continue
            name = f"radio_{len(output_rows):04d}_{Path(source).stem}_{int(span_start * 100):08d}_{int(span_end * 100):08d}.wav"
            if name in EXCLUDED_OUTPUT_NAMES or "safe_word_" in name:
                counters["excluded_safe_word"] += 1
                continue
            output = ACCEPTED_DIR / name
            write_clip(wav, span_start, span_end, output)
            duration = span_end - span_start
            if duration < MIN_SEC or duration > MAX_SEC + 0.2:
                output.unlink(missing_ok=True)
                counters["invalid_duration"] += 1
                continue
            duplicate = False
            for old in output_rows:
                if old["source"] != source:
                    continue
                overlap = max(0.0, min(span_end, old["end"]) - max(span_start, old["start"]))
                smaller = min(span_end - span_start, old["end"] - old["start"])
                if smaller > 0 and overlap / smaller >= 0.75:
                    duplicate = True
                    break
            if duplicate:
                output.unlink(missing_ok=True)
                counters["overlapping_duplicate"] += 1
                continue
            list_lines.append(f"{output.resolve()}|radio_quality_v3|JA|{text}")
            output_rows.append({**candidate, "file": str(output), "start": span_start, "end": span_end, "duration": duration, "text": text, "cleared_word_count": bad_count})

    CANDIDATE_LIST.write_text("/n".join(list_lines) + ("/n" if list_lines else ""), encoding="utf-8")
    report = {
        "version": "radio_quality_v3",
        "note": "High-score sliding-window candidates, split at ASR boundaries; clear non-target words and confirmed contaminated intervals removed.",
        "source_count": len(SOURCE_PATHS),
        "source_duration_sec": sum(wav.shape[1] / SAMPLE_RATE for wav in waves.values()),
        "report_inputs": [str(cfg["report"]) for cfg in REPORT_INPUTS],
        "candidate_windows_before_dedup": len(candidates),
        "candidate_windows_after_dedup": len(unique),
        "output_count": len(output_rows),
        "output_duration_sec": sum(row["duration"] for row in output_rows),
        "counters": dict(counters),
        "excluded_safe_word_names": sorted(EXCLUDED_OUTPUT_NAMES),
        "outputs": output_rows,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    paths = [Path(line.split("|", 1)[0]) for line in list_lines]
    validation = {
        "candidate_count": len(list_lines),
        "missing_paths": [str(path) for path in paths if not path.exists()],
        "duplicate_paths": [str(path) for path in sorted(set(str(p) for p in paths)) if sum(str(q) == str(path) for q in paths) > 1],
        "safe_word_hits": [line for line in list_lines if "safe_word_000" in line or "safe_word_001" in line],
        "short_outputs": [
            {"path": str(path), "duration": sf.info(str(path)).duration}
            for path in paths
            if path.exists() and sf.info(str(path)).duration < MIN_SEC
        ],
    }
    validation["passed"] = not any(validation[key] for key in ("missing_paths", "duplicate_paths", "safe_word_hits", "short_outputs"))
    VALIDATION_PATH.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"candidate_windows={len(unique)} outputs={len(output_rows)} seconds={sum(row['duration'] for row in output_rows):.1f}", flush=True)
    print(f"saved_list={CANDIDATE_LIST}", flush=True)
    print(f"validation_passed={validation['passed']}", flush=True)
    if not validation["passed"]:
        raise RuntimeError("Validation failed: " + str(VALIDATION_PATH))


if __name__ == "__main__":
    main()
