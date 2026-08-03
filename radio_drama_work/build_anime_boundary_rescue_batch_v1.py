# -*- coding: utf-8 -*-
"""Build two original-vocals rescue variants for each human boundary failure."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import build_human_calibration_v1 as page_base


BATCH_LABELS = ROOT / "human_anime_semantic_review_batch_v1" / "labels_resolved.json"
WORDS_DIR = ROOT / "anime_mining_v1" / "asr" / "source_reports"
HARD_MASKS = ROOT / "human_anime_speaker_calibration_v3" / "hard_exclusion_intervals.json"
OUT = ROOT / "anime_boundary_rescue_batch_v1"
AUDIO_DIR = OUT / "audio"
RESCUE_VERSION = "anime_boundary_rescue_batch_v1"
EXPECTED_BOUNDARY_COUNT = 26
OUTPUT_SR = 32000
PAD_SEC = 0.22
MAX_SEC = 6.0


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8"
    )


def overlap(a: float, b: float, row: dict) -> bool:
    return max(a, float(row["start"])) < min(b, float(row["end"]))


def micro_fade(audio: np.ndarray) -> np.ndarray:
    output = np.asarray(audio, dtype=np.float32).copy()
    count = min(len(output) // 4, max(1, round(OUTPUT_SR * 0.002)))
    if count:
        ramp = np.linspace(0.0, 1.0, count, dtype=np.float32)
        output[:count] *= ramp
        output[-count:] *= ramp[::-1]
    return output


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return np.asarray(audio.mean(axis=1), dtype=np.float32), int(rate)


def resample_1d(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)
    divisor = math.gcd(int(source_rate), int(target_rate))
    up = int(target_rate) // divisor
    down = int(source_rate) // divisor
    return np.asarray(resample_poly(audio, up, down), dtype=np.float32)


def segment(audio: np.ndarray, rate: int, start: float, end: float) -> np.ndarray:
    left = max(0, int(round(start * rate)))
    count = max(1, int(round((end - start) * rate)))
    right = min(len(audio), left + count)
    clip = np.asarray(audio[left:right], dtype=np.float32)
    if len(clip) < count:
        clip = np.pad(clip, (0, count - len(clip)))
    return clip[:count]


def load_words(source_id: str) -> list[dict]:
    path = WORDS_DIR / f"{source_id}.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    return document.get("word_rows") or document.get("words") or []


def human_direction(row: dict) -> tuple[bool, bool, bool]:
    text = f"{row.get('error_type', '')} {row.get('human_note', '')}"
    head = any(token in text for token in ("头", "开头", "開始", "切頭"))
    tail = any(token in text for token in ("尾", "结尾", "終了", "切尾"))
    if not head and not tail:
        head = True
        tail = True
    trim_non_target_tail = any(
        token in text for token in ("非目标", "非目標", "非伊蕾娜", "other voice")
    ) and tail
    return head, tail, trim_non_target_tail


def choose_bounds(
    row: dict, words: list[dict], variant: str
) -> tuple[float, float, str, int, int]:
    head, tail, trim_non_target_tail = human_direction(row)
    left = int(row["word_start_index"])
    right = int(row["word_end_index"])
    start = float(row["start"])
    end = float(row["end"])
    if variant == "word_expand":
        if head and left > 0:
            previous = words[left - 1]
            if float(words[left]["start"]) - float(previous["end"]) <= 0.75:
                start = float(previous["start"])
                left -= 1
        if tail and right + 1 < len(words):
            following = words[right + 1]
            if float(following["start"]) - float(words[right]["end"]) <= 0.75:
                end = float(following["end"])
                right += 1
    else:
        side_count = int(head) + int(tail)
        pad = PAD_SEC
        original_duration = float(row["end"]) - float(row["start"])
        if side_count and original_duration + pad * side_count > MAX_SEC:
            # Leave a small margin below the 6s training limit so a capped
            # word-expanded variant remains a genuinely different waveform.
            available = max(0.05, MAX_SEC - original_duration)
            pad = max(0.05, available / side_count * 0.85)
        if head:
            start = max(0.0, start - pad)
        if tail:
            end += pad

    if trim_non_target_tail and right > left:
        end = min(end, float(words[right - 1]["end"]))
        right -= 1

    if end - start > MAX_SEC:
        if head and tail:
            excess = end - start - MAX_SEC
            start += excess * 0.5
            end -= excess * 0.5
        elif tail:
            end = start + MAX_SEC
        else:
            start = end - MAX_SEC
    start = max(0.0, start)
    if end - start < 1.5:
        end = start + 1.5
    selected = words[left : right + 1] if 0 <= left <= right < len(words) else []
    text = "".join(str(item.get("text") or "") for item in selected).strip()
    if not text:
        text = str(row["text"])
    return start, end, text, left, right


def main() -> None:
    labels = json.loads(BATCH_LABELS.read_text(encoding="utf-8"))["samples"]
    boundary = [row for row in labels if row["final_class"] == "boundary_cut"]
    if len(boundary) != EXPECTED_BOUNDARY_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_BOUNDARY_COUNT} boundary rows, got {len(boundary)}"
        )
    hard = json.loads(HARD_MASKS.read_text(encoding="utf-8"))["intervals"]
    words_cache = {}
    audio_cache = {}
    candidates = []
    for row in boundary:
        source_id = str(row["source_id"])
        if source_id not in words_cache:
            words_cache[source_id] = load_words(source_id)
        words = words_cache[source_id]
        for variant in ("word_expand", "pad_220ms"):
            start, end, text, left, right = choose_bounds(row, words, variant)
            hits = [
                item
                for item in hard
                if str(item["source_id"]) == source_id and overlap(start, end, item)
            ]
            if hits:
                raise RuntimeError(
                    f"{row['blind_id']} {variant} overlaps human hard mask: {hits}"
                )
            if not 1.5 <= end - start <= MAX_SEC:
                raise RuntimeError(f"Invalid bounds {row['blind_id']} {variant}: {start}-{end}")
            if source_id not in audio_cache:
                audio_cache[source_id] = read_mono(Path(row["source"]))
            source_audio, source_rate = audio_cache[source_id]
            clip = segment(source_audio, source_rate, start, end)
            clip = micro_fade(resample_1d(clip, source_rate, OUTPUT_SR))
            candidates.append(
                {
                    "origin_batch_blind_id": str(row["blind_id"]),
                    "origin_pool_blind_id": str(row["origin_blind_id"]),
                    "source_id": source_id,
                    "source": str(row["source"]),
                    "variant": variant,
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "word_start_index": left,
                    "word_end_index": right,
                    "text": text,
                    "human_note": str(row.get("human_note") or ""),
                    "human_error_type": str(row.get("error_type") or ""),
                    "_clip": clip,
                }
            )

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(candidates, 1):
        blind_id = f"R{index:03d}"
        clip = row.pop("_clip")
        output = AUDIO_DIR / f"{blind_id}.wav"
        sf.write(str(output), clip, OUTPUT_SR, subtype="PCM_16")
        info = sf.info(str(output))
        if (
            info.samplerate != OUTPUT_SR
            or info.channels != 1
            or info.subtype != "PCM_16"
            or not 1.5 <= info.duration <= MAX_SEC
        ):
            raise RuntimeError(f"Invalid rescue audio: {output} {info}")
        row["blind_id"] = blind_id
        row["audio"] = str(output.resolve())
        row["audio_rel"] = f"audio/{output.name}"
        row["output_duration"] = float(info.duration)
        row["wave"] = page_base.waveform(np.asarray(clip, dtype=np.float32))

    report = {
        "version": f"{RESCUE_VERSION}_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_boundary_count": len(boundary),
        "variant_count": len(candidates),
        "duration_sec": sum(float(row["output_duration"]) for row in candidates),
        "variant_counts": dict(Counter(row["variant"] for row in candidates)),
        "training_started": False,
        "existing_training_lists_modified": False,
        "score_90_claimed": False,
    }
    audit = {
        "version": f"{RESCUE_VERSION}_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "input": f"human boundary_cut rows from {RESCUE_VERSION}",
            "source_audio": "original BS-Roformer vocals only",
            "variants": "ASR adjacent-word expansion and 220ms raw-vocals padding",
            "special_non_target_tail": "trim tail before expanding head when user note flags non-target tail",
            "status": "review only; not training data",
        },
        "sample_count": len(candidates),
        "samples": candidates,
    }
    write_json(OUT / "audit_manifest.json", audit)
    write_json(OUT / "private_manifest.json", audit)
    write_json(OUT / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
