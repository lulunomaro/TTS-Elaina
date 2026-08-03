# -*- coding: utf-8 -*-
"""Re-cut complete phrases for the boundary failures in acceptance round 1.

This is deliberately a small calibration rescue pack.  It restores complete
ASR words/phrases from the original BS-Roformer vocals, never reconstructs a
voice, and does not modify any training list.  Once the rescue behavior is
human-validated it can replace the over-aggressive global hard-split rule.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_resource_policy import apply_cpu_friendly_policy


RESOURCE_POLICY = apply_cpu_friendly_policy(2)

import numpy as np
import soundfile as sf

import build_human_calibration_v1 as page_base
import full_speaker_diarize_new_v1 as audio_base


LABELS = ROOT / "human_full_semantic_acceptance_v1" / "labels_resolved.json"
HARD_EXCLUSIONS = (
    ROOT / "human_full_semantic_acceptance_v1" / "hard_exclusion_intervals.json"
)
WORDS_DIR = ROOT / "full_source_local_word_scores_v1" / "source_reports"
OUT = ROOT / "full_semantic_acceptance_boundary_rescue_v1"
AUDIO_DIR = OUT / "audio"

OUTPUT_SR = 32000
FADE_SEC = 0.002
RANDOM_SEED = 202608021745
MAX_REVIEW_SEC = 12.0

# Human feedback says every listed source clip contains the correct target
# speaker but is phonetically/semantically incomplete.  Bounds below restore
# the nearest complete target phrase.  A001 and A012 get one conservative and
# one fuller alternative because a seamless speaker boundary makes one edge
# ambiguous; the listener decides which, if either, is safe.
RESCUE_BOUNDS: dict[str, list[tuple[str, int, int]]] = {
    "A001": [
        ("word_complete", 1650, 1664),
        ("sentence_complete", 1648, 1664),
    ],
    "A004": [("sentence_complete_long_quality_recheck", 1521, 1560)],
    "A007": [("sentence_complete_gold_recut", 739, 752)],
    "A008": [("sentence_complete", 1700, 1715)],
    "A009": [("sentence_complete", 4444, 4474)],
    "A010": [("restore_terminal_word", 1280, 1295)],
    "A011": [("sentence_complete", 2514, 2536)],
    "A012": [
        ("eroded_words_restored", 4894, 4902),
        ("sentence_complete", 4886, 4902),
    ],
    "A017": [("restore_split_run_tail", 213, 241)],
    "A019": [("sentence_complete", 1532, 1549)],
    "A021": [("restore_terminal_word", 1690, 1706)],
    "A023": [("restore_sentence_head", 23, 34)],
    "A024": [("sentence_complete", 1496, 1521)],
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8"
    )


def overlaps(start: float, end: float, left: float, right: float) -> bool:
    return max(start, left) < min(end, right)


def read_mono(path: Path, start: float, end: float) -> tuple[np.ndarray, int]:
    with sf.SoundFile(str(path), "r") as handle:
        rate = int(handle.samplerate)
        left = max(0, round(start * rate))
        right = min(int(handle.frames), round(end * rate))
        handle.seek(left)
        audio = handle.read(max(0, right - left), dtype="float32", always_2d=True)
    return audio.mean(axis=1).astype(np.float32, copy=False), rate


def frame_rms(audio: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    frame = max(1, round(rate * 0.020))
    hop = max(1, round(rate * 0.005))
    if len(audio) < frame:
        audio = np.pad(audio, (0, frame - len(audio)))
    starts = np.arange(0, len(audio) - frame + 1, hop, dtype=np.int64)
    squared = np.square(audio, dtype=np.float64)
    prefix = np.empty(len(squared) + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(squared, out=prefix[1:])
    values = np.sqrt((prefix[starts + frame] - prefix[starts]) / frame + 1e-12)
    return (starts + frame / 2.0) / rate, values


def local_valley(path: Path, search_start: float, search_end: float) -> float:
    if search_end <= search_start + 0.010:
        return 0.5 * (search_start + search_end)
    audio, rate = read_mono(path, search_start, search_end)
    times, values = frame_rms(audio, rate)
    if not len(values):
        return 0.5 * (search_start + search_end)
    # Prefer the quietest point; ties go to the middle rather than an edge.
    minimum = float(np.min(values))
    eligible = np.flatnonzero(values <= minimum * 1.08 + 1e-7)
    center = 0.5 * (search_end - search_start)
    index = int(eligible[np.argmin(np.abs(times[eligible] - center))])
    return float(search_start + times[index])


def word_aware_bounds(path: Path, words: list[dict], left: int, right: int) -> tuple[float, float, dict]:
    word_start = float(words[left]["start"])
    word_end = float(words[right]["end"])
    previous_end = float(words[left - 1]["end"]) if left > 0 else max(0.0, word_start - 0.25)
    next_start = (
        float(words[right + 1]["start"])
        if right + 1 < len(words)
        else word_end + 0.25
    )
    left_gap = max(0.0, word_start - previous_end)
    right_gap = max(0.0, next_start - word_end)

    if left_gap >= 0.060:
        search_start = max(previous_end + 0.010, word_start - 0.200)
        search_end = word_start - 0.018
        start = local_valley(path, search_start, search_end)
        left_mode = "local_energy_valley_in_gap"
    else:
        start = max(0.0, word_start - 0.025)
        left_mode = "seamless_word_onset_minus_25ms"

    if right_gap >= 0.060:
        search_start = word_end + 0.018
        search_end = min(next_start - 0.010, word_end + 0.200)
        end = local_valley(path, search_start, search_end)
        right_mode = "local_energy_valley_in_gap"
    else:
        end = word_end + 0.025
        right_mode = "seamless_word_offset_plus_25ms"

    start = min(start, word_start - 0.012)
    end = max(end, word_end + 0.012)
    return start, end, {
        "left_gap_sec": left_gap,
        "right_gap_sec": right_gap,
        "left_mode": left_mode,
        "right_mode": right_mode,
        "word_start": word_start,
        "word_end": word_end,
    }


def micro_fade(audio: np.ndarray) -> np.ndarray:
    result = np.asarray(audio, dtype=np.float32).copy()
    count = min(len(result) // 4, max(1, round(OUTPUT_SR * FADE_SEC)))
    if count:
        ramp = np.linspace(0.0, 1.0, count, dtype=np.float32)
        result[:count] *= ramp
        result[-count:] *= ramp[::-1]
    return result


def main() -> None:
    labels = json.loads(LABELS.read_text(encoding="utf-8"))["samples"]
    label_by_id = {str(row["blind_id"]): row for row in labels}
    actual_boundary = {
        str(row["blind_id"])
        for row in labels
        if row["final_class"] == "boundary_cut"
    }
    if actual_boundary != set(RESCUE_BOUNDS):
        raise RuntimeError(
            f"Boundary override mismatch: labels={sorted(actual_boundary)} bounds={sorted(RESCUE_BOUNDS)}"
        )
    hard = json.loads(HARD_EXCLUSIONS.read_text(encoding="utf-8"))["intervals"]

    candidates = []
    for origin_id, variants in RESCUE_BOUNDS.items():
        human = label_by_id[origin_id]
        source_id = str(human["source_id"])
        source = Path(human["source"]).resolve()
        words = json.loads(
            (WORDS_DIR / f"{source_id}.json").read_text(encoding="utf-8")
        )["word_rows"]
        for variant, left, right in variants:
            if not (0 <= left <= right < len(words)):
                raise IndexError(f"Invalid word bounds for {origin_id}: {left}-{right}")
            start, end, boundary = word_aware_bounds(source, words, left, right)
            duration = end - start
            if duration <= 0.80 or duration > MAX_REVIEW_SEC:
                raise RuntimeError(
                    f"Unexpected rescue duration for {origin_id}/{variant}: {duration:.3f}s"
                )
            hard_hits = [
                row["reason"]
                for row in hard
                if str(row["source_id"]) == source_id
                and overlaps(start, end, float(row["start"]), float(row["end"]))
            ]
            if hard_hits:
                raise RuntimeError(
                    f"Rescue {origin_id}/{variant} overlaps human unsafe mask: {hard_hits}"
                )
            raw, rate = read_mono(source, start, end)
            clip = micro_fade(audio_base.resample_1d(raw, rate, OUTPUT_SR))
            selected_words = words[left : right + 1]
            text = "".join(str(row.get("text") or "") for row in selected_words)
            if not text.strip() or "/ufffd" in text:
                raise RuntimeError(f"Invalid ASR text for {origin_id}/{variant}")
            candidates.append(
                {
                    "kind": "full_semantic_boundary_rescue",
                    "origin_blind_id": origin_id,
                    "origin_human_key": human["human_key"],
                    "origin_role": human["role"],
                    "origin_start": float(human["start"]),
                    "origin_end": float(human["end"]),
                    "origin_text": str(human.get("text") or ""),
                    "origin_feedback_note": str(human.get("human_note") or ""),
                    "source_id": source_id,
                    "source": str(source),
                    "variant": variant,
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "word_start_index": left,
                    "word_end_index": right,
                    "text": text,
                    "boundary": boundary,
                    "restored_left_sec": max(0.0, float(human["start"]) - start),
                    "restored_right_sec": max(0.0, end - float(human["end"])),
                    "strict_word_fraction": float(
                        np.mean([bool(row["strict_speaker_pass"]) for row in selected_words])
                    ),
                    "minimum_context_target_fraction": float(
                        min(float(row["context_target_fraction"]) for row in selected_words)
                    ),
                    "minimum_cluster_margin": float(
                        min(float(row["cluster_margin"]) for row in selected_words)
                    ),
                    "minimum_anchor_margin": float(
                        min(float(row["anchor_margin"]) for row in selected_words)
                    ),
                    "hard_mask_hits": hard_hits,
                    "_clip": clip,
                }
            )

    random.Random(RANDOM_SEED).shuffle(candidates)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(candidates, 1):
        blind_id = f"R{index:03d}"
        clip = row.pop("_clip")
        output = AUDIO_DIR / f"{blind_id}.wav"
        sf.write(str(output), clip, OUTPUT_SR, subtype="PCM_16")
        info = sf.info(str(output))
        if info.samplerate != OUTPUT_SR or info.channels != 1 or info.subtype != "PCM_16":
            raise RuntimeError(f"Invalid export format: {output}")
        row["blind_id"] = blind_id
        row["audio"] = str(output.resolve())
        row["audio_rel"] = f"audio/{output.name}"
        row["output_duration"] = float(info.duration)
        row["wave"] = page_base.waveform(clip)

    origin_counts = Counter(str(row["origin_blind_id"]) for row in candidates)
    audit = {
        "version": "full_semantic_acceptance_boundary_rescue_v1_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "input": "13 human-labelled boundary failures from full-semantic acceptance v1",
            "source_audio": "original BS-Roformer vocals",
            "strategy": "restore complete ASR words/phrases; word-aware local valley; no blind 180ms erosion",
            "seamless_boundary": "25ms word padding only; never cross a human-confirmed mixed/non-target mask",
            "fade": "2ms click-prevention micro-fade",
            "status": "human review only; not training data",
            "boundary_failure_is_not_speaker_negative": True,
            "local_resources": RESOURCE_POLICY,
        },
        "origin_count": len(origin_counts),
        "sample_count": len(candidates),
        "origin_variant_counts": dict(sorted(origin_counts.items())),
        "samples": candidates,
    }
    public = {
        "version": "full_semantic_acceptance_boundary_rescue_v1_blind",
        "sample_count": len(candidates),
        "samples": [
            {
                "blind_id": row["blind_id"],
                "audio": row["audio_rel"],
                "duration_sec": round(float(row["output_duration"]), 3),
            }
            for row in candidates
        ],
    }
    source_stats: dict[str, dict] = defaultdict(lambda: {"samples": 0, "duration": 0.0})
    for row in candidates:
        source_stats[row["source_id"]]["samples"] += 1
        source_stats[row["source_id"]]["duration"] += float(row["duration"])
    report = {
        "version": "full_semantic_acceptance_boundary_rescue_v1_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "boundary_origins": len(origin_counts),
        "review_samples": len(candidates),
        "total_duration_sec": sum(float(row["duration"]) for row in candidates),
        "longer_than_6s": sum(float(row["duration"]) > 6.0 for row in candidates),
        "source_stats": dict(sorted(source_stats.items())),
        "hard_mask_overlap_count": sum(bool(row["hard_mask_hits"]) for row in candidates),
        "training_started": False,
        "existing_training_lists_modified": False,
        "score_90_claimed": False,
    }
    write_json(OUT / "audit_manifest.json", audit)
    write_json(OUT / "blind_manifest.json", public)
    write_json(OUT / "report.json", report)
    (OUT / "README_试听说明.txt").write_text(
        "这批专门修复上一轮大量切头/切尾。音频从原始 BS-Roformer vocals 按完整词句重切，未做声音重建。/n"
        "每条必须听到结尾：只有整条都是伊蕾娜、首尾字完整、音质可训练时才不要勾错误。/n"
        "若仍切字，选“切头/切尾”；若扩边带入别人，选“混入/无缝接话”；音质糊则选“杂音/音质问题”。/n"
        "同一问题可能有保守/完整两个盲测版本，任一版本通过即可保留通过版。/n",
        encoding="utf-8",
    )
    (OUT / "开始试听.cmd").write_text(
        '@echo off/r/nstart "" "%~dp0index.html"/r/n', encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
