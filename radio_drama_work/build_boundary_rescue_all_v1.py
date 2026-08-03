# -*- coding: utf-8 -*-
"""Re-cut every human-confirmed boundary-only clip from its full vocals track."""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

BOOTSTRAP_DIR = Path(__file__).resolve().parent
if str(BOOTSTRAP_DIR) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_DIR))
from local_resource_policy import apply_cpu_friendly_policy

RESOURCE_POLICY = apply_cpu_friendly_policy(2)

import numpy as np
import soundfile as sf

import build_human_calibration_v1 as page_base


ROOT = Path(__file__).resolve().parent
LABELS = ROOT / "human_calibration_combined_v3" / "labels_resolved.json"
MANIFEST = ROOT / "full_track_manifest_v1.json"
WORDS_DIR = ROOT / "word_target_all_v1" / "source_reports"
MODELS_DIR = ROOT / "source_local_cluster_anchor_v3" / "models"
OUT = ROOT / "boundary_rescue_all_v5"
AUDIO_DIR = OUT / "audio"
OUTPUT_SR = 32000
RANDOM_SEED = 20260802
MIN_DURATION = 1.50
MAX_DURATION = 6.50
MAX_CONTEXT_DURATION = 9.00

SPEAKER_UNSAFE = {"pure_non_target", "mixed", "overlap", "target_variant_exclude"}
TERMINAL_SUFFIXES = (
    "でした",
    "ました",
    "まして",
    "ません",
    "でしょう",
    "ください",
    "ましょう",
    "ですよ",
    "ですね",
    "ですか",
    "ますか",
    "です",
    "ます",
    "だよ",
    "だね",
    "なの",
    "のよ",
    "か",
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8")


def frame_rms(audio: np.ndarray, sr: int, frame_sec: float = 0.020, hop_sec: float = 0.010) -> tuple[np.ndarray, np.ndarray]:
    frame = max(1, round(sr * frame_sec))
    hop = max(1, round(sr * hop_sec))
    working = np.pad(audio, (0, max(0, frame - len(audio))))
    count = max(1, 1 + (len(working) - frame) // hop)
    starts = np.arange(count, dtype=np.int64) * hop
    ends = starts + frame
    squared = np.square(working, dtype=np.float64)
    prefix = np.empty(len(squared) + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(squared, out=prefix[1:])
    rms = np.sqrt((prefix[ends] - prefix[starts]) / frame + 1e-12)
    return (starts + frame / 2.0) / sr, rms


def terminal_strength(words: list[dict], index: int) -> float:
    left = max(0, index - 4)
    text = "".join(str(words[idx].get("text", "")) for idx in range(left, index + 1)).rstrip("。．.!！?？、,")
    for suffix in TERMINAL_SUFFIXES:
        if text.endswith(suffix):
            return 1.0 if len(suffix) >= 2 else 0.55
    return 0.0


def valley(
    times: np.ndarray,
    rms: np.ndarray,
    search_start: float,
    search_end: float,
    speech_level: float,
) -> tuple[float, float]:
    if search_end <= search_start:
        center = (search_start + search_end) / 2.0
        return center, 1.0
    indices = np.flatnonzero((times >= search_start) & (times <= search_end))
    if not len(indices):
        center = (search_start + search_end) / 2.0
        return center, 1.0
    index = int(indices[np.argmin(rms[indices])])
    return float(times[index]), float(rms[index] / max(speech_level, 1e-7))


def start_boundary(words: list[dict], word_index: int, times: np.ndarray, rms: np.ndarray, speech_level: float) -> dict:
    word_start = float(words[word_index]["start"])
    previous_end = float(words[word_index - 1]["end"]) if word_index > 0 else max(0.0, word_start - 0.35)
    gap = max(0.0, word_start - previous_end)
    if gap >= 0.025:
        search_start = previous_end + min(0.015, gap * 0.15)
        search_end = word_start - min(0.015, gap * 0.15)
    else:
        search_start = max(0.0, word_start - 0.16)
        search_end = word_start + 0.04
    point, ratio = valley(times, rms, search_start, search_end, speech_level)
    return {
        "time": max(0.0, min(point - 0.010, word_start - 0.015)),
        "energy_ratio": ratio,
        "gap_sec": gap,
        "terminal_before": terminal_strength(words, word_index - 1) if word_index > 0 else 1.0,
    }


def end_boundary(words: list[dict], word_index: int, times: np.ndarray, rms: np.ndarray, speech_level: float) -> dict:
    word_end = float(words[word_index]["end"])
    next_start = float(words[word_index + 1]["start"]) if word_index + 1 < len(words) else word_end + 0.35
    gap = max(0.0, next_start - word_end)
    if gap >= 0.025:
        search_start = word_end + min(0.015, gap * 0.15)
        search_end = next_start - min(0.015, gap * 0.15)
    else:
        search_start = word_end - 0.04
        search_end = word_end + 0.16
    point, ratio = valley(times, rms, search_start, search_end, speech_level)
    return {
        "time": max(word_end + 0.015, point + 0.010),
        "energy_ratio": ratio,
        "gap_sec": gap,
        "terminal_after": terminal_strength(words, word_index),
    }


def target_fraction(model: dict | None, start: float, end: float) -> float | None:
    if model is None:
        return None
    centers = model["specs"].mean(axis=1)
    indices = np.flatnonzero((centers >= start) & (centers <= end))
    if not len(indices):
        return None
    return float(np.mean([int(model["labels"][idx]) in model["targets"] for idx in indices]))


def overlaps_unsafe(start: float, end: float, rows: list[dict], own_key: str) -> list[str]:
    hits = []
    for row in rows:
        if row["human_key"] == own_key or row["final_class"] not in SPEAKER_UNSAFE | {"quality_issue"}:
            continue
        left = float(row["start"]) - 0.12
        right = float(row["end"]) + 0.12
        if max(start, left) < min(end, right):
            hits.append(str(row["human_key"]))
    return hits


def seed_words(words: list[dict], start: float, end: float) -> tuple[int, int]:
    indices = [
        idx
        for idx, row in enumerate(words)
        if float(row.get("end", 0.0)) > start + 0.01
        and float(row.get("start", 0.0)) < end - 0.01
        and str(row.get("text", "")).strip()
    ]
    if not indices:
        raise RuntimeError(f"No ASR words overlap {start:.3f}-{end:.3f}")
    return min(indices), max(indices)


def sanitize_seed_range(words: list[dict], first: int, last: int, model: dict | None) -> tuple[int, int]:
    """Drop a tiny adjacent sentence accidentally included at a cut edge."""
    split_after = []
    for idx in range(first, last):
        gap = float(words[idx + 1]["start"]) - float(words[idx]["end"])
        if terminal_strength(words, idx) >= 0.90 or gap >= 0.24:
            split_after.append(idx)
    if not split_after:
        return first, last
    ranges = []
    left = first
    for split in split_after:
        ranges.append((left, split))
        left = split + 1
    ranges.append((left, last))
    scored = []
    for left, right in ranges:
        start = float(words[left]["start"])
        end = float(words[right]["end"])
        duration = end - start
        fraction = target_fraction(model, start, end)
        speaker_score = 0.65 if fraction is None else 2.2 * fraction
        terminal_score = 0.55 * terminal_strength(words, right)
        scored.append((speaker_score + terminal_score + min(duration, 2.5) / 2.5, left, right))
    _, chosen_left, chosen_right = max(scored)
    return chosen_left, chosen_right


def tight_word_range(words: list[dict], first: int, last: int) -> tuple[int, int]:
    left, right = first, last
    seed_start = float(words[first]["start"])
    seed_end = float(words[last]["end"])
    while left > 0:
        gap = float(words[left]["start"]) - float(words[left - 1]["end"])
        if (
            gap >= 0.085
            or terminal_strength(words, left - 1) >= 0.90
            or seed_start - float(words[left - 1]["start"]) > 1.10
        ):
            break
        left -= 1
    while right + 1 < len(words):
        gap = float(words[right + 1]["start"]) - float(words[right]["end"])
        next_text = str(words[right + 1].get("text", "")).strip()
        if terminal_strength(words, right) >= 0.90 and gap < 0.085 and next_text in {"よ", "ね", "か", "な", "ぞ", "わ"}:
            right += 1
            continue
        if (
            gap >= 0.085
            or terminal_strength(words, right) >= 0.90
            or float(words[right + 1]["end"]) - seed_end > 1.10
        ):
            break
        right += 1
    return left, right


def make_candidate(
    *,
    row: dict,
    words: list[dict],
    left: int,
    right: int,
    times: np.ndarray,
    rms: np.ndarray,
    speech_level: float,
    model: dict | None,
    source_rows: list[dict],
    variant: str,
) -> dict | None:
    left_boundary = start_boundary(words, left, times, rms, speech_level)
    right_boundary = end_boundary(words, right, times, rms, speech_level)
    start = float(left_boundary["time"])
    end = float(right_boundary["time"])
    duration = end - start
    maximum = MAX_CONTEXT_DURATION if variant == "phrase_context" else MAX_DURATION
    if duration < MIN_DURATION or duration > maximum:
        return None
    hard_hits = overlaps_unsafe(start, end, source_rows, str(row["human_key"]))
    if hard_hits:
        return None
    fraction = target_fraction(model, start, end)
    if fraction is not None and fraction < 0.50:
        return None
    first_token = str(words[left].get("text", "")).strip()
    if (
        variant == "phrase_context"
        and float(left_boundary["gap_sec"]) < 0.04
        and float(left_boundary["terminal_before"]) < 0.50
        and (
            first_token in {"ら", "けど", "ので", "て", "が", "を", "に", "は", "も", "と"}
            or (len(first_token) == 1 and "ァ" <= first_token <= "ヿ")
        )
    ):
        return None
    text = "".join(str(words[idx].get("text", "")) for idx in range(left, right + 1))
    if not text.strip() or "�" in text:
        return None
    return {
        "kind": "boundary_rescue_candidate",
        "origin_human_key": row["human_key"],
        "origin_blind_id": row["blind_id"],
        "origin_round": row["human_round"],
        "source_id": row["source_id"],
        "source": row["full_vocals"],
        "origin_start": float(row["start"]),
        "origin_end": float(row["end"]),
        "variant": variant,
        "start": start,
        "end": end,
        "duration": duration,
        "word_start_index": left,
        "word_end_index": right,
        "text": text,
        "target_cluster_fraction": fraction,
        "left_boundary": left_boundary,
        "right_boundary": right_boundary,
        "hard_overlap": hard_hits,
    }


def phrase_candidates(
    row: dict,
    words: list[dict],
    first: int,
    last: int,
    times: np.ndarray,
    rms: np.ndarray,
    speech_level: float,
    model: dict | None,
    source_rows: list[dict],
    tight: dict,
) -> list[dict]:
    left_limit = max(0, first - 55)
    right_limit = min(len(words) - 1, last + 55)
    left_options = []
    for idx in range(left_limit, first + 1):
        boundary = start_boundary(words, idx, times, rms, speech_level)
        if tight["start"] - float(boundary["time"]) > 6.5:
            continue
        gap_score = min(1.4, float(boundary["gap_sec"]) / 0.28)
        silence_score = max(0.0, 1.0 - min(1.0, float(boundary["energy_ratio"])))
        strength = gap_score + silence_score + 0.85 * float(boundary["terminal_before"])
        left_options.append((strength, idx))
    right_options = []
    for idx in range(last, right_limit + 1):
        boundary = end_boundary(words, idx, times, rms, speech_level)
        if float(boundary["time"]) - tight["end"] > 6.5:
            continue
        gap_score = min(1.4, float(boundary["gap_sec"]) / 0.28)
        silence_score = max(0.0, 1.0 - min(1.0, float(boundary["energy_ratio"])))
        strength = gap_score + silence_score + 0.95 * float(boundary["terminal_after"])
        right_options.append((strength, idx))

    ranked = []
    for left_strength, left in left_options:
        for right_strength, right in right_options:
            if right < left:
                continue
            candidate = make_candidate(
                row=row,
                words=words,
                left=left,
                right=right,
                times=times,
                rms=rms,
                speech_level=speech_level,
                model=model,
                source_rows=source_rows,
                variant="phrase_context",
            )
            if candidate is None:
                continue
            if (
                float(candidate["left_boundary"]["gap_sec"]) < 0.04
                and float(candidate["left_boundary"]["terminal_before"]) < 0.50
                and float(candidate["left_boundary"]["energy_ratio"]) > 0.25
            ):
                continue
            tight_fraction = tight["target_cluster_fraction"]
            candidate_fraction = candidate["target_cluster_fraction"]
            if (
                tight_fraction is not None
                and candidate_fraction is not None
                and candidate_fraction < max(0.50, float(tight_fraction) - 0.15)
            ):
                continue
            if (
                terminal_strength(words, int(tight["word_end_index"])) >= 0.90
                and int(candidate["word_end_index"]) > int(tight["word_end_index"])
            ):
                continue
            if (
                int(tight["word_start_index"]) > 0
                and terminal_strength(words, int(tight["word_start_index"]) - 1) >= 0.90
                and int(candidate["word_start_index"]) < int(tight["word_start_index"])
            ):
                continue
            extension = max(0.0, tight["start"] - candidate["start"]) + max(0.0, candidate["end"] - tight["end"])
            if extension < 0.28:
                continue
            fraction = candidate["target_cluster_fraction"]
            speaker_score = 0.8 if fraction is None else 1.7 * fraction
            duration_score = min(candidate["duration"], 4.8) / 4.8
            score = left_strength + right_strength + speaker_score + 0.55 * duration_score
            ranked.append((score, candidate))
    return [item[1] for item in sorted(ranked, key=lambda item: item[0], reverse=True)]


def load_model(source_id: str) -> dict | None:
    path = MODELS_DIR / f"{source_id}.npz"
    if not path.is_file():
        return None
    data = np.load(str(path), allow_pickle=True)
    return {
        "specs": np.asarray(data["specs"], dtype=np.float32),
        "labels": np.asarray(data["labels"], dtype=np.int64),
        "targets": set(int(value) for value in data["target_clusters"].tolist()),
    }


def main() -> None:
    label_rows = json.loads(LABELS.read_text(encoding="utf-8"))["samples"]
    boundary_rows = [dict(row) for row in label_rows if row["final_class"] == "boundary_cut"]
    tracks = {str(row["id"]): row for row in json.loads(MANIFEST.read_text(encoding="utf-8"))["tracks"]}
    by_source: dict[str, list[dict]] = defaultdict(list)
    all_rows_by_source: dict[str, list[dict]] = defaultdict(list)
    for row in label_rows:
        if row.get("source_id") is not None:
            all_rows_by_source[str(row["source_id"])].append(row)
    for row in boundary_rows:
        row["full_vocals"] = str(Path(tracks[str(row["source_id"])]["vocals"]).resolve())
        by_source[str(row["source_id"])].append(row)

    selected: list[dict] = []
    source_stats = {}
    for source_id, rows in sorted(by_source.items()):
        source_path = Path(rows[0]["full_vocals"])
        audio, sr = sf.read(str(source_path), dtype="float32", always_2d=True)
        mono = audio.mean(axis=1).astype(np.float32, copy=False)
        times, rms = frame_rms(mono, sr)
        words = json.loads((WORDS_DIR / f"{source_id}.json").read_text(encoding="utf-8"))["word_rows"]
        model = load_model(source_id)
        built = 0
        for row in sorted(rows, key=lambda item: float(item["start"])):
            first, last = seed_words(words, float(row["start"]), float(row["end"]))
            first, last = sanitize_seed_range(words, first, last, model)
            context_indices = np.flatnonzero((times >= float(row["start"]) - 1.5) & (times <= float(row["end"]) + 1.5))
            speech_level = float(np.quantile(rms[context_indices], 0.80)) if len(context_indices) else float(np.quantile(rms, 0.80))
            tight_left, tight_right = tight_word_range(words, first, last)
            tight = make_candidate(
                row=row,
                words=words,
                left=tight_left,
                right=tight_right,
                times=times,
                rms=rms,
                speech_level=speech_level,
                model=model,
                source_rows=all_rows_by_source[source_id],
                variant="tight_word_complete",
            )
            if tight is None:
                tight = make_candidate(
                    row=row,
                    words=words,
                    left=first,
                    right=last,
                    times=times,
                    rms=rms,
                    speech_level=speech_level,
                    model=model,
                    source_rows=all_rows_by_source[source_id],
                    variant="tight_word_complete",
                )
            if tight is None:
                print(f"[{row['human_key']}] no safe tight candidate", flush=True)
                continue
            candidates = [tight]
            contexts = phrase_candidates(
                row,
                words,
                first,
                last,
                times,
                rms,
                speech_level,
                model,
                all_rows_by_source[source_id],
                tight,
            )
            if contexts:
                candidates.append(contexts[0])
            for candidate in candidates:
                start_sample = max(0, round(float(candidate["start"]) * sr))
                end_sample = min(len(mono), round(float(candidate["end"]) * sr))
                clip = mono[start_sample:end_sample].copy()
                if sr != OUTPUT_SR:
                    import torch
                    import torchaudio

                    clip = torchaudio.functional.resample(torch.from_numpy(clip), sr, OUTPUT_SR).numpy()
                candidate["_audio_clip"] = clip
                candidate["source_sample_rate"] = sr
                selected.append(candidate)
                built += 1
        source_stats[source_id] = {"boundary_origins": len(rows), "rescue_candidates": built}
        del audio, mono

    random.Random(RANDOM_SEED).shuffle(selected)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(selected, 1):
        blind_id = f"B{idx:03d}"
        clip = row.pop("_audio_clip")
        output = AUDIO_DIR / f"{blind_id}.wav"
        sf.write(str(output), clip, OUTPUT_SR, subtype="PCM_16")
        row["blind_id"] = blind_id
        row["audio"] = str(output.resolve())
        row["audio_rel"] = f"audio/{output.name}"
        row["output_duration"] = len(clip) / OUTPUT_SR
        row["wave"] = page_base.waveform(clip)

    origin_counts = defaultdict(int)
    for row in selected:
        origin_counts[str(row["origin_human_key"])] += 1
    audit = {
        "version": "boundary_rescue_all_v5_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "policy": {
            "source": "original full BS-Roformer vocals only",
            "tight": "complete every overlapping ASR word and move both edges to local energy valleys",
            "context": "expand toward ASR/semantic/energy phrase boundaries, maximum 6.5 seconds",
            "speaker": "optional source-local v3 target-cluster score; every result still requires human review",
            "hard_exclusion": "do not overlap any other known non-target, mixed, overlap, variant, or quality interval",
            "dedupe": "after feedback retain at most one accepted rescue per original boundary clip",
            "status": "review-only; not training data",
            "local_resources": RESOURCE_POLICY,
        },
        "boundary_origin_count": len(boundary_rows),
        "sample_count": len(selected),
        "source_stats": source_stats,
        "origin_candidate_counts": dict(sorted(origin_counts.items())),
        "samples": selected,
    }
    public = {
        "version": "boundary_rescue_all_v5_blind",
        "sample_count": len(selected),
        "samples": [
            {"blind_id": row["blind_id"], "audio": row["audio_rel"], "duration_sec": round(row["output_duration"], 3)}
            for row in selected
        ],
    }
    write_json(OUT / "audit_manifest.json", audit)
    write_json(OUT / "blind_manifest.json", public)
    (OUT / "README_试听说明.txt").write_text(
        "这是三轮人工标出的全部切头/切尾素材，从整轨原始 vocals 重新裁切后的救援复核。/n"
        "只要仍有断字、半句话、非伊蕾娜、接入他人或杂音，就勾有错误；完整自然才保持不勾错误。/n"
        "同一原始片段可能有两个盲测版本，最终只保留通过且边界更完整的一条。/n",
        encoding="utf-8",
    )
    (OUT / "开始试听.cmd").write_text('@echo off/r/nstart "" "%~dp0index.html"/r/n', encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUT),
                "boundary_origins": len(boundary_rows),
                "rescue_candidates": len(selected),
                "origins_with_two_variants": sum(value == 2 for value in origin_counts.values()),
                "origins_with_one_variant": sum(value == 1 for value in origin_counts.values()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
