# -*- coding: utf-8 -*-
"""Rebuild all 12 tracks with calibrated soft-fail hysteresis and whole phrases.

The v1 rule treated every non-strict short word as a speaker switch and then
eroded 180 ms.  Human review showed that this systematically cut target words.
V2 keeps only calibrated strong/soft/edge-safe words, hard-breaks sustained
negative or human-masked regions, and requires a plausible complete phrase
boundary before a candidate can enter the strict review pool.
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
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

import build_boundary_rescue_all_v1 as boundary
import build_full_semantic_candidates_v1 as v1
import build_human_calibration_v1 as page_base
import full_speaker_diarize_new_v1 as audio_base


MANIFEST = ROOT / "full_track_manifest_v1.json"
BASE_LABELS = ROOT / "human_calibration_combined_v3" / "labels_resolved.json"
LEGACY_RESCUES = ROOT / "human_verified_boundary_rescues_v1" / "audit_manifest.json"
WORD_SCORES = ROOT / "full_source_local_word_scores_v1" / "source_reports"
MODEL_REPORT = ROOT / "source_local_cluster_anchor_v4" / "report.json"
ACCEPTANCE_LABELS = ROOT / "human_full_semantic_acceptance_v1" / "labels_resolved.json"
ACCEPTANCE_PRIVATE = ROOT / "full_semantic_acceptance_v1" / "private_manifest.json"
SEMANTIC_RESCUES = ROOT / "human_verified_semantic_rescues_v3" / "audit_manifest.json"

EXTRA_MASK_FILES = (
    ROOT / "human_full_semantic_acceptance_v1" / "hard_exclusion_intervals.json",
    ROOT / "human_full_semantic_boundary_rescue_v1" / "hard_exclusion_intervals.json",
    ROOT / "human_verified_semantic_rescues_v3" / "boundary_exclusion_intervals.json",
    ROOT / "human_verified_semantic_rescues_v3" / "quality_exclusion_intervals.json",
)

OUT = ROOT / "full_semantic_mining_v2"
STRICT_AUDIO = OUT / "audio"
REVIEW_AUDIO = OUT / "semantic_review_audio"
MACHINE_LIST = ROOT / "full_semantic_machine_strict_review_v2.list"
HUMAN_LIST = ROOT / "human_confirmed_broadcast_v2.list"
HUMAN_TRAIN_SAFE_LIST = ROOT / "human_confirmed_broadcast_v2_train_1p5_6s.list"

OUTPUT_SR = 32000
FADE_SEC = 0.002
MIN_STRICT_SEC = 1.50
MAX_STRICT_SEC = 6.00
MIN_REVIEW_SEC = 0.80
MAX_REVIEW_SEC = 6.25
MAX_WORD_GAP_SEC = 0.35
SEMANTIC_GAP_SEC = 0.18
SEAMLESS_GAP_SEC = 0.10

HIGH_RISK_SOURCES = {"1405372904-1-192", "306672160-1-192"}

BAD_START_TOKENS = {
    "ら", "けど", "ので", "て", "が", "を", "に", "は", "も", "と", "で",
    "ば", "から", "れる", "さ", "し", "く", "の", "ん", "り", "え",
}
BAD_END_TOKENS = {
    "を", "が", "に", "は", "も", "と", "で", "て", "し", "さ", "く", "の",
    "へ", "や", "か", "ば", "から", "ので", "けど", "なら", "たり", "って",
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8"
    )


def add_mask(masks: dict[str, list[dict]], row: dict, prefix: str) -> None:
    source_id = str(row["source_id"])
    start = float(row["start"])
    end = float(row["end"])
    if end > start:
        masks[source_id].append(
            {
                "start": max(0.0, start),
                "end": end,
                "reason": f"{prefix}:{row.get('reason', 'interval')}",
            }
        )


def build_masks(manifest: dict, base_rows: list[dict], rescue_rows: list[dict]) -> dict[str, list[dict]]:
    masks = v1.build_masks(manifest, base_rows, rescue_rows)
    for path in EXTRA_MASK_FILES:
        document = json.loads(path.read_text(encoding="utf-8"))
        for row in document["intervals"]:
            add_mask(masks, row, path.parent.name)

    # Acceptance samples explicitly marked for quality stay out of new output,
    # but they are not speaker negatives.  A004's old quality note was
    # superseded by a later complete-phrase pass and is deliberately omitted.
    acceptance = json.loads(ACCEPTANCE_LABELS.read_text(encoding="utf-8"))["samples"]
    for row in acceptance:
        if row["final_class"] == "quality_issue":
            add_mask(
                masks,
                {
                    "source_id": row["source_id"],
                    "start": float(row["start"]),
                    "end": float(row["end"]),
                    "reason": f"acceptance_quality:{row['blind_id']}",
                },
                "quality",
            )
    for source_id in masks:
        masks[source_id].sort(key=lambda item: (float(item["start"]), float(item["end"])))
    return masks


def overlap_ratio(a: dict, b: dict) -> float:
    if str(a["source_id"]) != str(b["source_id"]):
        return 0.0
    amount = max(0.0, min(float(a["end"]), float(b["end"])) - max(float(a["start"]), float(b["start"])))
    shorter = min(float(a["end"]) - float(a["start"]), float(b["end"]) - float(b["start"]))
    return amount / shorter if shorter > 0 else 0.0


def load_human_confirmed(
    base_rows: list[dict], legacy_rows: list[dict], scores_by_source: dict[str, list[dict]]
) -> tuple[list[dict], dict]:
    legacy = v1.confirmed_rows(base_rows, legacy_rows, scores_by_source)
    rows = []
    for row in legacy:
        rows.append({**row, "priority": 1, "human_source": row["provenance"]})

    acceptance = json.loads(ACCEPTANCE_LABELS.read_text(encoding="utf-8"))["samples"]
    for row in acceptance:
        if row["final_class"] != "target" or row["role"] != "machine_candidate":
            continue
        audio = ROOT / "full_semantic_acceptance_v1" / "audio" / f"{row['blind_id']}.wav"
        rows.append(
            {
                "source_id": str(row["source_id"]),
                "start": float(row["start"]),
                "end": float(row["end"]),
                "duration": float(row["duration"]),
                "audio": str(audio.resolve()),
                "text": str(row["text"]),
                "human_key": str(row["human_key"]),
                "provenance": "human_full_semantic_acceptance_direct_target",
                "human_source": "human_full_semantic_acceptance_direct_target",
                "priority": 3,
            }
        )

    semantic = json.loads(SEMANTIC_RESCUES.read_text(encoding="utf-8"))["samples"]
    for row in semantic:
        rows.append(
            {
                "source_id": str(row["source_id"]),
                "start": float(row["start"]),
                "end": float(row["end"]),
                "duration": float(row["duration"]),
                "audio": str(row["audio"]),
                "text": str(row["text"]),
                "human_key": str(row.get("human_key") or row["final_id"]),
                "provenance": "human_verified_semantic_rescue_v3",
                "human_source": "human_verified_semantic_rescue_v3",
                "priority": 4,
            }
        )

    missing = [row["audio"] for row in rows if not Path(row["audio"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing human-confirmed audio: {missing[:3]}")

    kept = []
    removed = []
    for row in sorted(rows, key=lambda item: (-int(item["priority"]), item["source_id"], float(item["start"]))):
        duplicate = next((other for other in kept if overlap_ratio(row, other) >= 0.80), None)
        if duplicate is not None:
            removed.append(
                {
                    "dropped": row["human_key"],
                    "kept": duplicate["human_key"],
                    "source_id": row["source_id"],
                    "overlap_ratio": overlap_ratio(row, duplicate),
                }
            )
            continue
        kept.append(row)
    kept.sort(key=lambda item: (item["source_id"], float(item["start"]), float(item["end"])))
    for row in kept:
        row.pop("priority", None)
    return kept, {"input": len(rows), "kept": len(kept), "removed": removed}


def word_features(word: dict) -> dict:
    probability = float(word.get("probability", 1.0) or 0.0)
    context = float(word["context_target_fraction"])
    cluster = float(word["cluster_margin"])
    anchor = float(word["anchor_margin"])
    target = float(word["target_center_similarity"])
    delta = target - float(word["non_target_center_similarity"])
    strict = bool(word["strict_speaker_pass"] and probability >= 0.20)
    soft = bool(
        probability >= 0.10
        and context >= 0.50
        and cluster >= -0.060
        and anchor >= 0.020
        and delta >= -0.030
    )
    edge = bool(
        probability >= 0.10
        and cluster >= -0.040
        and anchor >= 0.030
        and delta >= -0.040
        and target >= 0.580
    )
    hard_negative = bool(
        context <= 0.00
        and cluster <= -0.040
        and anchor <= 0.030
        and delta <= 0.000
    )
    return {
        "probability": probability,
        "context": context,
        "cluster": cluster,
        "anchor": anchor,
        "target": target,
        "delta": delta,
        "strict": strict,
        "soft": soft,
        "edge": edge,
        "hard_negative": hard_negative,
    }


def classify_words(words: list[dict], masks: list[dict]) -> list[dict]:
    states = []
    for index, word in enumerate(words):
        hits = v1.mask_hits(float(word["start"]), float(word["end"]), masks)
        features = word_features(word)
        if hits:
            state = "mask"
        elif features["hard_negative"]:
            state = "hard_negative"
        elif features["strict"]:
            state = "strong"
        elif features["soft"]:
            state = "soft"
        elif features["edge"]:
            state = "edge_soft"
        else:
            state = "unknown"
        states.append(
            {
                "word_index": index,
                "state": state,
                "eligible": state in {"strong", "soft", "edge_soft"},
                "mask_hits": hits,
                **features,
            }
        )
    return states


def eligible_runs(words: list[dict], states: list[dict]) -> list[list[int]]:
    runs = []
    current = []
    for index, state in enumerate(states):
        if not state["eligible"]:
            if current:
                runs.append(current)
                current = []
            continue
        if current:
            gap = float(words[index]["start"]) - float(words[current[-1]]["end"])
            if gap > MAX_WORD_GAP_SEC:
                runs.append(current)
                current = []
        current.append(index)
    if current:
        runs.append(current)
    return runs


def punctuation_end(text: str) -> bool:
    return bool(re.search(r"[。．.!！?？]$", text.strip()))


def token_is_bad_start(token: str) -> bool:
    token = token.strip("、,。.!！?？ ")
    return not token or token in BAD_START_TOKENS


def token_is_bad_end(token: str) -> bool:
    token = token.strip("、,。.!！?？ ")
    return not token or token in BAD_END_TOKENS


def endpoint_metrics(words: list[dict], index: int) -> dict:
    gap = (
        max(0.0, float(words[index + 1]["start"]) - float(words[index]["end"]))
        if index + 1 < len(words)
        else 0.40
    )
    terminal = float(boundary.terminal_strength(words, index))
    text = str(words[index].get("text") or "")
    punct = punctuation_end(text)
    complete = bool(terminal >= 0.55 or gap >= SEMANTIC_GAP_SEC or punct)
    return {"gap": gap, "terminal": terminal, "punctuation": punct, "complete": complete}


def split_run(run: list[int], words: list[dict], states: list[dict]) -> list[dict]:
    result = []
    position = 0
    while position < len(run):
        left = run[position]
        options = []
        for offset in range(position, len(run)):
            right = run[offset]
            duration = float(words[right]["end"]) - float(words[left]["start"])
            if duration > MAX_REVIEW_SEC - 0.02:
                break
            if duration < MIN_REVIEW_SEC - 0.05:
                continue
            endpoint = endpoint_metrics(words, right)
            length_preference = 1.0 - min(1.0, abs(duration - 4.0) / 4.0)
            score = (
                3.2 * float(endpoint["complete"])
                + 1.8 * endpoint["terminal"]
                + min(1.8, endpoint["gap"] / 0.16)
                + 0.55 * length_preference
                + 0.25 * (offset - position) / max(1, len(run))
            )
            if token_is_bad_end(str(words[right].get("text") or "")):
                score -= 1.7
            options.append((score, offset, right, duration, endpoint))
        if not options:
            break

        strict_options = [
            item
            for item in options
            if item[3] >= MIN_STRICT_SEC - 0.05 and item[4]["complete"]
        ]
        if strict_options:
            chosen = max(strict_options, key=lambda item: (item[0], item[1]))
            semantic_complete = True
        else:
            review_options = [item for item in options if item[3] >= MIN_REVIEW_SEC]
            if not review_options:
                break
            chosen = max(review_options, key=lambda item: (item[0], item[1]))
            semantic_complete = False
        _, chosen_offset, right, _, endpoint = chosen
        result.append(
            {
                "word_start_index": left,
                "word_end_index": right,
                "semantic_complete_end": semantic_complete,
                "endpoint": endpoint,
            }
        )
        position = chosen_offset + 1
    return result


def nearest_mask_clamps(start: float, end: float, masks: list[dict]) -> tuple[float, float]:
    for row in masks:
        left = float(row["start"])
        right = float(row["end"])
        if start < right <= end:
            start = max(start, right)
        if start <= left < end:
            end = min(end, left)
    return start, end


def candidate_bounds(
    words: list[dict],
    states: list[dict],
    left: int,
    right: int,
    energy_times: np.ndarray,
    energy_rms: np.ndarray,
    speech_level: float,
    masks: list[dict],
) -> tuple[float, float, dict, dict, dict]:
    left_boundary = boundary.start_boundary(words, left, energy_times, energy_rms, speech_level)
    right_boundary = boundary.end_boundary(words, right, energy_times, energy_rms, speech_level)
    start = float(left_boundary["time"])
    end = float(right_boundary["time"])
    previous_state = states[left - 1]["state"] if left > 0 else "track_start"
    next_state = states[right + 1]["state"] if right + 1 < len(states) else "track_end"
    left_gap = float(left_boundary["gap_sec"])
    right_gap = float(right_boundary["gap_sec"])
    left_mode = "energy_valley"
    right_mode = "energy_valley"

    if previous_state in {"mask", "hard_negative"} and left_gap < SEAMLESS_GAP_SEC:
        # Human-calibrated A021: exact ASR onset retained the preceding vowel,
        # while +20 ms kept the target word complete.
        start = max(start, float(words[left]["start"]) + 0.020)
        left_mode = "human_calibrated_20ms_hysteresis"
    elif previous_state == "unknown" and left_gap < SEAMLESS_GAP_SEC:
        start = max(start, float(words[left]["start"]) - 0.015)
        left_mode = "unknown_neighbor_tight_word_edge"

    if next_state in {"mask", "hard_negative"} and right_gap < SEAMLESS_GAP_SEC:
        end = min(end, float(words[right]["end"]))
        right_mode = "hard_transition_exact_word_end"
    elif next_state == "unknown" and right_gap < SEAMLESS_GAP_SEC:
        end = min(end, float(words[right]["end"]) + 0.015)
        right_mode = "unknown_neighbor_tight_word_edge"

    start, end = nearest_mask_clamps(start, end, masks)
    modes = {
        "left": left_mode,
        "right": right_mode,
        "previous_state": previous_state,
        "next_state": next_state,
    }
    return start, end, left_boundary, right_boundary, modes


def export_clip(
    source_audio: np.ndarray,
    source_rate: int,
    start: float,
    end: float,
    output: Path,
) -> tuple[np.ndarray, dict]:
    left = max(0, round(start * source_rate))
    right = min(len(source_audio), round(end * source_rate))
    raw = np.asarray(source_audio[left:right], dtype=np.float32)
    clip = audio_base.resample_1d(raw, source_rate, OUTPUT_SR)
    fade = min(round(FADE_SEC * OUTPUT_SR), len(clip) // 4)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        clip[:fade] *= ramp
        clip[-fade:] *= ramp[::-1]
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), clip, OUTPUT_SR, subtype="PCM_16")
    return clip, v1.quality_metrics(raw, source_rate)


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    STRICT_AUDIO.mkdir(parents=True, exist_ok=True)
    REVIEW_AUDIO.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    base_rows = json.loads(BASE_LABELS.read_text(encoding="utf-8"))["samples"]
    legacy_rows = json.loads(LEGACY_RESCUES.read_text(encoding="utf-8"))["samples"]
    model_report = json.loads(MODEL_REPORT.read_text(encoding="utf-8"))
    model_by_source = {str(row["source_id"]): row for row in model_report["sources"]}
    masks = build_masks(manifest, base_rows, legacy_rows)

    scores_by_source = {}
    for track in manifest["tracks"]:
        source_id = str(track["id"])
        document = json.loads((WORD_SCORES / f"{source_id}.json").read_text(encoding="utf-8"))
        scores_by_source[source_id] = document["word_rows"]

    confirmed, human_dedupe = load_human_confirmed(base_rows, legacy_rows, scores_by_source)
    HUMAN_LIST.write_text(
        "/n".join(
            f"{row['audio']}|ireina_human_confirmed_v2|JA|{row['text']}" for row in confirmed
        )
        + "/n",
        encoding="utf-8",
    )
    human_train_safe = [row for row in confirmed if 1.5 <= float(row["duration"]) <= 6.0]
    HUMAN_TRAIN_SAFE_LIST.write_text(
        "/n".join(
            f"{row['audio']}|ireina_human_confirmed_v2|JA|{row['text']}" for row in human_train_safe
        )
        + "/n",
        encoding="utf-8",
    )

    all_candidates = []
    source_stats = []
    word_state_report = {}
    for track in manifest["tracks"]:
        source_id = str(track["id"])
        source = Path(track["vocals"])
        source_audio, source_rate = audio_base.read_mono(source)
        words = scores_by_source[source_id]
        source_masks = masks.get(source_id, [])
        states = classify_words(words, source_masks)
        runs = eligible_runs(words, states)
        energy_times, energy_rms = boundary.frame_rms(source_audio, source_rate)
        positive_energy = energy_rms[energy_rms > 1e-5]
        speech_level = float(np.percentile(positive_energy, 70)) if len(positive_energy) else 1e-5
        counters = Counter()
        source_candidates = []

        for run in runs:
            if not any(states[index]["state"] == "strong" for index in run):
                counters["run_without_strong_anchor"] += 1
                continue
            for seed in split_run(run, words, states):
                left = int(seed["word_start_index"])
                right = int(seed["word_end_index"])
                start, end, left_boundary, right_boundary, modes = candidate_bounds(
                    words,
                    states,
                    left,
                    right,
                    energy_times,
                    energy_rms,
                    speech_level,
                    source_masks,
                )
                duration = end - start
                if duration < MIN_REVIEW_SEC or duration > MAX_REVIEW_SEC:
                    counters["duration_rejected"] += 1
                    continue
                hits = v1.mask_hits(start, end, source_masks)
                if hits:
                    counters["hard_mask_rejected"] += 1
                    continue
                selected_words = words[left : right + 1]
                selected_states = states[left : right + 1]
                text = "".join(str(word.get("text") or "") for word in selected_words)
                if not text.strip() or "/ufffd" in text or "|" in text:
                    counters["text_rejected"] += 1
                    continue
                probabilities = [float(item["probability"]) for item in selected_states]
                strong_fraction = float(np.mean([item["state"] == "strong" for item in selected_states]))
                mean_delta = float(np.mean([item["delta"] for item in selected_states]))
                mean_anchor = float(np.mean([item["anchor"] for item in selected_states]))
                first_token = str(selected_words[0].get("text") or "")
                last_token = str(selected_words[-1].get("text") or "")
                semantic_start = not token_is_bad_start(first_token)
                semantic_end = bool(seed["semantic_complete_end"] and not token_is_bad_end(last_token))
                raw_left = max(0, round(start * source_rate))
                raw_right = min(len(source_audio), round(end * source_rate))
                quality = v1.quality_metrics(source_audio[raw_left:raw_right], source_rate)
                strict_duration = MIN_STRICT_SEC <= duration <= MAX_STRICT_SEC
                strong_minimum = 0.60 if source_id in HIGH_RISK_SOURCES else 0.50
                delta_minimum = 0.035 if source_id in HIGH_RISK_SOURCES else 0.020
                anchor_minimum = 0.100 if source_id in HIGH_RISK_SOURCES else 0.080
                machine_strict = bool(
                    strict_duration
                    and semantic_start
                    and semantic_end
                    and quality["pass"]
                    and strong_fraction >= strong_minimum
                    and mean_delta >= delta_minimum
                    and mean_anchor >= anchor_minimum
                    and float(np.mean(probabilities)) >= 0.62
                    and min(probabilities) >= 0.10
                )
                tier = "machine_strict_v2" if machine_strict else "semantic_review_v2"
                candidate = {
                    "source_id": source_id,
                    "source": str(source),
                    "word_start_index": left,
                    "word_end_index": right,
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "text": text,
                    "word_count": len(selected_words),
                    "word_state_counts": dict(Counter(item["state"] for item in selected_states)),
                    "strong_word_fraction": strong_fraction,
                    "mean_target_center_delta": mean_delta,
                    "minimum_target_center_delta": min(item["delta"] for item in selected_states),
                    "mean_anchor_margin": mean_anchor,
                    "minimum_anchor_margin": min(item["anchor"] for item in selected_states),
                    "mean_asr_probability": float(np.mean(probabilities)),
                    "minimum_asr_probability": min(probabilities),
                    "semantic_start": semantic_start,
                    "semantic_complete_end": semantic_end,
                    "endpoint": seed["endpoint"],
                    "left_boundary": left_boundary,
                    "right_boundary": right_boundary,
                    "boundary_modes": modes,
                    "quality": quality,
                    "tier": tier,
                    "hard_mask_hits": hits,
                    "source_model_known_unsafe_rejected": bool(
                        model_by_source[source_id]["selected"]["negative_rejected"]
                        == model_by_source[source_id]["selected"]["negative_total"]
                        and model_by_source[source_id]["selected"]["mixed_rejected"]
                        == model_by_source[source_id]["selected"]["mixed_total"]
                    ),
                }
                source_candidates.append(candidate)
                counters[tier] += 1

        all_candidates.extend(source_candidates)
        state_counts = Counter(row["state"] for row in states)
        word_state_report[source_id] = dict(state_counts)
        source_stats.append(
            {
                "source_id": source_id,
                "word_count": len(words),
                "state_counts": dict(state_counts),
                "eligible_run_count": len(runs),
                "candidate_counts": dict(counters),
                "hard_mask_count": len(source_masks),
            }
        )
        print(
            f"[{source_id}] runs={len(runs)} strict={counters['machine_strict_v2']} "
            f"review={counters['semantic_review_v2']} states={dict(state_counts)}",
            flush=True,
        )
        del source_audio, energy_times, energy_rms

    # Human-heard clips always win.  Machine duplicates prefer semantic
    # completeness, stronger speaker evidence, and longer useful duration.
    duplicate_human = 0
    candidate_pool = []
    for row in sorted(
        all_candidates,
        key=lambda item: (
            item["source_id"],
            float(item["start"]),
            -float(item["strong_word_fraction"]),
            -float(item["duration"]),
        ),
    ):
        if any(overlap_ratio(row, human) >= 0.80 for human in confirmed):
            duplicate_human += 1
            continue
        duplicate = next(
            (other for other in candidate_pool[-12:] if overlap_ratio(row, other) >= 0.80),
            None,
        )
        if duplicate is not None:
            continue
        candidate_pool.append(row)

    strict_rows = [row for row in candidate_pool if row["tier"] == "machine_strict_v2"]
    review_rows = [row for row in candidate_pool if row["tier"] == "semantic_review_v2"]

    tracks = {str(row["id"]): Path(row["vocals"]) for row in manifest["tracks"]}

    def materialize(rows: list[dict], folder: Path, prefix: str) -> None:
        loaded = {}
        for index, row in enumerate(rows, 1):
            source_id = row["source_id"]
            if source_id not in loaded:
                loaded[source_id] = audio_base.read_mono(tracks[source_id])
            source_audio, source_rate = loaded[source_id]
            blind_id = f"{prefix}{index:04d}"
            output = folder / f"{blind_id}.wav"
            clip, export_quality = export_clip(
                source_audio, source_rate, float(row["start"]), float(row["end"]), output
            )
            info = sf.info(str(output))
            if info.samplerate != OUTPUT_SR or info.channels != 1 or info.subtype != "PCM_16":
                raise RuntimeError(f"Invalid export: {output}")
            row["blind_id"] = blind_id
            row["audio"] = str(output.resolve())
            row["audio_rel"] = output.relative_to(OUT).as_posix()
            row["output_duration"] = float(info.duration)
            row["wave"] = page_base.waveform(clip)
            row["export_quality"] = export_quality

    materialize(strict_rows, STRICT_AUDIO, "V2M")
    materialize(review_rows, REVIEW_AUDIO, "V2R")

    MACHINE_LIST.write_text(
        "/n".join(
            f"{row['audio']}|ireina_machine_strict_review_v2|JA|{row['text']}" for row in strict_rows
        )
        + ("/n" if strict_rows else ""),
        encoding="utf-8",
    )

    audit = {
        "version": "full_semantic_mining_v2_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "source": "original BS-Roformer vocals",
            "calibration": "988 human-target words vs 510 pure-negative words",
            "word_gate": "strong + calibrated soft_balanced/edge_soft; no arbitrary unknown-word bridge",
            "soft_balanced_rates": "89.37% human target / 0.20% pure negative",
            "hard_break": "human masks or calibrated sustained-negative word",
            "seamless_start": "human-calibrated +20ms hysteresis after hard transition",
            "semantic_gate": "complete terminal/gap plus non-fragment start/end tokens",
            "status": "machine-selected review candidates; human listening still required",
        },
        "sample_count": len(strict_rows),
        "samples": strict_rows,
    }
    write_json(OUT / "audit_manifest.json", audit)
    write_json(
        OUT / "semantic_review_manifest.json",
        {
            "version": "full_semantic_mining_v2_review",
            "sample_count": len(review_rows),
            "samples": review_rows,
        },
    )
    write_json(OUT / "hard_exclusion_masks.json", dict(masks))
    write_json(OUT / "word_state_report.json", word_state_report)

    strict_duration = sum(float(row["output_duration"]) for row in strict_rows)
    review_duration = sum(float(row["output_duration"]) for row in review_rows)
    human_duration = sum(float(row["duration"]) for row in confirmed)
    human_train_duration = sum(float(row["duration"]) for row in human_train_safe)
    report = {
        "version": "full_semantic_mining_v2_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime_sec": time.time() - started,
        "local_resources": RESOURCE_POLICY,
        "coverage": {"expected_sources": 12, "completed_sources": len(source_stats)},
        "calibration": json.loads(
            (ROOT / "full_semantic_word_state_calibration_v2.json").read_text(encoding="utf-8")
        )["condition_rates"],
        "human_confirmed": {
            "count": len(confirmed),
            "duration_sec": human_duration,
            "train_1p5_6s_count": len(human_train_safe),
            "train_1p5_6s_duration_sec": human_train_duration,
            "list": str(HUMAN_LIST),
            "train_safe_list": str(HUMAN_TRAIN_SAFE_LIST),
            "dedupe": human_dedupe,
        },
        "machine_strict_review": {
            "count": len(strict_rows),
            "duration_sec": strict_duration,
            "list": str(MACHINE_LIST),
            "status": "not training-approved until human review",
        },
        "semantic_review": {"count": len(review_rows), "duration_sec": review_duration},
        "combined_potential_after_interval_dedupe": {
            "count": len(human_train_safe) + len(strict_rows),
            "duration_sec": human_train_duration + strict_duration,
            "human_confirmed_count": len(human_train_safe),
            "machine_pending_count": len(strict_rows),
        },
        "machine_duplicates_removed_by_human": duplicate_human,
        "source_stats": source_stats,
        "training_started": False,
        "existing_training_lists_modified": False,
        "score_90_claimed": False,
    }
    write_json(OUT / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
