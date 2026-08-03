# -*- coding: utf-8 -*-
"""Build complete, inward-trimmed target phrases from full-track word scores.

The segmentation rules are derived from the 12 successful boundary rescues:
never keep a whole silence slice merely because most of it is the target;
break at every failed word; erode seamless speaker-transition edges inward;
then cut only on ASR word boundaries plus a local energy valley.  The output
is a fresh machine-review pool, not automatically approved training data.
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
import build_human_calibration_v1 as page_base
import full_speaker_diarize_new_v1 as audio_base


MANIFEST = ROOT / "full_track_manifest_v1.json"
BASE_LABELS = ROOT / "human_calibration_combined_v3" / "labels_resolved.json"
RESCUES = ROOT / "human_verified_boundary_rescues_v1" / "audit_manifest.json"
WORD_SCORES = ROOT / "full_source_local_word_scores_v1" / "source_reports"
MODEL_REPORT = ROOT / "source_local_cluster_anchor_v4" / "report.json"
OUT = ROOT / "full_semantic_mining_v1"
STRICT_AUDIO = OUT / "audio"
REVIEW_AUDIO = OUT / "boundary_review_audio"

OUTPUT_SR = 32000
MIN_STRICT_SEC = 1.50
MAX_STRICT_SEC = 6.00
MIN_REVIEW_SEC = 0.80
MAX_REVIEW_SEC = 6.50
MAX_MERGE_GAP_SEC = 0.35
SEAMLESS_GAP_SEC = 0.10
EDGE_EROSION_SEC = 0.18
FADE_SEC = 0.005

UNSAFE_CLASSES = {
    "pure_non_target",
    "mixed",
    "overlap",
    "quality_issue",
    "target_variant_exclude",
}
BAD_START_TOKENS = {
    "ら",
    "けど",
    "ので",
    "て",
    "が",
    "を",
    "に",
    "は",
    "も",
    "と",
    "で",
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8"
    )


def overlaps(start: float, end: float, left: float, right: float) -> bool:
    return max(start, left) < min(end, right)


def overlap_ratio(a: dict, b: dict) -> float:
    if a["source_id"] != b["source_id"]:
        return 0.0
    amount = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    shorter = min(a["end"] - a["start"], b["end"] - b["start"])
    return amount / shorter if shorter > 0 else 0.0


def add_mask(
    masks: dict[str, list[dict]],
    source_id: str,
    start: float,
    end: float,
    reason: str,
) -> None:
    if end > start:
        masks[source_id].append(
            {"start": max(0.0, start), "end": end, "reason": reason}
        )


def build_masks(
    manifest: dict, base_rows: list[dict], rescue_rows: list[dict]
) -> dict[str, list[dict]]:
    masks: dict[str, list[dict]] = defaultdict(list)
    rescues_by_root = {
        row["root_origin_human_key"]: row for row in rescue_rows
    }
    for row in base_rows:
        source_id = str(row["source_id"])
        class_name = str(row["final_class"])
        start = float(row["start"])
        end = float(row["end"])
        human_key = str(row["human_key"])
        if class_name in UNSAFE_CLASSES:
            add_mask(masks, source_id, start - 0.10, end + 0.10, human_key)
        elif class_name == "boundary_cut":
            rescue = rescues_by_root.get(human_key)
            if rescue is None:
                add_mask(
                    masks,
                    source_id,
                    start - 0.10,
                    end + 0.10,
                    f"unrescued_boundary:{human_key}",
                )
            else:
                safe_start = float(rescue["source_start"])
                safe_end = float(rescue["source_end"])
                if start < safe_start:
                    add_mask(
                        masks,
                        source_id,
                        start - 0.10,
                        min(end, safe_start) + 0.025,
                        f"rescued_dirty_left:{human_key}",
                    )
                if end > safe_end:
                    add_mask(
                        masks,
                        source_id,
                        max(start, safe_end) - 0.025,
                        end + 0.10,
                        f"rescued_dirty_right:{human_key}",
                    )

    for item in manifest.get("hard_exclusions", []):
        add_mask(
            masks,
            str(item["source_id"]),
            float(item["start"]) - 0.10,
            float(item["end"]) + 0.10,
            f"manifest:{item['reason']}",
        )

    for basename in manifest.get("excluded_basenames", []):
        source_id = next(
            (str(track["id"]) for track in manifest["tracks"] if str(track["id"]) in basename),
            None,
        )
        numbers = [int(value) for value in re.findall(r"(?<!/d)(/d{7,12})(?!/d)", basename)]
        if source_id and len(numbers) >= 2:
            add_mask(
                masks,
                source_id,
                numbers[0] / 32000.0 - 0.10,
                numbers[1] / 32000.0 + 0.10,
                f"forced_exclusion:{basename}",
            )
    for source_id in masks:
        masks[source_id].sort(key=lambda row: (row["start"], row["end"]))
    return masks


def mask_hits(start: float, end: float, masks: list[dict]) -> list[str]:
    return [
        row["reason"]
        for row in masks
        if overlaps(start, end, float(row["start"]), float(row["end"]))
    ]


def group_pass_words(words: list[dict], allowed: list[bool]) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = []
    for index, passed in enumerate(allowed):
        if not passed:
            if current:
                groups.append(current)
                current = []
            continue
        if current:
            gap = float(words[index]["start"]) - float(words[current[-1]]["end"])
            if gap > MAX_MERGE_GAP_SEC:
                groups.append(current)
                current = []
        current.append(index)
    if current:
        groups.append(current)
    return groups


def edge_erode(run: list[int], words: list[dict], allowed: list[bool]) -> tuple[list[int], dict]:
    original = list(run)
    removed_left: list[int] = []
    removed_right: list[int] = []
    if not run:
        return run, {"removed_left": [], "removed_right": []}

    left_neighbor = run[0] - 1
    if left_neighbor >= 0 and not allowed[left_neighbor]:
        gap = float(words[run[0]]["start"]) - float(words[left_neighbor]["end"])
        if gap < SEAMLESS_GAP_SEC:
            edge = float(words[run[0]]["start"])
            while run and float(words[run[0]]["end"]) - edge < EDGE_EROSION_SEC:
                removed_left.append(run.pop(0))

    if run:
        right_neighbor = run[-1] + 1
        if right_neighbor < len(words) and not allowed[right_neighbor]:
            gap = float(words[right_neighbor]["start"]) - float(words[run[-1]]["end"])
            if gap < SEAMLESS_GAP_SEC:
                edge = float(words[run[-1]]["end"])
                while run and edge - float(words[run[-1]]["start"]) < EDGE_EROSION_SEC:
                    removed_right.append(run.pop())

    while run:
        token = str(words[run[0]].get("text", "")).strip()
        if token not in BAD_START_TOKENS:
            break
        removed_left.append(run.pop(0))

    return run, {
        "original_first": original[0],
        "original_last": original[-1],
        "removed_left": removed_left,
        "removed_right": removed_right,
    }


def split_run(run: list[int], words: list[dict]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    position = 0
    while position < len(run):
        left = run[position]
        endpoints = []
        for offset in range(position, len(run)):
            right = run[offset]
            duration = float(words[right]["end"]) - float(words[left]["start"])
            if duration > MAX_STRICT_SEC - 0.04:
                break
            if duration < MIN_STRICT_SEC - 0.10:
                continue
            next_gap = (
                float(words[right + 1]["start"]) - float(words[right]["end"])
                if right + 1 < len(words)
                else 0.35
            )
            terminal = boundary.terminal_strength(words, right)
            preferred = 1.0 - min(1.0, abs(duration - 4.0) / 4.0)
            score = 2.2 * terminal + min(1.6, max(0.0, next_gap) / 0.18) + 0.55 * preferred
            endpoints.append((score, offset, right, duration))

        if not endpoints:
            break
        # Prefer a real semantic/pause boundary.  With no such boundary, take
        # the farthest legal endpoint and leave it for boundary review.
        semantic = [row for row in endpoints if row[0] >= 1.05]
        chosen = max(semantic or endpoints, key=lambda row: (row[0], row[1]))
        _, chosen_offset, right, _ = chosen
        result.append((left, right))
        position = chosen_offset + 1

    return result


def quality_metrics(audio: np.ndarray, sample_rate: int) -> dict:
    if not len(audio):
        return {"pass": False, "reason": "empty"}
    values = np.asarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(values, dtype=np.float64)) + 1e-12))
    peak = float(np.max(np.abs(values)))
    clipping = float(np.mean(np.abs(values) >= 0.998))
    dc = float(abs(np.mean(values)))
    difference = np.diff(values)
    click_ratio = float(np.max(np.abs(difference)) / max(rms, 1e-6)) if len(difference) else 0.0
    frame = max(1, round(0.020 * sample_rate))
    hop = max(1, round(0.010 * sample_rate))
    frame_values = []
    for start in range(0, max(1, len(values) - frame + 1), hop):
        part = values[start : start + frame]
        if len(part):
            frame_values.append(float(np.sqrt(np.mean(np.square(part, dtype=np.float64)) + 1e-12)))
    frame_values_np = np.asarray(frame_values or [0.0])
    speech_level = float(np.percentile(frame_values_np, 75))
    silence_fraction = float(np.mean(frame_values_np < max(0.0012, speech_level * 0.10)))
    reasons = []
    if rms < 0.0025:
        reasons.append("low_rms")
    if clipping > 0.001:
        reasons.append("clipping")
    if dc > 0.04:
        reasons.append("dc_offset")
    if click_ratio > 42.0:
        reasons.append("impulsive_click")
    if silence_fraction > 0.58:
        reasons.append("too_much_silence")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "rms": rms,
        "peak": peak,
        "clipping_fraction": clipping,
        "dc_absolute": dc,
        "click_ratio": click_ratio,
        "silence_fraction": silence_fraction,
    }


def export_clip(
    source_audio: np.ndarray,
    source_rate: int,
    start: float,
    end: float,
    output: Path,
) -> tuple[np.ndarray, dict]:
    left = max(0, round(start * source_rate))
    right = min(len(source_audio), round(end * source_rate))
    clip = np.asarray(source_audio[left:right], dtype=np.float32)
    clip32 = audio_base.resample_1d(clip, source_rate, OUTPUT_SR)
    fade = min(round(FADE_SEC * OUTPUT_SR), len(clip32) // 4)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        clip32[:fade] *= ramp
        clip32[-fade:] *= ramp[::-1]
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), clip32, OUTPUT_SR, subtype="PCM_16")
    return clip32, quality_metrics(clip, source_rate)


def confirmed_rows(
    base_rows: list[dict], rescue_rows: list[dict], scores_by_source: dict[str, list[dict]]
) -> list[dict]:
    rows = []
    for row in base_rows:
        if row["final_class"] != "target":
            continue
        words = scores_by_source[str(row["source_id"])]
        text = "".join(
            str(word["text"])
            for word in words
            if 0.5 * (float(word["start"]) + float(word["end"])) >= float(row["start"])
            and 0.5 * (float(word["start"]) + float(word["end"])) <= float(row["end"])
        )
        if not text or "/ufffd" in text:
            continue
        rows.append(
            {
                "source_id": str(row["source_id"]),
                "start": float(row["start"]),
                "end": float(row["end"]),
                "duration": float(row["duration"]),
                "audio": str(row["audio"]),
                "text": text,
                "human_key": str(row["human_key"]),
                "provenance": "human_calibration_target",
            }
        )
    for row in rescue_rows:
        rows.append(
            {
                "source_id": str(row["source_id"]),
                "start": float(row["source_start"]),
                "end": float(row["source_end"]),
                "duration": float(row["duration"]),
                "audio": str(row["audio"]),
                "text": str(row["text"]),
                "human_key": str(row["root_origin_human_key"]),
                "provenance": "human_verified_boundary_rescue",
            }
        )
    rows.sort(key=lambda row: (row["source_id"], row["start"], row["end"]))
    return rows


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    STRICT_AUDIO.mkdir(parents=True, exist_ok=True)
    REVIEW_AUDIO.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    base_rows = json.loads(BASE_LABELS.read_text(encoding="utf-8"))["samples"]
    rescue_rows = json.loads(RESCUES.read_text(encoding="utf-8"))["samples"]
    model_report = json.loads(MODEL_REPORT.read_text(encoding="utf-8"))
    model_by_source = {row["source_id"]: row for row in model_report["sources"]}
    masks = build_masks(manifest, base_rows, rescue_rows)

    scores_by_source = {}
    score_documents = {}
    for track in manifest["tracks"]:
        source_id = str(track["id"])
        path = WORD_SCORES / f"{source_id}.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        score_documents[source_id] = document
        scores_by_source[source_id] = document["word_rows"]

    confirmed = confirmed_rows(base_rows, rescue_rows, scores_by_source)
    if len(confirmed) != 57:
        raise RuntimeError(f"Expected 57 human-confirmed rows, got {len(confirmed)}")
    confirmed_paths = [Path(row["audio"]) for row in confirmed]
    if not all(path.is_file() for path in confirmed_paths):
        raise FileNotFoundError("One or more human-confirmed audio paths are missing")
    confirmed_list = ROOT / "human_confirmed_broadcast_v1.list"
    confirmed_list.write_text(
        "/n".join(
            f"{row['audio']}|ireina_human_confirmed|JA|{row['text']}" for row in confirmed
        )
        + "/n",
        encoding="utf-8",
    )

    candidates: list[dict] = []
    source_stats = []
    for track in manifest["tracks"]:
        source_id = str(track["id"])
        source = Path(track["vocals"])
        source_audio, source_rate = audio_base.read_mono(source)
        words = scores_by_source[source_id]
        source_masks = masks.get(source_id, [])
        allowed = []
        word_mask_reasons = []
        for word in words:
            hits = mask_hits(float(word["start"]), float(word["end"]), source_masks)
            probability = float(word.get("probability", 1.0) or 0.0)
            passed = bool(word["strict_speaker_pass"] and probability >= 0.20 and not hits)
            allowed.append(passed)
            word_mask_reasons.append(hits)

        runs = group_pass_words(words, allowed)
        counters = Counter()
        source_candidates = []
        for run in runs:
            eroded, erosion = edge_erode(list(run), words, allowed)
            counters["edge_words_removed"] += len(erosion["removed_left"]) + len(
                erosion["removed_right"]
            )
            if not eroded:
                counters["empty_after_erosion"] += 1
                continue
            for left, right in split_run(eroded, words):
                source_candidates.append(
                    {
                        "source_id": source_id,
                        "source": str(source),
                        "word_start_index": left,
                        "word_end_index": right,
                        "erosion": erosion,
                    }
                )

        # Compute the energy timeline once and finish every candidate.
        energy_times, energy_rms = boundary.frame_rms(source_audio, source_rate)
        positive = energy_rms[energy_rms > 1e-5]
        speech_level = float(np.percentile(positive, 70)) if len(positive) else 1e-5
        finalized = []
        for seed in source_candidates:
            left = seed["word_start_index"]
            right = seed["word_end_index"]
            left_boundary = boundary.start_boundary(
                words, left, energy_times, energy_rms, speech_level
            )
            right_boundary = boundary.end_boundary(
                words, right, energy_times, energy_rms, speech_level
            )
            start = float(left_boundary["time"])
            end = float(right_boundary["time"])

            previous_failed = left > 0 and not allowed[left - 1]
            next_failed = right + 1 < len(words) and not allowed[right + 1]
            if previous_failed and float(words[left]["start"]) - float(words[left - 1]["end"]) < SEAMLESS_GAP_SEC:
                start = max(start, float(words[left]["start"]) - 0.010)
            if next_failed and float(words[right + 1]["start"]) - float(words[right]["end"]) < SEAMLESS_GAP_SEC:
                end = min(end, float(words[right]["end"]) + 0.010)

            duration = end - start
            if duration < MIN_REVIEW_SEC or duration > MAX_REVIEW_SEC:
                counters["duration_rejected"] += 1
                continue
            hits = mask_hits(start, end, source_masks)
            if hits:
                counters["hard_mask_rejected"] += 1
                continue
            selected_words = words[left : right + 1]
            text = "".join(str(word["text"]) for word in selected_words)
            if not text.strip() or "/ufffd" in text or "|" in text:
                counters["text_rejected"] += 1
                continue

            raw_left = max(0, round(start * source_rate))
            raw_right = min(len(source_audio), round(end * source_rate))
            quality = quality_metrics(source_audio[raw_left:raw_right], source_rate)
            start_safe = bool(
                float(left_boundary["gap_sec"]) >= 0.035
                or float(left_boundary["energy_ratio"]) <= 0.42
                or seed["erosion"]["removed_left"]
            )
            end_safe = bool(
                float(right_boundary["gap_sec"]) >= 0.035
                or float(right_boundary["energy_ratio"]) <= 0.42
                or seed["erosion"]["removed_right"]
            )
            probabilities = [float(word.get("probability", 1.0) or 0.0) for word in selected_words]
            strict_duration = MIN_STRICT_SEC <= duration <= MAX_STRICT_SEC
            machine_strict = bool(
                strict_duration
                and start_safe
                and end_safe
                and quality["pass"]
                and np.mean(probabilities) >= 0.62
                and min(probabilities) >= 0.10
            )
            tier = "machine_strict" if machine_strict else "boundary_review"
            candidate = {
                **seed,
                "start": start,
                "end": end,
                "duration": duration,
                "text": text,
                "word_count": len(selected_words),
                "minimum_cluster_margin": min(
                    float(word["cluster_margin"]) for word in selected_words
                ),
                "minimum_context_target_fraction": min(
                    float(word["context_target_fraction"]) for word in selected_words
                ),
                "minimum_anchor_margin": min(
                    float(word["anchor_margin"]) for word in selected_words
                ),
                "mean_asr_probability": float(np.mean(probabilities)),
                "minimum_asr_probability": min(probabilities),
                "left_boundary": left_boundary,
                "right_boundary": right_boundary,
                "start_safe": start_safe,
                "end_safe": end_safe,
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
            finalized.append(candidate)
            counters[tier] += 1

        candidates.extend(finalized)
        source_stats.append(
            {
                "source_id": source_id,
                "word_count": len(words),
                "strict_word_count_before_masks": sum(
                    bool(word["strict_speaker_pass"]) for word in words
                ),
                "strict_word_count_after_masks": sum(allowed),
                "hard_mask_count": len(source_masks),
                "pass_run_count": len(runs),
                "candidate_counts": dict(counters),
            }
        )
        print(
            f"[{source_id}] runs={len(runs)} strict={counters['machine_strict']} "
            f"review={counters['boundary_review']}",
            flush=True,
        )

    # Prefer human-confirmed material whenever a machine segment substantially
    # duplicates the same source interval.
    deduped = []
    duplicate_count = 0
    for candidate in sorted(candidates, key=lambda row: (row["source_id"], row["start"], row["end"])):
        if any(overlap_ratio(candidate, row) >= 0.80 for row in confirmed):
            duplicate_count += 1
            continue
        if deduped and any(overlap_ratio(candidate, row) >= 0.80 for row in deduped[-8:]):
            duplicate_count += 1
            continue
        deduped.append(candidate)
    candidates = deduped

    strict_rows = [row for row in candidates if row["tier"] == "machine_strict"]
    review_rows = [row for row in candidates if row["tier"] == "boundary_review"]

    def materialize(rows: list[dict], folder: Path, prefix: str) -> None:
        tracks = {str(row["id"]): Path(row["vocals"]) for row in manifest["tracks"]}
        loaded = {}
        for index, row in enumerate(rows, start=1):
            source_id = row["source_id"]
            if source_id not in loaded:
                loaded[source_id] = audio_base.read_mono(tracks[source_id])
            source_audio, source_rate = loaded[source_id]
            blind_id = f"{prefix}{index:04d}"
            output = folder / f"{blind_id}.wav"
            clip32, export_quality = export_clip(
                source_audio, source_rate, row["start"], row["end"], output
            )
            row["blind_id"] = blind_id
            row["audio"] = str(output)
            row["audio_rel"] = output.relative_to(OUT).as_posix()
            row["output_duration"] = float(sf.info(str(output)).duration)
            row["wave"] = page_base.waveform(clip32)
            row["export_quality"] = export_quality

    materialize(strict_rows, STRICT_AUDIO, "M")
    materialize(review_rows, REVIEW_AUDIO, "B")

    machine_list = ROOT / "full_semantic_machine_strict_review_v1.list"
    machine_list.write_text(
        "/n".join(
            f"{row['audio']}|ireina_machine_strict_review_v1|JA|{row['text']}"
            for row in strict_rows
        )
        + ("/n" if strict_rows else ""),
        encoding="utf-8",
    )

    audit = {
        "version": "full_semantic_mining_v1_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "source": "original BS-Roformer vocals",
            "word_gate": "every retained word passed source-local CAM++ strict scoring",
            "failed_word": "hard split; never bridged or filled",
            "seamless_transition": "erode at least 0.18 seconds inward before energy-valley cut",
            "hard_masks": "all known non-target/mixed/quality intervals and dirty boundary remnants",
            "status": "machine-selected review candidates; human listening still required",
        },
        "sample_count": len(strict_rows),
        "samples": strict_rows,
    }
    write_json(OUT / "audit_manifest.json", audit)
    write_json(
        OUT / "boundary_review_manifest.json",
        {
            "version": "full_semantic_mining_v1_boundary_review",
            "sample_count": len(review_rows),
            "samples": review_rows,
        },
    )
    write_json(OUT / "hard_exclusion_masks.json", dict(masks))

    confirmed_duration = sum(float(row["duration"]) for row in confirmed)
    strict_duration = sum(float(row["output_duration"]) for row in strict_rows)
    review_duration = sum(float(row["output_duration"]) for row in review_rows)
    report = {
        "version": "full_semantic_mining_v1_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime_sec": time.time() - started,
        "local_resources": RESOURCE_POLICY,
        "coverage": {"expected_sources": 12, "completed_sources": len(source_stats)},
        "human_confirmed": {
            "count": len(confirmed),
            "duration_sec": confirmed_duration,
            "list": str(confirmed_list),
        },
        "machine_strict_review": {
            "count": len(strict_rows),
            "duration_sec": strict_duration,
            "list": str(machine_list),
            "status": "not training-approved until human review",
        },
        "boundary_review": {
            "count": len(review_rows),
            "duration_sec": review_duration,
        },
        "combined_potential_after_interval_dedupe": {
            "count": len(confirmed) + len(strict_rows),
            "duration_sec": confirmed_duration + strict_duration,
            "human_confirmed_count": len(confirmed),
            "machine_pending_count": len(strict_rows),
        },
        "dedupe_removed": duplicate_count,
        "source_stats": source_stats,
        "training_started": False,
        "existing_training_lists_modified": False,
        "score_90_claimed": False,
    }
    write_json(OUT / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
