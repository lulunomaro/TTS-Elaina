# -*- coding: utf-8 -*-
"""Build complete-phrase anime candidates from human-anchored word scores.

Outputs remain machine review material. No clip is added to a training list
until a later human acceptance pass resolves identity, boundaries, and quality.
"""
from __future__ import annotations

import json
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

import build_full_semantic_candidates_v1 as quality_base
import build_full_semantic_candidates_v2 as semantic
import build_human_calibration_v1 as page_base
import full_speaker_diarize_new_v1 as audio_base


MANIFEST = ROOT / "anime_batch_v1" / "manifest.json"
WORD_SCORES = ROOT / "anime_mining_v1" / "source_local_word_scores_v1" / "source_reports"
MODEL_REPORT = ROOT / "anime_mining_v1" / "source_local_cluster_anchor_v1" / "report.json"
ERES_REPORT = ROOT / "anime_mining_v1" / "eres_scan" / "interval_map.json"
HUMAN_LABELS = ROOT / "human_anime_speaker_calibration_v3" / "labels_resolved.json"
HARD_MASKS = ROOT / "human_anime_speaker_calibration_v3" / "hard_exclusion_intervals.json"

OUT = ROOT / "anime_semantic_mining_v1"
STRICT_AUDIO = OUT / "audio"
REVIEW_AUDIO = OUT / "semantic_review_audio"
MACHINE_LIST = ROOT / "anime_semantic_machine_strict_review_v1.list"

OUTPUT_SR = 32000
MIN_STRICT_SEC = 1.50
MAX_STRICT_SEC = 6.00
MAX_REVIEW_EXPORT_PER_SOURCE = 80


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8"
    )
    temporary.replace(path)


def load_masks() -> dict[str, list[dict]]:
    document = json.loads(HARD_MASKS.read_text(encoding="utf-8"))
    masks: dict[str, list[dict]] = defaultdict(list)
    for row in document["intervals"]:
        masks[str(row["source_id"])].append(
            {
                "start": float(row["start"]),
                "end": float(row["end"]),
                "reason": str(row["reason"]),
            }
        )
    for rows in masks.values():
        rows.sort(key=lambda row: (row["start"], row["end"]))
    return dict(masks)


def overlap_ratio(a: dict, b: dict) -> float:
    if str(a["source_id"]) != str(b["source_id"]):
        return 0.0
    overlap = max(
        0.0,
        min(float(a["end"]), float(b["end"]))
        - max(float(a["start"]), float(b["start"])),
    )
    shorter = min(
        float(a["end"]) - float(a["start"]),
        float(b["end"]) - float(b["start"]),
    )
    return overlap / shorter if shorter > 0 else 0.0


def eres2net_metrics(scan: dict, start: float, end: float) -> dict:
    if "_anime_runtime_thresholds" not in scan:
        all_centers = np.asarray(
            [float(item["center"]) for item in scan.get("windows", [])],
            dtype=np.float32,
        )
        all_margins = np.asarray(
            [float(item["margin"]) for item in scan.get("windows", [])],
            dtype=np.float32,
        )
        scan["_anime_runtime_thresholds"] = {
            "center_q70": float(np.quantile(all_centers, 0.70)),
            "margin_q50": float(np.quantile(all_margins, 0.50)),
            "margin_q25": float(np.quantile(all_margins, 0.25)),
        }
    thresholds = scan["_anime_runtime_thresholds"]
    windows = [
        item
        for item in scan.get("windows", [])
        if start <= float(item["center_sec"]) <= end
    ]
    if not windows:
        windows = [
            item
            for item in scan.get("windows", [])
            if max(start, float(item["start_sec"]))
            < min(end, float(item["end_sec"]))
        ]
    centers = [float(item["center"]) for item in windows]
    margins = [float(item["margin"]) for item in windows]
    median_center = float(np.median(centers)) if centers else -1.0
    median_margin = float(np.median(margins)) if margins else -1.0
    positive_fraction = (
        float(np.mean([value > 0.0 for value in margins])) if margins else 0.0
    )
    return {
        "window_count": len(windows),
        "median_target_center": median_center,
        "median_target_negative_margin": median_margin,
        "positive_margin_fraction": positive_fraction,
        "active_fraction": (
            float(np.mean([bool(item["active"]) for item in windows]))
            if windows
            else 0.0
        ),
        "support": bool(
            len(windows) >= 2
            and median_center >= max(0.56, thresholds["center_q70"])
            and median_margin >= max(-0.005, thresholds["margin_q50"])
            and positive_fraction >= 0.60
        ),
        "hard_negative_warning": bool(
            not windows
            or median_margin < thresholds["margin_q25"]
            or positive_fraction < 0.35
        ),
        "source_relative_thresholds": thresholds,
    }


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    STRICT_AUDIO.mkdir(parents=True, exist_ok=True)
    REVIEW_AUDIO.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tracks = manifest["tracks"]
    if len(tracks) != 12:
        raise RuntimeError(f"Expected 12 anime episodes, got {len(tracks)}")
    labels = json.loads(HUMAN_LABELS.read_text(encoding="utf-8"))["samples"]
    masks = load_masks()
    model_report = json.loads(MODEL_REPORT.read_text(encoding="utf-8"))
    model_by_source = {str(row["source_id"]): row for row in model_report["sources"]}
    eres_report = json.loads(ERES_REPORT.read_text(encoding="utf-8"))
    eres_by_source = {
        str(row["source_id"]): row for row in eres_report["sources"]
    }

    scores_by_source = {}
    for track in tracks:
        source_id = str(track["id"])
        document = json.loads(
            (WORD_SCORES / f"{source_id}.json").read_text(encoding="utf-8")
        )
        scores_by_source[source_id] = document["word_rows"]

    all_candidates = []
    source_stats = []
    word_state_report = {}
    for track in tracks:
        source_id = str(track["id"])
        source = Path(track["vocals"])
        source_audio, source_rate = audio_base.read_mono(source)
        words = scores_by_source[source_id]
        source_masks = masks.get(source_id, [])
        states = semantic.classify_words(words, source_masks)
        runs = semantic.eligible_runs(words, states)
        energy_times, energy_rms = semantic.boundary.frame_rms(source_audio, source_rate)
        positive_energy = energy_rms[energy_rms > 1e-5]
        speech_level = (
            float(np.percentile(positive_energy, 70)) if len(positive_energy) else 1e-5
        )
        selected_model = model_by_source[source_id]["selected"]
        model_calibration_pass = bool(
            selected_model["target_retained"] == selected_model["target_total"]
            and selected_model["negative_rejected"] == selected_model["negative_total"]
            and selected_model["mixed_rejected"] == selected_model["mixed_total"]
        )
        counters = Counter()
        source_candidates = []

        for run in runs:
            if not any(states[index]["state"] == "strong" for index in run):
                counters["run_without_strong_anchor"] += 1
                continue
            for seed in semantic.split_run(run, words, states):
                left = int(seed["word_start_index"])
                right = int(seed["word_end_index"])
                start, end, left_boundary, right_boundary, modes = semantic.candidate_bounds(
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
                if duration < semantic.MIN_REVIEW_SEC or duration > semantic.MAX_REVIEW_SEC:
                    counters["duration_rejected"] += 1
                    continue
                hits = quality_base.mask_hits(start, end, source_masks)
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
                strong_fraction = float(
                    np.mean([item["state"] == "strong" for item in selected_states])
                )
                mean_delta = float(np.mean([item["delta"] for item in selected_states]))
                mean_anchor = float(np.mean([item["anchor"] for item in selected_states]))
                direct_target_fraction = float(
                    np.mean([bool(word["direct_target"]) for word in selected_words])
                )
                minimum_delta = min(item["delta"] for item in selected_states)
                minimum_anchor = min(item["anchor"] for item in selected_states)
                semantic_start = not semantic.token_is_bad_start(
                    str(selected_words[0].get("text") or "")
                )
                semantic_end = bool(
                    seed["semantic_complete_end"]
                    and not semantic.token_is_bad_end(
                        str(selected_words[-1].get("text") or "")
                    )
                )
                raw_left = max(0, round(start * source_rate))
                raw_right = min(len(source_audio), round(end * source_rate))
                quality = quality_base.quality_metrics(
                    source_audio[raw_left:raw_right], source_rate
                )
                seamless_hard_transition = bool(
                    modes["left"] == "human_calibrated_20ms_hysteresis"
                    or modes["right"] == "hard_transition_exact_word_end"
                )
                eres = eres2net_metrics(
                    eres_by_source[source_id], float(start), float(end)
                )
                machine_strict = bool(
                    model_calibration_pass
                    and MIN_STRICT_SEC <= duration <= MAX_STRICT_SEC
                    and semantic_start
                    and semantic_end
                    and quality["pass"]
                    and strong_fraction >= 0.60
                    and direct_target_fraction >= 0.80
                    and mean_delta >= 0.035
                    and mean_anchor >= 0.100
                    and minimum_delta >= -0.015
                    and minimum_anchor >= 0.030
                    and float(np.mean(probabilities)) >= 0.62
                    and min(probabilities) >= 0.10
                    and eres["support"]
                    and not eres["hard_negative_warning"]
                    # Seamless speaker switches must first be heard in the
                    # boundary review pool; they never auto-enter strict.
                    and not seamless_hard_transition
                )
                tier = "anime_machine_strict_v1" if machine_strict else "anime_semantic_review_v1"
                row = {
                    "source_id": source_id,
                    "source": str(source.resolve()),
                    "word_start_index": left,
                    "word_end_index": right,
                    "start": float(start),
                    "end": float(end),
                    "duration": float(duration),
                    "text": text,
                    "word_count": len(selected_words),
                    "word_state_counts": dict(
                        Counter(item["state"] for item in selected_states)
                    ),
                    "strong_word_fraction": strong_fraction,
                    "direct_target_word_fraction": direct_target_fraction,
                    "mean_target_center_delta": mean_delta,
                    "minimum_target_center_delta": minimum_delta,
                    "mean_anchor_margin": mean_anchor,
                    "minimum_anchor_margin": minimum_anchor,
                    "mean_asr_probability": float(np.mean(probabilities)),
                    "minimum_asr_probability": min(probabilities),
                    "semantic_start": semantic_start,
                    "semantic_complete_end": semantic_end,
                    "endpoint": seed["endpoint"],
                    "left_boundary": left_boundary,
                    "right_boundary": right_boundary,
                    "boundary_modes": modes,
                    "seamless_hard_transition": seamless_hard_transition,
                    "quality": quality,
                    "eres2net": eres,
                    "model_calibration_pass": model_calibration_pass,
                    "tier": tier,
                    "hard_mask_hits": hits,
                }
                source_candidates.append(row)
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
                "model_calibration_pass": model_calibration_pass,
            }
        )
        print(
            f"[{source_id}] runs={len(runs)} "
            f"strict={counters['anime_machine_strict_v1']} "
            f"review={counters['anime_semantic_review_v1']} "
            f"model_calibration={model_calibration_pass}",
            flush=True,
        )
        del source_audio, energy_times, energy_rms

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
        duplicate = next(
            (
                old
                for old in candidate_pool[-12:]
                if overlap_ratio(row, old) >= 0.80
            ),
            None,
        )
        if duplicate is None:
            candidate_pool.append(row)

    strict_rows = [
        row for row in candidate_pool if row["tier"] == "anime_machine_strict_v1"
    ]
    review_all = [
        row for row in candidate_pool if row["tier"] == "anime_semantic_review_v1"
    ]
    review_rows = []
    for source_id in sorted({row["source_id"] for row in review_all}):
        subset = [row for row in review_all if row["source_id"] == source_id]
        subset.sort(
            key=lambda row: (
                row["model_calibration_pass"],
                row["semantic_start"],
                row["semantic_complete_end"],
                row["strong_word_fraction"],
                row["mean_anchor_margin"],
                row["duration"],
            ),
            reverse=True,
        )
        review_rows.extend(subset[:MAX_REVIEW_EXPORT_PER_SOURCE])

    track_paths = {str(row["id"]): Path(row["vocals"]) for row in tracks}

    def materialize(rows: list[dict], folder: Path, prefix: str) -> None:
        current_source_id = ""
        current_audio = None
        current_rate = 0
        for index, row in enumerate(rows, 1):
            source_id = str(row["source_id"])
            if source_id != current_source_id:
                current_audio, current_rate = audio_base.read_mono(track_paths[source_id])
                current_source_id = source_id
            assert current_audio is not None
            blind_id = f"{prefix}{index:04d}"
            output = folder / f"{blind_id}.wav"
            clip, export_quality = semantic.export_clip(
                current_audio,
                current_rate,
                float(row["start"]),
                float(row["end"]),
                output,
            )
            info = sf.info(str(output))
            if (
                info.samplerate != OUTPUT_SR
                or info.channels != 1
                or info.subtype != "PCM_16"
            ):
                raise RuntimeError(f"Invalid anime candidate: {output} {info}")
            row["blind_id"] = blind_id
            row["audio"] = str(output.resolve())
            row["audio_rel"] = output.relative_to(OUT).as_posix()
            row["output_duration"] = float(info.duration)
            row["wave"] = page_base.waveform(clip)
            row["export_quality"] = export_quality

    materialize(strict_rows, STRICT_AUDIO, "AM")
    materialize(review_rows, REVIEW_AUDIO, "AR")
    MACHINE_LIST.write_text(
        "/n".join(
            f"{row['audio']}|ireina_anime_machine_review_v1|JA|{row['text']}"
            for row in strict_rows
        )
        + ("/n" if strict_rows else ""),
        encoding="utf-8",
    )

    audit = {
        "version": "anime_semantic_mining_v1_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "source": "original anime BS-Roformer vocals",
            "speaker": "human-anchored episode-local CAM++ models",
            "word_gate": "guarded strong/soft/edge states with hard breaks",
            "semantic_gate": "complete phrase terminal and non-fragment tokens",
            "seamless_switch": "never strict; boundary review required",
            "status": "machine review candidates only; not training-approved",
        },
        "sample_count": len(strict_rows),
        "samples": strict_rows,
    }
    write_json(OUT / "audit_manifest.json", audit)
    write_json(
        OUT / "semantic_review_manifest.json",
        {
            "version": "anime_semantic_mining_v1_review",
            "sample_count": len(review_rows),
            "unmaterialized_review_count": len(review_all) - len(review_rows),
            "samples": review_rows,
        },
    )
    write_json(OUT / "word_state_report.json", word_state_report)
    write_json(OUT / "hard_exclusion_masks.json", masks)
    report = {
        "version": "anime_semantic_mining_v1_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime_sec": time.time() - started,
        "local_resources": RESOURCE_POLICY,
        "coverage": {"expected_sources": 12, "completed_sources": len(source_stats)},
        "human_identity_calibration": {
            "sample_count": len(labels),
            "class_counts": dict(Counter(row["final_class"] for row in labels)),
        },
        "machine_strict_review": {
            "count": len(strict_rows),
            "duration_sec": sum(float(row["output_duration"]) for row in strict_rows),
            "list": str(MACHINE_LIST),
            "status": "pending human semantic acceptance",
        },
        "semantic_review": {
            "all_count": len(review_all),
            "materialized_count": len(review_rows),
            "materialized_duration_sec": sum(
                float(row["output_duration"]) for row in review_rows
            ),
        },
        "source_stats": source_stats,
        "training_started": False,
        "existing_training_lists_modified": False,
        "score_90_claimed": False,
    }
    write_json(OUT / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
