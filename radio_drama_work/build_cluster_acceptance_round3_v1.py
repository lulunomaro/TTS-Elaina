# -*- coding: utf-8 -*-
"""Build fresh blind samples from human-anchored source-local target clusters."""
from __future__ import annotations

import json
import math
import random
import sys
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
MANIFEST = ROOT / "full_track_manifest_v1.json"
LABELS = ROOT / "human_calibration_combined_v2" / "labels_resolved.json"
CLUSTER_DIR = ROOT / "source_local_cluster_anchor_v2" / "models"
WORD_DIR = ROOT / "word_target_all_v1" / "source_reports"
OUT = ROOT / "cluster_acceptance_round3_v1"
AUDIO_DIR = OUT / "audio"
OUTPUT_SR = 32000
RANDOM_SEED = 20260804
HUMAN_EXCLUSION_PAD = 3.0
MIN_SAMPLE_SEPARATION = 10.0
MIN_DURATION = 1.50
MAX_DURATION = 4.20


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8")


def norm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def target_runs(specs: np.ndarray, labels: np.ndarray, target_clusters: set[int]) -> list[dict]:
    centers = specs.mean(axis=1)
    order = np.argsort(centers)
    runs = []
    current = []
    previous_center = None
    for idx in order:
        center = float(centers[idx])
        is_target = int(labels[idx]) in target_clusters
        if is_target and (previous_center is None or center - previous_center <= 0.95):
            current.append(int(idx))
        elif is_target:
            if current:
                runs.append(current)
            current = [int(idx)]
        else:
            if current:
                runs.append(current)
                current = []
        previous_center = center
    if current:
        runs.append(current)
    result = []
    for indices in runs:
        start = float(np.min(specs[indices, 0]))
        end = float(np.max(specs[indices, 1]))
        if end - start >= 1.45:
            result.append({"indices": indices, "start": start, "end": end})
    return result


def overlap(start: float, end: float, intervals: list[tuple[float, float]]) -> bool:
    return any(max(start, left) < min(end, right) for left, right in intervals)


def word_group(words: list[dict], anchor_time: float, run_start: float, run_end: float) -> tuple[float, float, list[int], str] | None:
    eligible = [
        idx
        for idx, row in enumerate(words)
        if float(row.get("end", 0.0)) > run_start + 0.03
        and float(row.get("start", 0.0)) < run_end - 0.03
        and str(row.get("text", "")).strip()
    ]
    if not eligible:
        return None
    anchor = min(
        eligible,
        key=lambda idx: abs((float(words[idx]["start"]) + float(words[idx]["end"])) / 2.0 - anchor_time),
    )
    left = right = anchor
    while float(words[right]["end"]) - float(words[left]["start"]) < 1.65:
        choices = []
        if left > 0:
            gap = float(words[left]["start"]) - float(words[left - 1]["end"])
            proposed = float(words[right]["end"]) - float(words[left - 1]["start"])
            if gap <= 0.38 and float(words[left - 1]["start"]) >= run_start and proposed <= 3.45:
                choices.append((proposed, "left"))
        if right + 1 < len(words):
            gap = float(words[right + 1]["start"]) - float(words[right]["end"])
            proposed = float(words[right + 1]["end"]) - float(words[left]["start"])
            if gap <= 0.38 and float(words[right + 1]["end"]) <= run_end and proposed <= 3.45:
                choices.append((proposed, "right"))
        if not choices:
            break
        _, direction = min(choices)
        if direction == "left":
            left -= 1
        else:
            right += 1
    core_start = float(words[left]["start"])
    core_end = float(words[right]["end"])
    if core_end - core_start < 1.20:
        return None
    indices = list(range(left, right + 1))
    text = "".join(str(words[idx].get("text", "")) for idx in indices)
    return core_start, core_end, indices, text


def frame_rms(audio: np.ndarray, sr: int, frame_sec: float = 0.020, hop_sec: float = 0.010) -> tuple[np.ndarray, np.ndarray]:
    frame = max(1, round(sr * frame_sec))
    hop = max(1, round(sr * hop_sec))
    if len(audio) < frame:
        working = np.pad(audio, (0, frame - len(audio)))
    else:
        working = audio
    count = max(1, 1 + (len(working) - frame) // hop)
    starts = np.arange(count, dtype=np.int64) * hop
    ends = starts + frame
    squared = np.square(working, dtype=np.float64)
    prefix = np.empty(len(squared) + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(squared, out=prefix[1:])
    rms = np.sqrt((prefix[ends] - prefix[starts]) / frame + 1e-12)
    times = (starts + frame / 2.0) / sr
    return times, rms


def snap_boundaries(
    audio: np.ndarray,
    sr: int,
    rms_times: np.ndarray,
    rms_values: np.ndarray,
    core_start: float,
    core_end: float,
    run_start: float,
    run_end: float,
) -> tuple[float, float, dict] | None:
    times, rms = rms_times, rms_values
    speech_region = rms[(times >= core_start) & (times <= core_end)]
    if not len(speech_region):
        return None
    speech_level = max(float(np.quantile(speech_region, 0.80)), 1e-6)

    left_min = max(run_start, core_start - 0.55)
    left_max = max(left_min, core_start - 0.025)
    right_min = min(run_end, core_end + 0.025)
    right_max = min(run_end, core_end + 0.55)
    left_indices = np.flatnonzero((times >= left_min) & (times <= left_max))
    right_indices = np.flatnonzero((times >= right_min) & (times <= right_max))
    if not len(left_indices) or not len(right_indices):
        return None
    left_idx = int(left_indices[np.argmin(rms[left_indices])])
    right_idx = int(right_indices[np.argmin(rms[right_indices])])
    left_ratio = float(rms[left_idx] / speech_level)
    right_ratio = float(rms[right_idx] / speech_level)
    if left_ratio > 0.30 or right_ratio > 0.30:
        return None
    start = max(run_start, float(times[left_idx] - 0.010))
    end = min(run_end, float(times[right_idx] + 0.010))
    if not MIN_DURATION <= end - start <= MAX_DURATION:
        return None
    start_samples = max(1, round(sr * 0.030))
    left_audio = audio[round(start * sr) : round(start * sr) + start_samples]
    right_audio = audio[max(0, round(end * sr) - start_samples) : round(end * sr)]
    edge_left = float(np.sqrt(np.mean(np.square(left_audio, dtype=np.float64)) + 1e-12) / speech_level)
    edge_right = float(np.sqrt(np.mean(np.square(right_audio, dtype=np.float64)) + 1e-12) / speech_level)
    if edge_left > 0.38 or edge_right > 0.38:
        return None
    return start, end, {
        "left_valley_ratio": left_ratio,
        "right_valley_ratio": right_ratio,
        "left_edge_ratio_30ms": edge_left,
        "right_edge_ratio_30ms": edge_right,
        "speech_level": speech_level,
    }


def target_fraction(specs: np.ndarray, labels: np.ndarray, target_clusters: set[int], start: float, end: float) -> float:
    centers = specs.mean(axis=1)
    indices = np.flatnonzero((centers >= start) & (centers <= end))
    if not len(indices):
        return 0.0
    return float(np.mean([int(labels[idx]) in target_clusters for idx in indices]))


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    track_map = {str(row["id"]): row for row in manifest["tracks"]}
    human_rows = json.loads(LABELS.read_text(encoding="utf-8"))["samples"]
    old_refs = norm(np.load(str(ROOT / "full_funasr_target_recall_v2" / "embedding_cache" / "all_target_references.npz"), allow_pickle=True)["embeddings"])
    candidates_by_source = {}
    rejected_reasons = {}
    for model_path in sorted(CLUSTER_DIR.glob("*.npz")):
        source_id = model_path.stem
        track = track_map[source_id]
        source_path = Path(track["vocals"])
        model = np.load(str(model_path), allow_pickle=True)
        specs = np.asarray(model["specs"], dtype=np.float32)
        embeddings = norm(model["embeddings"])
        labels = np.asarray(model["labels"], dtype=np.int64)
        centers = norm(model["centers"])
        target_clusters = set(int(value) for value in model["target_clusters"].tolist())
        words = json.loads((WORD_DIR / f"{source_id}.json").read_text(encoding="utf-8"))["word_rows"]
        audio, sr = sf.read(str(source_path), dtype="float32", always_2d=True)
        mono = audio.mean(axis=1).astype(np.float32, copy=False)
        rms_times, rms_values = frame_rms(mono, sr)
        exclusions = [
            (float(row["start"]) - HUMAN_EXCLUSION_PAD, float(row["end"]) + HUMAN_EXCLUSION_PAD)
            for row in human_rows
            if str(row["source_id"]) == source_id
        ]
        for row in manifest.get("hard_exclusions", []):
            if str(row["source_id"]) == source_id:
                exclusions.append((float(row["start"]) - 0.5, float(row["end"]) + 0.5))
        source_candidates = []
        reason_counts = {}
        for run_idx, run in enumerate(target_runs(specs, labels, target_clusters)):
            run_embeddings = embeddings[run["indices"]]
            run_labels = labels[run["indices"]]
            run_similarity = np.asarray(
                [float(run_embeddings[idx] @ centers[int(run_labels[idx])]) for idx in range(len(run_embeddings))]
            )
            order = np.argsort(-run_similarity)
            for local_idx in order[: min(5, len(order))]:
                spec_idx = int(run["indices"][int(local_idx)])
                anchor_time = float(np.mean(specs[spec_idx]))
                group = word_group(words, anchor_time, run["start"], run["end"])
                if group is None:
                    reason_counts["no_complete_word_group"] = reason_counts.get("no_complete_word_group", 0) + 1
                    continue
                core_start, core_end, word_indices, text = group
                snapped = snap_boundaries(
                    mono,
                    sr,
                    rms_times,
                    rms_values,
                    core_start,
                    core_end,
                    run["start"],
                    run["end"],
                )
                if snapped is None:
                    reason_counts["no_low_energy_boundary"] = reason_counts.get("no_low_energy_boundary", 0) + 1
                    continue
                start, end, boundary = snapped
                if overlap(start, end, exclusions):
                    reason_counts["human_or_hard_exclusion"] = reason_counts.get("human_or_hard_exclusion", 0) + 1
                    continue
                fraction = target_fraction(specs, labels, target_clusters, start, end)
                if fraction < 0.90:
                    reason_counts["target_cluster_fraction"] = reason_counts.get("target_cluster_fraction", 0) + 1
                    continue
                old_similarities = embeddings[spec_idx] @ old_refs.T
                old_top4 = float(np.mean(np.sort(old_similarities)[-4:]))
                distance_to_human = min(
                    [
                        max(0.0, abs((start + end) / 2.0 - (left + right) / 2.0))
                        for left, right in exclusions
                    ]
                    or [9999.0]
                )
                source_candidates.append(
                    {
                        "kind": "fresh_cluster_candidate",
                        "source_id": source_id,
                        "source": str(source_path.resolve()),
                        "start": start,
                        "end": end,
                        "duration": end - start,
                        "text": text,
                        "word_indices": word_indices,
                        "run_index": run_idx,
                        "cluster": int(labels[spec_idx]),
                        "cluster_similarity": float(embeddings[spec_idx] @ centers[int(labels[spec_idx])]),
                        "old_target_top4": old_top4,
                        "target_cluster_fraction": fraction,
                        "distance_to_human_sec": distance_to_human,
                        "boundary": boundary,
                    }
                )
        deduplicated = []
        for row in sorted(
            source_candidates,
            key=lambda item: (item["target_cluster_fraction"], item["cluster_similarity"], item["distance_to_human_sec"]),
            reverse=True,
        ):
            if any(max(row["start"], old["start"]) < min(row["end"], old["end"]) for old in deduplicated):
                continue
            deduplicated.append(row)
        candidates_by_source[source_id] = deduplicated
        rejected_reasons[source_id] = reason_counts
        print(f"[{source_id}] valid fresh candidates={len(deduplicated)} rejected={reason_counts}", flush=True)

    selected = []
    for source_id, candidates in sorted(candidates_by_source.items()):
        if len(candidates) < 3:
            raise RuntimeError(f"Only {len(candidates)} valid fresh cluster candidates for {source_id}")
        source_selected = []
        profiles = [
            ("deep_cluster", sorted(candidates, key=lambda row: row["cluster_similarity"], reverse=True)),
            ("far_from_human", sorted(candidates, key=lambda row: row["distance_to_human_sec"], reverse=True)),
            ("lower_cluster_margin", sorted(candidates, key=lambda row: row["cluster_similarity"])),
        ]
        for profile, ranked in profiles:
            chosen = next(
                (
                    row
                    for row in ranked
                    if all(abs((row["start"] + row["end"]) / 2.0 - (old["start"] + old["end"]) / 2.0) >= MIN_SAMPLE_SEPARATION for old in source_selected)
                ),
                None,
            )
            if chosen is None:
                chosen = next(row for row in ranked if row not in source_selected)
            item = dict(chosen)
            item["selection_profile"] = profile
            source_selected.append(item)
        selected.extend(source_selected)

    if len(selected) != 21:
        raise RuntimeError(f"Expected 21 acceptance samples, got {len(selected)}")
    random.Random(RANDOM_SEED).shuffle(selected)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(selected, 1):
        blind_id = f"A{idx:03d}"
        source_audio, sr = sf.read(str(row["source"]), dtype="float32", always_2d=True)
        start_sample = max(0, round(row["start"] * sr))
        end_sample = min(len(source_audio), round(row["end"] * sr))
        mono = source_audio[start_sample:end_sample].mean(axis=1).astype(np.float32, copy=False)
        if sr != OUTPUT_SR:
            import torch
            import torchaudio

            mono = torchaudio.functional.resample(torch.from_numpy(mono), sr, OUTPUT_SR).numpy()
        output = AUDIO_DIR / f"{blind_id}.wav"
        sf.write(str(output), mono, OUTPUT_SR, subtype="PCM_16")
        row["blind_id"] = blind_id
        row["audio"] = str(output.resolve())
        row["audio_rel"] = f"audio/{output.name}"
        row["output_duration"] = len(mono) / OUTPUT_SR
        row["wave"] = page_base.waveform(mono)
    audit = {
        "version": "cluster_acceptance_round3_v1_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "policy": {
            "fresh": "no output overlaps either human-review round or manifest hard exclusions",
            "source_scope": sorted(candidates_by_source),
            "per_source": 3,
            "boundary": "ASR word core expanded to low-energy valleys within a continuous target-cluster run",
            "speaker": "source-local human-anchored target clusters; >=90% local target-cluster coverage",
            "local_resources": RESOURCE_POLICY,
            "status": "blind acceptance only; not training data until user confirms",
        },
        "candidate_pool_counts": {source: len(rows) for source, rows in candidates_by_source.items()},
        "rejected_reasons": rejected_reasons,
        "samples": selected,
    }
    public = {
        "version": "cluster_acceptance_round3_v1_blind",
        "sample_count": len(selected),
        "samples": [
            {"blind_id": row["blind_id"], "audio": row["audio_rel"], "duration_sec": round(row["output_duration"], 3)}
            for row in selected
        ],
    }
    write_json(OUT / "audit_manifest.json", audit)
    write_json(OUT / "blind_manifest.json", public)
    (OUT / "README_试听说明.txt").write_text(
        "这是局部说话人簇的全新验收样本，共 21 条。整条纯伊蕾娜且边界完整时只标已听；任何非目标字、混合、切边或杂音都勾错误。/n"
        "完成后保存 JSON 并交给我。任何明确非目标都将使对应来源回炉。/n",
        encoding="utf-8",
    )
    (OUT / "开始试听.cmd").write_text('@echo off/r/nstart "" "%~dp0index.html"/r/n', encoding="utf-8")
    print(json.dumps({"output": str(OUT), "samples": len(selected)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
