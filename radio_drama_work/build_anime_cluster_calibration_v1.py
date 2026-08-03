# -*- coding: utf-8 -*-
"""Build a blind, identity-only anime speaker calibration pack.

This first anime pass is deliberately not a training-data exporter. It uses
the 168 old clean Ireina references to rank source-local CAM++ clusters, then
asks the user to identify target/non-target/mixed samples in every episode.
Those labels become the episode-local anchors for the semantic mining pass.
"""
from __future__ import annotations

import json
import random
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
import torch

import build_human_calibration_v1 as page_base
import full_speaker_diarize_new_v1 as audio_base
import source_local_cluster_anchor_v2 as cluster_base


MANIFEST = ROOT / "anime_batch_v1" / "manifest.json"
ASR_REPORT = ROOT / "anime_mining_v1" / "asr" / "report.json"
ERES_REPORT = ROOT / "anime_mining_v1" / "eres_scan" / "interval_map.json"
CAM_ROOT = ROOT / "anime_mining_v1" / "source_local_campp"
CAM_CACHE = CAM_ROOT / "embedding_cache"
TARGET_REFS = CAM_CACHE / "all_target_references.npz"
HARD_NEGATIVES = (
    ROOT / "full_speaker_diarize_new_v3" / "embedding_cache" / "hard_negatives.npz"
)
OUT = ROOT / "anime_speaker_calibration_v1"
AUDIO_DIR = OUT / "audio"

OUTPUT_SR = 32000
FADE_SEC = 0.002
K_GRID = (8, 12, 16, 20)
TOP_CLUSTERS_PER_K = 2
SAMPLES_PER_SOURCE = 3
TARGET_SAMPLES_PER_SOURCE = 2
MIN_SEC = 1.50
MAX_SEC = 5.00
MAX_GAP_SEC = 0.32
RANDOM_SEED = 202608022030


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8"
    )
    temporary.replace(path)


def norm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def vector_scores(
    values: np.ndarray, targets: np.ndarray, negatives: np.ndarray
) -> dict[str, np.ndarray]:
    target_matrix = values @ targets.T
    top_count = min(4, target_matrix.shape[1])
    target_top4 = np.partition(
        target_matrix, target_matrix.shape[1] - top_count, axis=1
    )[:, -top_count:].mean(axis=1)
    target_max = target_matrix.max(axis=1)
    negative_max = (
        (values @ negatives.T).max(axis=1)
        if len(negatives)
        else np.full(len(values), -1.0, dtype=np.float32)
    )
    return {
        "target_top4": target_top4.astype(np.float32),
        "target_max": target_max.astype(np.float32),
        "negative_max": negative_max.astype(np.float32),
        "margin": (target_top4 - negative_max).astype(np.float32),
    }


def fit_multik(
    embeddings: np.ndarray, targets: np.ndarray, negatives: np.ndarray
) -> tuple[list[dict], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    values = torch.from_numpy(embeddings).to(device)
    runs = []
    votes = np.zeros(len(embeddings), dtype=np.int16)
    primary_labels = np.full(len(embeddings), -1, dtype=np.int32)
    direct = vector_scores(embeddings, targets, negatives)
    for k in K_GRID:
        labels, centers, objective = cluster_base.best_kmeans(values, min(k, len(values) - 1))
        centers = norm(centers)
        scores = vector_scores(centers, targets, negatives)
        cluster_rows = []
        for index in range(len(centers)):
            members = labels == index
            cluster_rows.append(
                {
                    "cluster": index,
                    "count": int(np.sum(members)),
                    "target_top4": float(scores["target_top4"][index]),
                    "target_max": float(scores["target_max"][index]),
                    "negative_max": float(scores["negative_max"][index]),
                    "margin": float(scores["margin"][index]),
                    "direct_top_half_fraction": float(
                        np.mean(
                            direct["target_top4"][members]
                            >= np.quantile(direct["target_top4"], 0.50)
                        )
                    ),
                }
            )
        ranked = sorted(
            cluster_rows,
            key=lambda row: (
                row["target_top4"] + 0.25 * row["target_max"] - 0.45 * row["negative_max"],
                row["direct_top_half_fraction"],
                row["count"],
            ),
            reverse=True,
        )
        target_clusters = {row["cluster"] for row in ranked[:TOP_CLUSTERS_PER_K]}
        votes += np.isin(labels, list(target_clusters)).astype(np.int16)
        if k == 16:
            primary_labels = labels.astype(np.int32)
        runs.append(
            {
                "k": int(k),
                "objective": float(objective),
                "target_clusters": sorted(target_clusters),
                "clusters": cluster_rows,
                "labels": labels.tolist(),
            }
        )
    if np.all(primary_labels < 0):
        primary_labels = np.asarray(runs[len(runs) // 2]["labels"], dtype=np.int32)
    del values
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return runs, votes, primary_labels, direct


def map_words(
    words: list[dict],
    specs: np.ndarray,
    votes: np.ndarray,
    labels: np.ndarray,
    direct: dict[str, np.ndarray],
) -> list[dict]:
    centers = specs.mean(axis=1)
    target_q55 = float(np.quantile(direct["target_top4"], 0.55))
    margin_q45 = float(np.quantile(direct["margin"], 0.45))
    target_cluster_ids = set(
        int(value)
        for value in labels[np.asarray(votes) >= max(3, len(K_GRID) - 1)].tolist()
    )
    output = []
    for index, word in enumerate(words):
        midpoint = 0.5 * (float(word["start"]) + float(word["end"]))
        mapped = np.flatnonzero((specs[:, 0] <= midpoint) & (specs[:, 1] >= midpoint))
        if not len(mapped):
            nearest = int(np.argmin(np.abs(centers - midpoint)))
            if abs(float(centers[nearest]) - midpoint) <= 0.85:
                mapped = np.asarray([nearest], dtype=np.int64)
        if len(mapped):
            mean_vote = float(np.mean(votes[mapped]))
            mean_target = float(np.mean(direct["target_top4"][mapped]))
            mean_margin = float(np.mean(direct["margin"][mapped]))
            primary = Counter(int(labels[item]) for item in mapped).most_common(1)[0][0]
        else:
            mean_vote = 0.0
            mean_target = -1.0
            mean_margin = -1.0
            primary = -1
        if (
            mean_vote >= 3.0
            and mean_target >= target_q55
            and mean_margin >= margin_q45
        ):
            tier = "target_like"
        elif mean_vote <= 1.0 and primary not in target_cluster_ids:
            tier = "competitor"
        else:
            tier = "uncertain"
        output.append(
            {
                **word,
                "word_index": index,
                "cluster_vote": mean_vote,
                "target_top4": mean_target,
                "target_negative_margin": mean_margin,
                "primary_cluster": primary,
                "predicted_tier": tier,
            }
        )
    return output


def group_words(words: list[dict], tier: str) -> list[list[dict]]:
    groups: list[list[dict]] = []
    current: list[dict] = []
    for word in words:
        same = word["predicted_tier"] == tier
        if tier == "competitor" and current:
            same = same and word["primary_cluster"] == current[-1]["primary_cluster"]
        if current and (
            not same
            or float(word["start"]) - float(current[-1]["end"]) > MAX_GAP_SEC
            or float(word["end"]) - float(current[0]["start"]) > MAX_SEC - 0.16
        ):
            groups.append(current)
            current = []
        if same:
            current.append(word)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return [
        group
        for group in groups
        if group and float(group[-1]["end"]) - float(group[0]["start"]) >= 1.20
    ]


def candidate_from_group(source_id: str, tier: str, group: list[dict]) -> dict:
    start = max(0.0, float(group[0]["start"]) - 0.08)
    end = float(group[-1]["end"]) + 0.10
    if end - start < MIN_SEC:
        missing = MIN_SEC - (end - start)
        start = max(0.0, start - missing / 2.0)
        end += missing / 2.0
        if end - start < MIN_SEC:
            end = start + MIN_SEC
    if end - start > MAX_SEC:
        end = start + MAX_SEC
    return {
        "source_id": source_id,
        "predicted_tier": tier,
        "start": start,
        "end": end,
        "duration": end - start,
        "word_start_index": int(group[0]["word_index"]),
        "word_end_index": int(group[-1]["word_index"]),
        "text": "".join(str(word["text"]) for word in group),
        "word_count": len(group),
        "minimum_cluster_vote": min(float(word["cluster_vote"]) for word in group),
        "mean_cluster_vote": float(np.mean([word["cluster_vote"] for word in group])),
        "mean_target_top4": float(np.mean([word["target_top4"] for word in group])),
        "mean_target_negative_margin": float(
            np.mean([word["target_negative_margin"] for word in group])
        ),
        "primary_cluster": Counter(
            int(word["primary_cluster"]) for word in group
        ).most_common(1)[0][0],
    }


def add_eres2net_metrics(row: dict, scan: dict) -> None:
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
    start = float(row["start"])
    end = float(row["end"])
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
    positive_fraction = (
        float(np.mean([value > 0.0 for value in margins])) if margins else 0.0
    )
    median_center = float(np.median(centers)) if centers else -1.0
    median_margin = float(np.median(margins)) if margins else -1.0
    row["eres2net"] = {
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


def nonoverlap_select(rows: list[dict], count: int) -> list[dict]:
    chosen: list[dict] = []
    for row in rows:
        if any(
            max(float(row["start"]), float(old["start"]))
            < min(float(row["end"]), float(old["end"])) + 1.0
            for old in chosen
        ):
            continue
        chosen.append(row)
        if len(chosen) >= count:
            break
    return chosen


def micro_fade(audio: np.ndarray) -> np.ndarray:
    output = np.asarray(audio, dtype=np.float32).copy()
    count = min(len(output) // 4, max(1, round(OUTPUT_SR * FADE_SEC)))
    if count:
        ramp = np.linspace(0.0, 1.0, count, dtype=np.float32)
        output[:count] *= ramp
        output[-count:] *= ramp[::-1]
    return output


def main() -> None:
    started = time.time()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    asr = json.loads(ASR_REPORT.read_text(encoding="utf-8"))
    asr_by_source = {str(row["source_id"]): row for row in asr["sources"]}
    eres = json.loads(ERES_REPORT.read_text(encoding="utf-8"))
    eres_by_source = {str(row["source_id"]): row for row in eres["sources"]}
    targets = norm(np.load(str(TARGET_REFS), allow_pickle=True)["embeddings"])
    negatives = norm(np.load(str(HARD_NEGATIVES), allow_pickle=True)["embeddings"])

    candidates: list[dict] = []
    source_reports = []
    for track in manifest["tracks"]:
        source_started = time.time()
        source_id = str(track["id"])
        source = Path(track["vocals"])
        cache = CAM_CACHE / f"{source.stem}.npz"
        data = np.load(str(cache), allow_pickle=True)
        specs = np.asarray(data["specs"], dtype=np.float32)
        embeddings = norm(data["embeddings"])
        words = asr_by_source[source_id]["words"]
        runs, votes, labels, direct = fit_multik(embeddings, targets, negatives)
        mapped = map_words(words, specs, votes, labels, direct)

        target_rows = [
            candidate_from_group(source_id, "target_like", group)
            for group in group_words(mapped, "target_like")
        ]
        for row in target_rows:
            add_eres2net_metrics(row, eres_by_source[source_id])
        target_rows.sort(
            key=lambda row: (
                row["eres2net"]["support"],
                -row["eres2net"]["hard_negative_warning"],
                row["minimum_cluster_vote"],
                row["mean_target_negative_margin"],
                row["duration"],
            ),
            reverse=True,
        )
        competitor_rows = [
            candidate_from_group(source_id, "competitor", group)
            for group in group_words(mapped, "competitor")
        ]
        for row in competitor_rows:
            add_eres2net_metrics(row, eres_by_source[source_id])
        competitor_rows.sort(
            key=lambda row: (
                row["eres2net"]["hard_negative_warning"],
                row["word_count"],
                -row["mean_target_top4"],
                row["duration"],
            ),
            reverse=True,
        )
        selected_target = nonoverlap_select(target_rows, TARGET_SAMPLES_PER_SOURCE)
        selected_competitor = nonoverlap_select(
            [
                row
                for row in competitor_rows
                if all(
                    max(float(row["start"]), float(old["start"]))
                    >= min(float(row["end"]), float(old["end"])) + 1.0
                    for old in selected_target
                )
            ],
            SAMPLES_PER_SOURCE - len(selected_target),
        )
        selected = selected_target + selected_competitor
        if len(selected) < SAMPLES_PER_SOURCE:
            fallback = sorted(
                target_rows + competitor_rows,
                key=lambda row: (row["duration"], row["word_count"]),
                reverse=True,
            )
            for row in fallback:
                if row in selected:
                    continue
                if any(
                    max(float(row["start"]), float(old["start"]))
                    < min(float(row["end"]), float(old["end"])) + 0.5
                    for old in selected
                ):
                    continue
                selected.append(row)
                if len(selected) >= SAMPLES_PER_SOURCE:
                    break
        if len(selected) != SAMPLES_PER_SOURCE:
            raise RuntimeError(
                f"{source_id}: only {len(selected)} calibration samples could be built"
            )
        for row in selected:
            row["source"] = str(source.resolve())
            candidates.append(row)
        source_reports.append(
            {
                "source_id": source_id,
                "source": str(source.resolve()),
                "cam_window_count": len(specs),
                "asr_word_count": len(words),
                "target_like_word_count": sum(
                    row["predicted_tier"] == "target_like" for row in mapped
                ),
                "competitor_word_count": sum(
                    row["predicted_tier"] == "competitor" for row in mapped
                ),
                "selected_target_like": len(selected_target),
                "selected_competitor": len(selected) - len(selected_target),
                "target_candidate_pool": target_rows[:40],
                "competitor_candidate_pool": competitor_rows[:40],
                "k_runs": runs,
                "elapsed_sec": time.time() - source_started,
            }
        )
        print(
            f"[{source_id}] calibration target={len(selected_target)} "
            f"competitor={len(selected)-len(selected_target)}",
            flush=True,
        )

    random.Random(RANDOM_SEED).shuffle(candidates)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    current_source_id = ""
    current_audio: np.ndarray | None = None
    current_sample_rate = 0
    for index, row in enumerate(candidates, 1):
        blind_id = f"A{index:03d}"
        source_id = str(row["source_id"])
        if source_id != current_source_id:
            current_audio, current_sample_rate = audio_base.read_mono(Path(row["source"]))
            current_source_id = source_id
        assert current_audio is not None
        clip = audio_base.segment(
            current_audio,
            current_sample_rate,
            float(row["start"]),
            float(row["end"]),
        )
        clip = micro_fade(
            audio_base.resample_1d(clip, current_sample_rate, OUTPUT_SR)
        )
        output = AUDIO_DIR / f"{blind_id}.wav"
        sf.write(str(output), clip, OUTPUT_SR, subtype="PCM_16")
        info = sf.info(str(output))
        if (
            info.samplerate != OUTPUT_SR
            or info.channels != 1
            or info.subtype != "PCM_16"
            or not MIN_SEC <= info.duration <= MAX_SEC + 0.01
        ):
            raise RuntimeError(f"Invalid calibration audio: {output} {info}")
        row["blind_id"] = blind_id
        row["audio"] = str(output.resolve())
        row["audio_rel"] = f"audio/{output.name}"
        row["output_duration"] = float(info.duration)
        row["wave"] = page_base.waveform(np.asarray(clip, dtype=np.float32))

    audit = {
        "version": "anime_speaker_calibration_v1_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "purpose": "identity-only human calibration; not trainable audio",
            "input": "original anime BS-Roformer vocals",
            "target_reference_count": len(targets),
            "hard_negative_count": len(negatives),
            "clustering": f"source-local spherical k-means K={list(K_GRID)}",
            "selection": "two target-like plus one competitor per episode",
            "boundary": "temporary ASR-word bounds; edge quality is ignored in this identity round",
            "local_resources": RESOURCE_POLICY,
        },
        "sample_count": len(candidates),
        "source_count": len(source_reports),
        "samples": candidates,
    }
    report = {
        "version": "anime_speaker_calibration_v1_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime_sec": time.time() - started,
        "sources": source_reports,
        "sample_count": len(candidates),
        "sample_duration_sec": sum(float(row["output_duration"]) for row in candidates),
        "predicted_counts": dict(Counter(row["predicted_tier"] for row in candidates)),
        "training_started": False,
        "existing_training_lists_modified": False,
        "score_90_claimed": False,
    }
    write_json(OUT / "audit_manifest.json", audit)
    write_json(OUT / "private_manifest.json", audit)
    write_json(OUT / "report.json", report)
    print(
        json.dumps(
            {
                "samples": report["sample_count"],
                "sources": len(source_reports),
                "predicted_counts": report["predicted_counts"],
                "report": str(OUT / "report.json"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
