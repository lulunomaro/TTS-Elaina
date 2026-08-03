# -*- coding: utf-8 -*-
"""Audit source-local CAM++ anchors against the two human review rounds."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

BOOTSTRAP_DIR = Path(__file__).resolve().parent
if str(BOOTSTRAP_DIR) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_DIR))
from local_resource_policy import apply_cpu_friendly_policy

RESOURCE_POLICY = apply_cpu_friendly_policy(2)

import numpy as np


ROOT = Path(__file__).resolve().parent
LABELS = ROOT / "human_calibration_combined_v2" / "labels_resolved.json"
CACHE = ROOT / "full_funasr_target_recall_v2" / "embedding_cache"
OUT = ROOT / "source_anchor_campp_v1"
TARGET_REFS = CACHE / "all_target_references.npz"
AGGREGATORS = ("minimum", "p05", "p10", "p20", "p25", "median")
UNSAFE = {"pure_non_target", "mixed", "target_variant_exclude"}


def norm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def aggregate(values: np.ndarray, name: str) -> float:
    if name == "minimum":
        return float(np.min(values))
    if name == "median":
        return float(np.median(values))
    return float(np.quantile(values, {"p05": 0.05, "p10": 0.10, "p20": 0.20, "p25": 0.25}[name]))


def top_mean(similarities: np.ndarray, count: int) -> np.ndarray:
    count = min(count, similarities.shape[1])
    return np.mean(np.sort(similarities, axis=1)[:, -count:], axis=1)


def clip_indices(specs: np.ndarray, start: float, end: float) -> np.ndarray:
    centers = specs.mean(axis=1)
    indices = np.flatnonzero((centers >= start) & (centers <= end))
    if len(indices):
        return indices
    overlap = np.maximum(0.0, np.minimum(specs[:, 1], end) - np.maximum(specs[:, 0], start))
    return np.flatnonzero(overlap >= 0.20)


def main() -> None:
    rows = json.loads(LABELS.read_text(encoding="utf-8"))["samples"]
    old_refs = norm(np.load(str(TARGET_REFS), allow_pickle=True)["embeddings"])
    source_caches = {path.stem.replace("_vocals", ""): path for path in CACHE.glob("*_vocals.npz")}
    records = []
    mapping_report = {}
    for source_id, cache_path in sorted(source_caches.items()):
        data = np.load(str(cache_path), allow_pickle=True)
        specs = np.asarray(data["specs"], dtype=np.float32)
        embeddings = norm(data["embeddings"])
        source_rows = [row for row in rows if str(row["source_id"]) == source_id]
        mapped = {}
        for row in source_rows:
            indices = clip_indices(specs, float(row["start"]), float(row["end"]))
            if len(indices):
                mapped[row["blind_id"]] = indices
        mapping_report[source_id] = {
            "cache_windows": len(specs),
            "human_rows": len(source_rows),
            "mapped_rows": len(mapped),
            "unmapped": [row["blind_id"] for row in source_rows if row["blind_id"] not in mapped],
        }
        for row in source_rows:
            query_indices = mapped.get(row["blind_id"])
            if query_indices is None or row["final_class"] not in {"target", *UNSAFE}:
                continue
            positive_indices = np.concatenate(
                [
                    mapped[other["blind_id"]]
                    for other in source_rows
                    if other["blind_id"] != row["blind_id"]
                    and other["final_class"] == "target"
                    and other["blind_id"] in mapped
                ]
                or [np.asarray([], dtype=np.int64)]
            )
            negative_indices = np.concatenate(
                [
                    mapped[other["blind_id"]]
                    for other in source_rows
                    if other["blind_id"] != row["blind_id"]
                    and other["final_class"] == "pure_non_target"
                    and other["blind_id"] in mapped
                ]
                or [np.asarray([], dtype=np.int64)]
            )
            query = embeddings[query_indices]
            old_top4 = top_mean(query @ old_refs.T, 4)
            if len(positive_indices):
                same_positive = top_mean(query @ embeddings[positive_indices].T, 3)
            else:
                same_positive = old_top4.copy()
            if len(negative_indices):
                same_negative = top_mean(query @ embeddings[negative_indices].T, 3)
            else:
                same_negative = np.full(len(query), -1.0, dtype=np.float32)
            records.append(
                {
                    "blind_id": row["blind_id"],
                    "human_round": row["human_round"],
                    "source_id": source_id,
                    "final_class": row["final_class"],
                    "window_count": len(query),
                    "old_top4": old_top4,
                    "same_positive_top3": same_positive,
                    "same_negative_top3": same_negative,
                }
            )

    candidates = []
    weights = (0.0, 0.25, 0.50, 0.75, 1.0)
    for old_weight, positive_weight, negative_weight in itertools.product(weights, repeat=3):
        if old_weight + positive_weight == 0.0:
            continue
        for aggregator in AGGREGATORS:
            scores = []
            for row in records:
                window_score = (
                    old_weight * row["old_top4"]
                    + positive_weight * row["same_positive_top3"]
                    - negative_weight * row["same_negative_top3"]
                )
                scores.append(aggregate(window_score, aggregator))
            score_array = np.asarray(scores, dtype=np.float64)
            unsafe_indices = [idx for idx, row in enumerate(records) if row["final_class"] in UNSAFE]
            target_indices = [idx for idx, row in enumerate(records) if row["final_class"] == "target"]
            threshold = float(np.max(score_array[unsafe_indices]) + 1e-7)
            target_retained = int(np.sum(score_array[target_indices] >= threshold))
            candidates.append(
                {
                    "old_weight": old_weight,
                    "same_positive_weight": positive_weight,
                    "same_negative_weight": negative_weight,
                    "aggregator": aggregator,
                    "threshold": threshold,
                    "target_retained": target_retained,
                    "target_total": len(target_indices),
                    "unsafe_rejected": len(unsafe_indices),
                    "unsafe_total": len(unsafe_indices),
                    "scores": score_array,
                }
            )
    selected = max(
        candidates,
        key=lambda row: (
            row["target_retained"],
            row["same_positive_weight"] > 0,
            row["same_negative_weight"] > 0,
            -row["old_weight"] - row["same_positive_weight"] - row["same_negative_weight"],
        ),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "version": "source_anchor_campp_v1",
        "local_resources": RESOURCE_POLICY,
        "scope": sorted(source_caches),
        "mapping": mapping_report,
        "policy": {
            "cross_fit": "the evaluated clip is excluded from same-source target/non-target anchor sets",
            "mixed": "evaluation-only, never an anchor",
            "threshold": "strictly above the highest cross-fitted unsafe score",
        },
        "selected": {key: value for key, value in selected.items() if key != "scores"},
        "top_candidates": [
            {key: value for key, value in row.items() if key != "scores"}
            for row in sorted(candidates, key=lambda row: row["target_retained"], reverse=True)[:30]
        ],
        "scores": [
            {
                "blind_id": row["blind_id"],
                "human_round": row["human_round"],
                "source_id": row["source_id"],
                "final_class": row["final_class"],
                "window_count": row["window_count"],
                "score": float(selected["scores"][idx]),
                "threshold": float(selected["threshold"]),
                "accepted": bool(selected["scores"][idx] >= selected["threshold"]),
            }
            for idx, row in enumerate(records)
        ],
    }
    write_path = OUT / "report.json"
    write_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "/n", encoding="utf-8")
    print(json.dumps({"selected": report["selected"], "mapping": mapping_report, "saved": str(write_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
