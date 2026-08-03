# -*- coding: utf-8 -*-
"""Anchor source-local spherical speaker clusters with human labels."""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

BOOTSTRAP_DIR = Path(__file__).resolve().parent
if str(BOOTSTRAP_DIR) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_DIR))
from local_resource_policy import apply_cpu_friendly_policy

RESOURCE_POLICY = apply_cpu_friendly_policy(2)

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
LABELS = ROOT / "human_calibration_combined_v2" / "labels_resolved.json"
CACHE = ROOT / "full_funasr_target_recall_v2" / "embedding_cache"
OUT = ROOT / "source_local_cluster_anchor_v2"
TARGET_REFS = CACHE / "all_target_references.npz"
K_GRID = tuple(range(3, 21))
TARGET_CLASSES = {"target"}
NEGATIVE_CLASSES = {"pure_non_target"}
MIXED_CLASSES = {"mixed", "target_variant_exclude"}


def norm_np(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def norm_torch(values: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(values, dim=1)


def clip_indices(specs: np.ndarray, start: float, end: float) -> np.ndarray:
    centers = specs.mean(axis=1)
    indices = np.flatnonzero((centers >= start) & (centers <= end))
    if len(indices):
        return indices
    overlap = np.maximum(0.0, np.minimum(specs[:, 1], end) - np.maximum(specs[:, 0], start))
    return np.flatnonzero(overlap >= 0.20)


def spherical_kmeans(values: torch.Tensor, k: int, seed: int, iterations: int = 45) -> tuple[torch.Tensor, torch.Tensor, float]:
    generator = torch.Generator(device=values.device)
    generator.manual_seed(seed)
    permutation = torch.randperm(len(values), generator=generator, device=values.device)
    centers = values[permutation[:k]].clone()
    labels = torch.zeros(len(values), dtype=torch.long, device=values.device)
    for _ in range(iterations):
        similarities = values @ centers.T
        new_labels = similarities.argmax(dim=1)
        if torch.equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        new_centers = []
        for cluster in range(k):
            members = values[labels == cluster]
            if len(members):
                new_centers.append(norm_torch(members.mean(dim=0, keepdim=True))[0])
            else:
                new_centers.append(values[permutation[(cluster + seed) % len(values)]])
        centers = torch.stack(new_centers)
    objective = float(torch.mean(torch.max(values @ centers.T, dim=1).values).item())
    return labels, centers, objective


def best_kmeans(values: torch.Tensor, k: int) -> tuple[np.ndarray, np.ndarray, float]:
    best = None
    for restart in range(5):
        labels, centers, objective = spherical_kmeans(values, k, 20260802 + 1009 * k + restart)
        if best is None or objective > best[2]:
            best = (labels.detach().cpu().numpy(), centers.detach().cpu().numpy(), objective)
    assert best is not None
    return best


def main() -> None:
    started = time.time()
    rows = json.loads(LABELS.read_text(encoding="utf-8"))["samples"]
    old_refs = norm_np(np.load(str(TARGET_REFS), allow_pickle=True)["embeddings"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    source_results = []
    for cache_path in sorted(CACHE.glob("*_vocals.npz")):
        source_id = cache_path.stem.replace("_vocals", "")
        data = np.load(str(cache_path), allow_pickle=True)
        specs = np.asarray(data["specs"], dtype=np.float32)
        embeddings = norm_np(data["embeddings"])
        values = torch.from_numpy(embeddings).to(device)
        source_rows = [
            row
            for row in rows
            if str(row["source_id"]) == source_id
            and row["final_class"] in TARGET_CLASSES | NEGATIVE_CLASSES | MIXED_CLASSES
        ]
        mapped = {
            row["blind_id"]: clip_indices(specs, float(row["start"]), float(row["end"])) for row in source_rows
        }
        if any(not len(indices) for indices in mapped.values()):
            raise RuntimeError(f"Unmapped human clip in {source_id}")
        target_window_indices = np.unique(
            np.concatenate([mapped[row["blind_id"]] for row in source_rows if row["final_class"] in TARGET_CLASSES])
        )
        negative_window_indices = np.unique(
            np.concatenate([mapped[row["blind_id"]] for row in source_rows if row["final_class"] in NEGATIVE_CLASSES])
            if any(row["final_class"] in NEGATIVE_CLASSES for row in source_rows)
            else np.asarray([], dtype=np.int64)
        )
        configurations = []
        clustering_cache = {}
        for k in K_GRID:
            labels, centers, objective = best_kmeans(values, k)
            clustering_cache[k] = (labels, centers, objective)
            target_counts = np.bincount(labels[target_window_indices], minlength=k)
            negative_counts = np.bincount(labels[negative_window_indices], minlength=k) if len(negative_window_indices) else np.zeros(k, dtype=int)
            for min_target_windows, max_negative_windows, target_clip_fraction in itertools.product(
                (1, 2, 3), (0, 1), (0.50, 0.65, 0.80, 0.95)
            ):
                target_clusters = np.flatnonzero(
                    (target_counts >= min_target_windows) & (negative_counts <= max_negative_windows)
                )
                if not len(target_clusters):
                    continue
                target_cluster_set = set(int(value) for value in target_clusters)
                evaluations = []
                for row in source_rows:
                    indices = mapped[row["blind_id"]]
                    fraction = float(np.mean([int(labels[idx]) in target_cluster_set for idx in indices]))
                    accepted = fraction >= target_clip_fraction
                    evaluations.append(
                        {
                            "blind_id": row["blind_id"],
                            "final_class": row["final_class"],
                            "window_count": len(indices),
                            "target_cluster_fraction": fraction,
                            "accepted": accepted,
                        }
                    )
                target_total = sum(row["final_class"] in TARGET_CLASSES for row in evaluations)
                negative_total = sum(row["final_class"] in NEGATIVE_CLASSES for row in evaluations)
                mixed_total = sum(row["final_class"] in MIXED_CLASSES for row in evaluations)
                target_retained = sum(row["final_class"] in TARGET_CLASSES and row["accepted"] for row in evaluations)
                negative_rejected = sum(row["final_class"] in NEGATIVE_CLASSES and not row["accepted"] for row in evaluations)
                mixed_rejected = sum(row["final_class"] in MIXED_CLASSES and not row["accepted"] for row in evaluations)
                cluster_old_top4 = []
                for cluster in target_clusters:
                    similarities = centers[int(cluster)] @ old_refs.T
                    cluster_old_top4.append(float(np.mean(np.sort(similarities)[-4:])))
                configurations.append(
                    {
                        "k": k,
                        "objective": objective,
                        "min_target_windows": min_target_windows,
                        "max_negative_windows": max_negative_windows,
                        "target_clip_fraction": target_clip_fraction,
                        "target_clusters": target_clusters.tolist(),
                        "target_cluster_chunk_count": int(np.sum(np.isin(labels, target_clusters))),
                        "target_cluster_old_top4_min": min(cluster_old_top4),
                        "target_cluster_old_top4_median": float(np.median(cluster_old_top4)),
                        "target_retained": target_retained,
                        "target_total": target_total,
                        "negative_rejected": negative_rejected,
                        "negative_total": negative_total,
                        "mixed_rejected": mixed_rejected,
                        "mixed_total": mixed_total,
                        "evaluations": evaluations,
                    }
                )
        selected = max(
            configurations,
            key=lambda row: (
                row["negative_rejected"] == row["negative_total"],
                row["mixed_rejected"] == row["mixed_total"],
                row["negative_rejected"],
                row["mixed_rejected"],
                row["target_retained"],
                row["target_cluster_old_top4_min"],
                -row["target_cluster_chunk_count"],
            ),
        )
        selected_labels, selected_centers, selected_objective = clustering_cache[int(selected["k"])]
        model_path = OUT / "models" / f"{source_id}.npz"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(model_path),
            specs=specs,
            embeddings=embeddings,
            labels=selected_labels,
            centers=selected_centers,
            target_clusters=np.asarray(selected["target_clusters"], dtype=np.int64),
        )
        public_selected = {key: value for key, value in selected.items() if key != "evaluations"}
        source_results.append(
            {
                "source_id": source_id,
                "cache": str(cache_path),
                "model": str(model_path),
                "chunk_count": len(specs),
                "human_clip_count": len(source_rows),
                "target_anchor_windows": len(target_window_indices),
                "negative_anchor_windows": len(negative_window_indices),
                "selected": public_selected,
                "evaluations": selected["evaluations"],
                "top_configurations": [
                    {key: value for key, value in row.items() if key != "evaluations"}
                    for row in sorted(
                        configurations,
                        key=lambda row: (
                            row["negative_rejected"] == row["negative_total"],
                            row["mixed_rejected"] == row["mixed_total"],
                            row["negative_rejected"],
                            row["mixed_rejected"],
                            row["target_retained"],
                        ),
                        reverse=True,
                    )[:20]
                ],
            }
        )
        print(
            f"[{source_id}] K={selected['k']} target={selected['target_retained']}/{selected['target_total']} "
            f"negative={selected['negative_rejected']}/{selected['negative_total']} "
            f"mixed={selected['mixed_rejected']}/{selected['mixed_total']} chunks={selected['target_cluster_chunk_count']}",
            flush=True,
        )
        del values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report = {
        "version": "source_local_cluster_anchor_v2",
        "runtime_sec": time.time() - started,
        "device": device,
        "local_resources": RESOURCE_POLICY,
        "policy": {
            "clustering": "source-local spherical k-means; K and anchor policy selected per source",
            "target_anchor": "human target clips",
            "negative_anchor": "human pure non-target clips",
            "mixed": "validation-only; never a cluster anchor",
            "status": "active-learning model; fresh blind target-cluster samples required before promotion",
        },
        "sources": source_results,
        "aggregate": {
            "target_retained": sum(row["selected"]["target_retained"] for row in source_results),
            "target_total": sum(row["selected"]["target_total"] for row in source_results),
            "negative_rejected": sum(row["selected"]["negative_rejected"] for row in source_results),
            "negative_total": sum(row["selected"]["negative_total"] for row in source_results),
            "mixed_rejected": sum(row["selected"]["mixed_rejected"] for row in source_results),
            "mixed_total": sum(row["selected"]["mixed_total"] for row in source_results),
            "target_cluster_chunks": sum(row["selected"]["target_cluster_chunk_count"] for row in source_results),
        },
        "deployable": False,
        "recommendation": "build_fresh_blind_cluster_acceptance_pack",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "/n", encoding="utf-8")
    print(json.dumps({"aggregate": report["aggregate"], "saved": str(OUT / "report.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
