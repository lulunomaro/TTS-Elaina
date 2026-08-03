# -*- coding: utf-8 -*-
"""Print target-cluster and ASR context around round-3 boundary-only clips."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "cluster_acceptance_round3_v1" / "audit_manifest.json"
LABELS = ROOT / "human_calibration_round3_v1" / "labels_resolved.json"
MODELS = ROOT / "source_local_cluster_anchor_v3" / "models"
WORDS = ROOT / "word_target_all_v1" / "source_reports"


def cluster_runs(specs: np.ndarray, labels: np.ndarray, targets: set[int], left: float, right: float) -> list[tuple]:
    centers = specs.mean(axis=1)
    selected = np.flatnonzero((centers >= left) & (centers <= right))
    result: list[tuple] = []
    current: list[int] = []
    current_target = None
    previous = None
    for idx in selected:
        status = int(labels[idx]) in targets
        center = float(centers[idx])
        if current and (status != current_target or center - float(previous) > 0.95):
            result.append(
                (
                    bool(current_target),
                    float(np.min(specs[current, 0])),
                    float(np.max(specs[current, 1])),
                    len(current),
                    sorted(set(int(labels[value]) for value in current)),
                )
            )
            current = []
        current.append(int(idx))
        current_target = status
        previous = center
    if current:
        result.append(
            (
                bool(current_target),
                float(np.min(specs[current, 0])),
                float(np.max(specs[current, 1])),
                len(current),
                sorted(set(int(labels[value]) for value in current)),
            )
        )
    return result


def fraction(specs: np.ndarray, labels: np.ndarray, targets: set[int], start: float, end: float) -> float:
    centers = specs.mean(axis=1)
    selected = np.flatnonzero((centers >= start) & (centers <= end))
    return float(np.mean([int(labels[idx]) in targets for idx in selected])) if len(selected) else 0.0


def main() -> None:
    audit_rows = {row["blind_id"]: row for row in json.loads(AUDIT.read_text(encoding="utf-8"))["samples"]}
    labels = json.loads(LABELS.read_text(encoding="utf-8"))["samples"]
    for human in labels:
        if human["final_class"] != "boundary_cut":
            continue
        row = audit_rows[human["blind_id"]]
        model = np.load(str(MODELS / f"{row['source_id']}.npz"), allow_pickle=True)
        specs = np.asarray(model["specs"], dtype=np.float32)
        cluster_labels = np.asarray(model["labels"], dtype=np.int64)
        targets = set(int(value) for value in model["target_clusters"].tolist())
        words = json.loads((WORDS / f"{row['source_id']}.json").read_text(encoding="utf-8"))["word_rows"]
        first = min(int(value) for value in row["word_indices"])
        last = max(int(value) for value in row["word_indices"])
        print(f"/n=== {human['blind_id']} {row['source_id']} original={row['start']:.3f}-{row['end']:.3f} ===")
        print("cluster runs (T=target):")
        for status, start, end, count, clusters in cluster_runs(
            specs, cluster_labels, targets, float(row["start"]) - 9.0, float(row["end"]) + 9.0
        ):
            print(f"  {'T' if status else 'N'} {start:.3f}-{end:.3f} n={count} clusters={clusters}")
        proposals = [("original", float(row["start"]), float(row["end"]))]
        left_idx = first
        while left_idx > 0 and float(words[left_idx]["start"]) - float(words[left_idx - 1]["end"]) < 0.35:
            left_idx -= 1
        right_idx = last
        while right_idx + 1 < len(words) and float(words[right_idx + 1]["start"]) - float(words[right_idx]["end"]) < 0.35:
            right_idx += 1
        proposals.append(("nearest_asr_pauses", float(words[left_idx]["start"]), float(words[right_idx]["end"])))
        if human["blind_id"] == "A013":
            proposals.extend(
                [
                    ("semantic_short", float(words[140]["start"]), float(words[156]["end"])),
                    ("semantic_long", float(words[125]["start"]), float(words[156]["end"])),
                ]
            )
        for name, start, end in proposals:
            print(f"  proposal {name:20s} {start:.3f}-{end:.3f} dur={end-start:.3f} target={fraction(specs, cluster_labels, targets, start, end):.3f}")


if __name__ == "__main__":
    main()
