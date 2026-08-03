# -*- coding: utf-8 -*-
"""Retrain the discriminator in the probe directory and emit v2 scored pools."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")
if str(PROBE) not in sys.path:
    sys.path.insert(0, str(PROBE))

import rank_all_review_pools as base


OUT = PROBE / "discriminator_rank_v2"
FEEDBACK_LABELS = PROBE / "discriminator_priority_review_v1" / "human_feedback" / "training_intervals_for_next_discriminator.json"


def load_labels_with_feedback() -> list[dict]:
    labels = base.load_labels()
    if FEEDBACK_LABELS.exists():
        document = json.loads(FEEDBACK_LABELS.read_text(encoding="utf-8"))
        added = document.get("samples") or []
        labels.extend(
            {
                "source_id": str(row["source_id"]),
                "start": float(row["start"]),
                "end": float(row["end"]),
                "label": int(row["label"]),
            }
            for row in added
            if int(row.get("label", -1)) in {0, 1}
        )
    return labels


def score_segment(clf, source_id: str, w_start: int, w_end: int) -> float:
    embs = base.load_embs(source_id)
    if embs is None or w_end >= len(embs) or w_start < 0:
        return float("nan")
    segment = embs[w_start : w_end + 1]
    if not len(segment):
        return float("nan")
    segment = segment / np.maximum(np.linalg.norm(segment, axis=1, keepdims=True), 1e-8)
    return float(np.mean(clf.predict_proba(segment)[:, 1]))


def main() -> None:
    base_interval_label_count = len(base.load_labels())
    labels = load_labels_with_feedback()
    X, y = base.build_training(labels)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(X, y)
    OUT.mkdir(parents=True, exist_ok=True)
    print(
        f"trained on {len(y)} words (pos={int(np.sum(y == 1))}, neg={int(np.sum(y == 0))}) with {len(labels)} interval labels",
        flush=True,
    )

    list_paths = {}
    list_counts = {}
    list_tiers = {}
    for name, manifest_path, out_name in (
        ("anime", base.ROOT / "anime_semantic_mining_v1" / "semantic_review_manifest.json", "anime_review_431_scored_v2.list"),
        ("radio", base.ROOT / "full_semantic_mining_v2" / "semantic_review_manifest.json", "radio_review_448_scored_v2.list"),
    ):
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = document.get("samples") or document.get("rows") or []
        scored = []
        for row in rows:
            probability = score_segment(
                clf,
                str(row["source_id"]),
                int(row["word_start_index"]),
                int(row["word_end_index"]),
            )
            if probability != probability:
                continue
            scored.append(
                {
                    "prob": probability,
                    "source_id": str(row["source_id"]),
                    "start": float(row["start"]),
                    "end": float(row["end"]),
                    "duration": float(row["duration"]),
                    "text": str(row.get("text", "")).replace("|", " ")[:60],
                    "blind_id": str(row.get("blind_id", "")),
                    "audio": str(row.get("audio", "")),
                }
            )
        scored.sort(key=lambda row: -row["prob"])
        output = OUT / out_name
        output.write_text(
            "".join(
                f"{row['audio']}|prob={row['prob']:.4f}|{row['source_id']}|{row['start']:.2f}-{row['end']:.2f}|{row['duration']:.2f}s|{row['text']}\n"
                for row in scored
            ),
            encoding="utf-8",
        )
        buckets = {
            "p>=0.9": sum(row["prob"] >= 0.9 for row in scored),
            "0.7-0.9": sum(0.7 <= row["prob"] < 0.9 for row in scored),
            "0.5-0.7": sum(0.5 <= row["prob"] < 0.7 for row in scored),
            "0.3-0.5": sum(0.3 <= row["prob"] < 0.5 for row in scored),
            "<0.3": sum(row["prob"] < 0.3 for row in scored),
        }
        list_paths[name] = str(output.resolve())
        list_counts[name] = len(scored)
        list_tiers[name] = buckets
        print(f"{name}: scored={len(scored)} buckets={buckets}; saved={output}", flush=True)

    summary = {
        "version": "discriminator_scored_lists_v2",
        "model": "LogisticRegression on CAM++ 192d word embeddings",
        "base_training_interval_labels": base_interval_label_count,
        "added_training_interval_labels": len(labels) - base_interval_label_count,
        "training_words": int(len(y)),
        "positive_words": int(np.sum(y == 1)),
        "negative_words": int(np.sum(y == 0)),
        "lists": list_paths,
        "list_counts": list_counts,
        "list_tiers": list_tiers,
        "policy": "speaker probability only; retain semantic, quality, boundary and hard-mask gates; no training list written",
        "training_started": False,
        "existing_training_lists_modified": False,
        "score_90_claimed": False,
    }
    (OUT / "scored_lists_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"summary saved={OUT / 'scored_lists_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
