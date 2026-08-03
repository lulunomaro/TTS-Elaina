# -*- coding: utf-8 -*-
"""Experiment 1: speaker-discriminator feasibility on existing word embeddings.

Reads ONLY existing embedding caches and human labels (never modifies
radio_drama_work). Builds positive/negative word-level samples, runs
leave-one-label-out logistic regression on CAM++ embeddings, reports AUC and
threshold grids. Pure CPU; no training, no GPU contention.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(r"F:\GPTSOVITS\radio_drama_work")
OUT = Path(r"F:\GPTSOVITS\discriminator_probe")
OUT.mkdir(parents=True, exist_ok=True)

RADIO_EMB = ROOT / "full_source_local_word_scores_v1" / "embedding_cache"
RADIO_WORDS = ROOT / "full_source_local_word_scores_v1" / "source_reports"
ANIME_EMB = ROOT / "anime_mining_v1" / "source_local_word_scores_v1" / "embedding_cache"
ANIME_WORDS = ROOT / "anime_mining_v1" / "source_local_word_scores_v1" / "source_reports"

LABEL_SOURCES = [
    # (path, class_map)
    (
        ROOT / "human_calibration_combined_v3" / "labels_resolved.json",
        {"target": 1, "pure_non_target": 0},
    ),
    (
        ROOT / "human_anime_speaker_calibration_v3" / "labels_resolved.json",
        {"target": 1, "pure_non_target": 0},
    ),
]


def load_labels() -> list[dict]:
    rows: list[dict] = []
    for path, class_map in LABEL_SOURCES:
        if not path.exists():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        samples = doc.get("samples", doc) if isinstance(doc, dict) else doc
        for row in samples:
            cls = str(row.get("final_class", ""))
            if cls not in class_map:
                continue
            rows.append(
                {
                    "source_id": str(row["source_id"]),
                    "start": float(row["start"]),
                    "end": float(row["end"]),
                    "label": class_map[cls],
                    "human_key": str(row.get("human_key") or row.get("blind_id") or ""),
                }
            )
    return rows


def load_words_and_embs(source_id: str, words_dir: Path, emb_dir: Path):
    report = words_dir / f"{source_id}.json"
    cache = emb_dir / f"{source_id}.npz"
    if not report.exists() or not cache.exists():
        return None, None
    doc = json.loads(report.read_text(encoding="utf-8"))
    word_rows = doc.get("word_rows", [])
    data = np.load(str(cache), allow_pickle=True)
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    if len(word_rows) != len(embeddings):
        return None, None
    starts = np.asarray([float(w["start"]) for w in word_rows])
    ends = np.asarray([float(w["end"]) for w in word_rows])
    centers = (starts + ends) / 2.0
    return word_rows, (embeddings, centers)


def build_dataset(labels: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in labels:
        by_source[row["source_id"]].append(row)

    X_list, y_list, group_list = [], [], []
    used = defaultdict(int)
    for source_id, rows in by_source.items():
        if source_id.startswith("majo_"):
            words, embs = load_words_and_embs(source_id, ANIME_WORDS, ANIME_EMB)
        else:
            words, embs = load_words_and_embs(source_id, RADIO_WORDS, RADIO_EMB)
        if words is None:
            continue
        embeddings, centers = embs
        for row in rows:
            idx = np.flatnonzero((centers >= row["start"]) & (centers <= row["end"]))
            for i in idx:
                X_list.append(embeddings[i])
                y_list.append(row["label"])
                group_list.append(row["human_key"])
                used[row["human_key"]] += 1
    X = np.stack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=int)
    groups = np.asarray(group_list)
    return X, y, groups, used


def main() -> None:
    labels = load_labels()
    print(f"labels loaded: {len(labels)} (target={sum(1 for r in labels if r['label']==1)}, "
          f"non_target={sum(1 for r in labels if r['label']==0)})", flush=True)
    X, y, groups, used = build_dataset(labels)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    print(f"word samples: {len(y)} (pos={pos}, neg={neg}), labels_used={len(used)}, "
          f"avg_words_per_label={len(y)/max(1, len(used)):.1f}", flush=True)
    if pos < 20 or neg < 20:
        print("TOO_FEW_SAMPLES", flush=True)
        return

    # Normalize embeddings to unit length (cosine space)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)

    # Leave-one-human-label-out logistic regression
    logo = LeaveOneGroupOut()
    preds = np.zeros(len(y), dtype=np.float64)
    for train_idx, test_idx in logo.split(X, y, groups):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X[train_idx], y[train_idx])
        preds[test_idx] = clf.predict_proba(X[test_idx])[:, 1]

    auc = roc_auc_score(y, preds)
    fpr, tpr, thr = roc_curve(y, preds)
    # Report operating points
    ops = {}
    for t in (0.3, 0.5, 0.7, 0.9):
        idx = np.argmin(np.abs(thr - t))
        fp = int(np.sum((preds >= t) & (y == 0)))
        tp = int(np.sum((preds >= t) & (y == 1)))
        ops[str(t)] = {"threshold": round(float(thr[idx]), 3), "tp": tp, "fp": fp,
                       "precision": round(tp / max(1, tp + fp), 3),
                       "recall": round(tp / max(1, pos), 3)}

    report = {
        "version": "discriminator_probe_exp1",
        "samples": {"total": int(len(y)), "pos": pos, "neg": neg,
                    "labels_used": len(used)},
        "auc_logo": round(float(auc), 4),
        "operating_points": ops,
        "note": "CAM++ 192-dim word embeddings; leave-one-human-label-out logistic regression",
    }
    (OUT / "exp1_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"saved={OUT / 'exp1_report.json'}", flush=True)


if __name__ == "__main__":
    main()
