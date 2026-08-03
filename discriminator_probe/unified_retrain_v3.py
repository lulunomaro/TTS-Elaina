# -*- coding: utf-8 -*-
"""Unified discriminator retrain: ALL human labels (base + v1 + v2 intervals).

Merges 204 base labels + 48 v1 + 44 v2 training intervals, dedupes by
(source_id, start, end), rebuilds word-level training set, LOO-validates,
and re-scores both review pools. Outputs v3 scored lists. Isolated in
discriminator_probe; never touches radio_drama_work.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(r"F:\GPTSOVITS\radio_drama_work")
PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")
OUT = PROBE / "discriminator_rank_v3"
RADIO_EMB = ROOT / "full_source_local_word_scores_v1" / "embedding_cache"
RADIO_WORDS = ROOT / "full_source_local_word_scores_v1" / "source_reports"
ANIME_EMB = ROOT / "anime_mining_v1" / "source_local_word_scores_v1" / "embedding_cache"
ANIME_WORDS = ROOT / "anime_mining_v1" / "source_local_word_scores_v1" / "source_reports"

BASE_LABEL_SOURCES = [
    (ROOT / "human_calibration_combined_v3" / "labels_resolved.json", {"target": 1, "pure_non_target": 0}),
    (ROOT / "human_anime_speaker_calibration_v3" / "labels_resolved.json", {"target": 1, "pure_non_target": 0}),
]
INTERVAL_SOURCES = [
    PROBE / "discriminator_priority_review_v1" / "human_feedback" / "training_intervals_for_next_discriminator.json",
    PROBE / "discriminator_priority_review_v2" / "human_feedback" / "training_intervals_for_next_discriminator.json",
]


def load_labels() -> list[dict]:
    rows: list[dict] = []
    for path, class_map in BASE_LABEL_SOURCES:
        if not path.exists():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        samples = doc.get("samples", doc) if isinstance(doc, dict) else doc
        for row in samples:
            cls = str(row.get("final_class", ""))
            if cls in class_map:
                rows.append({
                    "source_id": str(row["source_id"]),
                    "start": float(row["start"]),
                    "end": float(row["end"]),
                    "label": class_map[cls],
                    "human_key": str(row.get("human_key") or row.get("blind_id") or "base"),
                    "origin": "base",
                })
    for path in INTERVAL_SOURCES:
        if not path.exists():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        for row in doc.get("samples", []):
            rows.append({
                "source_id": str(row["source_id"]),
                "start": float(row["start"]),
                "end": float(row["end"]),
                "label": int(row["label"]),
                "human_key": str(row.get("blind_id") or "interval"),
                "origin": path.parent.parent.name,
            })
    # dedupe by (source_id, start, end) keeping first
    seen = set()
    out = []
    for row in rows:
        key = (row["source_id"], round(row["start"], 3), round(row["end"], 3))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def load_embs(source_id):
    cache = (ANIME_EMB if source_id.startswith("majo_") else RADIO_EMB) / f"{source_id}.npz"
    if not cache.exists():
        return None
    return np.asarray(np.load(str(cache), allow_pickle=True)["embeddings"], dtype=np.float32)


def word_centers(source_id):
    rep = (ANIME_WORDS if source_id.startswith("majo_") else RADIO_WORDS) / f"{source_id}.json"
    if not rep.exists():
        return None
    doc = json.loads(rep.read_text(encoding="utf-8"))
    return np.asarray([(float(w["start"]) + float(w["end"])) / 2.0 for w in doc.get("word_rows", [])])


def build_dataset(labels: list[dict]):
    X, y, groups = [], [], []
    by_src = defaultdict(list)
    for r in labels:
        by_src[r["source_id"]].append(r)
    for sid, rows in by_src.items():
        embs, centers = load_embs(sid), word_centers(sid)
        if embs is None or centers is None or len(embs) != len(centers):
            continue
        for row in rows:
            idx = np.flatnonzero((centers >= row["start"]) & (centers <= row["end"]))
            for i in idx:
                X.append(embs[i])
                y.append(row["label"])
                groups.append(row["human_key"])
    if not X:
        return np.zeros((0, 192), np.float32), np.zeros(0, int), np.zeros(0, object)
    X = np.stack(X).astype(np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    return X, np.asarray(y, int), np.asarray(groups, object)


def score_segment(clf, source_id, w_start, w_end):
    embs = load_embs(source_id)
    if embs is None or w_end >= len(embs) or w_start < 0:
        return float("nan")
    seg = embs[w_start:w_end + 1]
    if not len(seg):
        return float("nan")
    seg = seg / np.maximum(np.linalg.norm(seg, axis=1, keepdims=True), 1e-8)
    return float(np.mean(clf.predict_proba(seg)[:, 1]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    pos_l = sum(1 for r in labels if r["label"] == 1)
    neg_l = sum(1 for r in labels if r["label"] == 0)
    print(f"merged labels: {len(labels)} (pos={pos_l}, neg={neg_l})", flush=True)

    X, y, groups = build_dataset(labels)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    print(f"word samples: {len(y)} (pos={pos}, neg={neg}), groups={len(set(groups))}", flush=True)

    # LOO validation
    logo = LeaveOneGroupOut()
    preds = np.zeros(len(y), dtype=np.float64)
    for tr, te in logo.split(X, y, groups):
        clf = LogisticRegression(max_iter=3000, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        preds[te] = clf.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y, preds)
    print(f"LOO AUC = {auc:.4f}", flush=True)

    # final model on all data
    final = LogisticRegression(max_iter=3000, class_weight="balanced")
    final.fit(X, y)

    results = {}
    for name, manifest_path, out_name in (
        ("anime", ROOT / "anime_semantic_mining_v1" / "semantic_review_manifest.json", "anime_review_431_scored_v3.list"),
        ("radio", ROOT / "full_semantic_mining_v2" / "semantic_review_manifest.json", "radio_review_448_scored_v3.list"),
    ):
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = doc.get("samples") or doc.get("rows") or []
        scored = []
        for row in rows:
            p = score_segment(final, str(row["source_id"]),
                              int(row["word_start_index"]), int(row["word_end_index"]))
            if p != p:
                continue
            scored.append({"prob": p, "source_id": str(row["source_id"]),
                           "start": float(row["start"]), "end": float(row["end"]),
                           "duration": float(row["duration"]),
                           "text": str(row.get("text", "")).replace("|", " ")[:60],
                           "audio": str(row.get("audio", ""))})
        scored.sort(key=lambda r: -r["prob"])
        lines = [f"{r['audio']}|prob={r['prob']:.4f}|{r['source_id']}|{r['start']:.2f}-{r['end']:.2f}|{r['duration']:.2f}s|{r['text']}" for r in scored]
        (OUT / out_name).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        buckets = {"p>=0.9": sum(1 for r in scored if r["prob"] >= 0.9),
                   "0.7-0.9": sum(1 for r in scored if 0.7 <= r["prob"] < 0.9),
                   "0.5-0.7": sum(1 for r in scored if 0.5 <= r["prob"] < 0.7),
                   "<0.5": sum(1 for r in scored if r["prob"] < 0.5)}
        results[name] = {"scored": len(scored), "buckets": buckets}
        print(f"{name}: {results[name]}", flush=True)

    summary = {
        "version": "discriminator_rank_v3",
        "merged_labels": {"total": len(labels), "pos": pos_l, "neg": neg_l},
        "word_samples": {"total": int(len(y)), "pos": pos, "neg": neg},
        "loo_auc": round(float(auc), 4),
        "pools": results,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"saved={OUT}", flush=True)


if __name__ == "__main__":
    main()
