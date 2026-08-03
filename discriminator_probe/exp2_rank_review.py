# -*- coding: utf-8 -*-
"""Experiment 2: rank review pools with the trained discriminator.

Trains LR on all human labels (as exp1), scores every review-pool segment by
averaging word-level probabilities, and cross-checks against segments Luna
already human-accepted. Read-only on radio_drama_work.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(r"F:\GPTSOVITS\radio_drama_work")
OUT = Path(r"F:\GPTSOVITS\discriminator_probe")
OUT.mkdir(parents=True, exist_ok=True)

RADIO_EMB = ROOT / "full_source_local_word_scores_v1" / "embedding_cache"
RADIO_WORDS = ROOT / "full_source_local_word_scores_v1" / "source_reports"
ANIME_EMB = ROOT / "anime_mining_v1" / "source_local_word_scores_v1" / "embedding_cache"
ANIME_WORDS = ROOT / "anime_mining_v1" / "source_local_word_scores_v1" / "source_reports"

LABEL_SOURCES = [
    (ROOT / "human_calibration_combined_v3" / "labels_resolved.json",
     {"target": 1, "pure_non_target": 0}),
    (ROOT / "human_anime_speaker_calibration_v3" / "labels_resolved.json",
     {"target": 1, "pure_non_target": 0}),
]


def load_labels() -> list[dict]:
    rows = []
    for path, class_map in LABEL_SOURCES:
        if not path.exists():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        samples = doc.get("samples", doc) if isinstance(doc, dict) else doc
        for row in samples:
            cls = str(row.get("final_class", ""))
            if cls not in class_map:
                continue
            rows.append({
                "source_id": str(row["source_id"]),
                "start": float(row["start"]),
                "end": float(row["end"]),
                "label": class_map[cls],
                "human_key": str(row.get("human_key") or row.get("blind_id") or ""),
            })
    return rows


def load_embs(source_id: str):
    if source_id.startswith("majo_"):
        cache = ANIME_EMB / f"{source_id}.npz"
    else:
        cache = RADIO_EMB / f"{source_id}.npz"
    if not cache.exists():
        return None
    data = np.load(str(cache), allow_pickle=True)
    return np.asarray(data["embeddings"], dtype=np.float32)


def build_training(labels: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in labels:
        by_source[row["source_id"]].append(row)
    for source_id, rows in by_source.items():
        embs = load_embs(source_id)
        if embs is None:
            continue
        n = len(embs)
        starts = np.full(n, np.nan)
        ends = np.full(n, np.nan)
        # word indices are sequential in the embedding cache
        centers = np.arange(n, dtype=np.float64)  # placeholder; replaced below
        # Load word rows to get real centers
        if source_id.startswith("majo_"):
            rep = ANIME_WORDS / f"{source_id}.json"
        else:
            rep = RADIO_WORDS / f"{source_id}.json"
        if not rep.exists():
            continue
        doc = json.loads(rep.read_text(encoding="utf-8"))
        wrows = doc.get("word_rows", [])
        if len(wrows) != n:
            continue
        centers = np.asarray([(float(w["start"]) + float(w["end"])) / 2.0 for w in wrows])
        for row in rows:
            idx = np.flatnonzero((centers >= row["start"]) & (centers <= row["end"]))
            for i in idx:
                X.append(embs[i])
                y.append(row["label"])
    if not X:
        return np.zeros((0, 192), np.float32), np.zeros(0, int)
    X = np.stack(X).astype(np.float32)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    return X, np.asarray(y, dtype=int)


def score_segment(clf, source_id: str, w_start: int, w_end: int) -> float:
    embs = load_embs(source_id)
    if embs is None or w_end > len(embs):
        return float("nan")
    seg = embs[w_start:w_end + 1]
    if not len(seg):
        return float("nan")
    seg = seg / np.maximum(np.linalg.norm(seg, axis=1, keepdims=True), 1e-8)
    return float(np.mean(clf.predict_proba(seg)[:, 1]))


def load_review_manifest(path: Path) -> list[dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    return doc.get("samples") or doc.get("rows") or []


def main() -> None:
    labels = load_labels()
    X, y = build_training(labels)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    print(f"training: {len(y)} words (pos={pos}, neg={neg})", flush=True)
    if pos < 30 or neg < 30:
        print("TOO_FEW", flush=True)
        return

    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(X, y)

    results = {}
    for name, manifest_path in (
        ("anime_review_431", ROOT / "anime_semantic_mining_v1" / "semantic_review_manifest.json"),
        ("radio_review_448", ROOT / "full_semantic_mining_v2" / "semantic_review_manifest.json"),
    ):
        rows = load_review_manifest(manifest_path)
        scored = []
        for row in rows:
            p = score_segment(clf, str(row["source_id"]),
                              int(row["word_start_index"]), int(row["word_end_index"]))
            scored.append({
                "source_id": row["source_id"],
                "blind_id": row.get("blind_id", ""),
                "start": round(float(row["start"]), 3),
                "end": round(float(row["end"]), 3),
                "duration": round(float(row["duration"]), 3),
                "text": str(row.get("text", ""))[:40],
                "disc_prob": round(p, 4) if p == p else None,
            })
        scored.sort(key=lambda r: -(r["disc_prob"] or -1))
        results[name] = {
            "pool_size": len(scored),
            "scored": scored[:200],
            "counts_by_bucket": {
                "p>=0.9": sum(1 for r in scored if (r["disc_prob"] or 0) >= 0.9),
                "0.7<=p<0.9": sum(1 for r in scored if 0.7 <= (r["disc_prob"] or 0) < 0.9),
                "0.5<=p<0.7": sum(1 for r in scored if 0.5 <= (r["disc_prob"] or 0) < 0.7),
                "0.3<=p<0.5": sum(1 for r in scored if 0.3 <= (r["disc_prob"] or 0) < 0.5),
                "p<0.3": sum(1 for r in scored if (r["disc_prob"] or 0) < 0.3),
            },
        }
        print(f"\n=== {name} ===", flush=True)
        print("  buckets:", json.dumps(results[name]["counts_by_bucket"], ensure_ascii=False), flush=True)
        for r in scored[:10]:
            print(f"  {r['disc_prob']:.3f} {r['source_id']} {r['duration']:.1f}s {r['text']}", flush=True)

    (OUT / "exp2_ranking.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved={OUT / 'exp2_ranking.json'}", flush=True)


if __name__ == "__main__":
    main()
