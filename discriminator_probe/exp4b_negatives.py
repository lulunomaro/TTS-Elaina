# -*- coding: utf-8 -*-
"""Exp 4b: score the file-based confirmed negatives by mapping filename -> source/time."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(r"F:\GPTSOVITS\radio_drama_work")
OUT = Path(r"F:\GPTSOVITS\discriminator_probe")
RADIO_EMB = ROOT / "full_source_local_word_scores_v1" / "embedding_cache"
RADIO_WORDS = ROOT / "full_source_local_word_scores_v1" / "source_reports"
ANIME_EMB = ROOT / "anime_mining_v1" / "source_local_word_scores_v1" / "embedding_cache"
ANIME_WORDS = ROOT / "anime_mining_v1" / "source_local_word_scores_v1" / "source_reports"
LABEL_SOURCES = [
    (ROOT / "human_calibration_combined_v3" / "labels_resolved.json", {"target": 1, "pure_non_target": 0}),
    (ROOT / "human_anime_speaker_calibration_v3" / "labels_resolved.json", {"target": 1, "pure_non_target": 0}),
]


def load_labels():
    rows = []
    for path, class_map in LABEL_SOURCES:
        doc = json.loads(path.read_text(encoding="utf-8"))
        samples = doc.get("samples", doc) if isinstance(doc, dict) else doc
        for row in samples:
            cls = str(row.get("final_class", ""))
            if cls in class_map:
                rows.append({"source_id": str(row["source_id"]), "start": float(row["start"]),
                             "end": float(row["end"]), "label": class_map[cls]})
    return rows


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


def build_training(labels):
    X, y = [], []
    by_src = {}
    for r in labels:
        by_src.setdefault(r["source_id"], []).append(r)
    for sid, rows in by_src.items():
        embs, centers = load_embs(sid), word_centers(sid)
        if embs is None or centers is None or len(embs) != len(centers):
            continue
        for row in rows:
            idx = np.flatnonzero((centers >= row["start"]) & (centers <= row["end"]))
            for i in idx:
                X.append(embs[i]); y.append(row["label"])
    X = np.stack(X).astype(np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    return X, np.asarray(y, int)


def score_by_time(clf, source_id, start, end):
    embs, centers = load_embs(source_id), word_centers(source_id)
    if embs is None or centers is None:
        return float("nan")
    idx = np.flatnonzero((centers >= start) & (centers <= end))
    if not len(idx):
        return float("nan")
    seg = embs[idx]
    seg = seg / np.maximum(np.linalg.norm(seg, axis=1, keepdims=True), 1e-8)
    return float(np.mean(clf.predict_proba(seg)[:, 1]))


def parse_radio_clip(name: str) -> tuple[str, float, float] | None:
    # pattern: <id>-1-<res>_vocals_<start>_<end>.wav with 32k samples
    m = re.search(r"(\d{9,11})-1-(\d{2,3})_vocals_(\d{10})_(\d{10})\.wav$", name)
    if not m:
        return None
    return f"{m.group(1)}-1-{m.group(2)}", int(m.group(3)) / 32000.0, int(m.group(4)) / 32000.0


def parse_review_clip(name: str) -> tuple[str, float, float] | None:
    # <prefix><id>[_uN]-1-<res>_vocals_<a>_<b>_<s>_<e>.wav ; window = last 10-digit pair
    m = re.search(r"(\d{9,11}(?:_u\d)?)-1-\d{2,3}_vocals_(\d{10})_(\d{10})_(\d{10})_(\d{10})\.wav$", name)
    if not m:
        return None
    return m.group(1) + "-1-192", int(m.group(4)) / 32000.0, int(m.group(5)) / 32000.0


def main():
    labels = load_labels()
    X, y = build_training(labels)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(X, y)

    results = []
    for line in (ROOT / "confirmed_non_target_or_mixed.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|", 1)
        path = parts[0].strip()
        reason = parts[1].strip() if len(parts) > 1 else ""
        name = Path(path).name
        if "review_" in name:
            parsed = parse_review_clip(name)
        else:
            parsed = parse_radio_clip(name)
        if not parsed:
            results.append({"file": name, "reason": reason, "score": None, "note": "unparsed"})
            continue
        src, s, e = parsed
        score = score_by_time(clf, src, s, e)
        results.append({"file": name, "reason": reason, "source": src,
                        "start": round(s, 3), "end": round(e, 3), "score": score})

    print("=== confirmed negatives scored by discriminator ===")
    for r in results:
        print(f"  score={r.get('score')}  {r['reason'][:40]}  {r['file'][:70]}")

    scored = [r for r in results if r.get("score") is not None]
    if scored:
        arr = np.asarray([r["score"] for r in scored])
        print(f"\nscored: {len(scored)}, mean={arr.mean():.3f}, max={arr.max():.3f}, "
              f"frac<0.5={np.mean(arr<0.5):.3f}")
    (OUT / "exp4b_negatives.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={OUT / 'exp4b_negatives.json'}")


if __name__ == "__main__":
    main()
