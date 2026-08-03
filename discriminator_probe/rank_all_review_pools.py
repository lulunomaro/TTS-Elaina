# -*- coding: utf-8 -*-
"""Generate full scored lists for both review pools (all segments, not just top).

Read-only on radio_drama_work. Writes:
  anime_review_431_scored.list
  radio_review_448_scored.list
into discriminator_probe/. Each line: <blind/audio path>|<prob>|<source>|<start>-<end>|<duration>|<text>
"""
from __future__ import annotations

import json
from collections import defaultdict
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
                X.append(embs[i]); y.append(row["label"])
    X = np.stack(X).astype(np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    return X, np.asarray(y, int)


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
    labels = load_labels()
    X, y = build_training(labels)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(X, y)
    print(f"trained on {len(y)} words (pos={int(np.sum(y==1))}, neg={int(np.sum(y==0))})", flush=True)

    for name, manifest_path, out_name in (
        ("anime", ROOT / "anime_semantic_mining_v1" / "semantic_review_manifest.json", "anime_review_431_scored.list"),
        ("radio", ROOT / "full_semantic_mining_v2" / "semantic_review_manifest.json", "radio_review_448_scored.list"),
    ):
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = doc.get("samples") or doc.get("rows") or []
        scored = []
        for row in rows:
            p = score_segment(clf, str(row["source_id"]),
                              int(row["word_start_index"]), int(row["word_end_index"]))
            if p != p:
                continue
            scored.append({
                "prob": p,
                "source_id": str(row["source_id"]),
                "start": float(row["start"]),
                "end": float(row["end"]),
                "duration": float(row["duration"]),
                "text": str(row.get("text", "")).replace("|", " ")[:60],
                "blind_id": str(row.get("blind_id", "")),
                "audio": str(row.get("audio", "")),
            })
        scored.sort(key=lambda r: -r["prob"])
        lines = [
            f"{r['audio']}|prob={r['prob']:.4f}|{r['source_id']}|{r['start']:.2f}-{r['end']:.2f}|{r['duration']:.2f}s|{r['text']}"
            for r in scored
        ]
        out_path = OUT / out_name
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        buckets = {
            "p>=0.9": sum(1 for r in scored if r["prob"] >= 0.9),
            "0.7-0.9": sum(1 for r in scored if 0.7 <= r["prob"] < 0.9),
            "0.5-0.7": sum(1 for r in scored if 0.5 <= r["prob"] < 0.7),
            "0.3-0.5": sum(1 for r in scored if 0.3 <= r["prob"] < 0.5),
            "<0.3": sum(1 for r in scored if r["prob"] < 0.3),
        }
        print(f"{name}: scored={len(scored)} buckets={buckets}", flush=True)
        print(f"  saved={out_path}", flush=True)

    # also emit a compact summary json
    summary = {
        "version": "discriminator_scored_lists_v1",
        "model": "LogisticRegression on CAM++ 192d word embeddings",
        "training_words": int(len(y)),
        "lists": {
            "anime_431": str(OUT / "anime_review_431_scored.list"),
            "radio_448": str(OUT / "radio_review_448_scored.list"),
        },
        "suggested_policy": {
            "auto_pass": "prob >= 0.9 (precision ~0.99 at 0.9 per exp1; human-frozen recall 40%)",
            "auto_review": "0.7 <= prob < 0.9 (exp1 precision 0.956; human-frozen coverage to 91.7%)",
            "human_spot_check": "0.5 <= prob < 0.7",
            "auto_reject": "prob < 0.3",
            "note": "quality gate (NISQA/clicks/clipping) must still run on auto-pass; discriminator is speaker-only",
        },
    }
    (OUT / "scored_lists_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary saved={OUT / 'scored_lists_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
