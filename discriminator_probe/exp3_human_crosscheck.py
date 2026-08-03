# -*- coding: utf-8 -*-
"""Cross-check: discriminator score vs human labels on the 48+48 review batches."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

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
    wrows = doc.get("word_rows", [])
    return np.asarray([(float(w["start"]) + float(w["end"])) / 2.0 for w in wrows])


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


def score_clip(clf, sid, w_start, w_end):
    embs = load_embs(sid)
    if embs is None or w_end >= len(embs):
        return float("nan")
    seg = embs[w_start:w_end + 1]
    seg = seg / np.maximum(np.linalg.norm(seg, axis=1, keepdims=True), 1e-8)
    return float(np.mean(clf.predict_proba(seg)[:, 1]))


def main():
    labels = load_labels()
    X, y = build_training(labels)
    print(f"train: {len(y)} words (pos={int(np.sum(y==1))}, neg={int(np.sum(y==0))})", flush=True)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(X, y)

    def human_rows(batch: str) -> list[dict]:
        """Join feedback archive (blind_id -> error) with batch manifest (blind_id -> segment)."""
        fb_paths = list((ROOT / f"human_anime_semantic_review_batch_{batch}").glob("feedback_received*.json"))
        if not fb_paths:
            return []
        fb = json.loads(fb_paths[0].read_text(encoding="utf-8"))
        fb_by_id = {str(r["blind_id"]): r for r in fb.get("samples", [])}
        manifest = json.loads((ROOT / f"anime_semantic_review_batch_{batch}" / "private_manifest.json").read_text(encoding="utf-8"))
        rows = manifest.get("samples") or manifest.get("rows") or []
        out = []
        for row in rows:
            bid = str(row.get("batch_blind_id") or row.get("blind_id") or "")
            human = fb_by_id.get(bid)
            if not human:
                continue
            err = str(human.get("error_type") or "").strip()
            cls = "target" if not bool(human.get("is_error")) else \
                  ("boundary_cut" if err in {"开头被切", "结尾被切", "开头和结尾都被切", "切头/切尾"} else
                   "overlap" if err in {"混入/无缝接话", "重叠说话"} else
                   "quality_issue" if err == "音质问题" else "uncertain")
            prob = score_clip(clf, str(row["source_id"]),
                              int(row.get("word_start_index", -1)), int(row.get("word_end_index", -1)))
            out.append({"batch": batch, "blind_id": bid, "human_class": cls,
                        "error_type": err, "prob": prob})
        return out

    all_human = human_rows("v1") + human_rows("v2")

    classes = ["target", "boundary_cut", "overlap", "quality_issue", "uncertain"]
    print(f"\ncross-check: {len(all_human)} human-labeled clips (48+48)")
    print(f"{'human_class':<15} {'n':>3} {'mean_prob':>9} {'median':>7} {'p25':>6} {'p75':>6}")
    for c in classes:
        rows = [r for r in all_human if r["human_class"] == c]
        if not rows:
            continue
        ps = np.asarray([r["prob"] for r in rows])
        print(f"{c:<15} {len(rows):>3} {ps.mean():>9.3f} {np.median(ps):>7.3f} "
              f"{np.quantile(ps,0.25):>6.3f} {np.quantile(ps,0.75):>6.3f}")

    # AUC: target vs non-target (excluding boundary which is 'correct speaker, bad cut')
    clean = [r for r in all_human if r["human_class"] in ("target", "overlap", "quality_issue", "uncertain")]
    if clean:
        ys = np.asarray([1 if r["human_class"] == "target" else 0 for r in clean])
        ps = np.asarray([r["prob"] for r in clean])
        print(f"\nAUC (target vs overlap/quality/uncertain, n={len(clean)}): {roc_auc_score(ys, ps):.4f}")

    # What fraction of human 'boundary_cut' clips rank >= 0.7? (they are speaker-correct)
    bd = [r for r in all_human if r["human_class"] == "boundary_cut"]
    if bd:
        ps = np.asarray([r["prob"] for r in bd])
        print(f"boundary_cut clips: n={len(bd)}, frac>=0.7: {np.mean(ps>=0.7):.3f}, "
              f"frac>=0.9: {np.mean(ps>=0.9):.3f}, mean={ps.mean():.3f}")

    (OUT / "exp3_human_crosscheck.json").write_text(
        json.dumps({"human_clips": all_human}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={OUT / 'exp3_human_crosscheck.json'}", flush=True)


if __name__ == "__main__":
    main()
