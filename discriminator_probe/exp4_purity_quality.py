# -*- coding: utf-8 -*-
"""Experiment 4: purity/quality validation of discriminator high-scoring clips.

Uses validation clips that were NEVER in training:
  - negatives: confirmed_non_target_or_mixed.txt + review-batch overlap/quality/uncertain
  - positives: all human_verified_* frozen clips (direct 11 + review batches + rescues)
Measures: negative exclusion rate, positive recall, and acoustic quality of
discriminator p>=0.9 anime review clips. Read-only on radio_drama_work.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from sklearn.linear_model import LogisticRegression

ROOT = Path(r"F:\GPTSOVITS\radio_drama_work")
OUT = Path(r"F:\GPTSOVITS\discriminator_probe")
OUT.mkdir(parents=True, exist_ok=True)

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


def score_clip_embs(clf, embs, w_start, w_end):
    if embs is None or w_end >= len(embs) or w_start < 0:
        return float("nan")
    seg = embs[w_start:w_end + 1]
    if not len(seg):
        return float("nan")
    seg = seg / np.maximum(np.linalg.norm(seg, axis=1, keepdims=True), 1e-8)
    return float(np.mean(clf.predict_proba(seg)[:, 1]))


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


def acoustic_metrics(path: Path) -> dict:
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
    except Exception as exc:
        return {"error": str(exc)}
    if sr != 32000:
        return {"sr": sr, "note": "non-32k"}
    frame, hop = 1024, 256
    count = max(1, 1 + max(0, len(audio) - frame) // hop)
    padded = np.pad(audio, (0, max(0, frame + (count - 1) * hop - len(audio))))
    frames = np.stack([padded[i * hop:i * hop + frame] for i in range(count)])
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    diff = np.abs(np.diff(audio))
    med = np.median(diff)
    mad = np.median(np.abs(diff - med))
    z = (diff - med) / max(1.4826 * mad, 1e-8)
    positions = np.flatnonzero(z > 40.0)
    click_groups = 0 if len(positions) == 0 else 1 + int(np.sum(np.diff(positions) > max(8, round(sr * 0.004))))
    low = rms <= np.percentile(rms, 20.0)
    speech = max(float(np.percentile(rms, 85.0)), 1e-7)
    floor = max(float(np.percentile(rms[low], 50.0)) if np.any(low) else float(np.percentile(rms, 20.0)), 1e-7)
    return {
        "duration": round(len(audio) / sr, 3),
        "clipping_fraction": round(float(np.mean(np.abs(audio) >= 0.999)), 5),
        "clicks_per_sec": round(click_groups / max(len(audio) / sr, 0.1), 2),
        "snr_proxy_db": round(float(20 * np.log10(speech / floor)), 1),
    }


def load_verified_positives() -> list[dict]:
    """All human-frozen anime clips, never in training (they were accepted AFTER training labels)."""
    out = []
    manifests = [
        ROOT / "human_verified_anime_semantic_v1_direct" / "audit_manifest.json",
        ROOT / "human_verified_anime_review_batch_v1" / "audit_manifest.json",
        ROOT / "human_verified_anime_review_batch_v2" / "audit_manifest.json",
        ROOT / "human_verified_anime_boundary_rescue_v1" / "audit_manifest.json",
    ]
    for path in manifests:
        if not path.exists():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        rows = doc.get("samples") or doc.get("rows") or []
        for row in rows:
            out.append({
                "source_id": str(row.get("source_id", "")),
                "start": float(row.get("start", row.get("source_start", 0))),
                "end": float(row.get("end", row.get("source_end", 0))),
                "human_key": str(row.get("human_key") or row.get("final_id") or row.get("blind_id") or ""),
                "audio": str(row.get("audio", "")),
            })
    return out


def load_validation_negatives() -> list[dict]:
    """Confirmed negatives + review-batch non-target clips, none in training."""
    out = []
    neg_list = ROOT / "confirmed_non_target_or_mixed.txt"
    if neg_list.exists():
        for line in neg_list.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|", 1)
            if len(parts) < 2:
                continue
            reason = parts[1].lower()
            if "mixed" in reason or "混" in reason:
                continue  # mixed is not a pure negative
            out.append({"kind": "confirmed_non_target", "source_id": "", "start": 0, "end": 0,
                        "audio": parts[0].strip(), "human_key": parts[0].strip()})
    # review-batch non-target/quality/uncertain clips
    for batch in ("v1", "v2"):
        fb_paths = list((ROOT / f"human_anime_semantic_review_batch_{batch}").glob("feedback_received*.json"))
        if not fb_paths:
            continue
        fb = json.loads(fb_paths[0].read_text(encoding="utf-8"))
        fb_by_id = {str(r["blind_id"]): r for r in fb.get("samples", [])}
        manifest = json.loads((ROOT / f"anime_semantic_review_batch_{batch}" / "private_manifest.json").read_text(encoding="utf-8"))
        for row in manifest.get("samples") or manifest.get("rows") or []:
            bid = str(row.get("batch_blind_id") or row.get("blind_id") or "")
            human = fb_by_id.get(bid)
            if not human or not bool(human.get("is_error")):
                continue
            err = str(human.get("error_type") or "")
            if err in {"混入/无缝接话", "重叠说话", "音质问题"}:
                out.append({"kind": f"review_{err}", "source_id": str(row["source_id"]),
                            "start": float(row["start"]), "end": float(row["end"]),
                            "human_key": bid, "audio": ""})
            elif "不确定" in err:
                out.append({"kind": "review_uncertain", "source_id": str(row["source_id"]),
                            "start": float(row["start"]), "end": float(row["end"]),
                            "human_key": bid, "audio": ""})
    return out


def main():
    labels = load_labels()
    X, y = build_training(labels)
    print(f"train: {len(y)} words (pos={int(np.sum(y==1))}, neg={int(np.sum(y==0))})", flush=True)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(X, y)

    # ---- 1. negative exclusion ----
    negs = load_validation_negatives()
    neg_scores = []
    for n in negs:
        if n["audio"]:
            # file-based negative: score via re-embed? we only have word embs for known sources.
            # confirmed negatives live in radio sources already covered; map by filename is hard.
            neg_scores.append({"kind": n["kind"], "score": float("nan"), "note": "file-based, skipped"})
        else:
            s = score_by_time(clf, n["source_id"], n["start"], n["end"])
            neg_scores.append({"kind": n["kind"], "human_key": n["human_key"], "score": s})
    scored_negs = [x for x in neg_scores if x["score"] == x["score"]]
    neg_arr = np.asarray([x["score"] for x in scored_negs])
    print(f"\n=== negatives (excl mixed): {len(scored_negs)} scored ===")
    if len(neg_arr):
        print(f"  mean={neg_arr.mean():.3f} median={np.median(neg_arr):.3f} max={neg_arr.max():.3f}")
        print(f"  frac<0.5 (correctly low): {np.mean(neg_arr<0.5):.3f}")
        print(f"  frac<0.3: {np.mean(neg_arr<0.3):.3f}")
        for x in sorted(scored_negs, key=lambda r: r["score"])[:15]:
            print(f"    {x['score']:.3f} {x['kind']} {x.get('human_key','')}")

    # ---- 2. positive recall on human-frozen clips ----
    poss = load_verified_positives()
    pos_scores = []
    for p in poss:
        if not p["source_id"]:
            continue
        s = score_by_time(clf, p["source_id"], p["start"], p["end"])
        pos_scores.append({"human_key": p["human_key"], "score": s})
    scored_pos = [x for x in pos_scores if x["score"] == x["score"]]
    pos_arr = np.asarray([x["score"] for x in scored_pos])
    print(f"\n=== human-frozen positives: {len(scored_pos)} scored ===")
    if len(pos_arr):
        print(f"  mean={pos_arr.mean():.3f} median={np.median(pos_arr):.3f} min={pos_arr.min():.3f}")
        print(f"  frac>=0.7: {np.mean(pos_arr>=0.7):.3f}  frac>=0.9: {np.mean(pos_arr>=0.9):.3f}")

    # ---- 3. acoustic quality of discriminator p>=0.9 anime review clips ----
    review = json.loads((ROOT / "anime_semantic_mining_v1" / "semantic_review_manifest.json").read_text(encoding="utf-8"))
    rows = review.get("samples") or review.get("rows") or []
    q_ok = []
    for row in rows:
        s = score_clip_embs(clf, load_embs(str(row["source_id"])),
                            int(row["word_start_index"]), int(row["word_end_index"]))
        if s != s:
            continue
        if s >= 0.9:
            audio_path = Path(str(row.get("audio", "")))
            if audio_path.exists():
                q_ok.append({"prob": s, "source_id": row["source_id"],
                             "blind_id": row.get("blind_id", ""), **acoustic_metrics(audio_path)})
    print(f"\n=== discriminator p>=0.9 anime review clips: {len(q_ok)} with audio ===")
    if q_ok:
        clk = np.asarray([x["clicks_per_sec"] for x in q_ok])
        clip = np.asarray([x["clipping_fraction"] for x in q_ok])
        snr = np.asarray([x["snr_proxy_db"] for x in q_ok])
        dur = np.asarray([x["duration"] for x in q_ok])
        clean = sum(1 for x in q_ok if x["clicks_per_sec"] <= 3 and x["clipping_fraction"] == 0)
        print(f"  duration: mean={dur.mean():.2f}s min={dur.min():.2f} max={dur.max():.2f}")
        print(f"  clicks/s: mean={clk.mean():.2f} max={clk.max():.2f}  <=3: {np.mean(clk<=3):.3f}")
        print(f"  clipping==0: {np.mean(clip==0):.3f}")
        print(f"  snr_db: mean={snr.mean():.1f} min={snr.min():.1f}")
        print(f"  PASS(click<=3 & clip==0): {clean}/{len(q_ok)} = {clean/len(q_ok):.3f}")

    report = {
        "version": "exp4_purity_quality",
        "negatives": {"scored": len(scored_negs),
                      "mean": float(neg_arr.mean()) if len(neg_arr) else None,
                      "frac_lt_0.5": float(np.mean(neg_arr < 0.5)) if len(neg_arr) else None},
        "positives": {"scored": len(scored_pos),
                      "mean": float(pos_arr.mean()) if len(pos_arr) else None,
                      "frac_ge_0.7": float(np.mean(pos_arr >= 0.7)) if len(pos_arr) else None,
                      "frac_ge_0.9": float(np.mean(pos_arr >= 0.9)) if len(pos_arr) else None},
        "high_conf_quality": {"n": len(q_ok),
                              "pass_fraction": round(clean / len(q_ok), 3) if q_ok else None,
                              "mean_snr_db": float(snr.mean()) if q_ok else None},
        "neg_detail": scored_negs,
        "pos_detail": scored_pos,
        "quality_detail": q_ok,
    }
    (OUT / "exp4_purity_quality.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved={OUT / 'exp4_purity_quality.json'}", flush=True)


if __name__ == "__main__":
    main()
