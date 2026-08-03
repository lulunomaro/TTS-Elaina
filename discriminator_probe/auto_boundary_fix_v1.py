# -*- coding: utf-8 -*-
"""Auto boundary repair using discriminator word-level probability trajectory.

For every boundary-cut segment (v1/v2/v3), expands from the original vocals
in word steps and finds the maximal contiguous word run whose discriminator
probability stays above a threshold. Re-cuts the segment at the ASR word
boundaries of that run. No human listening required; produces a review page
only for the repaired output.

Read-only on radio_drama_work. Output in discriminator_probe/auto_boundary_fix_v1.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
from sklearn.linear_model import LogisticRegression

PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")
ROOT = Path(r"F:\GPTSOVITS\radio_drama_work")
OUT = PROBE / "auto_boundary_fix_v1"
AUDIO_DIR = OUT / "audio"
OUTPUT_SR = 32000
PROB_MIN = 0.70
MAX_SEC = 6.0
MIN_SEC = 1.0

RADIO_EMB = ROOT / "full_source_local_word_scores_v1" / "embedding_cache"
RADIO_WORDS = ROOT / "full_source_local_word_scores_v1" / "source_reports"
ANIME_EMB = ROOT / "anime_mining_v1" / "source_local_word_scores_v1" / "embedding_cache"
ANIME_WORDS = ROOT / "anime_mining_v1" / "source_local_word_scores_v1" / "source_reports"
RADIO_MANIFEST = ROOT / "full_track_manifest_v1.json"
ANIME_MANIFEST = ROOT / "anime_batch_v1" / "manifest.json"

BASE_LABEL_SOURCES = [
    (ROOT / "human_calibration_combined_v3" / "labels_resolved.json", {"target": 1, "pure_non_target": 0}),
    (ROOT / "human_anime_speaker_calibration_v3" / "labels_resolved.json", {"target": 1, "pure_non_target": 0}),
]
INTERVAL_SOURCES = [
    PROBE / "discriminator_priority_review_v1" / "human_feedback" / "training_intervals_for_next_discriminator.json",
    PROBE / "discriminator_priority_review_v2" / "human_feedback" / "training_intervals_for_next_discriminator.json",
]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_labels() -> list[dict]:
    rows = []
    for path, class_map in BASE_LABEL_SOURCES:
        doc = json.loads(path.read_text(encoding="utf-8"))
        samples = doc.get("samples", doc) if isinstance(doc, dict) else doc
        for row in samples:
            cls = str(row.get("final_class", ""))
            if cls in class_map:
                rows.append({"source_id": str(row["source_id"]), "start": float(row["start"]),
                             "end": float(row["end"]), "label": class_map[cls]})
    for path in INTERVAL_SOURCES:
        doc = json.loads(path.read_text(encoding="utf-8"))
        for row in doc.get("samples", []):
            rows.append({"source_id": str(row["source_id"]), "start": float(row["start"]),
                         "end": float(row["end"]), "label": int(row["label"])})
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
    if not X:
        return np.zeros((0, 192), np.float32), np.zeros(0, int)
    X = np.stack(X).astype(np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    return X, np.asarray(y, int)


def train_clf() -> LogisticRegression:
    labels = load_labels()
    X, y = build_training(labels)
    print(f"train: {len(y)} words (pos={int(np.sum(y==1))}, neg={int(np.sum(y==0))})", flush=True)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(X, y)
    return clf


def word_probs(clf, source_id) -> np.ndarray:
    embs = load_embs(source_id)
    if embs is None:
        return np.zeros(0)
    embs = embs / np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-8)
    return clf.predict_proba(embs)[:, 1]


def source_vocals(source_id) -> Path | None:
    manifest = json.loads((ANIME_MANIFEST if source_id.startswith("majo_") else RADIO_MANIFEST).read_text(encoding="utf-8"))
    for t in manifest.get("tracks", []):
        if t["id"] == source_id:
            return Path(t["vocals"])
    return None


def resample(audio, src, dst):
    if src == dst:
        return audio
    from scipy.signal import resample_poly
    g = np.gcd(src, dst)
    return resample_poly(audio, dst // g, src // g).astype(np.float32)


def micro_fade(audio: np.ndarray) -> np.ndarray:
    out = np.asarray(audio, dtype=np.float32).copy()
    count = min(len(out) // 4, max(1, round(OUTPUT_SR * 0.002)))
    if count:
        ramp = np.linspace(0.0, 1.0, count, dtype=np.float32)
        out[:count] *= ramp
        out[-count:] *= ramp[::-1]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    clf = train_clf()

    # collect all boundary-cut segments from v1/v2/v3 ingest
    segments = []
    for v, report_key in (("v1", "discriminator_priority_review_v1"), ("v2", "discriminator_priority_review_v2"),
                          ("v3", "discriminator_priority_review_v3")):
        rp = PROBE / report_key / "human_feedback" / "labels_resolved.json"
        if not rp.exists():
            continue
        doc = json.loads(rp.read_text(encoding="utf-8"))
        for row in doc.get("samples", []):
            if row.get("final_class") == "boundary_cut":
                segments.append({"source_id": row["source_id"], "start": float(row["start"]),
                                 "end": float(row["end"]), "origin": v,
                                 "blind_id": row.get("blind_id", "")})
    print(f"boundary segments: {len(segments)}", flush=True)

    repaired = []
    failed = []
    seen = set()
    for seg in segments:
        sid = seg["source_id"]
        key = (sid, round(seg["start"], 2), round(seg["end"], 2))
        if key in seen:
            continue
        seen.add(key)
        probs = word_probs(clf, sid)
        centers = word_centers(sid)
        if probs is None or len(probs) != len(centers) or not len(probs):
            failed.append({**seg, "reason": "no_embeddings"})
            continue
        # words whose center lies inside the original segment
        inside = np.flatnonzero((centers >= seg["start"]) & (centers <= seg["end"]))
        if not len(inside):
            failed.append({**seg, "reason": "no_words_inside"})
            continue
        # expand outward greedily while prob >= threshold
        lo_idx = int(inside[0])
        hi_idx = int(inside[-1])
        n = len(probs)
        while lo_idx > 0 and probs[lo_idx - 1] >= PROB_MIN:
            lo_idx -= 1
        while hi_idx + 1 < n and probs[hi_idx + 1] >= PROB_MIN:
            hi_idx += 1
        # trim trailing low-prob words inside run
        while hi_idx > lo_idx and probs[hi_idx] < PROB_MIN:
            hi_idx -= 1
        while lo_idx < hi_idx and probs[lo_idx] < PROB_MIN:
            lo_idx += 1

        rep = (ANIME_WORDS if sid.startswith("majo_") else RADIO_WORDS) / f"{sid}.json"
        doc = json.loads(rep.read_text(encoding="utf-8"))
        words = doc["word_rows"]
        new_start = max(0.0, float(words[lo_idx]["start"]) - 0.05)
        new_end = float(words[hi_idx]["end"]) + 0.05
        duration = new_end - new_start
        if duration < MIN_SEC or duration > MAX_SEC:
            failed.append({**seg, "reason": f"duration_{duration:.2f}"})
            continue

        vocals = source_vocals(sid)
        if vocals is None or not vocals.exists():
            failed.append({**seg, "reason": "no_vocals"})
            continue
        audio, sr = sf.read(str(vocals), dtype="float32", always_2d=True)
        mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
        a = max(0, round(new_start * sr))
        b = min(len(mono), round(new_end * sr))
        clip = resample(mono[a:b], sr, OUTPUT_SR)
        clip = micro_fade(clip)
        bid = f"AB{len(repaired) + 1:03d}"
        dest = AUDIO_DIR / f"{bid}.wav"
        sf.write(str(dest), clip, OUTPUT_SR, subtype="PCM_16")
        run_probs = probs[lo_idx:hi_idx + 1]
        repaired.append({
            "blind_id": bid,
            "source_id": sid,
            "origin_blind_id": seg["blind_id"],
            "origin": seg["origin"],
            "old_start": seg["start"], "old_end": seg["end"],
            "new_start": round(new_start, 3), "new_end": round(new_end, 3),
            "duration": round(duration, 3),
            "mean_prob": round(float(run_probs.mean()), 4),
            "min_prob": round(float(run_probs.min()), 4),
            "n_words": int(len(run_probs)),
            "audio": str(dest.resolve()),
        })
        print(f"  {bid} {sid} {seg['start']:.1f}-{seg['end']:.1f} -> {new_start:.2f}-{new_end:.2f} "
              f"p={run_probs.mean():.3f} n={len(run_probs)}", flush=True)

    write_json(OUT / "report.json", {
        "version": "auto_boundary_fix_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prob_min": PROB_MIN,
        "input_segments": len(segments),
        "unique_segments": len(seen),
        "repaired": len(repaired),
        "failed": failed,
        "samples": repaired,
    })
    # review page (for spot-check only, not required)
    items = []
    for r in repaired:
        src = f"audio/{Path(r['audio']).name}"
        items.append(f"<li><b>{r['blind_id']}</b> ({r['source_id']} {r['duration']:.1f}s p={r['mean_prob']:.3f}) "
                     f"<audio controls preload='none' src='{src}'></audio></li>")
    (OUT / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><h1>自动边界修复 v1</h1><p>共 %d 条自动修复。抽验即可。</p><ol>%s</ol>"
        % (len(repaired), "\n".join(items)), encoding="utf-8")
    print(json.dumps({"repaired": len(repaired), "failed": len(failed), "saved": str(OUT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
