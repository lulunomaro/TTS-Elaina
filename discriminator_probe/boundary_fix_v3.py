# -*- coding: utf-8 -*-
"""Boundary fix v3: pure energy-envelope cutting, ignore ASR word edges.

v1/v2 failed because faster-whisper Japanese word timestamps are single-kana
fragments whose edges cut inside content words. v3 never uses word edges.
For each boundary-cut segment it:
  1. loads the original BS-Roformer vocal waveform
  2. computes frame RMS (20ms/10ms)
  3. finds the true speech interval by scanning a wide window around the
     target [start-1.2s, end+1.2s]:
       - compute a smoothed energy profile
       - find leftmost and rightmost points where energy exceeds a floor
       - refine edges to nearest deep valleys outside those points
  4. also uses discriminator word-prob to bound the interval (speaker sanity)
  5. trims 40ms fade, exports 32k mono
Output: discriminator_probe/boundary_fix_v3/  (no radio_drama_work writes)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
from sklearn.linear_model import LogisticRegression

PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")
ROOT = Path(r"F:\GPTSOVITS\radio_drama_work")
OUT = PROBE / "boundary_fix_v3"
AUDIO_DIR = OUT / "audio"
OUTPUT_SR = 32000
MAX_SEC = 6.0
MIN_SEC = 1.2
FRAME = 0.020
HOP = 0.010

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


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_labels():
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


def load_word_rows(source_id):
    rep = (ANIME_WORDS if source_id.startswith("majo_") else RADIO_WORDS) / f"{source_id}.json"
    if not rep.exists():
        return []
    doc = json.loads(rep.read_text(encoding="utf-8"))
    return doc.get("word_rows", [])


def build_training(labels):
    X, y = [], []
    by_src = {}
    for r in labels:
        by_src.setdefault(r["source_id"], []).append(r)
    for sid, rows in by_src.items():
        embs = load_embs(sid)
        wrows = load_word_rows(sid)
        if embs is None or not wrows or len(embs) != len(wrows):
            continue
        centers = np.asarray([(float(w["start"]) + float(w["end"])) / 2.0 for w in wrows])
        for row in rows:
            idx = np.flatnonzero((centers >= row["start"]) & (centers <= row["end"]))
            for i in idx:
                X.append(embs[i]); y.append(row["label"])
    if not X:
        return np.zeros((0, 192), np.float32), np.zeros(0, int)
    X = np.stack(X).astype(np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    return X, np.asarray(y, int)


def source_vocals(source_id):
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


def micro_fade(audio):
    out = np.asarray(audio, dtype=np.float32).copy()
    count = min(len(out) // 4, max(1, round(OUTPUT_SR * 0.002)))
    if count:
        ramp = np.linspace(0.0, 1.0, count, dtype=np.float32)
        out[:count] *= ramp
        out[-count:] *= ramp[::-1]
    return out


def energy_edges(mono, sr, center_lo, center_hi, margin=1.2):
    """Find true speech interval around [center_lo, center_hi] by energy."""
    lo = max(0, round((center_lo - margin) * sr))
    hi = min(len(mono), round((center_hi + margin) * sr))
    seg = mono[lo:hi]
    times = np.arange(lo, hi) / sr
    # frame RMS
    frame = max(1, round(sr * FRAME))
    hop = max(1, round(sr * HOP))
    n_frames = max(1, 1 + (len(seg) - frame) // hop)
    starts = np.arange(n_frames, dtype=np.int64) * hop
    ends = starts + frame
    squared = np.square(seg, dtype=np.float64)
    prefix = np.empty(len(squared) + 1)
    prefix[0] = 0.0
    np.cumsum(squared, out=prefix[1:])
    rms = np.sqrt((prefix[ends] - prefix[starts]) / frame + 1e-12)
    ftimes = (starts + frame / 2.0) / sr + lo / sr

    # floor: 15th percentile of rms inside the target region only
    t_lo = max(ftimes[0], center_lo - 0.1)
    t_hi = min(ftimes[-1], center_hi + 0.1)
    inner = rms[(ftimes >= t_lo) & (ftimes <= t_hi)]
    floor = float(np.percentile(inner, 15)) if len(inner) else 1e-4
    active = rms > max(floor * 1.6, 1e-4)

    # find active span
    idx = np.flatnonzero(active)
    if not len(idx):
        return None, None
    left_i, right_i = int(idx[0]), int(idx[-1])

    # refine left: nearest valley in [ftimes[left_i]-0.3, ftimes[left_i]]
    l_idx = np.flatnonzero((ftimes >= ftimes[left_i] - 0.30) & (ftimes <= ftimes[left_i]))
    if len(l_idx):
        left_i = int(l_idx[np.argmin(rms[l_idx])])
    r_idx = np.flatnonzero((ftimes >= ftimes[right_i]) & (ftimes <= ftimes[right_i] + 0.30))
    if len(r_idx):
        right_i = int(r_idx[np.argmin(rms[r_idx])])

    return float(ftimes[left_i]), float(ftimes[right_i])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    X, y = build_training(labels)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(X, y)
    print(f"train: {len(y)} words", flush=True)

    segments = []
    for v, key in (("v1", "discriminator_priority_review_v1"), ("v2", "discriminator_priority_review_v2"),
                   ("v3", "discriminator_priority_review_v3")):
        rp = PROBE / key / "human_feedback" / "labels_resolved.json"
        if not rp.exists():
            continue
        doc = json.loads(rp.read_text(encoding="utf-8"))
        for row in doc.get("samples", []):
            if row.get("final_class") == "boundary_cut":
                segments.append({"source_id": row["source_id"], "start": float(row["start"]),
                                 "end": float(row["end"]), "origin": v, "blind_id": row.get("blind_id", "")})
    print(f"boundary segments: {len(segments)}", flush=True)

    repaired, failed = [], []
    seen = set()
    for seg in segments:
        sid = seg["source_id"]
        key = (sid, round(seg["start"], 2), round(seg["end"], 2))
        if key in seen:
            continue
        seen.add(key)
        vocals = source_vocals(sid)
        if vocals is None or not vocals.exists():
            failed.append({**seg, "reason": "no_vocals"}); continue
        audio, sr = sf.read(str(vocals), dtype="float32", always_2d=True)
        mono = np.asarray(audio.mean(axis=1), dtype=np.float32)

        new_start, new_end = energy_edges(mono, sr, seg["start"], seg["end"])
        if new_start is None:
            failed.append({**seg, "reason": "no_energy"}); continue
        # clamp by discriminator: expand to include prob>=0.7 words
        embs = load_embs(sid)
        wrows = load_word_rows(sid)
        if embs is not None and wrows and len(embs) == len(wrows):
            probs = clf.predict_proba(embs / np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-8))[:, 1]
            centers = np.asarray([(float(w["start"]) + float(w["end"])) / 2.0 for w in wrows])
            inside = np.flatnonzero((centers >= seg["start"]) & (centers <= seg["end"]))
            if len(inside):
                lo0, hi0 = int(inside[0]), int(inside[-1])
                n = len(probs)
                lo, hi = lo0, hi0
                while lo > 0 and probs[lo - 1] >= 0.7:
                    lo -= 1
                while hi + 1 < n and probs[hi + 1] >= 0.7:
                    hi += 1
                # energy edges must cover the prob interval
                new_start = min(new_start, float(wrows[lo]["start"]) - 0.03)
                new_end = max(new_end, float(wrows[hi]["end"]) + 0.03)

        duration = new_end - new_start
        if duration < MIN_SEC or duration > MAX_SEC:
            failed.append({**seg, "reason": f"duration_{duration:.2f}"}); continue

        a = max(0, round(new_start * sr))
        b = min(len(mono), round(new_end * sr))
        clip = micro_fade(resample(mono[a:b], sr, OUTPUT_SR))
        bid = f"BF{len(repaired) + 1:03d}"
        dest = AUDIO_DIR / f"{bid}.wav"
        sf.write(str(dest), clip, OUTPUT_SR, subtype="PCM_16")
        repaired.append({
            "blind_id": bid, "source_id": sid, "origin": seg["origin"],
            "origin_blind_id": seg["blind_id"],
            "old_start": seg["start"], "old_end": seg["end"],
            "new_start": round(new_start, 3), "new_end": round(new_end, 3),
            "duration": round(duration, 3), "audio": str(dest.resolve()),
        })
        print(f"  {bid} {sid} {seg['start']:.1f}-{seg['end']:.1f} -> {new_start:.2f}-{new_end:.2f}", flush=True)

    write_json(OUT / "report.json", {
        "version": "boundary_fix_v3", "created_at": datetime.now(timezone.utc).isoformat(),
        "input": len(segments), "unique": len(seen),
        "repaired": len(repaired), "failed": failed, "samples": repaired,
    })
    print(json.dumps({"repaired": len(repaired), "failed": len(failed), "saved": str(OUT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
