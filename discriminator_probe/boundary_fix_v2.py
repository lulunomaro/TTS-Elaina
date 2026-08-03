# -*- coding: utf-8 -*-
"""Boundary fix v2: energy-valley cuts + particle-avoidance + tail completion.

v1 failed (14/15 still cut) because it cut at ASR word boundaries, and
faster-whisper word timestamps for Japanese are syllable fragments (は, が,
って...) that land mid-sentence. v2:
  1. picks the target word run by discriminator probability
  2. refines BOTH edges with energy valleys from the original waveform
  3. rejects start/end on particles, expanding outward until non-particle
  4. completes the tail by extending past the last ASR word end to the next
     real speech boundary (energy valley after it)
  5. iterates up to 3 rounds, then reports remaining failures
No human listening. Output in discriminator_probe/boundary_fix_v2.
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
OUT = PROBE / "boundary_fix_v2"
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
RADIO_ASR = ROOT / "word_target_all_v1" / "asr_cache"
ANIME_ASR = ROOT / "anime_mining_v1" / "asr" / "source_reports"

BASE_LABEL_SOURCES = [
    (ROOT / "human_calibration_combined_v3" / "labels_resolved.json", {"target": 1, "pure_non_target": 0}),
    (ROOT / "human_anime_speaker_calibration_v3" / "labels_resolved.json", {"target": 1, "pure_non_target": 0}),
]
INTERVAL_SOURCES = [
    PROBE / "discriminator_priority_review_v1" / "human_feedback" / "training_intervals_for_next_discriminator.json",
    PROBE / "discriminator_priority_review_v2" / "human_feedback" / "training_intervals_for_next_discriminator.json",
]

# particles / fragments that can never start or end a clean segment
BAD_START = {"ら", "けど", "ので", "て", "が", "を", "に", "は", "も", "と", "で", "ば",
             "から", "れる", "さ", "し", "く", "の", "ん", "り", "え", "って", "だ", "でしょ"}
BAD_END = {"を", "が", "に", "は", "も", "と", "で", "て", "し", "さ", "く", "の", "へ",
           "や", "か", "ば", "から", "ので", "けど", "なら", "たり", "って", "だ"}


def write_json(path: Path, value: object) -> None:
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


def load_asr_words(source_id):
    rep = (ANIME_ASR if source_id.startswith("majo_") else RADIO_ASR) / f"{source_id}.json"
    if not rep.exists():
        return []
    doc = json.loads(rep.read_text(encoding="utf-8"))
    return doc.get("words", [])


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


def frame_rms(audio, sr, frame_sec=0.020, hop_sec=0.010):
    frame = max(1, round(sr * frame_sec))
    hop = max(1, round(sr * hop_sec))
    working = np.pad(audio, (0, max(0, frame - len(audio))))
    count = max(1, 1 + (len(working) - frame) // hop)
    starts = np.arange(count, dtype=np.int64) * hop
    ends = starts + frame
    squared = np.square(working, dtype=np.float64)
    prefix = np.empty(len(squared) + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(squared, out=prefix[1:])
    rms = np.sqrt((prefix[ends] - prefix[starts]) / frame + 1e-12)
    return (starts + frame / 2.0) / sr, rms


def valley_time(times, rms, lo, hi, speech_level):
    idx = np.flatnonzero((times >= lo) & (times <= hi))
    if not len(idx):
        return (lo + hi) / 2.0
    return float(times[idx[int(np.argmin(rms[idx]))]])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    X, y = build_training(labels)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(X, y)
    print(f"train: {len(y)} words", flush=True)

    # collect all boundary-cut segments
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
        embs = load_embs(sid)
        wrows = load_word_rows(sid)
        if embs is None or not wrows or len(embs) != len(wrows):
            failed.append({**seg, "reason": "no_embeddings"}); continue
        centers = np.asarray([(float(w["start"]) + float(w["end"])) / 2.0 for w in wrows])
        probs = clf.predict_proba(embs / np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-8))[:, 1]
        inside = np.flatnonzero((centers >= seg["start"]) & (centers <= seg["end"]))
        if not len(inside):
            failed.append({**seg, "reason": "no_words"}); continue
        lo0, hi0 = int(inside[0]), int(inside[-1])
        n = len(probs)
        lo, hi = lo0, hi0
        while lo > 0 and probs[lo - 1] >= PROB_MIN:
            lo -= 1
        while hi + 1 < n and probs[hi + 1] >= PROB_MIN:
            hi += 1
        # particle avoidance on start: move forward until non-particle
        while lo <= hi and str(wrows[lo].get("text", "")).strip() in BAD_START:
            lo += 1
        # particle avoidance on end: move backward until non-particle
        while hi >= lo and str(wrows[hi].get("text", "")).strip() in BAD_END:
            hi -= 1
        if hi <= lo:
            failed.append({**seg, "reason": "all_particles"}); continue

        vocals = source_vocals(sid)
        if vocals is None or not vocals.exists():
            failed.append({**seg, "reason": "no_vocals"}); continue
        audio, sr = sf.read(str(vocals), dtype="float32", always_2d=True)
        mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
        times, rms = frame_rms(mono, sr)
        speech_level = float(np.percentile(rms[rms > 1e-5], 70)) if np.any(rms > 1e-5) else 1e-5

        # start edge: search valley between previous word end and first word start
        first_word_start = float(wrows[lo]["start"])
        prev_end = float(wrows[lo - 1]["end"]) if lo > 0 else max(0.0, first_word_start - 0.35)
        gap = max(0.0, first_word_start - prev_end)
        if gap >= 0.025:
            s_lo = prev_end + min(0.015, gap * 0.15)
            s_hi = first_word_start - min(0.015, gap * 0.15)
        else:
            s_lo = max(0.0, first_word_start - 0.16)
            s_hi = first_word_start + 0.04
        new_start = max(0.0, valley_time(times, rms, s_lo, s_hi, speech_level) - 0.010)

        # end edge: search valley between last word end and next word start
        last_word_end = float(wrows[hi]["end"])
        next_start = float(wrows[hi + 1]["start"]) if hi + 1 < n else last_word_end + 0.35
        gap = max(0.0, next_start - last_word_end)
        if gap >= 0.025:
            e_lo = last_word_end + min(0.015, gap * 0.15)
            e_hi = next_start - min(0.015, gap * 0.15)
        else:
            e_lo = last_word_end - 0.04
            e_hi = last_word_end + 0.16
        new_end = valley_time(times, rms, e_lo, e_hi, speech_level) + 0.010

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
            "duration": round(duration, 3),
            "first_token": str(wrows[lo].get("text", "")), "last_token": str(wrows[hi].get("text", "")),
            "n_words": int(hi - lo + 1), "mean_prob": round(float(probs[lo:hi + 1].mean()), 4),
            "audio": str(dest.resolve()),
        })
        print(f"  {bid} {sid} {seg['start']:.1f}-{seg['end']:.1f} -> {new_start:.2f}-{new_end:.2f} "
              f"首={wrows[lo].get('text','')} 尾={wrows[hi].get('text','')} p={probs[lo:hi+1].mean():.3f}", flush=True)

    write_json(OUT / "report.json", {
        "version": "boundary_fix_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prob_min": PROB_MIN,
        "input": len(segments), "unique": len(seen),
        "repaired": len(repaired), "failed": failed, "samples": repaired,
    })
    print(json.dumps({"repaired": len(repaired), "failed": len(failed), "saved": str(OUT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
