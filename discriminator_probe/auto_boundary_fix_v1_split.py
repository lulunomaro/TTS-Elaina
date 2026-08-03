# -*- coding: utf-8 -*-
"""Second pass: split over-6s repaired runs into compliant sub-segments.

For runs that exceeded MAX_SEC, pick the best <=6s window inside the target
word run: maximize mean discriminator prob, require semantic completeness
(avoid starting/ending on particles), then export. Keeps everything in
discriminator_probe.
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
OUT = PROBE / "auto_boundary_fix_v1"
AUDIO_DIR = OUT / "audio"
OUTPUT_SR = 32000
MAX_SEC = 6.0
MIN_SEC = 1.5
PROB_MIN = 0.70

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


def main():
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    failed = report.get("failed", [])
    labels = load_labels()
    X, y = build_training(labels)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(X, y)

    added = []
    still_failed = []
    for row in failed:
        sid = row["source_id"]
        probs = clf.predict_proba(load_embs(sid) / np.maximum(
            np.linalg.norm(load_embs(sid), axis=1, keepdims=True), 1e-8))[:, 1]
        centers = word_centers(sid)
        rep = (ANIME_WORDS if sid.startswith("majo_") else RADIO_WORDS) / f"{sid}.json"
        doc = json.loads(rep.read_text(encoding="utf-8"))
        words = doc["word_rows"]
        if len(probs) != len(centers) or len(probs) != len(words):
            still_failed.append({**row, "reason": "mismatch"})
            continue
        # words in the expanded target run: those inside original segment
        inside = np.flatnonzero((centers >= row["start"]) & (centers <= row["end"]))
        if not len(inside):
            still_failed.append({**row, "reason": "no_words"})
            continue
        lo0, hi0 = int(inside[0]), int(inside[-1])
        # expand again same as pass1
        n = len(probs)
        lo, hi = lo0, hi0
        while lo > 0 and probs[lo - 1] >= PROB_MIN:
            lo -= 1
        while hi + 1 < n and probs[hi + 1] >= PROB_MIN:
            hi += 1
        while hi > lo and probs[hi] < PROB_MIN:
            hi -= 1
        while lo < hi and probs[lo] < PROB_MIN:
            lo += 1
        run = list(range(lo, hi + 1))
        if len(run) < 3:
            still_failed.append({**row, "reason": "run_too_short"})
            continue
        # find best <=6s sub-window maximizing mean prob (min 4 words)
        best = None
        for a in range(len(run)):
            for b in range(a + 3, len(run) + 1):
                seg = run[a:b]
                dur = float(words[seg[-1]]["end"]) - float(words[seg[0]]["start"]) + 0.1
                if dur > MAX_SEC or dur < MIN_SEC:
                    continue
                mean_p = float(probs[seg].mean())
                # semantic penalty: start/end particle
                s_tok = str(words[seg[0]].get("text", ""))
                e_tok = str(words[seg[-1]].get("text", ""))
                bad_start = s_tok in {"ら", "けど", "ので", "て", "が", "を", "に", "は", "も", "と", "で", "ば", "から", "し", "く", "の", "ん", "り", "え"}
                bad_end = e_tok in {"を", "が", "に", "は", "も", "と", "で", "て", "し", "さ", "く", "の", "へ", "や", "か", "ば", "から"}
                score = mean_p - (0.25 if bad_start else 0.0) - (0.25 if bad_end else 0.0)
                if best is None or score > best[0]:
                    best = (score, a, b, mean_p, bad_start, bad_end)
        if best is None:
            still_failed.append({**row, "reason": "no_window"})
            continue
        _, a, b, mean_p, bs, be = best
        sel = run[a:b]
        new_start = max(0.0, float(words[sel[0]]["start"]) - 0.05)
        new_end = float(words[sel[-1]]["end"]) + 0.05
        vocals = source_vocals(sid)
        if vocals is None or not vocals.exists():
            still_failed.append({**row, "reason": "no_vocals"})
            continue
        audio, sr = sf.read(str(vocals), dtype="float32", always_2d=True)
        mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
        a0 = max(0, round(new_start * sr))
        b0 = min(len(mono), round(new_end * sr))
        clip = resample(mono[a0:b0], sr, OUTPUT_SR)
        clip = micro_fade(clip)
        bid = f"AB{len(report['samples']) + len(added) + 1:03d}"
        dest = AUDIO_DIR / f"{bid}.wav"
        sf.write(str(dest), clip, OUTPUT_SR, subtype="PCM_16")
        added.append({
            "blind_id": bid,
            "source_id": sid,
            "origin_blind_id": row.get("blind_id", ""),
            "origin": row.get("origin", ""),
            "old_start": row["start"], "old_end": row["end"],
            "new_start": round(new_start, 3), "new_end": round(new_end, 3),
            "duration": round(new_end - new_start, 3),
            "mean_prob": round(float(mean_p), 4),
            "n_words": int(len(sel)),
            "audio": str(dest.resolve()),
            "sub_window": True,
        })
        print(f"  {bid} {sid} dur={new_end-new_start:.2f} p={mean_p:.3f} n={len(sel)}", flush=True)

    report["samples"].extend(added)
    report["failed"] = still_failed
    report["repaired"] = len(report["samples"])
    report["sub_window_added"] = len(added)
    report["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(report_path, report)
    print(json.dumps({"total_repaired": len(report["samples"]), "still_failed": len(still_failed),
                      "total_sec": round(sum(x["duration"] for x in report["samples"]), 1)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
