# -*- coding: utf-8 -*-
"""Boundary fix v4: sentence-level cutting (the reliable fix).

v1-v3 failed because faster-whisper *word* timestamps are kana fragments.
v4 cuts at *sentence* boundaries:
  - anime: use ASR segment_index (whisper sentence segments)
  - radio: reconstruct sentences by grouping words with gap>=0.30s
For each boundary-cut segment, pick the sentence(s) overlapping the target,
extend to full sentence span, verify speaker with discriminator (mean prob),
then export. No human listening needed for generation.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
from sklearn.linear_model import LogisticRegression

PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")
ROOT = Path(r"F:\GPTSOVITS\radio_drama_work")
OUT = PROBE / "boundary_fix_v4"
AUDIO_DIR = OUT / "audio"
OUTPUT_SR = 32000
MAX_SEC = 6.0
MIN_SEC = 1.0
SENT_GAP = 0.30

RADIO_EMB = ROOT / "full_source_local_word_scores_v1" / "embedding_cache"
RADIO_WORDS = ROOT / "full_source_local_word_scores_v1" / "source_reports"
ANIME_EMB = ROOT / "anime_mining_v1" / "source_local_word_scores_v1" / "embedding_cache"
ANIME_WORDS = ROOT / "anime_mining_v1" / "source_local_word_scores_v1" / "source_reports"
ANIME_ASR = ROOT / "anime_mining_v1" / "asr" / "source_reports"
RADIO_ASR = ROOT / "word_target_all_v1" / "asr_cache"
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


def sentences_from_words(words: list[dict]) -> list[tuple[float, float, str]]:
    """Group words into sentences by gap>=SENT_GAP."""
    out = []
    if not words:
        return out
    cur = [words[0]]
    for w in words[1:]:
        if w["start"] - cur[-1]["end"] >= SENT_GAP:
            out.append((cur[0]["start"], cur[-1]["end"], "".join(x["text"] for x in cur)))
            cur = [w]
        else:
            cur.append(w)
    out.append((cur[0]["start"], cur[-1]["end"], "".join(x["text"] for x in cur)))
    return out


def sentences_from_segments(seg_words: dict[int, list[dict]]) -> list[tuple[float, float, str]]:
    out = []
    for k in sorted(seg_words):
        ws = seg_words[k]
        out.append((float(ws[0]["start"]), float(ws[-1]["end"]), "".join(w["text"] for w in ws)))
    return out


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
        asr_words = load_asr_words(sid)
        if sid.startswith("majo_"):
            seg_groups = defaultdict(list)
            for w in asr_words:
                seg_groups[w.get("segment_index", 0)].append(w)
            sents = sentences_from_segments(seg_groups)
        else:
            sents = sentences_from_words(asr_words)
        if not sents:
            failed.append({**seg, "reason": "no_sentences"}); continue

        # pick sentence with max overlap with the target segment
        best = None
        for s, e, text in sents:
            ov = min(e, seg["end"]) - max(s, seg["start"])
            if ov > 0:
                if best is None or ov > best[0]:
                    best = (ov, s, e, text)
        if best is None:
            center = (seg["start"] + seg["end"]) / 2.0
            best = min(((abs((s + e) / 2 - center), s, e, text) for s, e, text in sents), key=lambda x: x[0])
            best = (0.0, best[1], best[2], best[3])
        _, s_start, s_end, s_text = best

        # ---- fix: roll sentence start back to energy valley (bounded 0.15s) ----
        vocals = source_vocals(sid)
        if vocals is None or not vocals.exists():
            failed.append({**seg, "reason": "no_vocals"}); continue
        audio, sr = sf.read(str(vocals), dtype="float32", always_2d=True)
        mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
        # frame RMS
        frame = max(1, round(sr * 0.020)); hop = max(1, round(sr * 0.010))
        n_fr = max(1, 1 + (len(mono) - frame) // hop)
        fr_starts = np.arange(n_fr, dtype=np.int64) * hop
        fr_ends = fr_starts + frame
        sq = np.square(mono, dtype=np.float64)
        pref = np.empty(len(sq) + 1); pref[0] = 0.0
        np.cumsum(sq, out=pref[1:])
        rms = np.sqrt((pref[fr_ends] - pref[fr_starts]) / frame + 1e-12)
        ftimes = (fr_starts + frame / 2.0) / sr
        if not sid.startswith("majo_"):
            # radio: allow 0.15s roll-back to valley
            w_lo = max(ftimes[0], s_start - 0.15)
            idx = np.flatnonzero((ftimes >= w_lo) & (ftimes <= s_start))
            if len(idx):
                inner = rms[(ftimes >= s_start - 0.05) & (ftimes <= s_start + 0.05)]
                floor = float(np.percentile(inner, 30)) if len(inner) else 1e-4
                valleys = idx[rms[idx] < max(floor * 1.5, 1e-4)]
                if len(valleys):
                    s_start = float(ftimes[valleys[-1]])
            w_lo2 = s_end
            w_hi2 = min(ftimes[-1], s_end + 0.15)
            idx2 = np.flatnonzero((ftimes >= w_lo2) & (ftimes <= w_hi2))
            if len(idx2):
                inner2 = rms[(ftimes >= s_end - 0.05) & (ftimes <= s_end + 0.05)]
                floor2 = float(np.percentile(inner2, 30)) if len(inner2) else 1e-4
                valleys2 = idx2[rms[idx2] < max(floor2 * 1.5, 1e-4)]
                if len(valleys2):
                    s_end = float(ftimes[valleys2[0]])

        # verify speaker with discriminator over the sentence
        embs = load_embs(sid)
        wrows = load_word_rows(sid)
        if embs is not None and wrows and len(embs) == len(wrows):
            centers = np.asarray([(float(w["start"]) + float(w["end"])) / 2.0 for w in wrows])
            idx = np.flatnonzero((centers >= s_start - 0.05) & (centers <= s_end + 0.05))
            if len(idx):
                seg_embs = embs[idx]
                seg_embs = seg_embs / np.maximum(np.linalg.norm(seg_embs, axis=1, keepdims=True), 1e-8)
                prob = float(clf.predict_proba(seg_embs)[:, 1].mean())
            else:
                prob = 0.0
        else:
            prob = 0.0

        duration = s_end - s_start
        if duration < MIN_SEC or duration > MAX_SEC:
            failed.append({**seg, "reason": f"duration_{duration:.2f}_prob_{prob:.2f}"}); continue
        if prob < 0.60:
            failed.append({**seg, "reason": f"low_prob_{prob:.2f}"}); continue

        pad = 0.03
        a = max(0, round((s_start - pad) * sr))
        b = min(len(mono), round((s_end + pad) * sr))
        clip = micro_fade(resample(mono[a:b], sr, OUTPUT_SR))
        bid = f"BF{len(repaired) + 1:03d}"
        dest = AUDIO_DIR / f"{bid}.wav"
        sf.write(str(dest), clip, OUTPUT_SR, subtype="PCM_16")
        repaired.append({
            "blind_id": bid, "source_id": sid, "origin": seg["origin"],
            "origin_blind_id": seg["blind_id"],
            "old_start": seg["start"], "old_end": seg["end"],
            "new_start": round(s_start, 3), "new_end": round(s_end, 3),
            "duration": round(duration, 3), "mean_prob": round(prob, 4),
            "text": s_text[:60], "audio": str(dest.resolve()),
        })
        print(f"  {bid} {sid} {seg['start']:.1f}-{seg['end']:.1f} -> {s_start:.2f}-{s_end:.2f} p={prob:.3f}", flush=True)

    write_json(OUT / "report.json", {
        "version": "boundary_fix_v4", "created_at": datetime.now(timezone.utc).isoformat(),
        "input": len(segments), "unique": len(seen),
        "repaired": len(repaired), "failed": failed, "samples": repaired,
    })
    print(json.dumps({"repaired": len(repaired), "failed": len(failed), "saved": str(OUT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
