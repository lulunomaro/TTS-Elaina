# -*- coding: utf-8 -*-
"""Apply boundary model: find human-like cut points for a target segment.

For a segment [lo, hi], scans candidate cut points around both edges using
the trained boundary model, picks the highest-probability accepted boundary,
then exports the cut. Verifies against human-marked boundaries where known.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import soundfile as sf

PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")
ROOT = Path(r"F:\GPTSOVITS\radio_drama_work")
OUT = PROBE / "boundary_model_v1"
OUTPUT_SR = 32000
RADIO_MANIFEST = ROOT / "full_track_manifest_v1.json"
ANIME_MANIFEST = ROOT / "anime_batch_v1" / "manifest.json"
RADIO_ASR = ROOT / "word_target_all_v1" / "asr_cache"
ANIME_ASR = ROOT / "anime_mining_v1" / "asr" / "source_reports"


def source_vocals(source_id):
    manifest = json.loads((ANIME_MANIFEST if source_id.startswith("majo_") else RADIO_MANIFEST).read_text(encoding="utf-8"))
    for t in manifest.get("tracks", []):
        if t["id"] == source_id:
            return Path(t["vocals"])
    return None


def asr_words(source_id):
    rep = (ANIME_ASR if source_id.startswith("majo_") else RADIO_ASR) / f"{source_id}.json"
    if not rep.exists():
        return []
    return json.loads(rep.read_text(encoding="utf-8")).get("words", [])


def frame_rms(mono, sr):
    frame = max(1, round(sr * 0.020))
    hop = max(1, round(sr * 0.010))
    n = max(1, 1 + (len(mono) - frame) // hop)
    starts = np.arange(n, dtype=np.int64) * hop
    ends = starts + frame
    sq = np.square(mono, dtype=np.float64)
    pref = np.empty(len(sq) + 1)
    pref[0] = 0.0
    np.cumsum(sq, out=pref[1:])
    rms = np.sqrt((pref[ends] - pref[starts]) / frame + 1e-12)
    ftimes = (starts + frame / 2.0) / sr
    return ftimes, rms


def rms_at(ftimes, rms, t, win):
    idx = np.flatnonzero((ftimes >= t - win) & (ftimes <= t + win))
    return float(rms[idx].mean()) if len(idx) else 0.0


def load_model():
    with (OUT / "boundary_model.pkl").open("rb") as f:
        return pickle.load(f)


def feats_for(model_pack, ftimes, rms, word_bounds, t):
    b50 = rms_at(ftimes, rms, t - 0.05, 0.0)
    a50 = rms_at(ftimes, rms, t + 0.05, 0.0)
    b150 = rms_at(ftimes, rms, t - 0.15, 0.0)
    a150 = rms_at(ftimes, rms, t + 0.15, 0.0)
    b300 = rms_at(ftimes, rms, t - 0.30, 0.0)
    a300 = rms_at(ftimes, rms, t + 0.30, 0.0)
    d = min(abs(t - b) for b in word_bounds) if word_bounds else 1.0
    local = rms_at(ftimes, rms, t, 0.0)
    surround = (b150 + a150) / 2.0
    depth = max(0.0, surround - local)
    return [b50, a50, b150, a150, b300, a300,
            a50 / max(b50, 1e-6), a150 / max(b150, 1e-6), d, depth]


def predict_boundary(model_pack, mono, sr, words, around, window=0.6):
    """Return best boundary time near 'around' (prob, time)."""
    clf = model_pack["model"]
    ftimes, rms = frame_rms(mono, sr)
    word_bounds = sorted(set([w["start"] for w in words] + [w["end"] for w in words]))
    cands = np.arange(around - window, around + window, 0.02)
    feats = np.asarray([feats_for(model_pack, ftimes, rms, word_bounds, t) for t in cands])
    prob = clf.predict_proba(feats)[:, 1]
    best = int(np.argmax(prob))
    return float(prob[best]), float(cands[best])


def main():
    pack = load_model()
    print("model loaded", flush=True)
    # human-confirmed boundaries: edge final report has abs times; calibrator
    # final list paths encode abs start/end? No - use feedback JSON + window_start.
    calib = json.loads(open(r"C:\Users\lulu\Downloads\boundary_calibrator_v1_2026-08-03T02-58-10-106Z.json", encoding="utf-8").read())
    calib_map = {s["id"]: s for s in calib["samples"]}
    ef = json.loads((PROBE / "boundary_calibrator_edge_final" / "report.json").read_text(encoding="utf-8"))
    errors = []
    tested = 0
    # edge final samples have abs_start/abs_end directly
    for r in ef.get("samples", []):
        sid = r["source_id"]
        vocals = source_vocals(sid)
        words = asr_words(sid)
        if vocals is None or not vocals.exists():
            continue
        audio, sr = sf.read(str(vocals), dtype="float32", always_2d=True)
        mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
        for kind, t_true, window in (("start", r["abs_start"], 0.8), ("end", r["abs_end"], 0.8)):
            p, t_pred = predict_boundary(pack, mono, sr, words, t_true, window)
            err = abs(t_pred - t_true)
            errors.append(err)
            tested += 1
            print(f"  {r['id']} {kind}: human={t_true:.3f} model={t_pred:.3f} err={err*1000:.0f}ms prob={p:.2f}", flush=True)
    # calibrator final: rebuild abs times from feedback + window_start
    cf_rows = []
    for line in (PROBE / "boundary_calibrator_final" / "boundary_calibrator_final.list").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = line.split("|")[0]
        import re
        m = re.search(r"CF(\d{3})", p)
        if not m:
            continue
        cid = "BC" + m.group(1)
        fb = calib_map.get(cid)
        if not fb:
            continue
        cf_rows.append({"id": cid, "source_id": fb["source_id"],
                        "abs_start": float(fb["window_start"]) + float(fb["start"]),
                        "abs_end": float(fb["window_start"]) + float(fb["end"])})
    for r in cf_rows:
        sid = r["source_id"]
        vocals = source_vocals(sid)
        words = asr_words(sid)
        if vocals is None or not vocals.exists():
            continue
        audio, sr = sf.read(str(vocals), dtype="float32", always_2d=True)
        mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
        for kind, t_true, window in (("start", r["abs_start"], 0.8), ("end", r["abs_end"], 0.8)):
            p, t_pred = predict_boundary(pack, mono, sr, words, t_true, window)
            err = abs(t_pred - t_true)
            errors.append(err)
            tested += 1
            print(f"  {r['id']} {kind}: human={t_true:.3f} model={t_pred:.3f} err={err*1000:.0f}ms prob={p:.2f}", flush=True)
    errs = np.asarray(errors)
    print(f"\ntested boundaries: {tested}")
    if len(errs):
        print(f"error: mean={np.mean(errs)*1000:.0f}ms median={np.median(errs)*1000:.0f}ms "
              f"p90={np.percentile(errs,90)*1000:.0f}ms", flush=True)
        (OUT / "eval_calibrator.json").write_text(json.dumps({
            "version": "boundary_model_eval_v1",
            "n_boundaries": int(len(errs)),
            "mean_err_ms": round(float(np.mean(errs)) * 1000, 1),
            "median_err_ms": round(float(np.median(errs)) * 1000, 1),
            "p90_err_ms": round(float(np.percentile(errs, 90)) * 1000, 1),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print("saved eval", flush=True)


if __name__ == "__main__":
    main()
