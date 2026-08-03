# -*- coding: utf-8 -*-
"""Hybrid boundary rule: ASR word boundary + energy-valley refinement.

Rule: candidate boundary = nearest ASR word boundary; if that boundary sits on
high energy (mid-word), snap to the nearest energy valley within 0.3s.
Evaluate on the 18 human-calibrated segments (36 boundaries).
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path

import soundfile as sf

PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")
ROOT = Path(r"F:\GPTSOVITS\radio_drama_work")
RADIO_MANIFEST = ROOT / "full_track_manifest_v1.json"
ANIME_MANIFEST = ROOT / "anime_batch_v1" / "manifest.json"
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


def hybrid_boundary(mono, sr, words, near, kind, window=0.30):
    """Return best boundary near 'near'.
    start: nearest word start boundary (snap to valley if mid-word energy high)
    end:   the end of the last word whose END is <= near (never borrow next word)
    """
    ftimes, rms = frame_rms(mono, sr)
    if kind == "start":
        bounds = sorted(set([w["start"] for w in words]))
        if not bounds:
            return near
        wb = min(bounds, key=lambda b: abs(b - near))
        idx = np.flatnonzero((ftimes >= wb - window) & (ftimes <= wb + window))
        if len(idx):
            local = rms[idx]
            floor = float(np.percentile(local, 30))
            valleys = idx[local <= max(floor * 1.3, 1e-4)]
            if len(valleys):
                vt = ftimes[valleys[np.argmin(np.abs(ftimes[valleys] - wb))]]
                r_wb = rms_at(ftimes, rms, wb, 0.0)
                r_vt = rms_at(ftimes, rms, vt, 0.0)
                if r_vt < r_wb * 0.8:
                    return float(vt)
        return float(wb)
    else:
        # end: last word whose end <= near
        ends = [w["end"] for w in words if w["end"] <= near + 0.02]
        if not ends:
            return float(near)
        wb = max(ends)
        # maybe extend slightly into trailing breath: check energy decay after
        idx = np.flatnonzero((ftimes >= wb) & (ftimes <= wb + 0.35))
        if len(idx):
            local = rms[idx]
            floor = float(np.percentile(local, 30))
            # find first deep valley after wb (trailing breath end)
            valleys = idx[local <= max(floor * 1.2, 1e-4)]
            if len(valleys):
                vt = float(ftimes[valleys[0]])
                # only extend if it's quiet (breath) not next speech
                r_before = rms_at(ftimes, rms, wb - 0.02, 0.0)
                r_at_vt = rms_at(ftimes, rms, vt, 0.0)
                if r_at_vt < r_before * 0.5 and vt - wb <= 0.25:
                    return vt
        return float(wb)


def rms_at(ftimes, rms, t, win):
    idx = np.flatnonzero((ftimes >= t - win) & (ftimes <= t + win))
    return float(rms[idx].mean()) if len(idx) else 0.0


def main():
    calib = json.load(open(r"C:\Users\lulu\Downloads\boundary_calibrator_v1_2026-08-03T02-58-10-106Z.json", encoding="utf-8"))
    calib_map = {s["id"]: s for s in calib["samples"]}
    ef = json.loads((PROBE / "boundary_calibrator_edge_final" / "report.json").read_text(encoding="utf-8"))

    pts = []
    for r in ef["samples"]:
        pts.append((r["id"], r["source_id"], r["abs_start"], "start"))
        pts.append((r["id"], r["source_id"], r["abs_end"], "end"))
    for line in (PROBE / "boundary_calibrator_final" / "boundary_calibrator_final.list").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        import re
        m = re.search(r"CF(\d{3})", line)
        if not m:
            continue
        fb = calib_map.get("BC" + m.group(1))
        if not fb:
            continue
        pts.append((fb["id"], fb["source_id"], float(fb["window_start"]) + float(fb["start"]), "start"))
        pts.append((fb["id"], fb["source_id"], float(fb["window_start"]) + float(fb["end"]), "end"))

    errors = []
    details = []
    for pid, sid, t_true, kind in pts:
        vocals = source_vocals(sid)
        words = asr_words(sid)
        if vocals is None or not vocals.exists():
            continue
        audio, sr = sf.read(str(vocals), dtype="float32", always_2d=True)
        mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
        t_pred = hybrid_boundary(mono, sr, words, t_true, kind)
        err = abs(t_pred - t_true)
        errors.append(err)
        details.append({"id": pid, "kind": kind, "human": round(t_true, 3),
                        "model": round(t_pred, 3), "err_ms": round(err * 1000, 0)})
    errs = np.asarray(errors)
    print(f"boundaries: {len(errs)}")
    print(f"hybrid rule error: mean={np.mean(errs)*1000:.0f}ms median={np.median(errs)*1000:.0f}ms "
          f"<=100ms: {100*np.mean(errs<=0.10):.0f}% <=50ms: {100*np.mean(errs<=0.05):.0f}%", flush=True)
    for d in details:
        print(f"  {d['id']} {d['kind']}: human={d['human']} model={d['model']} err={d['err_ms']}ms", flush=True)
    (PROBE / "boundary_model_v1" / "eval_hybrid.json").write_text(json.dumps({
        "version": "hybrid_boundary_rule_eval", "n": len(errs),
        "mean_err_ms": round(float(np.mean(errs)) * 1000, 1),
        "median_err_ms": round(float(np.median(errs)) * 1000, 1),
        "pct_le_50ms": round(float(np.mean(errs <= 0.05)) * 100, 1),
        "pct_le_100ms": round(float(np.mean(errs <= 0.10)) * 100, 1),
        "details": details,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", flush=True)


if __name__ == "__main__":
    main()
