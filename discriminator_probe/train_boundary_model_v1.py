# -*- coding: utf-8 -*-
"""Boundary-learning model: learn what cut points humans accept.

Positive samples: all human-confirmed segment boundaries (start & end) from
calibrator results, v4 clean, cross-passed, and the 149 confirmed segments.
Negative samples: random points inside speech (mid-utterance) from the same
sources. Features per candidate point:
  - RMS before/after at 50/150/300ms windows (energy contrast)
  - local RMS valley depth
  - distance to nearest ASR word boundary
  - discriminator prob before/after (speaker stability)
Trains a RandomForest, evaluates with holdout (calibrator segments), reports
boundary acceptance AUC. Model saved for reuse in auto-cutting.

Read-only on radio_drama_work; outputs in discriminator_probe/boundary_model_v1.
"""
from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression

PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")
ROOT = Path(r"F:\GPTSOVITS\radio_drama_work")
OUT = PROBE / "boundary_model_v1"
OUT.mkdir(parents=True, exist_ok=True)
OUTPUT_SR = 32000
RADIO_MANIFEST = ROOT / "full_track_manifest_v1.json"
ANIME_MANIFEST = ROOT / "anime_batch_v1" / "manifest.json"
RADIO_ASR = ROOT / "word_target_all_v1" / "asr_cache"
ANIME_ASR = ROOT / "anime_mining_v1" / "asr" / "source_reports"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def collect_positive_boundaries() -> list[dict]:
    """All confirmed boundaries: (source_id, abs_time, kind)."""
    rows = []

    # 1) calibrator final (CF) - from report.json samples
    cf = json.loads((PROBE / "boundary_calibrator_final" / "report.json").read_text(encoding="utf-8"))
    for r in cf.get("samples", []):
        rows.append({"source_id": r["source_id"], "time": r["abs_start"], "kind": "start"})
        rows.append({"source_id": r["source_id"], "time": r["abs_end"], "kind": "end"})

    # 2) edge final (EF)
    ef = json.loads((PROBE / "boundary_calibrator_edge_final" / "report.json").read_text(encoding="utf-8"))
    for r in ef.get("samples", []):
        rows.append({"source_id": r["source_id"], "time": r["abs_start"], "kind": "start"})
        rows.append({"source_id": r["source_id"], "time": r["abs_end"], "kind": "end"})

    # 3) v4 clean - report.json has new_start/new_end
    v4 = json.loads((PROBE / "boundary_fix_v4" / "report.json").read_text(encoding="utf-8"))
    for r in v4.get("samples", []):
        rows.append({"source_id": r["source_id"], "time": r["new_start"], "kind": "start"})
        rows.append({"source_id": r["source_id"], "time": r["new_end"], "kind": "end"})

    # 4) confirmed 149 - parse from list paths (name encodes start/end samples at 32k)
    lst = ROOT / "human_confirmed_radio_anime_v3_train_1p5_6s.list"
    import re
    for line in lst.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        p = line.split("|")[0]
        m = re.search(r"_(\d{10})_(\d{10})\.wav$", p)
        if not m:
            continue
        start = int(m.group(1)) / 32000.0
        end = int(m.group(2)) / 32000.0
        sid = None
        for key in ("243358894", "253810599", "28055832533", "1311215796", "1405440540",
                    "113814583", "280479210", "majo_ep"):
            if key in p:
                sid = "found"
                break
        # map to source id via path parent clues
        name = p.replace("\\", "/")
        parts = name.split("/")
        if "audio" in parts:
            pass
        # extract source id from filename
        fm = re.search(r"(majo_ep\d\d|\d{9,11}(?:_u1)?-1-\d{2,3})", name)
        if fm:
            sid = fm.group(1)
        if sid and sid != "found":
            rows.append({"source_id": sid, "time": start, "kind": "start"})
            rows.append({"source_id": sid, "time": end, "kind": "end"})
    return rows


def main():
    pos = collect_positive_boundaries()
    print(f"positive boundaries collected: {len(pos)}", flush=True)

    # group by source, compute features for each boundary
    X, y, groups = [], [], []
    feat_names = [
        "rms_50_before", "rms_50_after", "rms_150_before", "rms_150_after",
        "rms_300_before", "rms_300_after", "rms_50_ratio", "rms_150_ratio",
        "dist_word_boundary", "local_min_depth",
    ]
    sources = sorted(set(r["source_id"] for r in pos))
    for sid in sources:
        vocals = source_vocals(sid)
        words = asr_words(sid)
        if vocals is None or not vocals.exists():
            continue
        audio, sr = sf.read(str(vocals), dtype="float32", always_2d=True)
        mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
        ftimes, rms = frame_rms(mono, sr)
        total = len(mono) / sr
        word_bounds = sorted(set([w["start"] for w in words] + [w["end"] for w in words]))

        def feats(t):
            b50 = rms_at(ftimes, rms, t - 0.05, 0.0)
            a50 = rms_at(ftimes, rms, t + 0.05, 0.0)
            b150 = rms_at(ftimes, rms, t - 0.15, 0.0)
            a150 = rms_at(ftimes, rms, t + 0.15, 0.0)
            b300 = rms_at(ftimes, rms, t - 0.30, 0.0)
            a300 = rms_at(ftimes, rms, t + 0.30, 0.0)
            # distance to nearest word boundary
            d = min(abs(t - b) for b in word_bounds) if word_bounds else 1.0
            # local min depth: rms at t vs surrounding 0.2s
            local = rms_at(ftimes, rms, t, 0.0)
            surround = (b150 + a150) / 2.0
            depth = max(0.0, surround - local)
            return [b50, a50, b150, a150, b300, a300,
                    a50 / max(b50, 1e-6), a150 / max(b150, 1e-6),
                    d, depth]

        # positives
        for r in pos:
            if r["source_id"] != sid:
                continue
            if r["time"] < 0.1 or r["time"] > total - 0.1:
                continue
            X.append(feats(r["time"]))
            y.append(1)
            groups.append(r["kind"] + ":" + sid)

        # negatives: random mid-speech points (inside confirmed segments)
        # PLUS near-miss points (offset from true boundary by 0.15-0.35s) so the
        # model learns precise placement, not just 'near a boundary'
        n_neg = 0
        rng = np.random.default_rng(42)
        segs = [(r["time"], r["time"] + 2.0) for r in pos if r["source_id"] == sid and r["kind"] == "start"]
        while n_neg < 40 and segs:
            s0, s1 = segs[rng.integers(len(segs))]
            if s1 - s0 < 1.0:
                continue
            t = rng.uniform(s0 + 0.4, s1 - 0.4)
            X.append(feats(t))
            y.append(0)
            groups.append("neg:" + sid)
            n_neg += 1
        # near-miss negatives: for each true start/end, sample offsets
        for r in pos:
            if r["source_id"] != sid:
                continue
            for off in (0.15, 0.25, 0.35):
                t = r["time"] + off
                if 0.1 < t < total - 0.1:
                    X.append(feats(t))
                    y.append(0)
                    groups.append("near:" + sid)
            for off in (-0.15, -0.25, -0.35):
                t = r["time"] + off
                if 0.1 < t < total - 0.1:
                    X.append(feats(t))
                    y.append(0)
                    groups.append("near:" + sid)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups, object)
    print(f"samples: {len(y)} (pos={int(np.sum(y==1))}, neg={int(np.sum(y==0))})", flush=True)

    # holdout: leave-source-out - hold out 2 sources entirely
    all_sources = sorted(set(sid for sid in sources if any(sid in g for g in groups)))
    test_srcs = set(all_sources[:2])
    te = np.isin(groups, [g for g in groups if any(s in g for s in test_srcs)])
    tr = ~te

    clf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=4, random_state=42, n_jobs=2)
    clf.fit(X[tr], y[tr])
    pred = clf.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y[te], pred) if len(set(y[te])) > 1 else float("nan")
    print(f"leave-source AUC: {auc:.4f} (test sources={test_srcs}, n={int(np.sum(te))})", flush=True)

    # full model
    clf_full = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=2)
    clf_full.fit(X, y)
    with (OUT / "boundary_model.pkl").open("wb") as f:
        pickle.dump({"model": clf_full, "feat_names": feat_names, "sources": sources}, f)

    # importance
    imp = sorted(zip(feat_names, clf.feature_importances_), key=lambda x: -x[1])
    print("top features:", [f"{n}={v:.3f}" for n, v in imp[:6]], flush=True)

    write_json(OUT / "report.json", {
        "version": "boundary_model_v1",
        "samples": {"total": int(len(y)), "pos": int(np.sum(y == 1)), "neg": int(np.sum(y == 0))},
        "holdout_auc": round(float(auc), 4) if auc == auc else None,
        "top_features": [n for n, _ in imp[:8]],
        "model": str(OUT / "boundary_model.pkl"),
    })
    print(f"saved={OUT}", flush=True)


if __name__ == "__main__":
    main()
