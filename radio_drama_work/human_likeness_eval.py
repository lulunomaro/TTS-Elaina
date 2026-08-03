#!/usr/bin/env python3
"""Human-likeness proxy evaluator for Ireina v2Pro audio.

This is not a MOS replacement. It builds a baseline from real D_old clips and
scores generated clips by distribution distance plus artifact penalties. It is
intended for checkpoint/model ranking under a fixed test sentence set.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


SR = 16000
EPS = 1e-8
FEATURES = [
    "rms_cv", "rms_roughness", "spectral_centroid", "spectral_rolloff",
    "spectral_flatness", "high_band_flatness", "high_ratio", "zcr",
    "flux_mean", "flux_roughness", "f0_voiced_ratio", "f0_cents_std",
    "f0_jitter_cents", "f0_jump_rate", "spectral_peakiness",
]


def load_mono(path: str) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path, always_2d=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1)
    y = np.nan_to_num(y, copy=False)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        sr = SR
    return y, sr


def quantile(x: np.ndarray, q: float, default: float = 0.0) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if x.size else default


def audio_features(path: str) -> dict[str, float]:
    y, sr = load_mono(path)
    if y.size < int(sr * 0.25):
        return {k: float("nan") for k in FEATURES} | {"clip_ratio": float("nan"), "duration": y.size / sr}

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    clip_ratio = float(np.mean(np.abs(y) >= 0.985))
    # Spectral features are gain-normalized; clipping remains measured on raw audio.
    norm = max(quantile(np.abs(y), 0.995), 1e-4)
    z = np.clip(y / norm, -1.0, 1.0)

    hop = 160
    n_fft = 1024
    win = 800
    rms = librosa.feature.rms(y=z, frame_length=win, hop_length=hop, center=True)[0] + EPS
    log_rms = np.log(rms)
    stft = np.abs(librosa.stft(z, n_fft=n_fft, hop_length=hop, win_length=win, center=True)) + EPS
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    power = stft ** 2
    speech_band = (freqs >= 150) & (freqs <= 7600)
    high_band = (freqs >= 5000) & (freqs <= 7600)
    total = np.sum(power[speech_band], axis=0) + EPS
    high = np.sum(power[high_band], axis=0)
    high_ratio = float(np.median(high / total))

    centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(S=stft)[0]
    high_flatness = librosa.feature.spectral_flatness(S=stft[high_band])[0] if np.any(high_band) else flatness
    zcr = librosa.feature.zero_crossing_rate(z, hop_length=hop)[0]
    db = librosa.amplitude_to_db(stft, ref=np.max)
    flux = librosa.onset.onset_strength(S=db, sr=sr, hop_length=hop)

    # A narrow spectral peak / mean-neighbour ratio is a useful electric-tone proxy.
    mid = stft[(freqs >= 900) & (freqs <= 7000)]
    peakiness = np.max(mid, axis=0) / (np.median(mid, axis=0) + EPS) if mid.size else np.array([0.0])

    try:
        f0 = librosa.yin(z, fmin=70, fmax=500, sr=sr, frame_length=1024, hop_length=hop)
        valid = np.isfinite(f0) & (f0 >= 70) & (f0 <= 500)
        fv = f0[valid]
    except Exception:
        valid = np.zeros(1, dtype=bool)
        fv = np.array([], dtype=np.float32)
    if fv.size >= 3:
        med = float(np.median(fv))
        cents = 1200.0 * np.log2(fv / med)
        dcents = np.abs(np.diff(cents))
        f0_cents_std = float(np.std(cents))
        f0_jitter = float(np.median(dcents))
        f0_jump = float(np.mean(dcents > 90.0))
    else:
        f0_cents_std = float("nan")
        f0_jitter = float("nan")
        f0_jump = float("nan")

    drms = np.abs(np.diff(log_rms))
    dflux = np.abs(np.diff(flux)) if flux.size > 1 else np.array([0.0])
    return {
        "duration": float(y.size / sr),
        "clip_ratio": clip_ratio,
        "rms_cv": float(np.std(rms) / (np.mean(rms) + EPS)),
        "rms_roughness": quantile(drms, 0.95),
        "spectral_centroid": float(np.median(centroid) / sr),
        "spectral_rolloff": float(np.median(rolloff) / sr),
        "spectral_flatness": float(np.median(flatness)),
        "high_band_flatness": float(np.median(high_flatness)),
        "high_ratio": high_ratio,
        "zcr": float(np.mean(zcr)),
        "flux_mean": float(np.mean(flux)),
        "flux_roughness": quantile(dflux, 0.95),
        "f0_voiced_ratio": float(np.mean(valid)) if valid.size else 0.0,
        "f0_cents_std": f0_cents_std,
        "f0_jitter_cents": f0_jitter,
        "f0_jump_rate": f0_jump,
        "spectral_peakiness": float(np.median(peakiness)),
    }


def robust_stats(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    out = {}
    for key in FEATURES:
        vals = np.array([r.get(key, np.nan) for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        med = float(np.median(vals)) if vals.size else 0.0
        mad = float(np.median(np.abs(vals - med))) if vals.size else 1.0
        scale = max(1.4826 * mad, float(np.std(vals)) * 0.25 if vals.size > 1 else 1e-3, 1e-3)
        out[key] = {
            "median": med,
            "scale": scale,
            "p05": quantile(vals, .05, med),
            "p95": quantile(vals, .95, med),
        }
    return out


def score_one(feat: dict[str, float], stats: dict[str, dict[str, float]]) -> dict[str, float]:
    zs = []
    for key in FEATURES:
        x = feat.get(key, np.nan)
        if not np.isfinite(x):
            continue
        s = stats[key]
        zs.append(abs(float(x) - s["median"]) / s["scale"])
    mean_z = float(np.mean(np.minimum(zs, 8.0))) if zs else 8.0
    distribution_score = 100.0 * math.exp(-0.32 * mean_z)

    # Explicit artifact penalties. These are relative to real D_old p95, not fixed loudness thresholds.
    penalty = 0.0
    clip = feat.get("clip_ratio", 0.0)
    penalty += min(25.0, 25.0 * clip / 0.002)
    for key, weight in (("high_ratio", 16.0), ("high_band_flatness", 12.0), ("spectral_peakiness", 10.0),
                        ("rms_roughness", 8.0), ("f0_jitter_cents", 7.0), ("f0_jump_rate", 7.0)):
        x = feat.get(key, np.nan)
        if np.isfinite(x):
            excess = max(0.0, (x - stats[key]["p95"]) / stats[key]["scale"])
            penalty += min(weight, weight * excess / 3.0)
    artifact_score = max(0.0, 100.0 - penalty)
    score = 0.68 * distribution_score + 0.32 * artifact_score
    return {"human_like_score": score, "distribution_score": distribution_score,
            "artifact_score": artifact_score, "mean_robust_z": mean_z, "artifact_penalty": penalty}


def group_name(path: Path) -> tuple[str, str]:
    stem = path.stem
    if "__" in stem:
        return stem.rsplit("__", 1)[0], stem.rsplit("__", 1)[1]
    return path.parent.name, stem


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/human_likeness_eval")
    real_source = sys.argv[2] if len(sys.argv) > 2 else "/root/autodl-tmp/dataset_v1/wavs"
    gen_dirs = [Path(x) for x in (sys.argv[3:] or ["/root/autodl-tmp/infer_refine_v3", "/root/autodl-tmp/infer_plus18_compare"])]
    out_dir.mkdir(parents=True, exist_ok=True)

    if real_source.endswith(".list"):
        real_paths = []
        for line in Path(real_source).read_text(encoding="utf-8").splitlines():
            if line.strip():
                real_paths.append(Path(line.split("|", 1)[0].replace("//", "/")))
        real_paths = sorted(real_paths)
    else:
        real_paths = sorted(Path(real_source).glob("*.wav"))
    real_rows = []
    for p in real_paths:
        try: real_rows.append(audio_features(str(p)))
        except Exception as e: print("real_fail", p, repr(e), flush=True)
    stats = robust_stats(real_rows)
    print(f"real={len(real_rows)}", flush=True)

    files = []
    for d in gen_dirs:
        files.extend(sorted(d.glob("*.wav")))
    rows = []
    for i, p in enumerate(files, 1):
        try:
            feat = audio_features(str(p))
            group, scene = group_name(p)
            score = score_one(feat, stats)
            rows.append({"file": str(p), "name": p.name, "group": group, "scene": scene, **feat, **score})
            print(f"{i}/{len(files)} {group} {scene} score={score['human_like_score']:.1f}", flush=True)
        except Exception as e:
            print("gen_fail", p, repr(e), flush=True)

    groups = defaultdict(list)
    for r in rows: groups[r["group"]].append(r)
    summary = []
    for group, rs in groups.items():
        summary.append({"group": group, "n": len(rs),
                        "human_like_score": float(np.mean([r["human_like_score"] for r in rs])),
                        "distribution_score": float(np.mean([r["distribution_score"] for r in rs])),
                        "artifact_score": float(np.mean([r["artifact_score"] for r in rs])),
                        "mean_robust_z": float(np.mean([r["mean_robust_z"] for r in rs])),
                        "artifact_penalty": float(np.mean([r["artifact_penalty"] for r in rs]))})
    summary.sort(key=lambda x: x["human_like_score"], reverse=True)
    with open(out_dir / "human_likeness_scores.json", "w", encoding="utf-8") as f:
        json.dump({"note": "proxy score; rank under fixed test set, not an absolute MOS", "real_count": len(real_rows), "real_stats": stats, "files": rows, "groups": summary}, f, ensure_ascii=False, indent=2)
    with open(out_dir / "human_likeness_groups.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "n", "human_like_score", "distribution_score", "artifact_score", "mean_robust_z", "artifact_penalty"], delimiter="/t")
        w.writeheader(); w.writerows(summary)
    print("/nRANKING")
    for r in summary: print(f"{r['human_like_score']:6.1f}/t{r['group']}/tn={r['n']}/tartifact={r['artifact_score']:.1f}")
    print("saved", out_dir, flush=True)


if __name__ == "__main__":
    main()
