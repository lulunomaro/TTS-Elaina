#!/usr/bin/env python3
"""Naturalness / human-likeness ranking for matched GPT-SoVITS samples.

This is a ranking instrument, not an absolute MOS predictor.  It combines:

1. artifact score from the existing open-source-style acoustic features;
2. distance from the real D_old distribution;
3. paired energy/F0 trajectory similarity when a manifest is supplied;
4. paired duration and terminal-intonation similarity.

Generated files use ``<model>__<sample_id>.wav``.  A manifest is JSONL with
``{"sample_id": ..., "real_path": ..., "scene": ..., "text": ...}``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np

from human_likeness_eval import FEATURES, audio_features, load_mono, robust_stats, score_one


def finite(value: float, default: float = 0.0) -> float:
    return float(value) if np.isfinite(value) else default


def corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 8:
        return float("nan")
    x, y = a[mask], b[mask]
    if np.std(x) < 1e-6 or np.std(y) < 1e-6:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def trajectory(path: str, n: int = 64) -> dict[str, np.ndarray | float]:
    y, sr = load_mono(path)
    hop = 160
    win = 800
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop, center=True)[0]
    log_rms = np.log(np.maximum(rms, 1e-7))
    log_rms = (log_rms - np.median(log_rms)) / max(np.std(log_rms), 1e-5)
    x = np.linspace(0.0, 1.0, max(2, log_rms.size))
    xx = np.linspace(0.0, 1.0, n)
    energy = np.interp(xx, x, log_rms)
    try:
        f0 = librosa.yin(np.clip(y, -1.0, 1.0), fmin=70, fmax=500,
                         sr=sr, frame_length=1024, hop_length=hop)
        valid = np.isfinite(f0) & (f0 >= 70) & (f0 <= 500)
        cents = np.full_like(f0, np.nan, dtype=np.float64)
        if int(valid.sum()) >= 3:
            med = float(np.median(f0[valid]))
            cents[valid] = 1200.0 * np.log2(f0[valid] / med)
        fx = np.linspace(0.0, 1.0, max(2, cents.size))
        fvalid = np.isfinite(cents)
        if int(fvalid.sum()) >= 3:
            ffill = np.interp(fx, fx[fvalid], cents[fvalid])
            ffill = np.interp(xx, fx, ffill)
        else:
            ffill = np.full(n, np.nan)
    except Exception:
        ffill = np.full(n, np.nan)
    tail = max(3, n // 8)
    prev = slice(-2 * tail, -tail)
    last = slice(-tail, None)
    return {
        "energy": energy,
        "f0": ffill,
        "duration": float(y.size / sr),
        "terminal_energy_slope": float(np.mean(energy[last]) - np.mean(energy[prev])),
        "terminal_f0_slope": float(np.nanmean(ffill[last]) - np.nanmean(ffill[prev]))
        if np.isfinite(ffill[last]).any() and np.isfinite(ffill[prev]).any() else float("nan"),
        "terminal_energy_jump": float(np.max(np.abs(np.diff(energy[-2 * tail:])))),
    }


def paired_scores(real_path: str, gen_path: str) -> dict[str, float]:
    real = trajectory(real_path)
    gen = trajectory(gen_path)
    energy_corr = corr(real["energy"], gen["energy"])
    f0_corr = corr(real["f0"], gen["f0"])
    corr_values = [x for x in (energy_corr, f0_corr) if np.isfinite(x)]
    if corr_values:
        trajectory_similarity = 100.0 * (0.5 + 0.5 * float(np.mean(corr_values)))
    else:
        trajectory_similarity = float("nan")

    duration_delta = abs(math.log(max(gen["duration"], 1e-3) / max(real["duration"], 1e-3)))
    terminal_energy_delta = abs(gen["terminal_energy_slope"] - real["terminal_energy_slope"])
    terminal_jump = abs(gen["terminal_energy_jump"] - real["terminal_energy_jump"])
    f0_delta = abs(gen["terminal_f0_slope"] - real["terminal_f0_slope"])
    acoustic_error = min(8.0, duration_delta / 0.30)
    acoustic_error += min(8.0, terminal_energy_delta / 0.70)
    acoustic_error += min(8.0, terminal_jump / 1.20)
    if np.isfinite(f0_delta):
        acoustic_error += min(8.0, f0_delta / 80.0)
    paired_acoustic = 100.0 * math.exp(-0.22 * acoustic_error)
    return {
        "energy_corr": finite(energy_corr, float("nan")),
        "f0_corr": finite(f0_corr, float("nan")),
        "trajectory_similarity": finite(trajectory_similarity, float("nan")),
        "duration_ratio": float(gen["duration"] / max(real["duration"], 1e-3)),
        "terminal_energy_delta": terminal_energy_delta,
        "terminal_f0_delta": finite(f0_delta, float("nan")),
        "paired_acoustic": paired_acoustic,
    }


def load_manifest(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[str(row["sample_id"])] = row
    return out


def real_paths(source: Path) -> list[Path]:
    if source.suffix.lower() == ".list":
        return [Path(line.split("|", 1)[0].replace("//", "/"))
                for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(source.glob("*.wav"))


def group_and_sample(path: Path) -> tuple[str, str]:
    stem = path.stem
    if "__" in stem:
        return stem.rsplit("__", 1)[0], stem.rsplit("__", 1)[1]
    return path.parent.name, stem


def mean(rows: list[dict], key: str) -> float:
    vals = [r.get(key, float("nan")) for r in rows]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("real_source", type=Path)
    ap.add_argument("generated_dirs", nargs="+", type=Path)
    ap.add_argument("--manifest", type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    real_rows = []
    for path in real_paths(args.real_source):
        try:
            real_rows.append(audio_features(str(path)))
        except Exception as exc:
            print(f"real_fail {path}: {exc}", flush=True)
    stats = robust_stats(real_rows)
    manifest = load_manifest(args.manifest)
    generated = []
    for directory in args.generated_dirs:
        for path in sorted(directory.glob("*.wav")):
            group, sample_id = group_and_sample(path)
            try:
                feat = audio_features(str(path))
                base = score_one(feat, stats)
                row = {"file": str(path), "name": path.name, "group": group,
                       "sample_id": sample_id, **feat, **base}
                pair = manifest.get(sample_id)
                if pair and Path(pair["real_path"]).exists():
                    row.update(paired_scores(pair["real_path"], str(path)))
                    row["scene"] = pair.get("scene", "")
                    row["text"] = pair.get("text", "")
                    row["paired"] = True
                else:
                    row.update({"paired": False, "trajectory_similarity": float("nan"),
                                "paired_acoustic": float("nan")})
                if row["paired"]:
                    row["naturalness_score_v1"] = (
                        0.45 * row["artifact_score"] +
                        0.25 * row["distribution_score"] +
                        0.20 * row["trajectory_similarity"] +
                        0.10 * row["paired_acoustic"]
                    )
                else:
                    row["naturalness_score_v1"] = (
                        0.65 * row["artifact_score"] + 0.35 * row["distribution_score"]
                    )
                generated.append(row)
                print(f"{path.name}/tnatural={row['naturalness_score_v1']:.1f}/tpaired={row['paired']}", flush=True)
            except Exception as exc:
                print(f"gen_fail {path}: {exc}", flush=True)

    groups = defaultdict(list)
    for row in generated:
        groups[row["group"]].append(row)
    summary = []
    for group, rows in groups.items():
        summary.append({
            "group": group,
            "n": len(rows),
            "paired_n": sum(bool(r.get("paired")) for r in rows),
            "naturalness_score_v1": mean(rows, "naturalness_score_v1"),
            "artifact_score": mean(rows, "artifact_score"),
            "distribution_score": mean(rows, "distribution_score"),
            "trajectory_similarity": mean(rows, "trajectory_similarity"),
            "paired_acoustic": mean(rows, "paired_acoustic"),
            "terminal_energy_delta": mean(rows, "terminal_energy_delta"),
            "terminal_f0_delta": mean(rows, "terminal_f0_delta"),
        })
    summary.sort(key=lambda row: row["naturalness_score_v1"], reverse=True)
    payload = {
        "note": "ranking proxy, not an absolute MOS; paired fields require the manifest",
        "scoring_standard": {
            "artifact_free": "45%: high-band/electric-tone/roughness/F0-jump penalties",
            "real_distribution": "25%: robust distance to real D_old acoustic distribution",
            "paired_trajectory": "20%: same-text normalized energy and F0 trajectory similarity",
            "paired_terminal": "10%: duration and final-intonation match",
            "interpretation": "85+ final candidate; 75-84 good; 60-74 usable with defects; below 60 reject",
        },
        "real_count": len(real_rows), "generated_count": len(generated),
        "groups": summary, "files": generated,
    }
    (args.out_dir / "naturalness_scores_v1.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
    )
    fields = list(summary[0].keys()) if summary else ["group"]
    with (args.out_dir / "naturalness_groups_v1.tsv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, delimiter="/t")
        writer.writeheader(); writer.writerows(summary)
    print("/nRANKING_V1")
    for row in summary:
        print(f"{row['naturalness_score_v1']:6.1f}/t{row['group']}/tpaired={row['paired_n']}/{row['n']}", flush=True)


if __name__ == "__main__":
    main()
