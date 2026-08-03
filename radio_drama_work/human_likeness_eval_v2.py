#!/usr/bin/env python3
"""Human-likeness ranking with a leave-one-model-out real/synthetic classifier.

This is a ranking aid for fixed, matched test sentences.  It is not an
absolute MOS predictor: the classifier learns the difference between the
real D_old recordings and the available generated candidates.  Each model
group is held out before fitting, so a candidate is not scored by a model
trained on its own files.

Usage:
  python human_likeness_eval_v2.py OUT_DIR REAL_LIST_OR_WAV_DIR GEN_DIR ...
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from human_likeness_eval import FEATURES, audio_features, robust_stats, score_one


def read_real_paths(source: str) -> list[Path]:
    p = Path(source)
    if p.suffix.lower() == ".list":
        return [Path(line.split("|", 1)[0].replace("//", "/"))
                for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(p.glob("*.wav"))


def group_name(path: Path) -> tuple[str, str]:
    stem = path.stem
    if "__" in stem:
        return stem.rsplit("__", 1)[0], stem.rsplit("__", 1)[1]
    return path.parent.name, stem


def matrix(rows: list[dict[str, float]]) -> np.ndarray:
    return np.asarray([[row.get(k, np.nan) for k in FEATURES] for row in rows], dtype=np.float64)


def leave_one_group_out_realness(real_rows: list[dict[str, float]], generated: list[dict]) -> dict[int, float]:
    """Return p(real) for every generated row, with its group held out."""
    groups = sorted({row["group"] for row in generated})
    if len(groups) < 2 or not real_rows:
        return {i: float("nan") for i in range(len(generated))}

    real_x = matrix(real_rows)
    out: dict[int, float] = {}
    for held_group in groups:
        held = [i for i, row in enumerate(generated) if row["group"] == held_group]
        train = [i for i, row in enumerate(generated) if row["group"] != held_group]
        if not train:
            continue
        gen_x = matrix([generated[i] for i in train])
        x = np.vstack([real_x, gen_x])
        y = np.r_[np.ones(len(real_x), dtype=np.int8), np.zeros(len(gen_x), dtype=np.int8)]
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            RobustScaler(),
            LogisticRegression(C=0.35, class_weight="balanced", max_iter=2000, random_state=1234),
        )
        try:
            model.fit(x, y)
            p_real = model.predict_proba(matrix([generated[i] for i in held]))[:, 1]
            for i, p in zip(held, p_real):
                out[i] = float(np.clip(p, 0.0, 1.0))
        except Exception as exc:
            print(f"classifier_fail group={held_group}: {exc}", flush=True)
            for i in held:
                out[i] = float("nan")
    return out


def mean_finite(values: list[float], default: float = 0.0) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else default


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/human_likeness_eval_v2")
    real_source = sys.argv[2] if len(sys.argv) > 2 else "/root/autodl-tmp/ireina_v2/data/D_old.list"
    gen_dirs = [Path(x) for x in sys.argv[3:]]
    if not gen_dirs:
        gen_dirs = [Path("/root/autodl-tmp/infer_refine_v3"), Path("/root/autodl-tmp/infer_plus18_compare")]
    out_dir.mkdir(parents=True, exist_ok=True)

    real_rows = []
    for path in read_real_paths(real_source):
        try:
            real_rows.append(audio_features(str(path)))
        except Exception as exc:
            print(f"real_fail {path}: {exc}", flush=True)
    stats = robust_stats(real_rows)
    print(f"real={len(real_rows)}", flush=True)

    generated = []
    paths = []
    for directory in gen_dirs:
        paths.extend(sorted(directory.glob("*.wav")))
    for index, path in enumerate(paths, 1):
        try:
            feat = audio_features(str(path))
            group, scene = group_name(path)
            base = score_one(feat, stats)
            generated.append({"file": str(path), "name": path.name, "group": group,
                              "scene": scene, **feat, **base})
            print(f"{index}/{len(paths)} {group} {scene} base={base['human_like_score']:.1f}", flush=True)
        except Exception as exc:
            print(f"gen_fail {path}: {exc}", flush=True)

    realness = leave_one_group_out_realness(real_rows, generated)
    for i, row in enumerate(generated):
        row["classifier_realness"] = float(100.0 * realness.get(i, np.nan))
        # Keep the handcrafted score visible, but let the held-out classifier
        # carry slightly more weight because it can detect multi-feature
        # combinations that the old p95 penalties missed.
        row["human_like_score_v2"] = float(
            0.42 * row["human_like_score"] + 0.58 * row["classifier_realness"]
        ) if np.isfinite(row["classifier_realness"]) else row["human_like_score"]

    groups = defaultdict(list)
    for row in generated:
        groups[row["group"]].append(row)
    summary = []
    for group, rows in groups.items():
        summary.append({
            "group": group,
            "n": len(rows),
            "human_like_score_v2": mean_finite([r["human_like_score_v2"] for r in rows]),
            "classifier_realness": mean_finite([r["classifier_realness"] for r in rows], float("nan")),
            "human_like_score": mean_finite([r["human_like_score"] for r in rows]),
            "distribution_score": mean_finite([r["distribution_score"] for r in rows]),
            "artifact_score": mean_finite([r["artifact_score"] for r in rows]),
            "mean_robust_z": mean_finite([r["mean_robust_z"] for r in rows]),
        })
    summary.sort(key=lambda row: row["human_like_score_v2"], reverse=True)

    payload = {
        "note": "ranking proxy only; classifier is leave-one-generated-group-out",
        "real_count": len(real_rows),
        "generated_count": len(generated),
        "real_stats": stats,
        "files": generated,
        "groups": summary,
    }
    (out_dir / "human_likeness_scores_v2.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out_dir / "human_likeness_groups_v2.tsv").open("w", encoding="utf-8", newline="") as fp:
        fields = ["group", "n", "human_like_score_v2", "classifier_realness",
                  "human_like_score", "distribution_score", "artifact_score", "mean_robust_z"]
        writer = csv.DictWriter(fp, fieldnames=fields, delimiter="/t")
        writer.writeheader()
        writer.writerows(summary)

    print("/nRANKING_V2")
    for row in summary:
        print(f"{row['human_like_score_v2']:6.1f}/t{row['group']}/tn={row['n']}/tclassifier={row['classifier_realness']:.1f}")
    print(f"saved {out_dir}", flush=True)


if __name__ == "__main__":
    main()
