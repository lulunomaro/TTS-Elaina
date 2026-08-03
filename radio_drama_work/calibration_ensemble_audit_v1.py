# -*- coding: utf-8 -*-
"""Audit whether independent speaker encoders can rescue Phase-1 calibration.

The 52 v9 files are retained as the plan's provisional positives, but they are
explicitly marked machine-derived.  A leave-one-unsafe-out check prevents a
flexible ensemble from merely memorising all observed unsafe examples.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import calibrate_classifier_v1 as base


OUT = base.OUT / "ensemble_audit_v1.json"
CACHE_NAMES = {
    "CAM++": "campp_v1.npz",
    "ERes2NetV2": "eres2net_v1.npz",
    "SpeechBrain-ECAPA-VoxCeleb": "speechbrain_ecapa_voxceleb_v1.npz",
}
TARGET_ONLY_METRICS = ("center", "median", "p25", "top4")
ORIGINAL_METRICS = ("center", "median", "p25", "margin")


def load_embeddings() -> tuple[list[Path], dict[str, np.ndarray]]:
    common_paths: list[str] | None = None
    result: dict[str, np.ndarray] = {}
    for name, filename in CACHE_NAMES.items():
        data = np.load(str(base.CACHE / filename), allow_pickle=True)
        paths = [str(value) for value in data["paths"].tolist()]
        if common_paths is None:
            common_paths = paths
        elif common_paths != paths:
            raise RuntimeError(f"Embedding cache path mismatch: {name}")
        result[name] = base.normalize_rows(data["embeddings"])
    assert common_paths is not None
    return [Path(path) for path in common_paths], result


def row_map(embeddings: np.ndarray, assets: dict) -> dict[str, dict]:
    refs, gold, negatives = base.make_metric_rows(embeddings, assets)
    return {str(row["path"]).lower(): row for row in refs + gold + negatives}


def candidate_thresholds(values: np.ndarray) -> list[float]:
    result = [-1.000001]
    for value in sorted(set(float(x) for x in values)):
        result.append(float(np.nextafter(np.float32(value), np.float32(np.inf))))
    return result


def exact_conjunction(rows: list[dict], metrics: tuple[str, ...], train_negative_paths: set[str]) -> dict:
    matrix = np.asarray([[float(row[name]) for name in metrics] for row in rows], dtype=np.float32)
    categories = [str(row["category"]) for row in rows]
    paths = [str(row["path"]).lower() for row in rows]
    gold = np.asarray([category == "gold" for category in categories])
    refs = np.asarray([category in {"old_reference_loo", "confirmed_target_loo"} for category in categories])
    all_neg = np.asarray([category in {"pure_non_target", "mixed_validation_only", "explicit_hard_negative"} for category in categories])
    train_neg = np.asarray([path in train_negative_paths for path in paths])
    grids = [candidate_thresholds(matrix[train_neg, idx]) for idx in range(len(metrics))]
    best: dict | None = None
    best_key: tuple | None = None
    for values in itertools.product(*grids):
        thresholds = np.asarray(values, dtype=np.float32)
        predicted = np.all(matrix >= thresholds, axis=1)
        if np.any(predicted & train_neg):
            continue
        gold_count = int(np.sum(predicted & gold))
        ref_count = int(np.sum(predicted & refs))
        all_negative_rejected = int(np.sum(all_neg & ~predicted))
        key = (gold_count, ref_count, all_negative_rejected, float(np.sum(thresholds)))
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "thresholds": {name: float(value) for name, value in zip(metrics, values)},
                "gold_target_count": gold_count,
                "reference_target_count": ref_count,
                "reference_count": int(np.sum(refs)),
                "negative_rejected": all_negative_rejected,
                "negative_count": int(np.sum(all_neg)),
                "predicted": predicted,
            }
    if best is None:
        raise RuntimeError("No conjunction found")
    return best


def serialise_rule(rule: dict) -> dict:
    return {key: value for key, value in rule.items() if key != "predicted"}


def weighted_embeddings(all_embeddings: dict[str, np.ndarray], weights: tuple[float, float, float]) -> np.ndarray:
    blocks = []
    for weight, values in zip(weights, all_embeddings.values()):
        if weight > 0:
            blocks.append(math.sqrt(weight) * values)
    return base.normalize_rows(np.concatenate(blocks, axis=1))


def weight_grid() -> list[tuple[float, float, float]]:
    values = []
    for first in range(11):
        for second in range(11 - first):
            third = 10 - first - second
            values.append((first / 10.0, second / 10.0, third / 10.0))
    return values


def audit_weighted_concat(assets: dict, all_embeddings: dict[str, np.ndarray]) -> dict:
    unsafe = {str(path).lower() for path in assets["validation_negatives"]}
    best_by_set: dict[str, dict] = {}
    row_cache: dict[tuple[float, float, float], list[dict]] = {}
    metric_sets = {
        "target_only": TARGET_ONLY_METRICS,
        "plan_original": ORIGINAL_METRICS,
    }
    for idx, weights in enumerate(weight_grid(), 1):
        embeddings = weighted_embeddings(all_embeddings, weights)
        rows = list(row_map(embeddings, assets).values())
        row_cache[weights] = rows
        for set_name, metrics in metric_sets.items():
            rule = exact_conjunction(rows, metrics, unsafe)
            key = (rule["gold_target_count"], rule["reference_target_count"], rule["negative_rejected"])
            current = best_by_set.get(set_name)
            if current is None or key > tuple(current["rank"]):
                best_by_set[set_name] = {
                    "rank": list(key),
                    "weights": dict(zip(all_embeddings.keys(), weights)),
                    "weights_tuple": weights,
                    "metrics": list(metrics),
                    "global_rule": serialise_rule(rule),
                }
        if idx % 10 == 0 or idx == len(weight_grid()):
            print(f"[weighted audit] {idx}/{len(weight_grid())}", flush=True)

    for set_name, result in best_by_set.items():
        weights = tuple(result.pop("weights_tuple"))
        rows = row_cache[weights]
        metrics = tuple(result["metrics"])
        folds = []
        for held_out in sorted(unsafe):
            train_negatives = unsafe - {held_out}
            rule = exact_conjunction(rows, metrics, train_negatives)
            held_row = next(row for row in rows if str(row["path"]).lower() == held_out)
            held_pass = all(float(held_row[name]) >= float(rule["thresholds"][name]) for name in metrics)
            folds.append(
                {
                    "held_out": held_row["path"],
                    "held_out_rejected": not held_pass,
                    "gold_target_count": rule["gold_target_count"],
                    "reference_target_count": rule["reference_target_count"],
                    "thresholds": rule["thresholds"],
                }
            )
        result["leave_one_unsafe_out"] = {
            "folds": folds,
            "all_held_out_rejected": all(fold["held_out_rejected"] for fold in folds),
            "minimum_gold_target_count": min(fold["gold_target_count"] for fold in folds),
            "minimum_reference_target_count": min(fold["reference_target_count"] for fold in folds),
        }
        loo = result["leave_one_unsafe_out"]
        global_rule = result["global_rule"]
        result["strict_gate_pass"] = bool(
            global_rule["gold_target_count"] >= 45
            and global_rule["negative_rejected"] == global_rule["negative_count"]
            and loo["all_held_out_rejected"]
            and loo["minimum_gold_target_count"] >= 45
        )
    return best_by_set


def logistic_leave_one_unsafe_out(assets: dict, all_embeddings: dict[str, np.ndarray]) -> dict:
    maps = {name: row_map(values, assets) for name, values in all_embeddings.items()}
    paths = list(next(iter(maps.values())).keys())
    categories = [next(iter(maps.values()))[path]["category"] for path in paths]
    feature_names = []
    columns = []
    for model_name, rows in maps.items():
        for metric in TARGET_ONLY_METRICS:
            feature_names.append(f"{model_name}:{metric}")
            columns.append([float(rows[path][metric]) for path in paths])
    matrix = np.asarray(columns, dtype=np.float64).T
    positive = np.asarray([category in {"old_reference_loo", "confirmed_target_loo", "gold"} for category in categories])
    gold = np.asarray([category == "gold" for category in categories])
    unsafe_indices = [idx for idx, category in enumerate(categories) if category in {"pure_non_target", "mixed_validation_only", "explicit_hard_negative"}]
    candidates = []
    for regularisation in (0.01, 0.1, 1.0, 10.0, 100.0):
        folds = []
        for held_idx in unsafe_indices:
            train_negative = [idx for idx in unsafe_indices if idx != held_idx]
            train_indices = np.asarray(list(np.flatnonzero(positive)) + train_negative)
            scaler = StandardScaler().fit(matrix[train_indices])
            train_x = scaler.transform(matrix[train_indices])
            train_y = positive[train_indices].astype(int)
            classifier = LogisticRegression(C=regularisation, class_weight="balanced", max_iter=5000, random_state=20260802)
            classifier.fit(train_x, train_y)
            decision = classifier.decision_function(scaler.transform(matrix))
            threshold = float(np.max(decision[train_negative]) + 1e-7)
            folds.append(
                {
                    "held_out": paths[held_idx],
                    "held_out_rejected": bool(decision[held_idx] < threshold),
                    "gold_target_count": int(np.sum(gold & (decision >= threshold))),
                    "threshold": threshold,
                }
            )
        candidate = {
            "C": regularisation,
            "all_held_out_rejected": all(fold["held_out_rejected"] for fold in folds),
            "minimum_gold_target_count": min(fold["gold_target_count"] for fold in folds),
            "folds": folds,
        }
        candidate["strict_gate_pass"] = bool(candidate["all_held_out_rejected"] and candidate["minimum_gold_target_count"] >= 45)
        candidates.append(candidate)
    best = max(candidates, key=lambda item: (item["strict_gate_pass"], item["all_held_out_rejected"], item["minimum_gold_target_count"]))
    return {"feature_names": feature_names, "candidates": candidates, "best": best}


def main() -> None:
    assets = base.build_assets()
    paths, embeddings = load_embeddings()
    expected = [str(path).lower() for path in assets["all_paths"]]
    if [str(path).lower() for path in paths] != expected:
        raise RuntimeError("Calibration assets no longer match cached path order")
    weighted = audit_weighted_concat(assets, embeddings)
    logistic = logistic_leave_one_unsafe_out(assets, embeddings)
    strict_pass = bool(any(item["strict_gate_pass"] for item in weighted.values()) or logistic["best"]["strict_gate_pass"])
    report = {
        "version": "calibration_ensemble_audit_v1",
        "policy": {
            "v9_gold_status": "machine-derived provisional positive; not human ground truth",
            "unsafe_validation": "8 unique user-confirmed unsafe clips; mixed clips remain validation-only",
            "features": "target-reference similarities only for strict leave-one-unsafe-out checks",
            "strict_gate": "every held-out unsafe rejected and at least 45/52 provisional gold retained in every fold",
        },
        "asset_counts": {
            "target_references": len(assets["target_refs"]),
            "provisional_gold": len(assets["gold"]),
            "validation_unsafe": len(assets["validation_negatives"]),
        },
        "weighted_concatenation": weighted,
        "logistic_leave_one_unsafe_out": logistic,
        "strict_gate_pass": strict_pass,
        "recommendation": "proceed_to_full_mining" if strict_pass else "do_not_auto-promote_without_new_human-ground-truth",
    }
    base.write_json(OUT, report)
    print(json.dumps({"strict_gate_pass": strict_pass, "weighted": {key: value["global_rule"] | {"strict_gate_pass": value["strict_gate_pass"]} for key, value in weighted.items()}, "logistic_best": logistic["best"], "saved": str(OUT)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
