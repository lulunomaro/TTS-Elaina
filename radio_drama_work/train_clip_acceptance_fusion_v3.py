# -*- coding: utf-8 -*-
"""Train same-domain clip-level speaker and boundary gates from 120 labels.

This is an active-learning/deployment candidate, not an independent final
validation.  Its output may only be used to create a fresh blind acceptance
pack.  It must not auto-promote full-track audio by itself.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

BOOTSTRAP_DIR = Path(__file__).resolve().parent
if str(BOOTSTRAP_DIR) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_DIR))
from local_resource_policy import apply_cpu_friendly_policy

RESOURCE_POLICY = apply_cpu_friendly_policy(2)

import joblib
import numpy as np
import soundfile as sf
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import calibrate_classifier_v1 as base
import train_human_temporal_fusion_v2 as temporal


ROOT = Path(__file__).resolve().parent
LABELS = ROOT / "human_calibration_combined_v2" / "labels_resolved.json"
TEMPORAL_DIR = ROOT / "human_temporal_fusion_combined_v3"
OUT = ROOT / "clip_acceptance_fusion_v3"
AGGREGATES = ("minimum", "p05", "p10", "p20", "p25", "median", "mean", "p75", "p90", "maximum", "std")
SPEAKER_NEGATIVE_CLASSES = {"pure_non_target", "mixed", "target_variant_exclude"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8")


def load_window_features(rows: list[dict], temporal_dir: Path) -> tuple[list[dict], np.ndarray]:
    manifest = read_json(temporal_dir / "window_manifest.json")
    windows = manifest["windows"]
    expected_ids = [row["blind_id"] for row in rows]
    window_ids = []
    for window in windows:
        clip_idx = int(window["clip_index"])
        if not 0 <= clip_idx < len(rows) or window["blind_id"] != expected_ids[clip_idx]:
            raise RuntimeError("Temporal window manifest no longer matches combined labels")
        window_ids.append(window["blind_id"])
    assets = base.build_assets()
    targets = temporal.load_old_target_embeddings(assets)
    embeddings = {}
    for model_name, cache_name in temporal.MODEL_CACHE_NAMES.items():
        data = np.load(str(temporal_dir / "embedding_cache" / cache_name), allow_pickle=True)
        paths = [str(value) for value in data["paths"].tolist()]
        expected_paths = [str(Path(window["audio"]).resolve()) for window in windows]
        if paths != expected_paths:
            raise RuntimeError(f"Temporal embedding path mismatch for {model_name}")
        embeddings[model_name] = base.normalize_rows(data["embeddings"])
    return windows, temporal.build_window_features(embeddings, targets)


def aggregate_vector(values: np.ndarray) -> list[float]:
    return [
        float(np.min(values)),
        float(np.quantile(values, 0.05)),
        float(np.quantile(values, 0.10)),
        float(np.quantile(values, 0.20)),
        float(np.quantile(values, 0.25)),
        float(np.median(values)),
        float(np.mean(values)),
        float(np.quantile(values, 0.75)),
        float(np.quantile(values, 0.90)),
        float(np.max(values)),
        float(np.std(values)),
    ]


def speaker_clip_features(rows: list[dict], windows: list[dict], window_features: np.ndarray) -> tuple[np.ndarray, list[str]]:
    sources = sorted({str(row["source_id"]) for row in rows})
    feature_names = [
        f"{window_name}:{aggregate_name}"
        for window_name in temporal.feature_names()
        for aggregate_name in AGGREGATES
    ]
    feature_names += ["active_window_count", "duration_sec"]
    feature_names += [f"source:{source}" for source in sources]
    matrix = []
    for clip_idx, row in enumerate(rows):
        indices = [
            idx for idx, window in enumerate(windows) if int(window["clip_index"]) == clip_idx and bool(window["active"])
        ]
        if not indices:
            raise RuntimeError(f"No active temporal windows for {row['blind_id']}")
        values = window_features[indices]
        features = []
        for column in range(values.shape[1]):
            features.extend(aggregate_vector(values[:, column]))
        features.extend([float(len(indices)), float(row["duration"])])
        features.extend(float(str(row["source_id"]) == source) for source in sources)
        matrix.append(features)
    return np.asarray(matrix, dtype=np.float64), feature_names


def rms_region(audio: np.ndarray, sr: int, start: int, end: int) -> float:
    chunk = audio[max(0, start) : min(len(audio), end)]
    return float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64)) + 1e-12)) if len(chunk) else 0.0


def boundary_one(path: Path) -> list[float]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32, copy=False)
    frame = max(1, round(sr * 0.020))
    hop = max(1, round(sr * 0.010))
    count = max(1, 1 + max(0, len(mono) - frame) // hop)
    padded = np.pad(mono, (0, max(0, frame + (count - 1) * hop - len(mono))))
    frame_rms = np.asarray(
        [np.sqrt(np.mean(np.square(padded[idx * hop : idx * hop + frame], dtype=np.float64)) + 1e-12) for idx in range(count)]
    )
    speech_level = max(float(np.quantile(frame_rms, 0.85)), 1e-6)
    threshold = 0.18 * speech_level
    leading_low = 0
    for value in frame_rms:
        if value > threshold:
            break
        leading_low += 1
    trailing_low = 0
    for value in frame_rms[::-1]:
        if value > threshold:
            break
        trailing_low += 1
    values = []
    for sec in (0.02, 0.05, 0.10, 0.20, 0.30):
        samples = max(1, round(sr * sec))
        values.append(rms_region(mono, sr, 0, samples) / speech_level)
        values.append(rms_region(mono, sr, len(mono) - samples, len(mono)) / speech_level)
    peak = max(float(np.max(np.abs(mono))) if len(mono) else 0.0, 1e-6)
    edge_samples = max(1, round(sr * 0.010))
    derivative = np.abs(np.diff(mono, prepend=mono[:1])) if len(mono) else np.zeros(1, dtype=np.float32)
    values.extend(
        [
            abs(float(mono[0])) / peak if len(mono) else 0.0,
            abs(float(mono[-1])) / peak if len(mono) else 0.0,
            float(np.max(derivative[:edge_samples])) / peak,
            float(np.max(derivative[-edge_samples:])) / peak,
            leading_low * 0.010,
            trailing_low * 0.010,
            float(np.mean(frame_rms[: min(10, len(frame_rms))] > threshold)),
            float(np.mean(frame_rms[-min(10, len(frame_rms)) :] > threshold)),
            len(mono) / sr,
        ]
    )
    return values


def boundary_features(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    names = []
    for sec in (20, 50, 100, 200, 300):
        names.extend([f"start_rms_ratio_{sec}ms", f"end_rms_ratio_{sec}ms"])
    names += [
        "start_sample_ratio",
        "end_sample_ratio",
        "start_derivative_ratio",
        "end_derivative_ratio",
        "leading_low_sec",
        "trailing_low_sec",
        "start_voiced_fraction_100ms",
        "end_voiced_fraction_100ms",
        "duration_sec",
    ]
    return np.asarray([boundary_one(Path(row["audio"])) for row in rows], dtype=np.float64), names


def model_specs() -> list[dict]:
    result = [{"type": "logistic", "C": value} for value in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)]
    for leaf in (2, 4):
        for max_features in ("sqrt", 0.5):
            result.append({"type": "extra_trees", "min_samples_leaf": leaf, "max_features": max_features})
    return result


def make_model(spec: dict, seed: int):
    if spec["type"] == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(spec["C"]), class_weight="balanced", max_iter=5000, random_state=seed
            ),
        )
    return ExtraTreesClassifier(
        n_estimators=240,
        min_samples_leaf=int(spec["min_samples_leaf"]),
        max_features=spec["max_features"],
        class_weight="balanced",
        n_jobs=2,
        random_state=seed,
    )


def cross_fitted_scores(matrix: np.ndarray, labels: np.ndarray, spec: dict) -> tuple[np.ndarray, list[dict]]:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=20260802)
    totals = np.zeros(len(labels), dtype=np.float64)
    counts = np.zeros(len(labels), dtype=np.int64)
    splits = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(matrix, labels)):
        seed = 20260802 + fold_idx
        model = make_model(spec, seed)
        model.fit(matrix[train_idx], labels[train_idx])
        probability = model.predict_proba(matrix[test_idx])[:, 1]
        totals[test_idx] += probability
        counts[test_idx] += 1
        splits.append({"fold": fold_idx, "train": train_idx.tolist(), "test": test_idx.tolist(), "seed": seed})
    if np.any(counts != 3):
        raise RuntimeError(f"Unexpected cross-fit counts: {np.unique(counts, return_counts=True)}")
    return totals / counts, splits


def evaluate_family(matrix: np.ndarray, labels: np.ndarray) -> tuple[dict, list[dict]]:
    candidates = []
    for spec in model_specs():
        scores, splits = cross_fitted_scores(matrix, labels, spec)
        threshold = float(np.max(scores[labels == 0]) + 1e-7)
        target_retained = int(np.sum(scores[labels == 1] >= threshold))
        candidate = {
            "spec": spec,
            "threshold": threshold,
            "target_retained": target_retained,
            "target_total": int(np.sum(labels == 1)),
            "unsafe_rejected": int(np.sum(scores[labels == 0] < threshold)),
            "unsafe_total": int(np.sum(labels == 0)),
            "auc": float(roc_auc_score(labels, scores)),
            "scores": scores,
            "splits": splits,
        }
        candidates.append(candidate)
    selected = max(candidates, key=lambda row: (row["target_retained"], row["auc"], row["spec"]["type"] == "logistic"))
    return selected, candidates


def fit_deployment_ensemble(matrix: np.ndarray, labels: np.ndarray, spec: dict, splits: list[dict]) -> list:
    models = []
    for split in splits:
        train_idx = np.asarray(split["train"], dtype=np.int64)
        model = make_model(spec, int(split["seed"]))
        model.fit(matrix[train_idx], labels[train_idx])
        models.append(model)
    return models


def public_candidate(candidate: dict) -> dict:
    return {key: value for key, value in candidate.items() if key not in {"scores", "splits"}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--temporal-dir", type=Path, default=TEMPORAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    started = time.time()
    rows = read_json(args.labels.resolve())["samples"]
    windows, window_matrix = load_window_features(rows, args.temporal_dir.resolve())
    speaker_matrix_all, speaker_feature_names = speaker_clip_features(rows, windows, window_matrix)
    boundary_matrix_all, boundary_feature_names = boundary_features(rows)

    speaker_indices = np.asarray(
        [idx for idx, row in enumerate(rows) if row["final_class"] == "target" or row["final_class"] in SPEAKER_NEGATIVE_CLASSES],
        dtype=np.int64,
    )
    speaker_labels = np.asarray([int(rows[idx]["final_class"] == "target") for idx in speaker_indices], dtype=np.int64)
    speaker_selected, speaker_candidates = evaluate_family(speaker_matrix_all[speaker_indices], speaker_labels)

    boundary_indices = np.asarray(
        [idx for idx, row in enumerate(rows) if row["final_class"] in {"target", "boundary_cut"}], dtype=np.int64
    )
    boundary_labels = np.asarray([int(rows[idx]["final_class"] == "target") for idx in boundary_indices], dtype=np.int64)
    boundary_selected, boundary_candidates = evaluate_family(boundary_matrix_all[boundary_indices], boundary_labels)

    speaker_models = fit_deployment_ensemble(
        speaker_matrix_all[speaker_indices], speaker_labels, speaker_selected["spec"], speaker_selected["splits"]
    )
    boundary_models = fit_deployment_ensemble(
        boundary_matrix_all[boundary_indices], boundary_labels, boundary_selected["spec"], boundary_selected["splits"]
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "version": "clip_acceptance_fusion_v3",
            "speaker": {
                "models": speaker_models,
                "threshold": speaker_selected["threshold"],
                "feature_names": speaker_feature_names,
                "known_sources": sorted({str(row["source_id"]) for row in rows}),
            },
            "boundary": {
                "models": boundary_models,
                "threshold": boundary_selected["threshold"],
                "feature_names": boundary_feature_names,
            },
            "policy": "active-learning model only; fresh blind acceptance is mandatory before full-track promotion",
        },
        output_dir / "clip_acceptance_models.joblib",
        compress=3,
    )
    report = {
        "version": "clip_acceptance_fusion_v3_report",
        "runtime_sec": time.time() - started,
        "local_resources": RESOURCE_POLICY,
        "policy": {
            "speaker_positive": "human target",
            "speaker_negative": sorted(SPEAKER_NEGATIVE_CLASSES),
            "boundary_training": "human target versus boundary_cut; no identity relabelling",
            "source_domain": "same 12 sources are represented in train folds; source one-hot is used",
            "validation_status": "cross-fitted development estimate only, not final independent acceptance",
            "next_gate": "fresh blind samples predicted accepted; any non-target invalidates deployment",
        },
        "speaker": {
            "sample_count": len(speaker_indices),
            "feature_count": speaker_matrix_all.shape[1],
            "selected": public_candidate(speaker_selected),
            "candidates": [public_candidate(row) for row in speaker_candidates],
            "scores": [
                {
                    "blind_id": rows[global_idx]["blind_id"],
                    "source_id": rows[global_idx]["source_id"],
                    "final_class": rows[global_idx]["final_class"],
                    "score": float(speaker_selected["scores"][local_idx]),
                    "threshold": float(speaker_selected["threshold"]),
                    "accepted": bool(speaker_selected["scores"][local_idx] >= speaker_selected["threshold"]),
                }
                for local_idx, global_idx in enumerate(speaker_indices)
            ],
        },
        "boundary": {
            "sample_count": len(boundary_indices),
            "feature_count": boundary_matrix_all.shape[1],
            "selected": public_candidate(boundary_selected),
            "candidates": [public_candidate(row) for row in boundary_candidates],
            "scores": [
                {
                    "blind_id": rows[global_idx]["blind_id"],
                    "source_id": rows[global_idx]["source_id"],
                    "final_class": rows[global_idx]["final_class"],
                    "score": float(boundary_selected["scores"][local_idx]),
                    "threshold": float(boundary_selected["threshold"]),
                    "accepted": bool(boundary_selected["scores"][local_idx] >= boundary_selected["threshold"]),
                }
                for local_idx, global_idx in enumerate(boundary_indices)
            ],
        },
        "deployable": False,
        "recommendation": "build_fresh_blind_acceptance_pack",
        "model_bundle": str(output_dir / "clip_acceptance_models.joblib"),
    }
    write_json(output_dir / "report.json", report)
    print(
        json.dumps(
            {
                "speaker": report["speaker"]["selected"],
                "boundary": report["boundary"]["selected"],
                "deployable": False,
                "saved": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
