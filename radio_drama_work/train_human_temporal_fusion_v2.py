# -*- coding: utf-8 -*-
"""Temporal three-encoder calibration for seamless speaker transitions.

Full-clip embeddings average two adjacent speakers.  This audit instead scores
0.8-second windows every 0.2 seconds, learns only from pure target/pure
non-target clips, and uses mixed clips solely to select a conservative temporal
aggregation rule.  Hidden controls remain outside model/rule selection.
"""
from __future__ import annotations

import argparse
import hashlib
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

import numpy as np
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import calibrate_classifier_v1 as base


HUMAN_DIR = SCRIPT_DIR / "human_calibration_v1"
LABELS_PATH = HUMAN_DIR / "labels_resolved.json"
OUT_DIR = SCRIPT_DIR / "human_temporal_fusion_v2"
WINDOW_DIR = OUT_DIR / "windows_0800ms_hop0200ms"
WINDOW_MANIFEST = OUT_DIR / "window_manifest.json"
WINDOW_CACHE = OUT_DIR / "embedding_cache"
WINDOW_SEC = 0.8
HOP_SEC = 0.2
ACTIVE_RMS = 0.003
MODEL_CACHE_NAMES = {
    "CAM++": "campp_v1.npz",
    "ERes2NetV2": "eres2net_v1.npz",
    "SpeechBrain-ECAPA-VoxCeleb": "speechbrain_ecapa_voxceleb_v1.npz",
}
METRICS = ("target_center", "target_median", "target_p25", "target_top4", "target_max")
REGULARISATION_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
AGGREGATORS = ("minimum", "p05", "p10", "p20", "p25", "mean_minus_std", "median")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8")


def extraction_fingerprint(rows: list[dict]) -> str:
    files = []
    for row in rows:
        path = Path(row["audio"])
        stat = path.stat()
        files.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    payload = {"window_sec": WINDOW_SEC, "hop_sec": HOP_SEC, "files": files}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def window_starts(duration: float) -> list[float]:
    if duration <= WINDOW_SEC + 1e-9:
        return [0.0]
    last = duration - WINDOW_SEC
    result = [idx * HOP_SEC for idx in range(int(math.floor(last / HOP_SEC)) + 1)]
    if last - result[-1] > 0.025:
        result.append(last)
    return result


def build_windows(rows: list[dict], resume: bool) -> list[dict]:
    fingerprint = extraction_fingerprint(rows)
    if resume and WINDOW_MANIFEST.exists():
        document = read_json(WINDOW_MANIFEST)
        cached = document.get("windows", [])
        if document.get("fingerprint") == fingerprint and cached and all(Path(row["audio"]).is_file() for row in cached):
            print(f"[window cache] {len(cached)} files", flush=True)
            return cached
    WINDOW_DIR.mkdir(parents=True, exist_ok=True)
    windows: list[dict] = []
    target_samples = max(1, int(round(WINDOW_SEC * base.SR)))
    for clip_idx, row in enumerate(rows, 1):
        wave = base.read_16k(Path(row["audio"]))
        duration = len(wave) / base.SR
        for window_idx, start in enumerate(window_starts(duration)):
            start_sample = int(round(start * base.SR))
            chunk = wave[start_sample : start_sample + target_samples]
            valid_samples = len(chunk)
            if len(chunk) < target_samples:
                chunk = np.pad(chunk, (0, target_samples - len(chunk)))
            rms = float(np.sqrt(np.mean(np.square(chunk[:valid_samples], dtype=np.float64)))) if valid_samples else 0.0
            name = f"{row['blind_id']}_w{window_idx:03d}_{int(round(start * 1000)):06d}.wav"
            output_path = WINDOW_DIR / name
            sf.write(str(output_path), chunk, base.SR, subtype="PCM_16")
            windows.append(
                {
                    "window_index": len(windows),
                    "clip_index": clip_idx - 1,
                    "blind_id": row["blind_id"],
                    "source_id": row["source_id"],
                    "kind": row["kind"],
                    "final_class": row["final_class"],
                    "start": float(start),
                    "end": float(min(duration, start + WINDOW_SEC)),
                    "rms": rms,
                    "active": rms >= ACTIVE_RMS,
                    "audio": str(output_path.resolve()),
                }
            )
        if clip_idx % 12 == 0 or clip_idx == len(rows):
            print(f"[window extraction] {clip_idx}/{len(rows)} clips, {len(windows)} windows", flush=True)
    write_json(
        WINDOW_MANIFEST,
        {
            "version": "human_temporal_windows_v1",
            "fingerprint": fingerprint,
            "window_sec": WINDOW_SEC,
            "hop_sec": HOP_SEC,
            "active_rms": ACTIVE_RMS,
            "windows": windows,
        },
    )
    return windows


def load_old_target_embeddings(assets: dict) -> dict[str, np.ndarray]:
    expected_all = [str(path).lower() for path in assets["all_paths"]]
    target_keys = {str(path).lower() for path in assets["target_refs"]}
    result: dict[str, np.ndarray] = {}
    for model_name, cache_name in MODEL_CACHE_NAMES.items():
        data = np.load(str(base.OUT / "embedding_cache" / cache_name), allow_pickle=True)
        paths = [str(value) for value in data["paths"].tolist()]
        if [path.lower() for path in paths] != expected_all:
            raise RuntimeError(f"Stale base cache for {model_name}")
        embeddings = base.normalize_rows(data["embeddings"])
        indices = [idx for idx, path in enumerate(paths) if path.lower() in target_keys]
        if len(indices) != len(target_keys):
            raise RuntimeError(f"Missing target references for {model_name}")
        result[model_name] = embeddings[indices]
    return result


def encode_windows(paths: list[Path], resume: bool) -> dict[str, np.ndarray]:
    original_cache = base.CACHE
    base.CACHE = WINDOW_CACHE
    try:
        return {
            "CAM++": base.encode_campp(paths, resume),
            "ERes2NetV2": base.encode_eres2net(paths, resume),
            "SpeechBrain-ECAPA-VoxCeleb": base.encode_ecapa(paths, resume),
        }
    finally:
        base.CACHE = original_cache


def feature_names() -> list[str]:
    return [f"{model_name}:{metric}" for model_name in MODEL_CACHE_NAMES for metric in METRICS]


def build_window_features(
    embeddings: dict[str, np.ndarray], target_embeddings: dict[str, np.ndarray]
) -> np.ndarray:
    blocks = []
    for model_name in MODEL_CACHE_NAMES:
        queries = embeddings[model_name]
        targets = target_embeddings[model_name]
        centroid = base.normalize_rows(targets.mean(axis=0, keepdims=True))[0]
        similarities = queries @ targets.T
        blocks.append(
            np.column_stack(
                [
                    queries @ centroid,
                    np.median(similarities, axis=1),
                    np.quantile(similarities, 0.25, axis=1),
                    np.mean(np.sort(similarities, axis=1)[:, -4:], axis=1),
                    np.max(similarities, axis=1),
                ]
            )
        )
    return np.concatenate(blocks, axis=1).astype(np.float64, copy=False)


def train_fold(
    held_source: str,
    rows: list[dict],
    windows: list[dict],
    features: np.ndarray,
    regularisation: float,
) -> tuple[StandardScaler, LogisticRegression]:
    pure_clip_indices = [
        idx
        for idx, row in enumerate(rows)
        if row["kind"] == "probe"
        and row["final_class"] in {"target", "pure_non_target"}
        and str(row["source_id"]) != held_source
    ]
    pure_set = set(pure_clip_indices)
    window_indices = [
        idx for idx, window in enumerate(windows) if window["active"] and int(window["clip_index"]) in pure_set
    ]
    labels = np.asarray(
        [int(rows[int(windows[idx]["clip_index"])]["final_class"] == "target") for idx in window_indices],
        dtype=np.int64,
    )
    if len(set(labels.tolist())) != 2:
        raise RuntimeError(f"Fold {held_source} has one window class")

    clip_window_counts: dict[int, int] = {}
    for idx in window_indices:
        clip_idx = int(windows[idx]["clip_index"])
        clip_window_counts[clip_idx] = clip_window_counts.get(clip_idx, 0) + 1
    class_clip_counts = {
        label: sum(rows[idx]["final_class"] == ("target" if label else "pure_non_target") for idx in pure_clip_indices)
        for label in (0, 1)
    }
    sample_weight = np.asarray(
        [
            0.5
            / class_clip_counts[int(label)]
            / clip_window_counts[int(windows[window_idx]["clip_index"])]
            for window_idx, label in zip(window_indices, labels)
        ],
        dtype=np.float64,
    )
    sample_weight *= len(sample_weight) / sample_weight.sum()
    matrix = features[window_indices]
    scaler = StandardScaler().fit(matrix, sample_weight=sample_weight)
    classifier = LogisticRegression(C=regularisation, max_iter=5000, random_state=20260802)
    classifier.fit(scaler.transform(matrix), labels, sample_weight=sample_weight)
    return scaler, classifier


def aggregate(values: np.ndarray, name: str) -> float:
    if name == "minimum":
        return float(np.min(values))
    if name == "p05":
        return float(np.quantile(values, 0.05))
    if name == "p10":
        return float(np.quantile(values, 0.10))
    if name == "p20":
        return float(np.quantile(values, 0.20))
    if name == "p25":
        return float(np.quantile(values, 0.25))
    if name == "mean_minus_std":
        return float(np.mean(values) - np.std(values))
    if name == "median":
        return float(np.median(values))
    raise KeyError(name)


def oof_window_scores(
    regularisation: float,
    rows: list[dict],
    windows: list[dict],
    features: np.ndarray,
) -> tuple[list[np.ndarray], dict[str, dict]]:
    sources = sorted({str(row["source_id"]) for row in rows if row["kind"] == "probe"})
    folds: dict[str, dict] = {}
    clip_scores: list[list[float]] = [[] for _ in rows]
    for source in sources:
        scaler, classifier = train_fold(source, rows, windows, features, regularisation)
        folds[source] = {"scaler": scaler, "classifier": classifier}
        source_window_indices = [
            idx for idx, window in enumerate(windows) if str(window["source_id"]) == source and window["active"]
        ]
        decisions = classifier.decision_function(scaler.transform(features[source_window_indices]))
        for window_idx, decision in zip(source_window_indices, decisions):
            clip_scores[int(windows[window_idx]["clip_index"])].append(float(decision))
    if any(not values for values in clip_scores):
        missing = [rows[idx]["blind_id"] for idx, values in enumerate(clip_scores) if not values]
        raise RuntimeError(f"No active windows for clips: {missing}")
    return [np.asarray(values, dtype=np.float64) for values in clip_scores], folds


def evaluate_rule(rows: list[dict], clip_scores: list[np.ndarray], aggregator: str, regularisation: float) -> dict:
    aggregate_scores = np.asarray([aggregate(values, aggregator) for values in clip_scores], dtype=np.float64)
    calibration_unsafe = [
        idx
        for idx, row in enumerate(rows)
        if row["kind"] == "probe" and row["final_class"] in {"pure_non_target", "mixed", "target_variant_exclude"}
    ]
    threshold = float(np.max(aggregate_scores[calibration_unsafe]) + 1e-7)
    accepted = aggregate_scores >= threshold

    def total(kind: str, label: str) -> int:
        return sum(row["kind"] == kind and row["final_class"] == label for row in rows)

    def accepted_count(kind: str, label: str) -> int:
        return sum(
            row["kind"] == kind and row["final_class"] == label and bool(accepted[idx]) for idx, row in enumerate(rows)
        )

    target_total = total("probe", "target")
    negative_total = total("probe", "pure_non_target")
    mixed_total = total("probe", "mixed")
    variant_total = total("probe", "target_variant_exclude")
    control_target_total = total("control", "target")
    control_unsafe_total = sum(row["kind"] == "control" and row["final_class"] != "target" for row in rows)
    result = {
        "C": regularisation,
        "aggregator": aggregator,
        "threshold": threshold,
        "probe_target_retained": accepted_count("probe", "target"),
        "probe_target_total": target_total,
        "probe_pure_negative_rejected": negative_total - accepted_count("probe", "pure_non_target"),
        "probe_pure_negative_total": negative_total,
        "probe_mixed_rejected": mixed_total - accepted_count("probe", "mixed"),
        "probe_mixed_total": mixed_total,
        "probe_target_variant_rejected": variant_total - accepted_count("probe", "target_variant_exclude"),
        "probe_target_variant_total": variant_total,
        "control_target_retained": accepted_count("control", "target"),
        "control_target_total": control_target_total,
        "control_unsafe_rejected": sum(
            row["kind"] == "control" and row["final_class"] != "target" and not accepted[idx]
            for idx, row in enumerate(rows)
        ),
        "control_unsafe_total": control_unsafe_total,
        "scores": [
            {
                "blind_id": row["blind_id"],
                "kind": row["kind"],
                "source_id": row["source_id"],
                "final_class": row["final_class"],
                "active_window_count": int(len(clip_scores[idx])),
                "aggregate_score": float(aggregate_scores[idx]),
                "threshold": threshold,
                "margin": float(aggregate_scores[idx] - threshold),
                "accepted": bool(accepted[idx]),
                "correct_safe_decision": bool(bool(accepted[idx]) == (row["final_class"] == "target")),
            }
            for idx, row in enumerate(rows)
        ],
    }
    result["selection_rank"] = [
        result["probe_pure_negative_rejected"],
        result["probe_mixed_rejected"],
        result["probe_target_variant_rejected"],
        result["probe_target_retained"],
        -abs(math.log10(regularisation)),
    ]
    return result


def serialise_fold(fold: dict) -> dict:
    scaler: StandardScaler = fold["scaler"]
    classifier: LogisticRegression = fold["classifier"]
    return {
        "feature_names": feature_names(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficient": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
    }


def main() -> None:
    global OUT_DIR, WINDOW_DIR, WINDOW_MANIFEST, WINDOW_CACHE
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    started = time.time()
    OUT_DIR = args.output_dir.resolve()
    WINDOW_DIR = OUT_DIR / "windows_0800ms_hop0200ms"
    WINDOW_MANIFEST = OUT_DIR / "window_manifest.json"
    WINDOW_CACHE = OUT_DIR / "embedding_cache"

    rows = read_json(args.labels.resolve())["samples"]
    windows = build_windows(rows, args.resume)
    window_paths = [Path(row["audio"]).resolve() for row in windows]
    assets = base.build_assets()
    target_embeddings = load_old_target_embeddings(assets)
    window_embeddings = encode_windows(window_paths, args.resume)
    features = build_window_features(window_embeddings, target_embeddings)

    candidates: list[dict] = []
    fold_cache: dict[float, dict[str, dict]] = {}
    for regularisation in REGULARISATION_GRID:
        clip_scores, folds = oof_window_scores(regularisation, rows, windows, features)
        fold_cache[regularisation] = folds
        for aggregator in AGGREGATORS:
            candidates.append(evaluate_rule(rows, clip_scores, aggregator, regularisation))
    selected = max(candidates, key=lambda item: tuple(item["selection_rank"]))
    selected_folds = fold_cache[float(selected["C"])]
    controls_pass = bool(
        selected["control_target_retained"] == selected["control_target_total"]
        and selected["control_unsafe_rejected"] == selected["control_unsafe_total"]
    )
    probes_pass = bool(
        selected["probe_pure_negative_rejected"] == selected["probe_pure_negative_total"]
        and selected["probe_mixed_rejected"] == selected["probe_mixed_total"]
        and selected["probe_target_variant_rejected"] == selected["probe_target_variant_total"]
        and selected["probe_target_retained"] >= math.ceil(0.90 * selected["probe_target_total"])
    )
    strict_gate = bool(probes_pass and controls_pass)

    selected_public = {key: value for key, value in selected.items() if key not in {"scores", "selection_rank"}}
    selected_public.update(
        {
            "source_held_out_probe_gate": probes_pass,
            "independent_hidden_control_gate": controls_pass,
            "strict_gate_pass": strict_gate,
        }
    )
    report = {
        "version": "human_temporal_three_encoder_fusion_v2",
        "runtime_sec": time.time() - started,
        "window": {"seconds": WINDOW_SEC, "hop_seconds": HOP_SEC, "active_rms": ACTIVE_RMS},
        "window_counts": {
            "all": len(windows),
            "active": sum(bool(row["active"]) for row in windows),
            "inactive": sum(not bool(row["active"]) for row in windows),
        },
        "policy": {
            "local_resources": RESOURCE_POLICY,
            "training_labels": "only pure target and pure non-target probe clips",
            "mixed_usage": "threshold calibration only; never a pure negative training window",
            "source_isolation": "every scored clip uses a window classifier trained without its source",
            "hidden_controls": "excluded from classifier and rule selection",
            "acceptance": "the selected lower-tail temporal aggregate must exceed the calibrated unsafe maximum",
        },
        "feature_names": feature_names(),
        "candidate_count": len(candidates),
        "candidates": [{key: value for key, value in row.items() if key not in {"scores", "selection_rank"}} for row in candidates],
        "selected": selected_public,
        "selected_scores": selected["scores"],
        "strict_gate_pass": strict_gate,
        "recommendation": "proceed_to_full_track_temporal_mining" if strict_gate else "do_not_deploy; inspect temporal hard errors",
    }
    write_json(OUT_DIR / "temporal_fusion_report.json", report)
    write_json(
        OUT_DIR / "source_held_out_window_models.json",
        {
            "version": "human_temporal_source_models_v2",
            "deployable": strict_gate,
            "window": report["window"],
            "C": selected["C"],
            "aggregator": selected["aggregator"],
            "threshold": selected["threshold"],
            "models": {source: serialise_fold(fold) for source, fold in selected_folds.items()},
        },
    )
    print(json.dumps({"selected": selected_public, "strict_gate_pass": strict_gate, "saved": str(OUT_DIR)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
