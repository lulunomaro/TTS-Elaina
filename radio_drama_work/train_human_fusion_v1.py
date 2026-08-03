# -*- coding: utf-8 -*-
"""Train and audit a three-encoder speaker classifier from blind human labels.

The deployable form is a bank of 12 leave-one-source-out models.  Audio from a
source is therefore scored by a model that never trained on human labels from
that source.  Mixed/boundary/variant clips are validation or hard-exclusion
data only; only fully non-target clips enter the negative speaker class.
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

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import calibrate_classifier_v1 as base


HUMAN_DIR = SCRIPT_DIR / "human_calibration_v1"
LABELS_PATH = HUMAN_DIR / "labels_resolved.json"
HUMAN_CACHE = HUMAN_DIR / "embedding_cache"
OUT_DIR = SCRIPT_DIR / "human_fusion_v1"
MODEL_CACHE_NAMES = {
    "CAM++": "campp_v1.npz",
    "ERes2NetV2": "eres2net_v1.npz",
    "SpeechBrain-ECAPA-VoxCeleb": "speechbrain_ecapa_voxceleb_v1.npz",
}
TARGET_METRICS = ("target_center", "target_median", "target_p25", "target_top4", "target_max")
NEGATIVE_METRICS = ("negative_center", "negative_median", "negative_p75", "negative_top3", "negative_max")
MARGIN_METRICS = ("center_minus_negative_max", "top4_minus_negative_top3")
REGULARISATION_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
THRESHOLD_BUFFER_GRID = (0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8")


def load_old_target_embeddings(assets: dict) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    expected_all = [str(path).lower() for path in assets["all_paths"]]
    target_keys = {str(path).lower() for path in assets["target_refs"]}
    for model_name, cache_name in MODEL_CACHE_NAMES.items():
        cache_path = base.OUT / "embedding_cache" / cache_name
        data = np.load(str(cache_path), allow_pickle=True)
        paths = [str(value) for value in data["paths"].tolist()]
        if [path.lower() for path in paths] != expected_all:
            raise RuntimeError(f"Stale base embedding cache for {model_name}")
        embeddings = base.normalize_rows(data["embeddings"])
        indices = [idx for idx, path in enumerate(paths) if path.lower() in target_keys]
        if len(indices) != len(target_keys):
            raise RuntimeError(f"Missing target-reference embeddings for {model_name}: {len(indices)}/{len(target_keys)}")
        result[model_name] = embeddings[indices]
    return result


def encode_human(paths: list[Path], resume: bool) -> dict[str, np.ndarray]:
    original_cache = base.CACHE
    base.CACHE = HUMAN_CACHE
    try:
        return {
            "CAM++": base.encode_campp(paths, resume),
            "ERes2NetV2": base.encode_eres2net(paths, resume),
            "SpeechBrain-ECAPA-VoxCeleb": base.encode_ecapa(paths, resume),
        }
    finally:
        base.CACHE = original_cache


def target_block(query: np.ndarray, references: np.ndarray) -> list[float]:
    centroid = base.normalize_rows(references.mean(axis=0, keepdims=True))[0]
    similarities = references @ query
    top = np.sort(similarities)[-min(4, len(similarities)) :]
    return [
        float(centroid @ query),
        float(np.median(similarities)),
        float(np.quantile(similarities, 0.25)),
        float(np.mean(top)),
        float(np.max(similarities)),
    ]


def negative_block(query: np.ndarray, references: np.ndarray) -> list[float]:
    if not len(references):
        return [-1.0] * len(NEGATIVE_METRICS)
    centroid = base.normalize_rows(references.mean(axis=0, keepdims=True))[0]
    similarities = references @ query
    top = np.sort(similarities)[-min(3, len(similarities)) :]
    return [
        float(centroid @ query),
        float(np.median(similarities)),
        float(np.quantile(similarities, 0.75)),
        float(np.mean(top)),
        float(np.max(similarities)),
    ]


def make_feature_names() -> list[str]:
    result: list[str] = []
    for model_name in MODEL_CACHE_NAMES:
        result.extend(f"{model_name}:{name}" for name in TARGET_METRICS + NEGATIVE_METRICS + MARGIN_METRICS)
    return result


def one_feature(
    query_idx: int,
    negative_indices: list[int],
    human_embeddings: dict[str, np.ndarray],
    target_embeddings: dict[str, np.ndarray],
    leave_self_out: bool,
) -> np.ndarray:
    result: list[float] = []
    usable_negative_indices = [idx for idx in negative_indices if not (leave_self_out and idx == query_idx)]
    for model_name in MODEL_CACHE_NAMES:
        query = human_embeddings[model_name][query_idx]
        target_values = target_block(query, target_embeddings[model_name])
        negative_refs = human_embeddings[model_name][usable_negative_indices]
        negative_values = negative_block(query, negative_refs)
        result.extend(target_values)
        result.extend(negative_values)
        result.append(target_values[0] - negative_values[4])
        result.append(target_values[3] - negative_values[3])
    return np.asarray(result, dtype=np.float64)


def fit_source_fold(
    held_source: str,
    rows: list[dict],
    human_embeddings: dict[str, np.ndarray],
    target_embeddings: dict[str, np.ndarray],
    regularisation: float,
    threshold_buffer: float,
) -> dict:
    pure_indices = [
        idx
        for idx, row in enumerate(rows)
        if row["kind"] == "probe"
        and row["final_class"] in {"target", "pure_non_target"}
        and row["source_id"] != held_source
    ]
    negative_indices = [idx for idx in pure_indices if rows[idx]["final_class"] == "pure_non_target"]
    if not negative_indices or not any(rows[idx]["final_class"] == "target" for idx in pure_indices):
        raise RuntimeError(f"Fold {held_source} has only one training class")
    matrix = np.stack(
        [
            one_feature(
                idx,
                negative_indices,
                human_embeddings,
                target_embeddings,
                leave_self_out=rows[idx]["final_class"] == "pure_non_target",
            )
            for idx in pure_indices
        ]
    )
    labels = np.asarray([int(rows[idx]["final_class"] == "target") for idx in pure_indices], dtype=np.int64)
    scaler = StandardScaler().fit(matrix)
    classifier = LogisticRegression(
        C=regularisation,
        class_weight="balanced",
        max_iter=5000,
        random_state=20260802,
    )
    classifier.fit(scaler.transform(matrix), labels)
    training_decisions = classifier.decision_function(scaler.transform(matrix))
    negative_mask = labels == 0
    threshold = float(np.max(training_decisions[negative_mask]) + threshold_buffer + 1e-7)
    return {
        "held_source": held_source,
        "pure_indices": pure_indices,
        "negative_indices": negative_indices,
        "scaler": scaler,
        "classifier": classifier,
        "threshold": threshold,
        "training_negative_max": float(np.max(training_decisions[negative_mask])),
        "training_positive_min": float(np.min(training_decisions[~negative_mask])),
    }


def score_with_fold(
    fold: dict,
    query_idx: int,
    human_embeddings: dict[str, np.ndarray],
    target_embeddings: dict[str, np.ndarray],
) -> tuple[float, bool]:
    feature = one_feature(
        query_idx,
        fold["negative_indices"],
        human_embeddings,
        target_embeddings,
        leave_self_out=False,
    )
    decision = float(fold["classifier"].decision_function(fold["scaler"].transform(feature.reshape(1, -1)))[0])
    return decision, decision >= float(fold["threshold"])


def evaluate_candidate(
    rows: list[dict],
    human_embeddings: dict[str, np.ndarray],
    target_embeddings: dict[str, np.ndarray],
    regularisation: float,
    threshold_buffer: float,
) -> tuple[dict, dict[str, dict]]:
    sources = sorted({str(row["source_id"]) for row in rows if row["kind"] == "probe"})
    folds = {
        source: fit_source_fold(source, rows, human_embeddings, target_embeddings, regularisation, threshold_buffer)
        for source in sources
    }
    scored: list[dict] = []
    for idx, row in enumerate(rows):
        source = str(row["source_id"])
        if source not in folds:
            raise RuntimeError(f"No source-held-out fold for {row['blind_id']}: {source}")
        decision, accepted = score_with_fold(folds[source], idx, human_embeddings, target_embeddings)
        scored.append(
            {
                "blind_id": row["blind_id"],
                "kind": row["kind"],
                "source_id": source,
                "final_class": row["final_class"],
                "decision": decision,
                "threshold": float(folds[source]["threshold"]),
                "margin": decision - float(folds[source]["threshold"]),
                "accepted": bool(accepted),
                "correct_safe_decision": bool(accepted == (row["final_class"] == "target")),
            }
        )

    def count(kind: str, label: str, accepted: bool) -> int:
        return sum(
            row["kind"] == kind and row["final_class"] == label and bool(row["accepted"]) == accepted for row in scored
        )

    probe_targets = count("probe", "target", True)
    probe_target_total = sum(row["kind"] == "probe" and row["final_class"] == "target" for row in scored)
    probe_neg_rejected = count("probe", "pure_non_target", False)
    probe_neg_total = sum(row["kind"] == "probe" and row["final_class"] == "pure_non_target" for row in scored)
    probe_mixed_rejected = count("probe", "mixed", False)
    probe_mixed_total = sum(row["kind"] == "probe" and row["final_class"] == "mixed" for row in scored)
    variant_rejected = count("probe", "target_variant_exclude", False)
    variant_total = sum(row["kind"] == "probe" and row["final_class"] == "target_variant_exclude" for row in scored)
    control_target_retained = count("control", "target", True)
    control_target_total = sum(row["kind"] == "control" and row["final_class"] == "target" for row in scored)
    control_unsafe_rejected = sum(
        row["kind"] == "control" and row["final_class"] != "target" and not row["accepted"] for row in scored
    )
    control_unsafe_total = sum(row["kind"] == "control" and row["final_class"] != "target" for row in scored)
    summary = {
        "C": regularisation,
        "threshold_buffer": threshold_buffer,
        "probe_target_retained": probe_targets,
        "probe_target_total": probe_target_total,
        "probe_pure_negative_rejected": probe_neg_rejected,
        "probe_pure_negative_total": probe_neg_total,
        "probe_mixed_rejected": probe_mixed_rejected,
        "probe_mixed_total": probe_mixed_total,
        "probe_target_variant_rejected": variant_rejected,
        "probe_target_variant_total": variant_total,
        "control_target_retained": control_target_retained,
        "control_target_total": control_target_total,
        "control_unsafe_rejected": control_unsafe_rejected,
        "control_unsafe_total": control_unsafe_total,
        "probe_pure_safe_accuracy": (probe_targets + probe_neg_rejected) / max(1, probe_target_total + probe_neg_total),
        "scored": scored,
    }
    # Controls are deliberately omitted from model-selection rank.  They remain
    # an independent final check after C/buffer selection.
    summary["selection_rank"] = [
        probe_neg_rejected,
        probe_mixed_rejected,
        probe_targets,
        variant_rejected,
        -threshold_buffer,
        -abs(math.log10(regularisation)),
    ]
    return summary, folds


def serialise_fold(fold: dict, rows: list[dict]) -> dict:
    scaler: StandardScaler = fold["scaler"]
    classifier: LogisticRegression = fold["classifier"]
    return {
        "held_source": fold["held_source"],
        "feature_names": make_feature_names(),
        "training_blind_ids": [rows[idx]["blind_id"] for idx in fold["pure_indices"]],
        "negative_reference_blind_ids": [rows[idx]["blind_id"] for idx in fold["negative_indices"]],
        "threshold": float(fold["threshold"]),
        "training_negative_max": float(fold["training_negative_max"]),
        "training_positive_min": float(fold["training_positive_min"]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficient": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    started = time.time()

    rows = read_json(args.labels.resolve())["samples"]
    paths = [Path(row["audio"]).resolve() for row in rows]
    if len(paths) != 72 or len({str(path).lower() for path in paths}) != 72:
        raise RuntimeError("Expected 72 unique human calibration audio paths")
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError([str(path) for path in paths if not path.is_file()])

    assets = base.build_assets()
    target_embeddings = load_old_target_embeddings(assets)
    human_embeddings = encode_human(paths, args.resume)
    dimensions = {name: int(values.shape[1]) for name, values in human_embeddings.items()}
    target_counts = {name: int(values.shape[0]) for name, values in target_embeddings.items()}

    candidates: list[dict] = []
    fold_cache: dict[tuple[float, float], dict[str, dict]] = {}
    for regularisation in REGULARISATION_GRID:
        for threshold_buffer in THRESHOLD_BUFFER_GRID:
            summary, folds = evaluate_candidate(
                rows,
                human_embeddings,
                target_embeddings,
                regularisation,
                threshold_buffer,
            )
            fold_cache[(regularisation, threshold_buffer)] = folds
            candidates.append(summary)
    selected = max(candidates, key=lambda item: tuple(item["selection_rank"]))
    selected_folds = fold_cache[(float(selected["C"]), float(selected["threshold_buffer"]))]

    control_gate = bool(
        selected["control_target_retained"] == selected["control_target_total"]
        and selected["control_unsafe_rejected"] == selected["control_unsafe_total"]
    )
    source_held_out_gate = bool(
        selected["probe_pure_negative_rejected"] == selected["probe_pure_negative_total"]
        and selected["probe_mixed_rejected"] == selected["probe_mixed_total"]
        and selected["probe_target_variant_rejected"] == selected["probe_target_variant_total"]
        and selected["probe_target_retained"] >= math.ceil(0.90 * selected["probe_target_total"])
    )
    strict_gate = bool(source_held_out_gate and control_gate)

    selected_public = {key: value for key, value in selected.items() if key not in {"scored", "selection_rank"}}
    selected_public["source_held_out_gate"] = source_held_out_gate
    selected_public["independent_hidden_control_gate"] = control_gate
    selected_public["strict_gate_pass"] = strict_gate
    report = {
        "version": "human_three_encoder_fusion_v1",
        "runtime_sec": time.time() - started,
        "policy": {
            "local_resources": RESOURCE_POLICY,
            "speaker_encoders": list(MODEL_CACHE_NAMES),
            "target_enrollment": "168 pre-existing target references; model-specific embeddings never mixed",
            "speaker_training": "23 human target probes versus 26 pure non-target probes",
            "validation_only": "mixed, overlap, boundary, quality, uncertainty, and child-target variant are not pure negatives",
            "source_isolation": "each source is evaluated and deployed with a model trained on the other 11 sources",
            "model_selection": "source-held-out probes only; 12 hidden controls excluded from selection rank",
            "strict_gate": "all held pure negatives, mixed probes and target variants rejected; >=90% held targets retained; all controls safe",
        },
        "embedding_dimensions": dimensions,
        "target_reference_counts": target_counts,
        "feature_names": make_feature_names(),
        "candidate_grid": {
            "regularisation_C": list(REGULARISATION_GRID),
            "threshold_buffer": list(THRESHOLD_BUFFER_GRID),
            "count": len(candidates),
        },
        "candidates": [{key: value for key, value in item.items() if key not in {"scored", "selection_rank"}} for item in candidates],
        "selected": selected_public,
        "selected_scores": selected["scored"],
        "strict_gate_pass": strict_gate,
        "recommendation": "proceed_to_full_track_mining" if strict_gate else "do_not_auto_mine; collect_or_revise_hard_examples",
    }
    output_dir = args.output_dir.resolve()
    write_json(output_dir / "fusion_classifier_report.json", report)
    write_json(
        output_dir / "source_held_out_models.json",
        {
            "version": "human_three_encoder_source_models_v1",
            "deployable": strict_gate,
            "C": selected["C"],
            "threshold_buffer": selected["threshold_buffer"],
            "models": {source: serialise_fold(fold, rows) for source, fold in selected_folds.items()},
        },
    )
    print(json.dumps({"selected": selected_public, "strict_gate_pass": strict_gate, "saved": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
