# -*- coding: utf-8 -*-
"""Cross-calibrate independent speaker encoders for Ireina mining.

This phase is deliberately independent from training and from all previous
candidate lists.  Mixed clips are evaluation-only unsafe examples; they are
never enrolled as pure negative speaker references.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


ROOT = Path(__file__).resolve().parent
PKG = ROOT.parent / "GPT-SoVITS-v2pro-20250604-nvidia50" / "GPT-SoVITS-v2pro-20250604-nvidia50"
OUT = ROOT / "calibrate_classifier_v1"
CACHE = OUT / "embedding_cache"
REPORT = OUT / "calibration_report.json"

OLD_REF_DIR = PKG / "logs" / "my_voice" / "5-wav32k"
CONFIRMED_TARGET = ROOT / "confirmed_target_pure.txt"
CONFIRMED_NEGATIVE = ROOT / "confirmed_non_target_or_mixed.txt"
GOLD_DIR = ROOT / "full_funasr_gold_v9" / "accepted_segments"
CAM_MODEL = ROOT / "modelscope_cache" / "models" / "iic" / "speech_campplus_sv_zh-cn_16k-common"
ERES_MODEL = PKG / "GPT_SoVITS" / "pretrained_models" / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt"
SPEECHBRAIN_PKG = ROOT / "third_party" / "speechbrain_pkg"
ECAPA_MODEL = ROOT / "third_party" / "models" / "spkrec-ecapa-voxceleb"
ECAPA_RUNTIME = ECAPA_MODEL / "sb_runtime_local"

EXPLICIT_HARD_NEGATIVES = [
    ROOT / "new_radio_quality_v1" / "accepted_segments" / "newradio_00001_243358894-1-192_vocals_0003417760_0003505760.wav",
    ROOT / "new_radio_quality_v1" / "accepted_segments" / "newradio_00003_243358894-1-192_vocals_0007518240_0007613280.wav",
    ROOT / "new_radio_quality_v1" / "accepted_segments" / "newradio_00004_243358894-1-192_vocals_0007968000_0008032000.wav",
]

SR = 16000
METRIC_NAMES = ("center", "median", "p25", "margin")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def listed_rows(path: Path) -> list[tuple[Path, str]]:
    rows: list[tuple[Path, str]] = []
    if not path.exists():
        return rows
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        left, reason = (raw.split("|", 1) + [""])[:2]
        item = Path(left.strip())
        if item.exists():
            rows.append((item, reason.strip()))
    return rows


def unique_paths(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve()).lower()
        if key not in seen:
            seen.add(key)
            result.append(path.resolve())
    return result


def build_assets() -> dict:
    old_refs = sorted(path.resolve() for path in OLD_REF_DIR.glob("*.wav"))
    confirmed_target = [path.resolve() for path, _ in listed_rows(CONFIRMED_TARGET)]
    gold = sorted(path.resolve() for path in GOLD_DIR.glob("*.wav"))
    confirmed_rows = listed_rows(CONFIRMED_NEGATIVE)
    pure_non_target: list[Path] = []
    mixed_validation: list[Path] = []
    negative_reasons: dict[str, str] = {}
    for path, reason in confirmed_rows:
        resolved = path.resolve()
        negative_reasons[str(resolved)] = reason
        lower = reason.lower()
        if "mixed" in lower or "混" in lower:
            mixed_validation.append(resolved)
        else:
            pure_non_target.append(resolved)
    explicit = [path.resolve() for path in EXPLICIT_HARD_NEGATIVES if path.exists()]
    target_refs = unique_paths(old_refs + confirmed_target)
    negative_refs = unique_paths(pure_non_target + explicit)
    validation_negatives = unique_paths(pure_non_target + mixed_validation + explicit)
    all_paths = unique_paths(target_refs + gold + validation_negatives)
    forbidden_training = [path for path in all_paths if "clearvoice" in str(path).lower() or "vasr" in str(path).lower() or "audio_super" in str(path).lower()]
    if forbidden_training:
        raise RuntimeError(f"Forbidden reconstructed assets entered calibration corpus: {forbidden_training}")
    missing_required = [path for path in [CAM_MODEL, ERES_MODEL, SPEECHBRAIN_PKG, ECAPA_MODEL] if not path.exists()]
    if missing_required:
        raise FileNotFoundError(missing_required)
    return {
        "old_refs": old_refs,
        "confirmed_target": confirmed_target,
        "gold": gold,
        "pure_non_target": unique_paths(pure_non_target),
        "mixed_validation": unique_paths(mixed_validation),
        "explicit_hard_negatives": explicit,
        "target_refs": target_refs,
        "negative_refs": negative_refs,
        "validation_negatives": validation_negatives,
        "all_paths": all_paths,
        "negative_reasons": negative_reasons,
    }


def fingerprint(paths: list[Path], model_path: Path | list[Path], tag: str) -> str:
    model_paths = [model_path] if isinstance(model_path, Path) else model_path
    records = []
    for path in paths + model_paths:
        stat = path.stat()
        records.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    raw = json.dumps({"tag": tag, "files": records}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_16k(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32, copy=False)
    if sr != SR:
        mono = torchaudio.functional.resample(torch.from_numpy(mono), int(sr), SR).numpy()
    if len(mono) < int(0.20 * SR):
        mono = np.pad(mono, (0, int(0.20 * SR) - len(mono)))
    return np.asarray(mono, dtype=np.float32)


def normalize_rows(values: np.ndarray | torch.Tensor) -> np.ndarray:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    tensor = torch.nn.functional.normalize(tensor, dim=-1)
    return tensor.cpu().numpy().astype(np.float32, copy=False)


def load_cache(name: str, paths: list[Path], expected_fingerprint: str) -> np.ndarray | None:
    path = CACHE / f"{name}.npz"
    if not path.exists():
        return None
    data = np.load(str(path), allow_pickle=True)
    cached_paths = [str(value) for value in data["paths"].tolist()]
    if str(data["fingerprint"].item()) != expected_fingerprint or cached_paths != [str(x) for x in paths]:
        return None
    return normalize_rows(data["embeddings"])


def save_cache(name: str, paths: list[Path], embeddings: np.ndarray, fp: str) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(CACHE / f"{name}.npz"),
        embeddings=normalize_rows(embeddings),
        paths=np.asarray([str(path) for path in paths]),
        fingerprint=np.asarray(fp),
    )


def encode_campp(paths: list[Path], resume: bool) -> np.ndarray:
    fp = fingerprint(paths, CAM_MODEL / "campplus_cn_common.bin", "CAM++-16k-fullclip-v1")
    cached = load_cache("campp_v1", paths, fp) if resume else None
    if cached is not None:
        print(f"[CAM++ cache] {len(paths)} files", flush=True)
        return cached
    from funasr.utils import install_model_requirements

    install_model_requirements.install_requirements = lambda path: print(f"[CAM++ deps] bundled; skip {path}", flush=True)
    from funasr import AutoModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModel(model=str(CAM_MODEL), device=device, check_latest=False, disable_pbar=True)
    output: list[np.ndarray] = []
    for idx, path in enumerate(paths, 1):
        wave = read_16k(path)
        result = model.generate(input=[wave], fs=SR, batch_size=1, disable_pbar=True)
        emb = result[0]["spk_embedding"]
        emb = torch.as_tensor(emb).reshape(1, -1)
        output.append(normalize_rows(emb)[0])
        if idx % 20 == 0 or idx == len(paths):
            print(f"[CAM++] {idx}/{len(paths)}", flush=True)
    embeddings = normalize_rows(np.stack(output))
    save_cache("campp_v1", paths, embeddings, fp)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embeddings


def encode_eres2net(paths: list[Path], resume: bool) -> np.ndarray:
    fp = fingerprint(paths, ERES_MODEL, "ERes2NetV2-fbank16k-fullclip-v1")
    cached = load_cache("eres2net_v1", paths, fp) if resume else None
    if cached is not None:
        print(f"[ERes2Net cache] {len(paths)} files", flush=True)
        return cached
    eres_dir = PKG / "GPT_SoVITS" / "eres2net"
    if str(eres_dir) not in sys.path:
        sys.path.insert(0, str(eres_dir))
    from ERes2NetV2 import ERes2NetV2
    import kaldi as Kaldi

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
    model.load_state_dict(torch.load(ERES_MODEL, map_location="cpu"))
    model.eval().to(device)
    output: list[np.ndarray] = []
    with torch.no_grad():
        for idx, path in enumerate(paths, 1):
            wave = torch.from_numpy(read_16k(path)).unsqueeze(0)
            feats = Kaldi.fbank(wave, num_mel_bins=80, sample_frequency=SR, dither=0).unsqueeze(0).to(device)
            emb = torch.nn.functional.normalize(model.forward(feats), dim=-1).cpu()
            output.append(emb.numpy()[0])
            if idx % 20 == 0 or idx == len(paths):
                print(f"[ERes2Net] {idx}/{len(paths)}", flush=True)
    embeddings = normalize_rows(np.stack(output))
    save_cache("eres2net_v1", paths, embeddings, fp)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embeddings


def encode_ecapa(paths: list[Path], resume: bool) -> np.ndarray:
    model_files = [
        ECAPA_MODEL / "embedding_model.ckpt",
        ECAPA_MODEL / "mean_var_norm_emb.ckpt",
        ECAPA_MODEL / "hyperparams.yaml",
    ]
    fp = fingerprint(paths, model_files, "SpeechBrain-ECAPA-VoxCeleb-16k-fullclip-v1")
    cached = load_cache("speechbrain_ecapa_voxceleb_v1", paths, fp) if resume else None
    if cached is not None:
        print(f"[ECAPA cache] {len(paths)} files", flush=True)
        return cached
    if str(SPEECHBRAIN_PKG) not in sys.path:
        sys.path.insert(0, str(SPEECHBRAIN_PKG))
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = EncoderClassifier.from_hparams(
        source=str(ECAPA_MODEL),
        savedir=str(ECAPA_RUNTIME),
        local_strategy=LocalStrategy.COPY,
        run_opts={"device": device},
    )
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for idx, path in enumerate(paths, 1):
            wave = torch.from_numpy(read_16k(path)).unsqueeze(0).to(device)
            emb = model.encode_batch(wave).reshape(1, -1)
            output.append(normalize_rows(emb)[0])
            if idx % 20 == 0 or idx == len(paths):
                print(f"[ECAPA] {idx}/{len(paths)}", flush=True)
    embeddings = normalize_rows(np.stack(output))
    save_cache("speechbrain_ecapa_voxceleb_v1", paths, embeddings, fp)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embeddings


def quantile_summary(values: list[float] | np.ndarray) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"n": 0}
    return {
        "n": int(len(array)),
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def make_metric_rows(embeddings: np.ndarray, assets: dict) -> tuple[list[dict], list[dict], list[dict]]:
    paths: list[Path] = assets["all_paths"]
    index = {str(path).lower(): idx for idx, path in enumerate(paths)}
    target_paths: list[Path] = assets["target_refs"]
    negative_paths: list[Path] = assets["negative_refs"]
    target_indices = [index[str(path).lower()] for path in target_paths]
    negative_indices = [index[str(path).lower()] for path in negative_paths]
    target_matrix = embeddings[target_indices]
    negative_matrix = embeddings[negative_indices]

    def one(path: Path, category: str, loo_target: bool = False, loo_negative: bool = False) -> dict:
        key = str(path).lower()
        emb = embeddings[index[key]]
        use_target = [i for i, item in enumerate(target_paths) if not (loo_target and str(item).lower() == key)]
        use_negative = [i for i, item in enumerate(negative_paths) if not (loo_negative and str(item).lower() == key)]
        targets = target_matrix[use_target]
        negatives = negative_matrix[use_negative] if use_negative else negative_matrix
        centroid = normalize_rows(targets.mean(axis=0, keepdims=True))[0]
        similarities = targets @ emb
        negative_similarities = negatives @ emb if len(negatives) else np.asarray([-1.0], dtype=np.float32)
        center = float(centroid @ emb)
        negative_max = float(np.max(negative_similarities))
        return {
            "path": str(path),
            "category": category,
            "center": center,
            "median": float(np.median(similarities)),
            "p25": float(np.quantile(similarities, 0.25)),
            "p75": float(np.quantile(similarities, 0.75)),
            "top4": float(np.mean(np.sort(similarities)[-min(4, len(similarities)) :])),
            "negative_max": negative_max,
            "margin": center - negative_max,
        }

    reference_rows = []
    confirmed_set = {str(path).lower() for path in assets["confirmed_target"]}
    for path in target_paths:
        category = "confirmed_target_loo" if str(path).lower() in confirmed_set else "old_reference_loo"
        reference_rows.append(one(path, category, loo_target=True))
    gold_rows = [one(path, "gold") for path in assets["gold"]]
    mixed_set = {str(path).lower() for path in assets["mixed_validation"]}
    explicit_set = {str(path).lower() for path in assets["explicit_hard_negatives"]}
    negative_rows = []
    for path in assets["validation_negatives"]:
        key = str(path).lower()
        if key in mixed_set:
            category = "mixed_validation_only"
        elif key in explicit_set:
            category = "explicit_hard_negative"
        else:
            category = "pure_non_target"
        negative_rows.append(one(path, category, loo_negative=True))
    return reference_rows, gold_rows, negative_rows


def classification_stats(rows: list[dict], thresholds: dict) -> dict:
    predicted = [all(float(row[name]) >= float(thresholds[name]) for name in METRIC_NAMES) for row in rows]
    gold = [idx for idx, row in enumerate(rows) if row["category"] == "gold"]
    positive = [idx for idx, row in enumerate(rows) if row["category"] in {"gold", "old_reference_loo", "confirmed_target_loo"}]
    negative = [idx for idx, row in enumerate(rows) if idx not in positive]
    tp = sum(predicted[idx] for idx in positive)
    fp = sum(predicted[idx] for idx in negative)
    gold_tp = sum(predicted[idx] for idx in gold)
    tpr = tp / max(1, len(positive))
    tnr = (len(negative) - fp) / max(1, len(negative))
    precision = tp / max(1, tp + fp)
    return {
        "positive_count": len(positive),
        "negative_count": len(negative),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "negative_rejected": int(len(negative) - fp),
        "gold_count": len(gold),
        "gold_target_count": int(gold_tp),
        "positive_recall": float(tpr),
        "negative_specificity": float(tnr),
        "balanced_accuracy": float((tpr + tnr) / 2.0),
        "precision": float(precision),
        "f1": float(2 * precision * tpr / max(precision + tpr, 1e-12)),
    }


def threshold_search(rows: list[dict]) -> dict:
    values = {name: np.asarray([float(row[name]) for row in rows], dtype=np.float32) for name in METRIC_NAMES}
    categories = [row["category"] for row in rows]
    pos = np.asarray([cat in {"gold", "old_reference_loo", "confirmed_target_loo"} for cat in categories])
    neg = ~pos
    gold = np.asarray([cat == "gold" for cat in categories])
    center_grid = np.round(np.arange(0.50, 0.851, 0.01), 2)
    median_grid = np.round(np.arange(0.35, 0.751, 0.01), 2)
    p25_grid = np.round(np.arange(0.20, 0.651, 0.01), 2)
    margin_grid = np.round(np.arange(0.00, 0.151, 0.01), 2)
    best_feasible = None
    best_feasible_key = None
    best_overall = None
    best_overall_key = None
    evaluated = 0
    for center in center_grid:
        center_pass = values["center"] >= center
        for median in median_grid:
            cm_pass = center_pass & (values["median"] >= median)
            for p25 in p25_grid:
                base_pass = cm_pass & (values["p25"] >= p25)
                for margin in margin_grid:
                    predicted = base_pass & (values["margin"] >= margin)
                    evaluated += 1
                    tp = int(np.sum(predicted & pos))
                    fp = int(np.sum(predicted & neg))
                    gold_tp = int(np.sum(predicted & gold))
                    neg_rejected = int(np.sum(neg) - fp)
                    tpr = tp / max(1, int(np.sum(pos)))
                    tnr = neg_rejected / max(1, int(np.sum(neg)))
                    balanced = (tpr + tnr) / 2.0
                    strength = float((center - 0.50) / 0.35 + (median - 0.35) / 0.40 + (p25 - 0.20) / 0.45 + margin / 0.15)
                    thresholds = {"center": float(center), "median": float(median), "p25": float(p25), "margin": float(margin)}
                    candidate = {"thresholds": thresholds, "gold_target_count": gold_tp, "negative_rejected": neg_rejected, "positive_recall": float(tpr), "negative_specificity": float(tnr), "balanced_accuracy": float(balanced)}
                    overall_key = (neg_rejected, gold_tp, balanced, tpr, strength)
                    if best_overall_key is None or overall_key > best_overall_key:
                        best_overall_key, best_overall = overall_key, candidate
                    feasible = gold_tp >= min(45, int(np.sum(gold))) and neg_rejected == int(np.sum(neg))
                    if feasible:
                        feasible_key = (gold_tp, tpr, balanced, strength)
                        if best_feasible_key is None or feasible_key > best_feasible_key:
                            best_feasible_key, best_feasible = feasible_key, candidate
    recommended = best_feasible or best_overall
    recommended["gate_pass"] = best_feasible is not None
    recommended["full_stats"] = classification_stats(rows, recommended["thresholds"])
    return {
        "search_space": {
            "center": [float(center_grid[0]), float(center_grid[-1]), 0.01],
            "median": [float(median_grid[0]), float(median_grid[-1]), 0.01],
            "p25": [float(p25_grid[0]), float(p25_grid[-1]), 0.01],
            "margin": [float(margin_grid[0]), float(margin_grid[-1]), 0.01],
            "combinations": int(evaluated),
        },
        "best_feasible": best_feasible,
        "best_overall": best_overall,
        "recommended": recommended,
    }


def auc_report(rows: list[dict]) -> dict:
    labels = np.asarray([int(row["category"] in {"gold", "old_reference_loo", "confirmed_target_loo"}) for row in rows])
    features = np.asarray([[float(row[name]) for name in METRIC_NAMES] for row in rows], dtype=np.float64)
    report = {f"{name}_auc": float(roc_auc_score(labels, features[:, idx])) for idx, name in enumerate(METRIC_NAMES)}
    folds = min(4, int(np.sum(labels == 0)))
    if folds >= 2:
        model = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=20260802)
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=20260802)
        probability = cross_val_predict(model, features, labels, cv=cv, method="predict_proba")[:, 1]
        report["multivariate_cv_auc"] = float(roc_auc_score(labels, probability))
        report["cv_folds"] = folds
    return report


def encoder_report(name: str, embeddings: np.ndarray, assets: dict) -> dict:
    reference_rows, gold_rows, negative_rows = make_metric_rows(embeddings, assets)
    all_rows = reference_rows + gold_rows + negative_rows
    search = threshold_search(all_rows)
    thresholds = search["recommended"]["thresholds"]
    for row in all_rows:
        row["predicted_target"] = all(float(row[key]) >= thresholds[key] for key in METRIC_NAMES)
    positive_rows = reference_rows + gold_rows
    distributions = {
        group: {metric: quantile_summary([float(row[metric]) for row in rows]) for metric in METRIC_NAMES + ("negative_max",)}
        for group, rows in (("target_reference_loo", reference_rows), ("gold", gold_rows), ("validation_negative", negative_rows))
    }
    separation = {
        metric: {
            "min_positive_minus_max_negative": float(min(row[metric] for row in positive_rows) - max(row[metric] for row in negative_rows)),
            "min_gold_minus_max_negative": float(min(row[metric] for row in gold_rows) - max(row[metric] for row in negative_rows)),
        }
        for metric in METRIC_NAMES
    }
    gate = search["recommended"]["full_stats"]
    gate["required_gold_target_count"] = min(45, len(gold_rows))
    gate["required_negative_rejected"] = len(negative_rows)
    gate["pass"] = bool(gate["gold_target_count"] >= gate["required_gold_target_count"] and gate["negative_rejected"] == gate["required_negative_rejected"])
    return {
        "encoder": name,
        "embedding_dim": int(embeddings.shape[1]),
        "auc": auc_report(all_rows),
        "distributions": distributions,
        "separation": separation,
        "threshold_search": search,
        "gate": gate,
        "gold_rows": gold_rows,
        "negative_rows": negative_rows,
        "target_reference_rows": reference_rows,
    }


def choose_encoder(results: dict[str, dict]) -> tuple[str, dict]:
    def rank(item: tuple[str, dict]):
        name, result = item
        gate = result["gate"]
        auc = float(result["auc"].get("multivariate_cv_auc", result["auc"]["margin_auc"]))
        return (int(gate["pass"]), gate["negative_rejected"], gate["gold_target_count"], auc, gate["balanced_accuracy"])

    return max(results.items(), key=rank)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("PYTHONUTF8", "1")
    OUT.mkdir(parents=True, exist_ok=True)
    assets = build_assets()
    print(
        f"assets old={len(assets['old_refs'])} confirmed_target={len(assets['confirmed_target'])} "
        f"gold={len(assets['gold'])} negative_refs={len(assets['negative_refs'])} "
        f"validation_negatives={len(assets['validation_negatives'])}",
        flush=True,
    )
    started = time.time()
    cam_embeddings = encode_campp(assets["all_paths"], args.resume)
    cam_report = encoder_report("CAM++", cam_embeddings, assets)
    del cam_embeddings
    gc.collect()
    eres_embeddings = encode_eres2net(assets["all_paths"], args.resume)
    eres_report = encoder_report("ERes2NetV2", eres_embeddings, assets)
    del eres_embeddings
    gc.collect()
    ecapa_embeddings = encode_ecapa(assets["all_paths"], args.resume)
    ecapa_report = encoder_report("SpeechBrain-ECAPA-VoxCeleb", ecapa_embeddings, assets)
    results = {
        "CAM++": cam_report,
        "ERes2NetV2": eres_report,
        "SpeechBrain-ECAPA-VoxCeleb": ecapa_report,
    }
    recommended_name, recommended = choose_encoder(results)
    gate_pass = bool(recommended["gate"]["pass"])
    thresholds = recommended["threshold_search"]["recommended"]["thresholds"]
    deviations = []
    if len(assets["confirmed_target"]) != 6:
        deviations.append(f"Plan expected 6 confirmed target clips; {len(assets['confirmed_target'])} exist on disk.")
    if len(assets["validation_negatives"]) != 7:
        deviations.append(f"Plan expected 7 validation negatives; {len(assets['validation_negatives'])} unique unsafe clips exist on disk.")
    report = {
        "version": "calibrate_classifier_v1",
        "created_unix": time.time(),
        "runtime_sec": time.time() - started,
        "policy": {
            "mixed_clips": "evaluation-only; never enrolled as pure negative embeddings",
            "reconstructed_audio": "ClearVoice/VASR/AudioSR excluded",
            "target_reference_loo": True,
            "negative_reference_loo": True,
            "gate": "at least 45/52 gold target and every observed validation negative rejected",
        },
        "asset_counts": {
            "old_reference": len(assets["old_refs"]),
            "confirmed_target": len(assets["confirmed_target"]),
            "target_reference_total": len(assets["target_refs"]),
            "gold": len(assets["gold"]),
            "pure_non_target": len(assets["pure_non_target"]),
            "mixed_validation_only": len(assets["mixed_validation"]),
            "explicit_hard_negative": len(assets["explicit_hard_negatives"]),
            "negative_reference_total": len(assets["negative_refs"]),
            "validation_negative_unique": len(assets["validation_negatives"]),
            "all_unique_audio": len(assets["all_paths"]),
        },
        "asset_deviations": deviations,
        "assets": {key: [str(path) for path in assets[key]] for key in ("confirmed_target", "pure_non_target", "mixed_validation", "explicit_hard_negatives", "target_refs", "negative_refs", "validation_negatives")},
        "encoders": results,
        "recommendation": {
            "encoder": recommended_name,
            "thresholds": thresholds,
            "gate_pass": gate_pass,
            "proceed_to_phase_2": gate_pass,
            "status": "calibrated" if gate_pass else "calibration_gate_failed",
            "phase2_environment": {
                "WTH_CENTER_MIN": thresholds["center"],
                "WTH_P25_MIN": thresholds["p25"],
                "WTH_MARGIN_MIN": thresholds["margin"],
            },
            "phase3_environment": {
                "SAW_CENTER_MIN": thresholds["center"],
                "SAW_MEDIAN_MIN": thresholds["median"],
                "SAW_P25_MIN": thresholds["p25"],
                "SAW_MARGIN_MIN": thresholds["margin"],
            },
        },
    }
    write_json(REPORT, report)
    print(json.dumps(report["recommendation"], ensure_ascii=False, indent=2), flush=True)
    print(f"saved={REPORT}", flush=True)


if __name__ == "__main__":
    main()
