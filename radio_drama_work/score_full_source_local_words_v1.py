# -*- coding: utf-8 -*-
"""GPU word-level scoring against the 12 source-local CAM++ models.

Every ASR word receives a fresh 1.2-second CAM++ embedding.  It is compared
only with clusters and human anchors from the same drama, avoiding the global
speaker-threshold failure seen in earlier passes.  This stage scores words; it
does not export training audio.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_resource_policy import apply_cpu_friendly_policy


RESOURCE_POLICY = apply_cpu_friendly_policy(2)

import numpy as np
import torch
from funasr import AutoModel

import full_speaker_diarize_new_v1 as base
import source_local_cluster_anchor_v2 as cluster_base


MANIFEST = ROOT / "full_track_manifest_v1.json"
WORDS_DIR = ROOT / "word_target_all_v1" / "source_reports"
MODEL_DIR = ROOT / "source_local_cluster_anchor_v4" / "models"
MODEL_REPORT = ROOT / "source_local_cluster_anchor_v4" / "report.json"
CLUSTER_LABELS = ROOT / "source_local_cluster_anchor_v4" / "cluster_labels_resolved.json"
OUT = ROOT / "full_source_local_word_scores_v1"
EMBED_CACHE = OUT / "embedding_cache"
SOURCE_REPORTS = OUT / "source_reports"
REPORT = OUT / "report.json"

CAM_SR = 16000
WORD_WINDOW_SEC = 1.20
WORD_BATCH = 256
GUARDED_SOURCES = {
    "113814583-1-160",      # no pure same-source negative anchor
    "28055832533-1-192",   # known seamless mixed transition survived clip aggregation
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8"
    )


def fingerprint(paths: list[Path], extra: str) -> str:
    parts = [extra]
    for path in paths:
        stat = path.stat()
        parts.append(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}")
    return hashlib.sha1("/n".join(parts).encode("utf-8")).hexdigest()


def norm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def valid_words(source_id: str) -> tuple[list[dict], Path]:
    path = WORDS_DIR / f"{source_id}.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    # New source domains may persist the raw ASR list as ``words`` while the
    # original radio pipeline used ``word_rows``. Both contain the same
    # validated word-timestamp schema at this stage.
    rows = document.get("word_rows", document.get("words"))
    if not isinstance(rows, list):
        raise ValueError(f"{source_id}: ASR cache has neither word_rows nor words")
    previous = -1.0
    for index, row in enumerate(rows):
        text = str(row.get("text", ""))
        start = float(row["start"])
        end = float(row["end"])
        if not text.strip() or "/ufffd" in text:
            raise ValueError(f"{source_id}: invalid ASR text at word {index}")
        if start < previous - 0.02 or end <= start:
            raise ValueError(f"{source_id}: non-monotonic ASR time at word {index}")
        previous = start
    return rows, path


def encode_words(
    cam: AutoModel,
    source: Path,
    words: list[dict],
    cache: Path,
    cache_fingerprint: str,
    resume: bool,
) -> np.ndarray:
    if resume and cache.is_file():
        data = np.load(str(cache), allow_pickle=True)
        cached_fingerprint = str(data["fingerprint"].tolist())
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        if cached_fingerprint == cache_fingerprint and embeddings.shape == (len(words), 192):
            print(f"  [word embeddings] resume {len(words)}", flush=True)
            return norm(embeddings)

    audio, sample_rate = base.read_mono(source)
    audio16 = base.resample_1d(audio, sample_rate, CAM_SR)
    output: list[np.ndarray] = []
    for offset in range(0, len(words), WORD_BATCH):
        batch_rows = words[offset : offset + WORD_BATCH]
        waves = [
            base.segment(
                audio16,
                CAM_SR,
                float(row["start"]),
                float(row["end"]),
                WORD_WINDOW_SEC,
            )
            for row in batch_rows
        ]
        output.append(base.cam_embed_batch(cam, waves).numpy().astype(np.float32))
        if offset == 0 or (offset // WORD_BATCH + 1) % 10 == 0:
            print(
                f"  [word embeddings] {min(offset + len(batch_rows), len(words))}/{len(words)}",
                flush=True,
            )
    embeddings = norm(np.concatenate(output, axis=0))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(cache),
        embeddings=embeddings,
        fingerprint=np.asarray(cache_fingerprint),
    )
    return embeddings


def mapped_indices(specs: np.ndarray, rows: list[dict]) -> np.ndarray:
    arrays = [
        cluster_base.clip_indices(specs, float(row["start"]), float(row["end"]))
        for row in rows
    ]
    arrays = [array for array in arrays if len(array)]
    return np.unique(np.concatenate(arrays)) if arrays else np.asarray([], dtype=np.int64)


def context_fraction(specs: np.ndarray, labels: np.ndarray, targets: set[int], words: list[dict]) -> np.ndarray:
    result = np.zeros(len(words), dtype=np.float32)
    spec_centers = specs.mean(axis=1)
    for index, word in enumerate(words):
        middle = 0.5 * (float(word["start"]) + float(word["end"]))
        candidates = np.flatnonzero((specs[:, 0] <= middle) & (specs[:, 1] >= middle))
        if not len(candidates):
            nearest = int(np.argmin(np.abs(spec_centers - middle)))
            if abs(float(spec_centers[nearest]) - middle) <= 0.80:
                candidates = np.asarray([nearest], dtype=np.int64)
        if len(candidates):
            result[index] = float(
                np.mean([int(labels[item]) in targets for item in candidates])
            )
    return result


def anchor_similarity(values: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    if not len(anchors):
        return np.full(len(values), -1.0, dtype=np.float32)
    similarities = values @ anchors.T
    top_count = min(3, similarities.shape[1])
    top = np.partition(similarities, similarities.shape[1] - top_count, axis=1)[
        :, -top_count:
    ]
    return top.mean(axis=1).astype(np.float32)


def evaluate_human(
    source_id: str,
    words: list[dict],
    scored: list[dict],
    label_rows: list[dict],
) -> list[dict]:
    evaluations = []
    centers = np.asarray(
        [0.5 * (float(row["start"]) + float(row["end"])) for row in words]
    )
    for label in label_rows:
        indices = np.flatnonzero(
            (centers >= float(label["start"])) & (centers <= float(label["end"]))
        )
        if not len(indices):
            continue
        evaluations.append(
            {
                "human_key": label.get("human_key", label["blind_id"]),
                "blind_id": label["blind_id"],
                "final_class": label["final_class"],
                "word_count": int(len(indices)),
                "direct_target_fraction": float(
                    np.mean([scored[index]["direct_target"] for index in indices])
                ),
                "strict_pass_fraction": float(
                    np.mean([scored[index]["strict_speaker_pass"] for index in indices])
                ),
                "median_cluster_margin": float(
                    np.median([scored[index]["cluster_margin"] for index in indices])
                ),
                "minimum_cluster_margin": float(
                    np.min([scored[index]["cluster_margin"] for index in indices])
                ),
            }
        )
    return evaluations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    EMBED_CACHE.mkdir(parents=True, exist_ok=True)
    SOURCE_REPORTS.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tracks = manifest["tracks"]
    model_report = json.loads(MODEL_REPORT.read_text(encoding="utf-8"))
    model_configs = {row["source_id"]: row for row in model_report["sources"]}
    labels = json.loads(CLUSTER_LABELS.read_text(encoding="utf-8"))["samples"]
    labels_by_source: dict[str, list[dict]] = defaultdict(list)
    for row in labels:
        labels_by_source[str(row["source_id"])].append(row)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cam = AutoModel(
        model=str(base.CAM_MODEL),
        device=device,
        check_latest=False,
        disable_pbar=True,
    )
    source_summaries = []
    for track in tracks:
        source_started = time.time()
        source_id = str(track["id"])
        source = Path(track["vocals"])
        model_path = MODEL_DIR / f"{source_id}.npz"
        words, words_path = valid_words(source_id)
        cache_fingerprint = fingerprint(
            [source, model_path, words_path],
            f"word_window={WORD_WINDOW_SEC}|cam={base.CAM_MODEL}",
        )
        source_report_path = SOURCE_REPORTS / f"{source_id}.json"
        if args.resume and source_report_path.is_file():
            prior = json.loads(source_report_path.read_text(encoding="utf-8"))
            if prior.get("fingerprint") == cache_fingerprint:
                source_summaries.append(prior["summary"])
                print(f"[{source_id}] resume report", flush=True)
                continue

        print(f"[{source_id}] score {len(words)} words", flush=True)
        model = np.load(str(model_path), allow_pickle=True)
        specs = np.asarray(model["specs"], dtype=np.float32)
        source_embeddings = norm(model["embeddings"])
        chunk_labels = np.asarray(model["labels"], dtype=np.int64)
        centers = norm(model["centers"])
        targets = set(int(value) for value in model["target_clusters"].tolist())
        target_center_indices = np.asarray(sorted(targets), dtype=np.int64)
        non_target_center_indices = np.asarray(
            [index for index in range(len(centers)) if index not in targets],
            dtype=np.int64,
        )
        if not len(target_center_indices) or not len(non_target_center_indices):
            raise RuntimeError(f"{source_id}: model lacks target/non-target centers")

        word_embeddings = encode_words(
            cam,
            source,
            words,
            EMBED_CACHE / f"{source_id}.npz",
            cache_fingerprint,
            args.resume,
        )
        similarities = word_embeddings @ centers.T
        direct_labels = similarities.argmax(axis=1)
        target_sim = similarities[:, target_center_indices].max(axis=1)
        non_target_sim = similarities[:, non_target_center_indices].max(axis=1)
        margins = target_sim - non_target_sim
        context = context_fraction(specs, chunk_labels, targets, words)

        source_label_rows = labels_by_source[source_id]
        clean_targets = [row for row in source_label_rows if row["final_class"] == "target"]
        pure_negatives = [
            row for row in source_label_rows if row["final_class"] == "pure_non_target"
        ]
        positive_indices = mapped_indices(specs, clean_targets)
        negative_indices = mapped_indices(specs, pure_negatives)
        positive_similarity = anchor_similarity(
            word_embeddings, source_embeddings[positive_indices]
        )
        negative_similarity = anchor_similarity(
            word_embeddings, source_embeddings[negative_indices]
        )
        anchor_margin = np.where(
            negative_similarity >= -0.5,
            positive_similarity - negative_similarity,
            positive_similarity - 0.55,
        )

        selected = model_configs[source_id]["selected"]
        guarded = source_id in GUARDED_SOURCES
        context_threshold = max(
            0.50, float(selected["target_clip_fraction"]) - 0.15
        )
        if guarded:
            context_threshold = max(context_threshold, 0.75)
        margin_threshold = 0.035 if guarded else 0.015
        similarity_threshold = 0.60 if guarded else 0.57
        anchor_margin_threshold = -0.015 if guarded else -0.045

        scored = []
        counters = Counter()
        for index, word in enumerate(words):
            direct_target = int(direct_labels[index]) in targets
            strict = bool(
                direct_target
                and float(context[index]) >= context_threshold
                and float(target_sim[index]) >= similarity_threshold
                and float(margins[index]) >= margin_threshold
                and float(anchor_margin[index]) >= anchor_margin_threshold
            )
            row = {
                **word,
                "word_index": index,
                "direct_cluster": int(direct_labels[index]),
                "direct_target": bool(direct_target),
                "target_center_similarity": float(target_sim[index]),
                "non_target_center_similarity": float(non_target_sim[index]),
                "cluster_margin": float(margins[index]),
                "context_target_fraction": float(context[index]),
                "positive_anchor_similarity": float(positive_similarity[index]),
                "negative_anchor_similarity": float(negative_similarity[index]),
                "anchor_margin": float(anchor_margin[index]),
                "strict_speaker_pass": strict,
            }
            scored.append(row)
            counters["strict_pass" if strict else "strict_reject"] += 1
            if direct_target:
                counters["direct_target"] += 1

        evaluations = evaluate_human(
            source_id, words, scored, source_label_rows
        )
        class_evaluation = {}
        for class_name in sorted({row["final_class"] for row in evaluations}):
            subset = [row for row in evaluations if row["final_class"] == class_name]
            class_evaluation[class_name] = {
                "clip_count": len(subset),
                "mean_strict_pass_fraction": float(
                    np.mean([row["strict_pass_fraction"] for row in subset])
                ),
                "fully_rejected_count": sum(
                    row["strict_pass_fraction"] == 0.0 for row in subset
                ),
            }

        summary = {
            "source_id": source_id,
            "source": str(source),
            "word_count": len(words),
            "direct_target_word_count": counters["direct_target"],
            "strict_pass_word_count": counters["strict_pass"],
            "guarded": guarded,
            "thresholds": {
                "context_target_fraction": context_threshold,
                "cluster_margin": margin_threshold,
                "target_center_similarity": similarity_threshold,
                "anchor_margin": anchor_margin_threshold,
            },
            "positive_anchor_window_count": int(len(positive_indices)),
            "negative_anchor_window_count": int(len(negative_indices)),
            "human_class_evaluation": class_evaluation,
            "elapsed_sec": time.time() - source_started,
        }
        source_report = {
            "version": "full_source_local_word_scores_v1_source",
            "fingerprint": cache_fingerprint,
            "summary": summary,
            "human_evaluations": evaluations,
            "word_rows": scored,
        }
        write_json(source_report_path, source_report)
        source_summaries.append(summary)
        print(
            f"  [done] strict={counters['strict_pass']}/{len(words)} "
            f"direct={counters['direct_target']} elapsed={summary['elapsed_sec']:.1f}s",
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del cam
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    report = {
        "version": "full_source_local_word_scores_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime_sec": time.time() - started,
        "device": device,
        "local_resources": RESOURCE_POLICY,
        "policy": {
            "embedding": "fresh 1.2-second CAM++ window for every ASR word",
            "speaker_domain": "source-local clusters and source-local human anchors only",
            "guarded_sources": sorted(GUARDED_SOURCES),
            "status": "word scores only; no training audio exported",
        },
        "coverage": {"expected_sources": 12, "completed_sources": len(source_summaries)},
        "total_word_count": sum(row["word_count"] for row in source_summaries),
        "strict_pass_word_count": sum(
            row["strict_pass_word_count"] for row in source_summaries
        ),
        "sources": source_summaries,
    }
    if report["coverage"]["completed_sources"] != 12:
        raise RuntimeError(f"Incomplete source coverage: {report['coverage']}")
    write_json(REPORT, report)
    print(
        json.dumps(
            {
                "coverage": report["coverage"],
                "total_words": report["total_word_count"],
                "strict_pass_words": report["strict_pass_word_count"],
                "report": str(REPORT),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
