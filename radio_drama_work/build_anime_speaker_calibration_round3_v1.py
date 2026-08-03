# -*- coding: utf-8 -*-
"""Build stable CAM++-window negative anchors for episode 12."""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_resource_policy import apply_cpu_friendly_policy


RESOURCE_POLICY = apply_cpu_friendly_policy(2)

import numpy as np
import soundfile as sf

import build_human_calibration_v1 as page_base
import full_speaker_diarize_new_v1 as audio_base


SOURCE_ID = "majo_ep12"
INITIAL = ROOT / "anime_speaker_calibration_v1"
ROUND2 = ROOT / "anime_speaker_calibration_round2_v1"
HUMAN_V2 = ROOT / "human_anime_speaker_calibration_v2"
CAM_CACHE = (
    ROOT
    / "anime_mining_v1"
    / "source_local_campp"
    / "embedding_cache"
    / "majo_ep12_vocals.npz"
)
OUT = ROOT / "anime_speaker_calibration_round3_v1"
AUDIO_DIR = OUT / "audio"
OUTPUT_SR = 32000
CLUSTER_COUNT = 4
SAMPLES_PER_CLUSTER = 3
RANDOM_SEED = 202608022244


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8"
    )


def norm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def overlaps(start: float, end: float, row: dict, padding: float = 0.0) -> bool:
    return max(start, float(row["start"]) - padding) < min(
        end, float(row["end"]) + padding
    )


def micro_fade(audio: np.ndarray) -> np.ndarray:
    output = np.asarray(audio, dtype=np.float32).copy()
    count = min(len(output) // 4, max(1, round(OUTPUT_SR * 0.002)))
    if count:
        ramp = np.linspace(0.0, 1.0, count, dtype=np.float32)
        output[:count] *= ramp
        output[-count:] *= ramp[::-1]
    return output


def main() -> None:
    initial_report = json.loads(
        (INITIAL / "report.json").read_text(encoding="utf-8")
    )
    initial_audit = json.loads(
        (INITIAL / "audit_manifest.json").read_text(encoding="utf-8")
    )
    round2_audit = json.loads(
        (ROUND2 / "audit_manifest.json").read_text(encoding="utf-8")
    )
    human = json.loads(
        (HUMAN_V2 / "labels_resolved.json").read_text(encoding="utf-8")
    )["samples"]
    human_report = json.loads((HUMAN_V2 / "report.json").read_text(encoding="utf-8"))
    if human_report["missing_pure_negative_sources"] != [SOURCE_ID]:
        raise RuntimeError(
            "Round3 is only valid when episode 12 is the sole missing negative source"
        )

    source_report = next(
        row for row in initial_report["sources"] if row["source_id"] == SOURCE_ID
    )
    k16 = next(row for row in source_report["k_runs"] if int(row["k"]) == 16)
    labels = np.asarray(k16["labels"], dtype=np.int32)
    cluster_stats = {int(row["cluster"]): row for row in k16["clusters"]}
    model_target_clusters = {int(value) for value in k16["target_clusters"]}

    audit_by_id = {}
    for row in initial_audit["samples"] + round2_audit["samples"]:
        audit_by_id[str(row["blind_id"])] = row
    reviewed = [row for row in human if str(row["source_id"]) == SOURCE_ID]
    human_target_clusters = set()
    human_mixed_clusters = set()
    for row in reviewed:
        audit_row = audit_by_id[str(row["blind_id"])]
        cluster = int(audit_row["primary_cluster"])
        if row["final_class"] == "target":
            human_target_clusters.add(cluster)
        elif row["final_class"] == "mixed":
            human_mixed_clusters.add(cluster)

    blocked_clusters = (
        model_target_clusters | human_target_clusters | human_mixed_clusters
    )
    available_clusters = [
        cluster for cluster in sorted(cluster_stats) if cluster not in blocked_clusters
    ]
    available_clusters.sort(
        key=lambda cluster: (
            float(cluster_stats[cluster]["direct_top_half_fraction"]),
            float(cluster_stats[cluster]["target_top4"]),
            float(cluster_stats[cluster]["margin"]),
            cluster,
        )
    )
    selected_clusters = available_clusters[:CLUSTER_COUNT]
    if len(selected_clusters) != CLUSTER_COUNT:
        raise RuntimeError(f"Only {len(selected_clusters)} safe clusters available")

    data = np.load(str(CAM_CACHE), allow_pickle=True)
    specs = np.asarray(data["specs"], dtype=np.float32)
    embeddings = norm(np.asarray(data["embeddings"], dtype=np.float32))
    if len(labels) != len(specs) or len(embeddings) != len(specs):
        raise RuntimeError("Episode 12 CAM++ cache and K=16 labels disagree")

    candidates_by_cluster: dict[int, list[dict]] = defaultdict(list)
    for cluster in selected_clusters:
        member_indices = np.flatnonzero(labels == cluster)
        center = norm(embeddings[member_indices].mean(axis=0, keepdims=True))[0]
        for index in member_indices.tolist():
            start, end = map(float, specs[index])
            if end - start < 1.45:
                continue
            end = min(end, start + 1.5)
            if any(overlaps(start, end, row, 0.45) for row in reviewed):
                continue
            local_left = max(0, index - 2)
            local_right = min(len(labels), index + 3)
            stability = int(np.sum(labels[local_left:local_right] == cluster))
            candidates_by_cluster[cluster].append(
                {
                    "source_id": SOURCE_ID,
                    "source": str(Path(source_report["source"]).resolve()),
                    "predicted_tier": "stable_unreviewed_competitor_window",
                    "primary_cluster": cluster,
                    "cam_window_index": index,
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "local_same_cluster_count": stability,
                    "center_similarity": float(embeddings[index] @ center),
                    "cluster_stats": cluster_stats[cluster],
                    "text": "",
                }
            )

    selected = []
    per_cluster = Counter()
    for cluster in selected_clusters:
        rows = sorted(
            candidates_by_cluster[cluster],
            key=lambda row: (
                -int(row["local_same_cluster_count"]),
                -float(row["center_similarity"]),
                float(row["start"]),
            ),
        )
        for minimum_separation in (8.0, 3.0, 0.75):
            for row in rows:
                if row in selected:
                    continue
                if any(
                    abs(
                        0.5 * (float(row["start"]) + float(row["end"]))
                        - 0.5 * (float(old["start"]) + float(old["end"]))
                    )
                    < minimum_separation
                    for old in selected
                ):
                    continue
                selected.append(row)
                per_cluster[cluster] += 1
                if per_cluster[cluster] >= SAMPLES_PER_CLUSTER:
                    break
            if per_cluster[cluster] >= SAMPLES_PER_CLUSTER:
                break
        if per_cluster[cluster] != SAMPLES_PER_CLUSTER:
            raise RuntimeError(
                f"Cluster {cluster}: only selected {per_cluster[cluster]} stable windows"
            )

    random.Random(RANDOM_SEED).shuffle(selected)
    source_audio, source_rate = audio_base.read_mono(Path(source_report["source"]))
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(selected, 1):
        blind_id = f"T{index:03d}"
        clip = audio_base.segment(
            source_audio, source_rate, float(row["start"]), float(row["end"])
        )
        clip = micro_fade(audio_base.resample_1d(clip, source_rate, OUTPUT_SR))
        output = AUDIO_DIR / f"{blind_id}.wav"
        sf.write(str(output), clip, OUTPUT_SR, subtype="PCM_16")
        info = sf.info(str(output))
        if (
            info.samplerate != OUTPUT_SR
            or info.channels != 1
            or info.subtype != "PCM_16"
            or not 1.45 <= info.duration <= 1.51
        ):
            raise RuntimeError(f"Invalid round3 window: {output} {info}")
        row["blind_id"] = blind_id
        row["audio"] = str(output.resolve())
        row["audio_rel"] = f"audio/{output.name}"
        row["output_duration"] = float(info.duration)
        row["wave"] = page_base.waveform(np.asarray(clip, dtype=np.float32))

    audit = {
        "version": "anime_speaker_calibration_round3_v1_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "purpose": "episode-12 pure-negative anchor recovery",
            "selection": "stable 1.5s CAM++ K=16 windows",
            "excluded_model_target_clusters": sorted(model_target_clusters),
            "excluded_human_target_clusters": sorted(human_target_clusters),
            "excluded_human_mixed_clusters": sorted(human_mixed_clusters),
            "selected_clusters": selected_clusters,
            "identity_only": True,
            "local_resources": RESOURCE_POLICY,
        },
        "sample_count": len(selected),
        "samples": selected,
    }
    report = {
        "version": "anime_speaker_calibration_round3_v1_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_id": SOURCE_ID,
        "selected_clusters": selected_clusters,
        "sample_count": len(selected),
        "duration_sec": sum(float(row["output_duration"]) for row in selected),
        "per_cluster": dict(sorted(per_cluster.items())),
        "training_started": False,
        "existing_training_lists_modified": False,
        "score_90_claimed": False,
    }
    write_json(OUT / "audit_manifest.json", audit)
    write_json(OUT / "private_manifest.json", audit)
    write_json(OUT / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
