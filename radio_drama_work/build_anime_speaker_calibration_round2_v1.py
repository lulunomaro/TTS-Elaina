# -*- coding: utf-8 -*-
"""Build a targeted negative-anchor review for episodes missing pure negatives."""
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


INITIAL = ROOT / "anime_speaker_calibration_v1"
HUMAN_V1 = ROOT / "human_anime_speaker_calibration_v1"
OUT = ROOT / "anime_speaker_calibration_round2_v1"
AUDIO_DIR = OUT / "audio"
OUTPUT_SR = 32000
SAMPLES_PER_SOURCE = 5
RANDOM_SEED = 202608022220


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8"
    )


def overlaps(a: dict, b: dict, padding: float = 0.0) -> bool:
    return max(float(a["start"]), float(b["start"])) < (
        min(float(a["end"]), float(b["end"])) + padding
    )


def micro_fade(audio: np.ndarray) -> np.ndarray:
    output = np.asarray(audio, dtype=np.float32).copy()
    count = min(len(output) // 4, max(1, round(OUTPUT_SR * 0.002)))
    if count:
        ramp = np.linspace(0.0, 1.0, count, dtype=np.float32)
        output[:count] *= ramp
        output[-count:] *= ramp[::-1]
    return output


def choose_source_rows(
    source_report: dict,
    reviewed: list[dict],
    target_cluster_ids: set[int],
) -> list[dict]:
    pool = []
    for row in source_report["competitor_candidate_pool"]:
        cluster = int(row["primary_cluster"])
        if cluster in target_cluster_ids:
            continue
        if any(overlaps(row, old, 0.15) for old in reviewed):
            continue
        if not 1.5 <= float(row["duration"]) <= 5.0:
            continue
        pool.append(dict(row))
    if not pool:
        raise RuntimeError(f"{source_report['source_id']}: no unreviewed competitor rows")

    by_cluster: dict[int, list[dict]] = defaultdict(list)
    for row in pool:
        by_cluster[int(row["primary_cluster"])].append(row)
    for rows in by_cluster.values():
        rows.sort(
            key=lambda row: (
                not bool(row["eres2net"]["hard_negative_warning"]),
                float(row["mean_target_top4"]),
                -int(row["word_count"]),
                -float(row["duration"]),
            )
        )
    cluster_order = sorted(
        by_cluster,
        key=lambda cluster: (
            min(float(row["mean_target_top4"]) for row in by_cluster[cluster]),
            -max(int(row["word_count"]) for row in by_cluster[cluster]),
            cluster,
        ),
    )

    selected = []
    for cluster in cluster_order:
        candidate = next(
            (
                row
                for row in by_cluster[cluster]
                if not any(overlaps(row, old, 0.75) for old in selected)
            ),
            None,
        )
        if candidate is not None:
            selected.append(candidate)
        if len(selected) >= SAMPLES_PER_SOURCE:
            break
    if len(selected) < SAMPLES_PER_SOURCE:
        remaining = sorted(
            pool,
            key=lambda row: (
                not bool(row["eres2net"]["hard_negative_warning"]),
                float(row["mean_target_top4"]),
                -int(row["word_count"]),
            ),
        )
        for row in remaining:
            if row in selected or any(overlaps(row, old, 0.75) for old in selected):
                continue
            selected.append(row)
            if len(selected) >= SAMPLES_PER_SOURCE:
                break
    if len(selected) != SAMPLES_PER_SOURCE:
        raise RuntimeError(
            f"{source_report['source_id']}: expected {SAMPLES_PER_SOURCE}, got {len(selected)}"
        )
    return selected


def main() -> None:
    report = json.loads((INITIAL / "report.json").read_text(encoding="utf-8"))
    audit = json.loads((INITIAL / "audit_manifest.json").read_text(encoding="utf-8"))
    human = json.loads((HUMAN_V1 / "labels_resolved.json").read_text(encoding="utf-8"))
    human_report = json.loads((HUMAN_V1 / "report.json").read_text(encoding="utf-8"))
    missing = list(human_report["missing_pure_negative_sources"])
    if not missing:
        raise RuntimeError("No source needs targeted negative calibration")
    if human_report["missing_target_sources"]:
        raise RuntimeError("This round only resolves pure-negative anchors")

    source_reports = {str(row["source_id"]): row for row in report["sources"]}
    initial_by_id = {str(row["blind_id"]): row for row in audit["samples"]}
    human_by_id = {str(row["blind_id"]): row for row in human["samples"]}
    reviewed_by_source: dict[str, list[dict]] = defaultdict(list)
    target_clusters: dict[str, set[int]] = defaultdict(set)
    for blind_id, row in initial_by_id.items():
        source_id = str(row["source_id"])
        reviewed_by_source[source_id].append(row)
        if human_by_id[blind_id]["final_class"] == "target":
            target_clusters[source_id].add(int(row["primary_cluster"]))

    selected = []
    for source_id in missing:
        rows = choose_source_rows(
            source_reports[source_id],
            reviewed_by_source[source_id],
            target_clusters[source_id],
        )
        for row in rows:
            row["source"] = str(Path(source_reports[source_id]["source"]).resolve())
            row["round2_reason"] = "missing_episode_local_pure_negative_anchor"
            row["excluded_human_target_clusters"] = sorted(target_clusters[source_id])
            selected.append(row)

    random.Random(RANDOM_SEED).shuffle(selected)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    audio_cache: dict[str, tuple[np.ndarray, int]] = {}
    for index, row in enumerate(selected, 1):
        source_id = str(row["source_id"])
        if source_id not in audio_cache:
            audio_cache[source_id] = audio_base.read_mono(Path(row["source"]))
        audio, rate = audio_cache[source_id]
        clip = audio_base.segment(audio, rate, float(row["start"]), float(row["end"]))
        clip = micro_fade(audio_base.resample_1d(clip, rate, OUTPUT_SR))
        blind_id = f"R{index:03d}"
        output = AUDIO_DIR / f"{blind_id}.wav"
        sf.write(str(output), clip, OUTPUT_SR, subtype="PCM_16")
        info = sf.info(str(output))
        if (
            info.samplerate != OUTPUT_SR
            or info.channels != 1
            or info.subtype != "PCM_16"
            or not 1.5 <= info.duration <= 5.01
        ):
            raise RuntimeError(f"Invalid round2 calibration audio: {output} {info}")
        row["blind_id"] = blind_id
        row["audio"] = str(output.resolve())
        row["audio_rel"] = f"audio/{output.name}"
        row["output_duration"] = float(info.duration)
        row["wave"] = page_base.waveform(np.asarray(clip, dtype=np.float32))

    output_audit = {
        "version": "anime_speaker_calibration_round2_v1_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "purpose": "targeted identity-only pure-negative anchor recovery",
            "sources": missing,
            "selection": "unreviewed competitor pools, excluding every human-target cluster",
            "samples_per_source": SAMPLES_PER_SOURCE,
            "boundary": "identity calibration only; edge quality ignored",
            "local_resources": RESOURCE_POLICY,
        },
        "sample_count": len(selected),
        "samples": selected,
    }
    output_report = {
        "version": "anime_speaker_calibration_round2_v1_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_count": len(missing),
        "sources": missing,
        "sample_count": len(selected),
        "duration_sec": sum(float(row["output_duration"]) for row in selected),
        "per_source": dict(Counter(str(row["source_id"]) for row in selected)),
        "excluded_human_target_clusters": {
            source_id: sorted(target_clusters[source_id]) for source_id in missing
        },
        "training_started": False,
        "existing_training_lists_modified": False,
        "score_90_claimed": False,
    }
    write_json(OUT / "audit_manifest.json", output_audit)
    write_json(OUT / "private_manifest.json", output_audit)
    write_json(OUT / "report.json", output_report)
    print(json.dumps(output_report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
