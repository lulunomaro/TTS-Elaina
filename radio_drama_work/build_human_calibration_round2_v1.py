# -*- coding: utf-8 -*-
"""Build 48 fresh source-stratified active-learning probes for round two."""
from __future__ import annotations

import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

BOOTSTRAP_DIR = Path(__file__).resolve().parent
if str(BOOTSTRAP_DIR) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_DIR))
from local_resource_policy import apply_cpu_friendly_policy

RESOURCE_POLICY = apply_cpu_friendly_policy(2)

import numpy as np
import soundfile as sf

import build_human_calibration_v1 as v1


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "human_calibration_round2_v1"
AUDIO_DIR = OUT / "audio"
MANIFEST_PATH = ROOT / "full_track_manifest_v1.json"
WORD_REPORT_DIR = ROOT / "word_target_all_v1" / "source_reports"
ROUND1_AUDIT = ROOT / "human_calibration_v1" / "audit_manifest.json"
RANDOM_SEED = 20260803
MIN_SEPARATION_SEC = 8.0
QUOTAS = {"model_disagreement": 2, "low_target_clear_speech": 1, "speaker_transition": 1}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8")


def prior_masks(manifest: dict, source_ids: list[str]) -> dict[str, list[tuple[float, float]]]:
    masks: dict[str, list[tuple[float, float]]] = {source_id: [] for source_id in source_ids}
    for item in manifest.get("hard_exclusions", []):
        masks[str(item["source_id"])].append((float(item["start"]) - 0.6, float(item["end"]) + 0.6))
    audit = json.loads(ROUND1_AUDIT.read_text(encoding="utf-8"))
    for row in audit["samples"]:
        if row.get("kind") == "probe":
            source_id, start, end = str(row["source_id"]), float(row["start"]), float(row["end"])
        else:
            interval = row.get("source_interval")
            if not interval:
                continue
            source_id, start, end = str(interval[0]), float(interval[1]), float(interval[2])
        if source_id in masks:
            masks[source_id].append((start - 3.0, end + 3.0))
    return masks


def source_profiles(source: dict) -> tuple[list[dict], float, dict[str, list[tuple[int, int | None]]], np.ndarray]:
    report = json.loads((WORD_REPORT_DIR / f"{source['id']}.json").read_text(encoding="utf-8"))
    rows: list[dict] = report["word_rows"]
    duration = float(report["duration_sec"])
    keys = ("direct_center", "direct_p25", "target_negative_margin", "frame_target", "cluster_vote")
    arrays = {key: np.asarray([v1.finite(row, key) for row in rows], dtype=np.float64) for key in keys}
    ranks = {key: v1.percentile_rank(values) for key, values in arrays.items()}
    score = (
        0.30 * ranks["direct_center"]
        + 0.22 * ranks["direct_p25"]
        + 0.18 * ranks["target_negative_margin"]
        + 0.18 * ranks["frame_target"]
        + 0.12 * ranks["cluster_vote"]
    )
    direct_profile = 0.60 * ranks["direct_center"] + 0.40 * ranks["direct_p25"]
    disagreement = np.abs(direct_profile - (0.65 * ranks["frame_target"] + 0.35 * ranks["cluster_vote"]))
    valid = []
    for row in rows:
        text = str(row.get("text", "")).strip()
        word_duration = v1.finite(row, "end") - v1.finite(row, "start")
        valid.append(
            bool(text)
            and bool(v1.JAPANESE_RE.search(text))
            and v1.finite(row, "probability", 0.0) >= 0.72
            and 0.08 <= word_duration <= 1.35
            and v1.finite(row, "start") >= 0.8
            and v1.finite(row, "end") <= duration - 0.8
        )
    valid_mask = np.asarray(valid, dtype=bool)
    disagreement_order = [
        int(idx)
        for idx in np.argsort(-(disagreement + 0.20 * np.maximum(direct_profile, ranks["frame_target"])))
        if valid_mask[idx] and max(direct_profile[idx], ranks["frame_target"][idx]) >= 0.58
    ]
    low_speech = [
        int(idx)
        for idx in np.argsort(score)
        if valid_mask[idx] and v1.finite(rows[idx], "probability") >= 0.90
    ]
    boundary_pairs: list[tuple[float, int]] = []
    for idx in range(len(rows) - 1):
        if not (valid_mask[idx] and valid_mask[idx + 1]):
            continue
        gap = v1.finite(rows[idx + 1], "start") - v1.finite(rows[idx], "end")
        if gap > 0.25:
            continue
        veto_change = 0.45 if bool(rows[idx].get("hard_veto")) != bool(rows[idx + 1].get("hard_veto")) else 0.0
        transition = abs(float(score[idx]) - float(score[idx + 1])) + veto_change
        if transition >= 0.22:
            boundary_pairs.append((transition, idx))
    boundary_pairs.sort(reverse=True)
    profiles = {
        "model_disagreement": [(idx, None) for idx in disagreement_order],
        "low_target_clear_speech": [(idx, None) for idx in low_speech],
        "speaker_transition": [(idx, idx + 1) for _, idx in boundary_pairs],
        "fallback_high_rank": [(int(idx), None) for idx in np.argsort(-score) if valid_mask[idx]],
    }
    return rows, duration, profiles, score


def make_item(source: dict, rows: list[dict], score: np.ndarray, profile: str, index: int, second: int | None, duration: float) -> dict:
    start, end, included = v1.build_interval(rows, index, second, duration)
    return {
        "kind": "probe",
        "source_id": source["id"],
        "source": str(Path(source["vocals"]).resolve()),
        "start": start,
        "end": end,
        "duration": end - start,
        "selection_profile": profile,
        "anchor_index": index,
        "included_word_indices": included,
        "private_text": "".join(str(rows[item].get("text", "")) for item in included),
        "private_scores": {
            "combined_rank": float(score[index]),
            "direct_center": v1.finite(rows[index], "direct_center"),
            "direct_p25": v1.finite(rows[index], "direct_p25"),
            "target_negative_margin": v1.finite(rows[index], "target_negative_margin"),
            "frame_target": v1.finite(rows[index], "frame_target"),
            "cluster_vote": v1.finite(rows[index], "cluster_vote"),
            "hard_veto": bool(rows[index].get("hard_veto")),
        },
    }


def select_source(source: dict, masks: list[tuple[float, float]]) -> list[dict]:
    rows, duration, profiles, score = source_profiles(source)
    selected: list[dict] = []
    centers: list[float] = []
    used_anchors: set[int] = set()

    def try_candidates(profile: str, candidates: list[tuple[int, int | None]], needed: int) -> None:
        for index, second in candidates:
            if len([row for row in selected if row["requested_profile"] == profile]) >= needed:
                break
            if index in used_anchors:
                continue
            item = make_item(source, rows, score, profile, index, second, duration)
            center = (item["start"] + item["end"]) / 2.0
            if v1.overlaps(item["start"], item["end"], masks):
                continue
            if any(abs(center - old) < MIN_SEPARATION_SEC for old in centers):
                continue
            item["requested_profile"] = profile
            selected.append(item)
            centers.append(center)
            used_anchors.add(index)

    for profile, needed in QUOTAS.items():
        try_candidates(profile, profiles[profile], needed)
    if len(selected) < 4:
        for index, second in profiles["fallback_high_rank"]:
            if len(selected) >= 4:
                break
            if index in used_anchors:
                continue
            item = make_item(source, rows, score, "fallback_high_rank", index, second, duration)
            center = (item["start"] + item["end"]) / 2.0
            if v1.overlaps(item["start"], item["end"], masks):
                continue
            if any(abs(center - old) < MIN_SEPARATION_SEC for old in centers):
                continue
            item["requested_profile"] = next(
                (profile for profile, needed in QUOTAS.items() if len([row for row in selected if row["requested_profile"] == profile]) < needed),
                "fallback_high_rank",
            )
            item["selection_profile"] = f"fallback_for_{item['requested_profile']}"
            selected.append(item)
            centers.append(center)
            used_anchors.add(index)
    if len(selected) != 4:
        raise RuntimeError(f"Could only select {len(selected)}/4 fresh probes for {source['id']}")
    return selected


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    tracks = manifest["tracks"]
    source_ids = [str(track["id"]) for track in tracks]
    masks = prior_masks(manifest, source_ids)
    probes: list[dict] = []
    for track in tracks:
        chosen = select_source(track, masks[str(track["id"])])
        probes.extend(chosen)
        print(f"[round2 select] {track['id']} -> {len(chosen)}", flush=True)
    if len(probes) != 48:
        raise RuntimeError(f"Expected 48 probes, got {len(probes)}")

    random.Random(RANDOM_SEED).shuffle(probes)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for idx, item in enumerate(probes, 1):
        blind_id = f"R{idx:03d}"
        output = AUDIO_DIR / f"{blind_id}.wav"
        audio, sr = v1.read_interval(Path(item["source"]), float(item["start"]), float(item["end"]))
        audio32 = v1.resample(audio, sr)
        sf.write(str(output), audio32, v1.OUTPUT_SR, subtype="PCM_16")
        item["blind_id"] = blind_id
        item["audio"] = str(output.resolve())
        item["audio_rel"] = f"audio/{output.name}"
        item["output_duration"] = len(audio32) / v1.OUTPUT_SR
        item["wave"] = v1.waveform(audio32)
        if idx % 12 == 0 or idx == len(probes):
            print(f"[round2 audio] {idx}/{len(probes)}", flush=True)

    public = {
        "version": "human_calibration_round2_v1_blind",
        "sample_count": len(probes),
        "samples": [
            {"blind_id": row["blind_id"], "audio": row["audio_rel"], "duration_sec": round(row["output_duration"], 3)}
            for row in probes
        ],
    }
    audit = {
        "version": "human_calibration_round2_v1_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "policy": {
            "fresh_probes_only": True,
            "round1_interval_exclusion_padding_sec": 3.0,
            "per_source": 4,
            "profile_quotas": QUOTAS,
            "local_resources": RESOURCE_POLICY,
            "blind": "HTML omits source, ASR text, model score, and selection profile",
        },
        "samples": probes,
    }
    write_json(OUT / "blind_manifest.json", public)
    write_json(OUT / "audit_manifest.json", audit)
    (OUT / "README_试听说明.txt").write_text(
        "双击 index.html 直接试听。每条听完；有任何非目标字、无缝接话、重叠、切边或音质问题就勾‘有错误’并选类型。/n"
        "全部 48 条完成后点击‘保存反馈文件’，把下载的 JSON 交给我。/n",
        encoding="utf-8",
    )
    (OUT / "开始试听.cmd").write_text('@echo off/r/nstart "" "%~dp0index.html"/r/n', encoding="utf-8")
    print(json.dumps({"output": str(OUT), "samples": len(probes)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
