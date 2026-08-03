"""Refine FunASR v2 word candidates with local boundary checks.

This pass does not reconstruct audio.  It reads the original BS-Roformer
vocals and writes new clips only from words that survive three local 0.8 s
CAM++ checks (word start, midpoint, and end).  A sharp local score drop next
to another candidate word is treated as a speaker boundary and is not merged.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from funasr import AutoModel

import sys
sys.path.insert(0, str(Path(r"radio_drama_work/radio_drama_work")))
import full_speaker_diarize_new_v1 as base


ROOT = Path(r"radio_drama_work/radio_drama_work")
PKG = Path(r"radio_drama_work/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50")
VOCALS = ROOT / "new_radio_batch_v1" / "vocals"
V2 = ROOT / "full_funasr_target_recall_v2"
V2_REPORT = V2 / "report.json"
V2_REF_CACHE = V2 / "embedding_cache" / "all_target_references.npz"
OUT = ROOT / "full_funasr_local_boundary_v7"
ACCEPTED = OUT / "accepted_segments"
REVIEW = OUT / "review_segments"
BOUNDARY = OUT / "boundary_review"
EMBED_CACHE = ROOT / "full_funasr_local_boundary_v4" / "embedding_cache"
REPORT = OUT / "report.json"
CANDIDATE_LIST = ROOT / "full_funasr_local_boundary_train_candidate_v7.list"
REVIEW_LIST = ROOT / "full_funasr_local_boundary_review_v7.list"

CAM_SR = 16000
OUTPUT_SR = 32000
LOCAL_WINDOW_SEC = 0.80
LOCAL_TOP4_MIN = 0.48
LOCAL_REVIEW_TOP4_MIN = 0.45
LOCAL_DIRECTION_DROP = 0.04
SAFE_SOURCE_DIRECTION_DROP = 0.10
NEIGHBOR_GAP_SEC = 0.25
MIN_CLIP_SEC = 1.50
MAX_CLIP_SEC = 10.00
MERGE_GAP_SEC = 0.65
EDGE_PAD_SEC = 0.16
TAIL_PAD_SEC = 0.12
MIN_CLIP_SEC = 1.80
SOURCE_MARGIN_MIN = 0.04


def overlap(a: float, b: float, c: float, d: float) -> bool:
    return max(a, c) < min(b, d)


def forbidden(row: dict, start: float, end: float) -> bool:
    return any(overlap(start, end, float(h["start"]), float(h["end"])) for h in row.get("hard_exclusions", []))


def source_name(path: str) -> str:
    return Path(path).name.replace("_vocals.wav", "")


def source_margin(source: dict) -> float:
    margins: list[float] = []
    for run in source.get("cluster_runs", []):
        values = sorted((float(cluster["target_top4"]) for cluster in run.get("clusters", [])), reverse=True)
        if len(values) >= 2:
            margins.append(values[0] - values[1])
    return min(margins) if margins else 0.0


def norm_rows(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)


def load_refs() -> torch.Tensor:
    if not V2_REF_CACHE.exists():
        raise FileNotFoundError(V2_REF_CACHE)
    return norm_rows(torch.from_numpy(np.load(str(V2_REF_CACHE))["embeddings"]).float())


def local_specs(row: dict) -> list[tuple[float, float]]:
    return [(float(row["start"]), float(row["start"])), ((float(row["start"]) + float(row["end"])) / 2.0, (float(row["start"]) + float(row["end"])) / 2.0), (float(row["end"]), float(row["end"]))]


def load_or_make_local_embeddings(cam: AutoModel, source: Path, audio16: np.ndarray, rows: list[dict]) -> torch.Tensor:
    cache = EMBED_CACHE / f"{source.stem}.npz"
    specs = np.asarray([[center for pair in local_specs(row) for center in pair] for row in rows], dtype=np.float32)
    if cache.exists():
        data = np.load(str(cache))
        if data["specs"].shape == specs.shape and np.allclose(data["specs"], specs, atol=1e-3):
            return norm_rows(torch.from_numpy(data["embeddings"]).float())
    waves = []
    for row in rows:
        for center, _ in local_specs(row):
            waves.append(base.segment(audio16, CAM_SR, center - LOCAL_WINDOW_SEC / 2, center + LOCAL_WINDOW_SEC / 2, LOCAL_WINDOW_SEC))
    embs = base.cam_embed_batch(cam, waves)
    np.savez_compressed(str(cache), specs=specs, embeddings=embs.numpy().astype(np.float32))
    return embs


def group_words(rows: list[dict], tier: str) -> list[list[dict]]:
    groups: list[list[dict]] = []
    current: list[dict] = []
    for row in rows:
        if row.get("local_decision") != tier:
            if current:
                groups.append(current)
                current = []
            continue
        if current:
            gap = float(row["start"]) - float(current[-1]["end"])
            if gap > MERGE_GAP_SEC:
                groups.append(current)
                current = []
            elif any(other.get("local_decision") != tier and float(current[-1]["end"]) <= float(other["start"]) < float(row["start"]) for other in rows):
                groups.append(current)
                current = []
        if not current or float(row["end"]) - float(current[0]["start"]) <= MAX_CLIP_SEC:
            current.append(row)
        else:
            groups.append(current)
            current = [row]
    if current:
        groups.append(current)
    return groups


def write_clip(audio: np.ndarray, sr: int, start: float, end: float, path: Path) -> None:
    raw = base.segment(audio, sr, start, end)
    out = base.resample_1d(raw, sr, OUTPUT_SR)
    sf.write(str(path), out, OUTPUT_SR, subtype="PCM_16")


def line(path: Path, label: str, text: str) -> str:
    return f"{path}|{label}|JA|{text.replace(chr(10), ' ').strip()}"


def main() -> None:
    if not V2_REPORT.exists():
        raise FileNotFoundError(V2_REPORT)
    for folder in (OUT, ACCEPTED, REVIEW, BOUNDARY, EMBED_CACHE):
        folder.mkdir(parents=True, exist_ok=True)
    report_v2 = json.loads(V2_REPORT.read_text(encoding="utf-8"))
    refs = load_refs()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cam = AutoModel(model=str(base.CAM_MODEL), device=device, check_latest=False, disable_pbar=True)
    candidate_lines: list[str] = []
    review_lines: list[str] = []
    report = {
        "version": "full_funasr_local_boundary_v7",
        "input_report": str(V2_REPORT),
        "method": "v2 per-word candidates + CAM++ 0.8 s start/mid/end local checks + speaker-boundary split",
        "input_waveform": "original BS-Roformer vocals",
        "target_reference_count": int(len(refs)),
        "parameters": {"local_window_sec": LOCAL_WINDOW_SEC, "local_top4_min": LOCAL_TOP4_MIN, "local_review_top4_min": LOCAL_REVIEW_TOP4_MIN, "local_direction_drop": LOCAL_DIRECTION_DROP, "neighbor_gap_sec": NEIGHBOR_GAP_SEC, "source_margin_min": SOURCE_MARGIN_MIN, "min_clip_sec": MIN_CLIP_SEC},
        "sources": [],
    }
    clip_index = 0
    only_source = os.environ.get("FUNASR_ONLY_SOURCE", "").strip()
    source_reports = [s for s in report_v2.get("sources", []) if not only_source or Path(s["source"]).name == only_source]
    for source_report in source_reports:
        source = Path(source_report["source"])
        started = time.time()
        print(f"[source] {source.name}", flush=True)
        audio, sr = base.read_mono(source)
        audio16 = base.resample_1d(audio, sr, CAM_SR)
        words = sorted(source_report.get("word_rows", []), key=lambda row: (float(row["start"]), float(row["end"])))
        candidates = [row for row in words if row.get("decision") in {"accepted", "expanded"}]
        local_embs = load_or_make_local_embeddings(cam, source, audio16, candidates)
        margin = source_margin(source_report)
        guarded = margin < SOURCE_MARGIN_MIN
        direction_drop = LOCAL_DIRECTION_DROP if guarded else SAFE_SOURCE_DIRECTION_DROP
        local_rows: list[dict] = []
        for index, row in enumerate(candidates):
            scores = []
            for emb in local_embs[index * 3 : index * 3 + 3]:
                values = torch.sort(emb @ refs.T).values
                scores.append(float(values[-4:].mean()))
            row_copy = dict(row)
            row_copy["local_start_top4"] = scores[0]
            row_copy["local_mid_top4"] = scores[1]
            row_copy["local_end_top4"] = scores[2]
            row_copy["local_min_top4"] = min(scores)
            local_rows.append(row_copy)
        by_start = {round(float(row["start"]), 5): row for row in local_rows}
        local_by_all: list[dict] = []
        candidate_set = {id(row): row for row in local_rows}
        for word in words:
            if word.get("decision") not in {"accepted", "expanded"}:
                row = dict(word)
                row["local_decision"] = "rejected"
                local_by_all.append(row)
                continue
            row = next(value for value in local_rows if abs(float(value["start"]) - float(word["start"])) < 1e-5 and abs(float(value["end"]) - float(word["end"])) < 1e-5)
            pos = local_rows.index(row)
            prev_row = local_rows[pos - 1] if pos > 0 else None
            next_row = local_rows[pos + 1] if pos + 1 < len(local_rows) else None
            left_drop = float(row["local_mid_top4"]) - float(row["local_start_top4"]) >= direction_drop and prev_row is not None and float(row["start"]) - float(prev_row["end"]) <= NEIGHBOR_GAP_SEC
            right_drop = float(row["local_mid_top4"]) - float(row["local_end_top4"]) >= direction_drop and next_row is not None and float(next_row["start"]) - float(row["end"]) <= NEIGHBOR_GAP_SEC
            row["local_boundary_drop"] = bool(left_drop or right_drop)
            if float(row["local_mid_top4"]) < LOCAL_TOP4_MIN or row["local_boundary_drop"]:
                row["local_decision"] = "boundary_review"
            elif row.get("decision") == "accepted":
                row["local_decision"] = "accepted"
            elif float(row["local_mid_top4"]) >= LOCAL_REVIEW_TOP4_MIN:
                row["local_decision"] = "expanded"
            else:
                row["local_decision"] = "rejected"
            local_by_all.append(row)

        source_accepted = []
        source_review = []
        source_boundary = []
        duration = len(audio) / sr
        def clip_bounds(group: list[dict]) -> tuple[float, float]:
            first_start = float(group[0]["start"])
            last_end = float(group[-1]["end"])
            previous = None
            following = None
            for word in words:
                if float(word["end"]) <= first_start + 1e-5:
                    previous = word
                if following is None and float(word["start"]) >= last_end - 1e-5 and word is not group[-1]:
                    following = word
            start = max(0.0, first_start - EDGE_PAD_SEC)
            end = min(duration, last_end + TAIL_PAD_SEC)
            if previous is not None:
                if float(previous["end"]) >= first_start - 0.02:
                    start = first_start
                else:
                    start = max(start, float(previous["end"]) + 0.02)
            if following is not None and float(following["start"]) <= end:
                end = min(end, max(last_end, float(following["start"]) - 0.02))
            return start, end

        for tier in ("accepted", "expanded"):
            for group in group_words(local_by_all, tier):
                start, end = clip_bounds(group)
                if forbidden(source_report, start, end):
                    continue
                if tier == "accepted" and end - start < MIN_CLIP_SEC:
                    continue
                folder = ACCEPTED if tier == "accepted" and not guarded else REVIEW
                name = f"local_{clip_index:05d}_{source.stem}_{int(round(start * OUTPUT_SR)):010d}_{int(round(end * OUTPUT_SR)):010d}.wav"
                path = folder / name
                write_clip(audio, sr, start, end, path)
                text = "".join(str(word["text"]) for word in group)
                record = {"file": str(path), "source": str(source), "start": start, "end": end, "duration": end - start, "text": text, "word_count": len(group), "tier": "accepted" if tier == "accepted" and not guarded else "review", "origin_tier": tier, "source_guarded": guarded}
                if tier == "accepted" and not guarded:
                    source_accepted.append(record)
                    candidate_lines.append(line(path, "full_funasr_local_boundary_v7", text))
                else:
                    source_review.append(record)
                    review_lines.append(line(path, "full_funasr_local_boundary_v7_review", text))
                clip_index += 1
        for row in local_by_all:
            if row.get("local_decision") != "boundary_review":
                continue
            start = max(0.0, float(row["start"]) - 0.20)
            end = min(duration, float(row["end"]) + 0.20)
            if forbidden(source_report, start, end):
                continue
            name = f"boundary_{clip_index:05d}_{source.stem}_{int(round(start * OUTPUT_SR)):010d}_{int(round(end * OUTPUT_SR)):010d}.wav"
            path = BOUNDARY / name
            write_clip(audio, sr, start, end, path)
            record = {"file": str(path), "source": str(source), "start": start, "end": end, "duration": end - start, "text": str(row["text"]), "tier": "boundary_review", "local_start_top4": row.get("local_start_top4"), "local_mid_top4": row.get("local_mid_top4"), "local_end_top4": row.get("local_end_top4")}
            source_boundary.append(record)
            review_lines.append(line(path, "full_funasr_local_boundary_v7_boundary_review", str(row["text"])))
            clip_index += 1
        report["sources"].append({"source": str(source), "duration_sec": duration, "source_margin_min": margin, "guarded_from_training": guarded, "candidate_word_count": len(candidates), "local_decision_counts": dict(Counter(row.get("local_decision") for row in local_by_all)), "accepted_segments": source_accepted, "review_segments": source_review, "boundary_review_segments": source_boundary, "elapsed_sec": time.time() - started})
        print(f"  [done] candidates={len(candidates)} accepted={len(source_accepted)} ({sum(row['duration'] for row in source_accepted):.1f}s) review={len(source_review)} boundary={len(source_boundary)} guarded={guarded}", flush=True)
    CANDIDATE_LIST.write_text("/n".join(candidate_lines) + ("/n" if candidate_lines else ""), encoding="utf-8")
    REVIEW_LIST.write_text("/n".join(review_lines) + ("/n" if review_lines else ""), encoding="utf-8")
    report["candidate_list"] = str(CANDIDATE_LIST)
    report["review_list"] = str(REVIEW_LIST)
    report["candidate_count"] = len(candidate_lines)
    report["candidate_total_sec"] = sum(sf.info(str(Path(line.split("|", 1)[0]))).duration for line in candidate_lines)
    report["review_count"] = len(review_lines)
    report["review_total_sec"] = sum(sf.info(str(Path(line.split("|", 1)[0]))).duration for line in review_lines)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"candidate_count": report["candidate_count"], "candidate_sec": report["candidate_total_sec"], "review_count": report["review_count"], "review_sec": report["review_total_sec"], "candidate_list": str(CANDIDATE_LIST), "review_list": str(REVIEW_LIST)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
