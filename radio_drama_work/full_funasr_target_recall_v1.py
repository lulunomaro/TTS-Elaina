# -*- coding: utf-8 -*-
"""Higher-recall target-speaker extraction using FunASR VAD + CAM++.

The previous v3 pass created speaker windows from Whisper ASR sentence spans
and required two target windows for every accepted word.  This version uses
FunASR's FSMN-VAD to cover all speech regions, runs the same CAM++ encoder
with FunASR's spectral clustering backend at several requested speaker counts,
and keeps only chunks that agree across clustering runs.  ASR word timestamps
are still used to cut the original BS-Roformer vocal waveform.

It is a separate candidate set.  It never edits old lists, reconstructs audio
with ClearVoice, or starts training.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from funasr import AutoModel
from funasr.models.campplus.cluster_backend import ClusterBackend

sys.path.insert(0, str(Path(r"radio_drama_work/radio_drama_work")))
import full_speaker_diarize_new_v1 as base  # noqa: E402


ROOT = Path(r"radio_drama_work/radio_drama_work")
PKG = Path(r"radio_drama_work/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50")
VOCALS = ROOT / "new_radio_batch_v1" / "vocals"
OTHER = ROOT / "new_radio_batch_v1" / "other"
VAD_PATH = PKG / "tools" / "asr" / "models" / "speech_fsmn_vad_zh-cn-16k-common-pytorch"
OUT = ROOT / "full_funasr_target_recall_v2"
ACCEPTED = OUT / "accepted_segments"
EXPANDED = OUT / "expanded_review"
REVIEW = OUT / "review"
REJECTED = OUT / "rejected"
ASR_CACHE = ROOT / "full_speaker_diarize_new_v3" / "asr_cache"
VAD_CACHE = OUT / "vad_cache"
EMBED_CACHE = OUT / "embedding_cache"
REPORT = OUT / "report.json"
CANDIDATE_LIST = ROOT / "full_funasr_target_recall_train_candidate_v2.list"
EXPANDED_LIST = ROOT / "full_funasr_target_recall_expanded_review_v2.list"
REVIEW_LIST = ROOT / "full_funasr_target_recall_review_v2.list"

CAM_SR = 16000
OUTPUT_SR = 32000
WINDOW_SEC = 1.50
HOP_SEC = 0.75
CAM_BATCH = 64
WORD_WINDOW_SEC = 1.20
MIN_WORD_SEC = 0.16
MIN_CLIP_SEC = 1.50
MAX_CLIP_SEC = 8.00
MERGE_GAP_SEC = 0.45
EDGE_PAD_SEC = 0.045
TARGET_SIM_MIN = 0.50
REVIEW_SIM_MIN = 0.46
HARD_NEGATIVE_SIM_VETO = 0.88
OTHER_RATIO_MAX = 1.15
CONSENSUS_MIN = 2
OLD_REFERENCE_DIR = PKG / "logs" / "my_voice" / "5-wav32k"
REFERENCE_CACHE = EMBED_CACHE / "all_target_references.npz"

HARD_EXCLUSIONS = {
    "243358894-1-192_vocals.wav": [
        (106.805, 109.555, "user rejected old newradio_00001"),
        (234.945, 237.915, "user rejected old newradio_00003"),
        (249.000, 251.000, "user rejected old newradio_00004"),
    ],
}


def overlap(a: float, b: float, c: float, d: float) -> bool:
    return max(a, c) < min(b, d)


def forbidden_reason(source_name: str, start: float, end: float) -> str | None:
    for a, b, reason in HARD_EXCLUSIONS.get(source_name, []):
        if overlap(start, end, a, b):
            return reason
    return None


def make_chunks(vad_rows: list[list[int]], audio16: np.ndarray) -> list[tuple[float, float, np.ndarray]]:
    chunks: list[tuple[float, float, np.ndarray]] = []
    chunk_len = int(round(WINDOW_SEC * CAM_SR))
    shift = int(round(HOP_SEC * CAM_SR))
    for start_ms, end_ms in vad_rows:
        start = float(start_ms) / 1000.0
        end = float(end_ms) / 1000.0
        a = max(0, int(round(start * CAM_SR)))
        b = min(len(audio16), int(round(end * CAM_SR)))
        data = audio16[a:b]
        if len(data) == 0:
            continue
        last_end = 0
        for chunk_start in range(0, len(data), shift):
            chunk_end = min(chunk_start + chunk_len, len(data))
            if chunk_end <= last_end:
                break
            last_end = chunk_end
            chunk_start = max(0, chunk_end - chunk_len)
            chunk = data[chunk_start:chunk_end]
            if len(chunk) < chunk_len:
                chunk = np.pad(chunk, (0, chunk_len - len(chunk)))
            chunks.append((start + chunk_start / CAM_SR, start + chunk_end / CAM_SR, chunk[:chunk_len]))
    return chunks


def get_vad(vad: AutoModel, source: Path) -> list[list[int]]:
    cache = VAD_CACHE / f"{source.stem}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))["value"]
    result = vad.generate(input=str(source))[0]
    rows = result.get("value", [])
    cache.write_text(json.dumps({"source": str(source), "value": rows}, ensure_ascii=False), encoding="utf-8")
    return rows


def load_or_make_embeddings(cam: AutoModel, source: Path, chunks: list[tuple[float, float, np.ndarray]]) -> torch.Tensor:
    cache = EMBED_CACHE / f"{source.stem}.npz"
    specs = np.asarray([[a, b] for a, b, _ in chunks], dtype=np.float32)
    if cache.exists():
        data = np.load(str(cache))
        if data["specs"].shape == specs.shape and np.allclose(data["specs"], specs, atol=1e-3):
            return base.norm_rows(torch.from_numpy(data["embeddings"]))
    embs = base.cam_embed_batch(cam, [chunk[2] for chunk in chunks])
    np.savez_compressed(str(cache), specs=specs, embeddings=embs.numpy().astype(np.float32))
    return embs


def all_target_reference_paths() -> list[Path]:
    """Return the full old pure-target bank plus the four confirmed refs.

    The previous v3/recall pass accidentally used only the four newly
    confirmed files.  The old bank is the strongest available target-speaker
    reference set, so it must be part of both calibration and direct scoring.
    """
    paths = sorted(OLD_REFERENCE_DIR.glob("*.wav")) if OLD_REFERENCE_DIR.exists() else []
    paths.extend(base.parse_reference_paths())
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve()).lower()
        if path.exists() and key not in seen:
            seen.add(key)
            result.append(path)
    if not result:
        raise RuntimeError(f"No target references found in {OLD_REFERENCE_DIR} or confirmed_target_pure.txt")
    return result


def load_all_target_reference_embeddings(cam: AutoModel) -> tuple[torch.Tensor, list[Path]]:
    """Embed all target references with a cache keyed by the exact file list."""
    refs = all_target_reference_paths()
    ref_names = [str(path) for path in refs]
    if REFERENCE_CACHE.exists():
        cached = np.load(str(REFERENCE_CACHE), allow_pickle=True)
        cached_names = [str(value) for value in cached["paths"].tolist()]
        if cached_names == ref_names:
            return base.norm_rows(torch.from_numpy(cached["embeddings"])), refs

    waves: list[np.ndarray] = []
    kept: list[Path] = []
    for path in refs:
        audio, sr = base.read_mono(path)
        if len(audio) < int(0.8 * sr):
            continue
        ref16 = base.resample_1d(audio, sr, CAM_SR)
        waves.append(base.segment(ref16, CAM_SR, 0.0, len(ref16) / CAM_SR, WINDOW_SEC))
        kept.append(path)
    if not waves:
        raise RuntimeError("All target references were too short to embed")
    embs = base.cam_embed_batch(cam, waves)
    np.savez_compressed(
        str(REFERENCE_CACHE),
        embeddings=embs.numpy().astype(np.float32),
        paths=np.asarray([str(path) for path in kept]),
    )
    return embs, kept


def cluster_runs(embs: torch.Tensor, refs: torch.Tensor, hard_negatives: torch.Tensor) -> tuple[list[dict], list[dict]]:
    n = len(embs)
    if n < 2:
        return [], []
    requested = [k for k in (3, 4, 5) if k < n]
    if not requested:
        requested = [2]
    runs: list[dict] = []
    ref_centroid = base.norm_rows(refs.mean(0, keepdim=True))[0]
    hard_sim = (embs @ hard_negatives.T).max(dim=1).values.numpy() if len(hard_negatives) else np.full(n, -1.0)
    direct_sim = (embs @ refs.T).max(dim=1).values.numpy()
    for requested_k in requested:
        labels = np.asarray(ClusterBackend()(embs.numpy(), oracle_num=requested_k), dtype=int)
        clusters = []
        for label in sorted(set(labels.tolist())):
            center = base.norm_rows(embs[labels == label].mean(0, keepdim=True))[0]
            sims = center @ refs.T
            clusters.append({
                "cluster": int(label),
                "count": int(np.sum(labels == label)),
                "target_sim": float(torch.max(sims).item()),
                "target_top4": float(torch.topk(sims, k=min(4, len(sims))).values.mean().item()),
            })
        clusters.sort(key=lambda row: row["target_top4"], reverse=True)
        target = clusters[0]["cluster"]
        target_mask = (labels == target) & (direct_sim >= TARGET_SIM_MIN) & (hard_sim < HARD_NEGATIVE_SIM_VETO)
        runs.append({
            "requested_k": requested_k,
            "labels": labels,
            "target_cluster": int(target),
            "clusters": clusters,
            "target_mask": target_mask,
        })
    target_votes = np.sum(np.stack([run["target_mask"] for run in runs], axis=0), axis=0)
    consensus = max(CONSENSUS_MIN, int(np.ceil(len(runs) * 0.67)))
    target_mask = target_votes >= consensus
    expanded_mask = (target_votes >= 1) & (direct_sim >= REVIEW_SIM_MIN) & (hard_sim < HARD_NEGATIVE_SIM_VETO)
    chunk_rows = []
    for idx in range(n):
        chunk_rows.append({
            "index": idx,
            "target_votes": int(target_votes[idx]),
            "run_count": len(runs),
            "direct_sim": float(direct_sim[idx]),
            "hard_negative_sim": float(hard_sim[idx]),
            "tier": "target" if target_mask[idx] else "expanded" if expanded_mask[idx] else "other",
        })
    return runs, chunk_rows


def load_words(source: Path) -> list[dict]:
    cache = ASR_CACHE / f"{source.stem}.json"
    if not cache.exists():
        raise FileNotFoundError(f"Missing v3 ASR cache: {cache}")
    return json.loads(cache.read_text(encoding="utf-8"))["words"]


def vote_word(word: dict, chunk_rows: list[dict], chunks: list[tuple[float, float, np.ndarray]]) -> dict:
    mid = (word["start"] + word["end"]) / 2.0
    candidates = [row for row, chunk in zip(chunk_rows, chunks) if chunk[0] <= mid <= chunk[1]]
    if not candidates:
        return {"target_vote": 0.0, "target_count": 0, "other_count": 0, "review_count": 0, "boundary": True}
    target = [row for row in candidates if row["tier"] in ("target", "expanded")]
    strong = [row for row in candidates if row["tier"] == "target"]
    other = [row for row in candidates if row["tier"] == "other"]
    expanded = [row for row in candidates if row["tier"] == "expanded"]
    return {
        "target_vote": len(target) / len(candidates),
        "target_count": len(target),
        "strong_count": len(strong),
        "expanded_count": len(expanded),
        "other_count": len(other),
        "boundary": bool(other or expanded),
    }


def group_words(all_rows: list[dict], tier: str) -> list[list[dict]]:
    groups: list[list[dict]] = []
    current: list[dict] = []
    for idx, row in enumerate(all_rows):
        if row["decision"] != tier:
            if current:
                groups.append(current)
                current = []
            continue
        if current:
            gap = row["start"] - current[-1]["end"]
            if gap > MERGE_GAP_SEC or any(other["decision"] != tier for other in all_rows if current[-1]["end"] <= other["start"] < row["start"]):
                groups.append(current)
                current = []
        if not current or row["end"] - current[0]["start"] <= MAX_CLIP_SEC:
            current.append(row)
        else:
            groups.append(current)
            current = [row]
    if current:
        groups.append(current)
    return groups


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + 1e-12)) if len(audio) else 0.0


def other_ratio(audio: np.ndarray, other: np.ndarray | None, sr: int, start: float, end: float) -> float:
    if other is None:
        return 0.0
    a = max(0, int(round(start * sr)))
    b = min(len(audio), len(other), int(round(end * sr)))
    return rms(other[a:b]) / max(rms(audio[a:b]), 1e-6) if b > a else 0.0


def write_clip(audio: np.ndarray, sr: int, start: float, end: float, path: Path) -> None:
    raw = base.segment(audio, sr, start, end)
    out = base.resample_1d(raw, sr, OUTPUT_SR)
    sf.write(str(path), out, OUTPUT_SR, subtype="PCM_16")


def validate(lines: list[str]) -> dict:
    paths = [Path(line.split("|", 1)[0]) for line in lines]
    missing = [str(p) for p in paths if not p.exists()]
    short = [str(p) for p in paths if p.exists() and sf.info(str(p)).duration < MIN_CLIP_SEC]
    duplicates = [p for p, count in Counter(str(x) for x in paths).items() if count > 1]
    return {"candidate_count": len(paths), "missing": missing, "short": short, "duplicates": duplicates, "passed": not missing and not short and not duplicates}


def main() -> None:
    for path in (VOCALS, OTHER, VAD_PATH, base.CAM_MODEL):
        if not path.exists():
            raise FileNotFoundError(path)
    for path in (OUT, ACCEPTED, EXPANDED, REVIEW, REJECTED, VAD_CACHE, EMBED_CACHE):
        path.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    vad = AutoModel(model=str(VAD_PATH), device=device, disable_update=True, disable_pbar=True)
    cam = AutoModel(model=str(base.CAM_MODEL), device=device, check_latest=False, disable_pbar=True)
    ref_embs, ref_paths = load_all_target_reference_embeddings(cam)
    hard_paths = base.parse_hard_negative_paths()
    hard_embs, hard_labels = base.load_or_make_hard_negative_embeddings(cam, hard_paths)

    report = {
        "version": "full_funasr_target_recall_v2",
        "method": "FunASR FSMN-VAD + CAM++ spectral clustering consensus + ASR word gate",
        "input_waveform": "original BS-Roformer vocals",
        "funasr_source": str(ROOT.parent / "FunASR"),
        "vad_model": str(VAD_PATH),
        "cam_model": str(base.CAM_MODEL),
        "target_reference_count": len(ref_paths),
        "target_reference_paths": [str(path) for path in ref_paths],
        "hard_negative_paths": [str(path) for path in hard_paths],
        "hard_negative_window_count": len(hard_labels),
        "parameters": {"window_sec": WINDOW_SEC, "hop_sec": HOP_SEC, "cluster_k": [3, 4, 5], "consensus_min": CONSENSUS_MIN, "target_sim_min": TARGET_SIM_MIN, "review_sim_min": REVIEW_SIM_MIN, "hard_negative_sim_veto": HARD_NEGATIVE_SIM_VETO, "min_clip_sec": MIN_CLIP_SEC},
        "sources": [],
    }
    candidate_lines: list[str] = []
    expanded_lines: list[str] = []
    review_lines: list[str] = []
    clip_index = 0

    source_paths = sorted(VOCALS.glob("*_vocals.wav"))
    only_source = os.environ.get("FUNASR_ONLY_SOURCE", "").strip()
    if only_source:
        source_paths = [source for source in source_paths if source.name == only_source]
    for source in source_paths:
        started = time.time()
        print(f"[source] {source.name}", flush=True)
        audio, sr = base.read_mono(source)
        duration = len(audio) / sr
        other_path = OTHER / source.name.replace("_vocals.wav", "_other.wav")
        other = base.read_mono(other_path)[0] if other_path.exists() else None
        vad_rows = get_vad(vad, source)
        audio16 = base.resample_1d(audio, sr, CAM_SR)
        chunks = make_chunks(vad_rows, audio16)
        embs = load_or_make_embeddings(cam, source, chunks)
        runs, chunk_rows = cluster_runs(embs, ref_embs, hard_embs)
        words = load_words(source)
        valid = [word for word in words if word["end"] - word["start"] >= MIN_WORD_SEC]
        word_waves = [base.segment(audio16, CAM_SR, word["start"], word["end"], WORD_WINDOW_SEC) for word in valid]
        word_embs = base.cam_embed_batch(cam, word_waves)
        word_rows = []
        counters = Counter()
        for word, emb in zip(valid, word_embs):
            vote = vote_word(word, chunk_rows, chunks)
            ref_sim = float(torch.max(emb @ ref_embs.T).item())
            hard_sim = float(torch.max(emb @ hard_embs.T).item()) if len(hard_embs) else -1.0
            exclusion = forbidden_reason(source.name, word["start"], word["end"])
            row = {**word, **vote, "reference_sim": ref_sim, "hard_negative_sim": hard_sim, "hard_exclusion": exclusion}
            if exclusion:
                row["decision"] = "rejected_hard_exclusion"
                counters["rejected_hard_exclusion"] += 1
            elif hard_sim >= HARD_NEGATIVE_SIM_VETO:
                row["decision"] = "rejected_hard_negative"
                counters["rejected_hard_negative"] += 1
            elif vote["target_vote"] >= 0.66 and vote["other_count"] == 0 and vote["expanded_count"] == 0 and vote["strong_count"] >= 1 and ref_sim >= TARGET_SIM_MIN:
                row["decision"] = "accepted"
                counters["accepted_word"] += 1
            elif vote["target_vote"] >= 0.66 and vote["other_count"] == 0 and ref_sim >= REVIEW_SIM_MIN:
                row["decision"] = "expanded"
                counters["expanded_word"] += 1
            elif vote["target_vote"] >= 0.50 and vote["other_count"] == 0 and ref_sim >= REVIEW_SIM_MIN:
                row["decision"] = "review"
                counters["review_word"] += 1
            else:
                row["decision"] = "rejected"
                counters["rejected_word"] += 1
            word_rows.append(row)

        source_outputs = []
        source_expanded = []
        source_reviews = []
        for tier, folder, output_list in (("accepted", ACCEPTED, candidate_lines), ("expanded", EXPANDED, expanded_lines), ("review", REVIEW, review_lines)):
            for group in group_words(word_rows, tier):
                start = max(0.0, group[0]["start"] - EDGE_PAD_SEC)
                end = min(duration, group[-1]["end"] + EDGE_PAD_SEC)
                if end - start < MIN_CLIP_SEC:
                    counters[f"{tier}_short_group"] += 1
                    continue
                exclusion = forbidden_reason(source.name, start, end)
                if exclusion:
                    counters[f"{tier}_hard_exclusion"] += 1
                    continue
                ratio = other_ratio(audio, other, sr, start, end)
                if ratio > OTHER_RATIO_MAX:
                    counters[f"{tier}_other_dominant"] += 1
                    continue
                name = f"funasr_{clip_index:05d}_{source.stem}_{int(round(start * OUTPUT_SR)):010d}_{int(round(end * OUTPUT_SR)):010d}.wav"
                path = folder / name
                write_clip(audio, sr, start, end, path)
                line = f"{path}|full_funasr_target_recall_v2|JA|{''.join(word['text'] for word in group)}"
                output_list.append(line)
                row = {"file": str(path), "source": str(source), "start": start, "end": end, "duration": end - start, "text": "".join(word["text"] for word in group), "other_ratio": ratio, "word_count": len(group), "min_reference_sim": min(word["reference_sim"] for word in group), "min_target_vote": min(word["target_vote"] for word in group), "tier": tier}
                if tier == "accepted":
                    source_outputs.append(row)
                elif tier == "expanded":
                    source_expanded.append(row)
                else:
                    source_reviews.append(row)
                clip_index += 1

        source_report = {
            "source": str(source),
            "other_source": str(other_path) if other_path.exists() else None,
            "duration_sec": duration,
            "sample_rate": sr,
            "vad_segment_count": len(vad_rows),
            "funasr_chunk_count": len(chunks),
            "cluster_runs": [{"requested_k": run["requested_k"], "target_cluster": run["target_cluster"], "clusters": run["clusters"]} for run in runs],
            "chunk_tiers": dict(Counter(row["tier"] for row in chunk_rows)),
            "asr_word_count": len(words),
            "counters": dict(counters),
            "accepted_segments": source_outputs,
            "expanded_segments": source_expanded,
            "review_segments": source_reviews,
            "word_rows": word_rows,
            "chunk_rows": chunk_rows,
            "hard_exclusions": [{"start": a, "end": b, "reason": reason} for a, b, reason in HARD_EXCLUSIONS.get(source.name, [])],
            "elapsed_sec": time.time() - started,
        }
        report["sources"].append(source_report)
        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [done] vad={len(vad_rows)} chunks={len(chunks)} target={source_report['chunk_tiers'].get('target', 0)} accepted={len(source_outputs)} ({sum(row['duration'] for row in source_outputs):.1f}s) expanded={len(source_expanded)} ({sum(row['duration'] for row in source_expanded):.1f}s) review={len(source_reviews)} ({sum(row['duration'] for row in source_reviews):.1f}s)", flush=True)

    CANDIDATE_LIST.write_text("/n".join(candidate_lines) + ("/n" if candidate_lines else ""), encoding="utf-8")
    EXPANDED_LIST.write_text("/n".join(expanded_lines) + ("/n" if expanded_lines else ""), encoding="utf-8")
    REVIEW_LIST.write_text("/n".join(review_lines) + ("/n" if review_lines else ""), encoding="utf-8")
    report["candidate_validation"] = validate(candidate_lines)
    report["candidate_count"] = len(candidate_lines)
    report["candidate_total_sec"] = sum(sf.info(line.split("|", 1)[0]).duration for line in candidate_lines if Path(line.split("|", 1)[0]).exists())
    report["expanded_count"] = len(expanded_lines)
    report["expanded_total_sec"] = sum(sf.info(line.split("|", 1)[0]).duration for line in expanded_lines if Path(line.split("|", 1)[0]).exists())
    report["review_count"] = len(review_lines)
    report["review_total_sec"] = sum(sf.info(line.split("|", 1)[0]).duration for line in review_lines if Path(line.split("|", 1)[0]).exists())
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"candidate_list": str(CANDIDATE_LIST), "candidate_count": len(candidate_lines), "candidate_sec": report["candidate_total_sec"], "expanded_list": str(EXPANDED_LIST), "expanded_count": len(expanded_lines), "expanded_sec": report["expanded_total_sec"], "review_count": len(review_lines), "review_sec": report["review_total_sec"], "validation": report["candidate_validation"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
