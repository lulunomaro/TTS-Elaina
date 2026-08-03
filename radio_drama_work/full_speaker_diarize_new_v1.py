# -*- coding: utf-8 -*-
"""Full-radio target-speaker extraction, version 3.

Pipeline:
    BS-Roformer vocals -> Japanese ASR word timestamps -> CAM++ speaker
    windows/clustering -> target-word gating -> clips from the original
    separated vocal waveform.

ClearVoice is intentionally not used to reconstruct training audio here.
The local ClearVoice source contains separation/TSE models, but it does not
provide an audio-only diarization API.  CAM++ is used for diarization and the
four user-confirmed pure clips are used as the target reference bank.

This script creates a new versioned output and does not touch any old list or
start training.  It writes 32 kHz clips by one final resample of the original
BS-Roformer vocal waveform; no ClearVoice reconstruction is used.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from faster_whisper import WhisperModel
from funasr import AutoModel
from sklearn.cluster import KMeans


ROOT = Path(r"radio_drama_work/radio_drama_work")
PKG = Path(r"radio_drama_work/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50")
VOCALS = ROOT / "new_radio_batch_v1" / "vocals"
OTHER = ROOT / "new_radio_batch_v1" / "other"
REF_LIST = ROOT / "confirmed_target_pure.txt"
CAM_MODEL = ROOT / "modelscope_cache" / "models" / "iic" / "speech_campplus_sv_zh-cn_16k-common"
OUT = ROOT / "full_speaker_diarize_new_v3"
ACCEPTED = OUT / "accepted_segments"
REVIEW = OUT / "review"
REJECTED = OUT / "rejected"
ASR_CACHE = OUT / "asr_cache"
EMBED_CACHE = OUT / "embedding_cache"
REPORT_PATH = OUT / "report.json"
CANDIDATE_LIST = ROOT / "full_speaker_diarize_new_train_candidate_v3.list"
REVIEW_LIST = ROOT / "full_speaker_diarize_new_review_v3.list"

CAM_SR = 16000
OUTPUT_SR = 32000
CAM_WINDOW_SEC = 1.50
CAM_HOP_SEC = 0.75
WORD_WINDOW_SEC = 1.20
CAM_BATCH = 64
MIN_WORD_SEC = 0.16
MIN_CLIP_SEC = 1.50
MAX_CLIP_SEC = 8.00
WORD_MERGE_GAP_SEC = 0.42
WORD_EDGE_PAD_SEC = 0.045
TARGET_VOTE_MIN = 0.66
REVIEW_VOTE_MIN = 0.50
TARGET_SIM_MIN = 0.50
TARGET_SIM_HIGH = 0.56
REVIEW_SIM_MIN = 0.46
HARD_NEGATIVE_SIM_VETO = 0.88
NEGATIVE_OTHER_RATIO = 1.15

# Explicit user-confirmed false positives from the previous 243358894 pass.
# The filenames encoded 32 kHz sample indices, hence the conversion to sec.
HARD_EXCLUSIONS: dict[str, list[tuple[float, float, str]]] = {
    "243358894-1-192_vocals.wav": [
        (106.805, 109.555, "user rejected old newradio_00001"),
        (234.945, 237.915, "user rejected old newradio_00003"),
        (249.000, 251.000, "user rejected old newradio_00004"),
    ],
}

EXCLUDED_BASENAMES = {
    "28055832533-1-192_vocals_0013268480_0013539200_safe_word_000.wav",
    "28055832533-1-192_vocals_0013268480_0013539200_safe_word_001.wav",
}

EXPLICIT_HARD_NEGATIVES = [
    ROOT / "new_radio_quality_v1" / "accepted_segments" / "newradio_00001_243358894-1-192_vocals_0003417760_0003505760.wav",
    ROOT / "new_radio_quality_v1" / "accepted_segments" / "newradio_00003_243358894-1-192_vocals_0007518240_0007613280.wav",
    ROOT / "new_radio_quality_v1" / "accepted_segments" / "newradio_00004_243358894-1-192_vocals_0007968000_0008032000.wav",
]


def norm_rows(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x.float(), dim=-1)


def overlap(a: float, b: float, c: float, d: float) -> bool:
    return max(a, c) < min(b, d)


def safe_text(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z/u3040-/u30ff/u4e00-/u9fff]", "_", text)[:32] or "speech"


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1), int(sr)


def resample_1d(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return np.asarray(audio, dtype=np.float32)
    tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32))
    out = torchaudio.functional.resample(tensor, source_sr, target_sr)
    return out.cpu().numpy().astype(np.float32, copy=False)


def segment(audio: np.ndarray, sr: int, start: float, end: float, length_sec: float | None = None) -> np.ndarray:
    if length_sec is not None:
        center = (start + end) / 2.0
        start = center - length_sec / 2.0
        end = start + length_sec
    n = max(1, int(round((end - start) * sr)))
    left = max(0, int(round(start * sr)))
    right = min(len(audio), left + n)
    data = audio[left:right]
    if len(data) < n:
        data = np.pad(data, (0, n - len(data)))
    return np.asarray(data[:n], dtype=np.float32)


def rms(audio: np.ndarray) -> float:
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + 1e-12))


def parse_reference_paths() -> list[Path]:
    if not REF_LIST.exists():
        raise FileNotFoundError(REF_LIST)
    paths = []
    for raw in REF_LIST.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        path_text = raw.split("|", 1)[0].strip()
        path = Path(path_text)
        if path.exists():
            paths.append(path)
    if not paths:
        raise RuntimeError("No confirmed pure target references exist")
    return paths


def parse_hard_negative_paths() -> list[Path]:
    """Load only user-confirmed non-target examples, plus explicit false clips."""
    paths: list[Path] = []
    negative_list = ROOT / "confirmed_non_target_or_mixed.txt"
    if negative_list.exists():
        for raw in negative_list.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw or raw.startswith("#") or "|" not in raw:
                continue
            path_text, reason = [part.strip() for part in raw.split("|", 1)]
            reason_lower = reason.lower()
            if "mixed" in reason_lower or "娣?" in reason_lower:
                continue
            if "non-target" in reason_lower or "非目标" in reason_lower:
                path = Path(path_text)
                if path.exists():
                    paths.append(path)
    paths.extend(path for path in EXPLICIT_HARD_NEGATIVES if path.exists())
    result = []
    seen = set()
    for path in paths:
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def parse_forbidden_intervals(source_name: str) -> list[dict]:
    return [
        {"start": float(a), "end": float(b), "reason": reason}
        for a, b, reason in HARD_EXCLUSIONS.get(source_name, [])
    ]


def make_asr(model: WhisperModel, path: Path) -> tuple[list[dict], list[dict]]:
    segments, _ = model.transcribe(
        str(path),
        language="ja",
        beam_size=5,
        temperature=0.0,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=180),
        condition_on_previous_text=False,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
    )
    words: list[dict] = []
    speech_segments: list[dict] = []
    for seg in segments:
        seg_start = float(seg.start)
        seg_end = float(seg.end)
        if seg_end <= seg_start:
            continue
        seg_words = []
        for word in seg.words or []:
            text = (word.word or "").strip()
            if not text or word.end <= word.start:
                continue
            row = {
                "text": text,
                "start": float(word.start),
                "end": float(word.end),
                "probability": float(getattr(word, "probability", 0.0)),
            }
            words.append(row)
            seg_words.append(row)
        speech_segments.append({"start": seg_start, "end": seg_end, "word_count": len(seg_words)})
    return words, speech_segments


def window_specs(speech_segments: list[dict], duration: float) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for row in speech_segments:
        a = max(0.0, row["start"] - 0.08)
        b = min(duration, row["end"] + 0.08)
        if b <= a:
            continue
        if b - a <= CAM_WINDOW_SEC:
            center = (a + b) / 2.0
            result.append((max(0.0, center - CAM_WINDOW_SEC / 2), min(duration, center + CAM_WINDOW_SEC / 2)))
            continue
        start = a
        while start + CAM_WINDOW_SEC < b - 0.02:
            result.append((start, start + CAM_WINDOW_SEC))
            start += CAM_HOP_SEC
        result.append((max(a, b - CAM_WINDOW_SEC), b))
    dedup = []
    seen = set()
    for a, b in result:
        key = (round(a, 3), round(b, 3))
        if key not in seen:
            seen.add(key)
            dedup.append((float(a), float(b)))
    return dedup


def cam_embed_batch(cam: AutoModel, waves: list[np.ndarray]) -> torch.Tensor:
    if not waves:
        return torch.zeros((0, 192), dtype=torch.float32)
    output: list[torch.Tensor] = []
    for start in range(0, len(waves), CAM_BATCH):
        batch = waves[start : start + CAM_BATCH]
        results = cam.generate(input=batch, fs=CAM_SR, batch_size=len(batch), disable_pbar=True)
        emb = results[0]["spk_embedding"]
        if not isinstance(emb, torch.Tensor):
            emb = torch.as_tensor(emb)
        if emb.ndim == 1:
            emb = emb.unsqueeze(0)
        output.append(norm_rows(emb.detach().cpu()))
    return torch.cat(output, dim=0)


def load_or_make_hard_negative_embeddings(cam: AutoModel, paths: list[Path]) -> tuple[torch.Tensor, list[str]]:
    cache_path = EMBED_CACHE / "hard_negatives.npz"
    path_names = [str(path) for path in paths]
    if cache_path.exists():
        cached = np.load(str(cache_path), allow_pickle=True)
        cached_paths = [str(value) for value in cached["paths"].tolist()]
        if cached_paths == path_names:
            return norm_rows(torch.from_numpy(cached["embeddings"])), cached_paths
    waves: list[np.ndarray] = []
    labels: list[str] = []
    for path in paths:
        audio, sr = read_mono(path)
        audio16 = resample_1d(audio, sr, CAM_SR)
        duration = len(audio16) / CAM_SR
        starts = [0.0] if duration <= CAM_WINDOW_SEC else list(np.arange(0.0, duration - CAM_WINDOW_SEC + 1e-5, 0.60))
        if duration > CAM_WINDOW_SEC and starts[-1] + CAM_WINDOW_SEC < duration:
            starts.append(duration - CAM_WINDOW_SEC)
        for start in starts[:64]:
            waves.append(segment(audio16, CAM_SR, float(start), float(start + CAM_WINDOW_SEC), CAM_WINDOW_SEC))
            labels.append(str(path))
    if not waves:
        return torch.zeros((0, 192), dtype=torch.float32), []
    embs = cam_embed_batch(cam, waves)
    np.savez_compressed(str(cache_path), embeddings=embs.numpy().astype(np.float32), paths=np.asarray(labels))
    return embs, labels


def load_or_make_embeddings(
    cam: AutoModel,
    audio16: np.ndarray,
    source: Path,
    specs: list[tuple[float, float]],
) -> torch.Tensor:
    cache_path = EMBED_CACHE / f"{source.stem}.npz"
    key = np.asarray([[a, b] for a, b in specs], dtype=np.float32)
    if cache_path.exists():
        cached = np.load(str(cache_path))
        cached_specs = cached["specs"]
        cached_embs = torch.from_numpy(cached["embeddings"])
        if cached_specs.shape == key.shape and np.allclose(cached_specs, key, atol=1e-3):
            print(f"  [cache] windows={len(specs)}", flush=True)
            return norm_rows(cached_embs)
    waves = [segment(audio16, CAM_SR, a, b, CAM_WINDOW_SEC) for a, b in specs]
    embs = cam_embed_batch(cam, waves)
    np.savez_compressed(str(cache_path), specs=key, embeddings=embs.numpy().astype(np.float32))
    return embs


def load_reference_embeddings(cam: AutoModel, refs: list[Path]) -> torch.Tensor:
    cache_path = EMBED_CACHE / "target_references.npz"
    if cache_path.exists():
        return norm_rows(torch.from_numpy(np.load(str(cache_path))["embeddings"]))
    waves = []
    kept = []
    for path in refs:
        audio, sr = read_mono(path)
        if len(audio) < int(0.8 * sr):
            continue
        waves.append(resample_1d(audio, sr, CAM_SR))
        kept.append(path)
    embs = cam_embed_batch(cam, [segment(w, CAM_SR, 0, len(w) / CAM_SR, CAM_WINDOW_SEC) for w in waves])
    np.savez_compressed(str(cache_path), embeddings=embs.numpy().astype(np.float32), paths=np.asarray([str(p) for p in kept]))
    return embs


def estimate_k(speech_segments: list[dict]) -> int:
    speech_sec = sum(max(0.0, x["end"] - x["start"]) for x in speech_segments)
    # Long radio dramas commonly contain several characters.  This floor
    # prevents a two-cluster fit from absorbing a distinct non-target voice.
    return int(min(8, max(3, 3 + round(speech_sec / 480.0))))


def cluster_windows(embs: torch.Tensor, refs: torch.Tensor, k: int) -> dict:
    if len(embs) < 2:
        return {"labels": np.zeros(len(embs), dtype=np.int32), "centers": embs.numpy(), "target_cluster": 0, "review_clusters": []}
    k = min(k, len(embs))
    km = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=100)
    labels = km.fit_predict(embs.numpy())
    centers = norm_rows(torch.from_numpy(km.cluster_centers_)).numpy()
    ref_scores = centers @ refs.numpy().T
    top_count = min(4, ref_scores.shape[1])
    target_scores = np.sort(ref_scores, axis=1)[:, -top_count:].mean(axis=1)
    best = int(np.argmax(target_scores))
    order = np.argsort(-target_scores)
    review = [int(idx) for idx in order[1:] if target_scores[idx] >= target_scores[best] - 0.045 and target_scores[idx] >= TARGET_SIM_MIN]
    counts = np.bincount(labels, minlength=k)
    return {
        "labels": labels,
        "centers": centers,
        "target_cluster": best,
        "review_clusters": review,
        "target_scores": target_scores,
        "reference_scores": ref_scores,
        "cluster_counts": counts,
        "k": k,
    }


def windows_to_rows(specs: list[tuple[float, float]], embs: torch.Tensor, clustering: dict, refs: torch.Tensor, hard_negatives: torch.Tensor) -> list[dict]:
    labels = clustering["labels"]
    target_cluster = clustering["target_cluster"]
    review_clusters = set(clustering.get("review_clusters", []))
    ref_centroid = norm_rows(refs.mean(dim=0, keepdim=True))[0]
    rows = []
    for idx, ((a, b), emb) in enumerate(zip(specs, embs)):
        cluster = int(labels[idx])
        target_sim = float(torch.max(emb @ refs.T).item())
        centroid_sim = float((emb @ ref_centroid).item())
        hard_negative_sim = float(torch.max(emb @ hard_negatives.T).item()) if len(hard_negatives) else -1.0
        hard_negative = hard_negative_sim >= HARD_NEGATIVE_SIM_VETO
        tier = "target" if cluster == target_cluster and target_sim >= TARGET_SIM_MIN and not hard_negative else "review" if cluster in review_clusters and target_sim >= REVIEW_SIM_MIN and not hard_negative else "other"
        rows.append({
            "index": idx,
            "start": float(a),
            "end": float(b),
            "cluster": cluster,
            "target_sim": target_sim,
            "centroid_sim": centroid_sim,
            "hard_negative_sim": hard_negative_sim,
            "hard_negative": hard_negative,
            "tier": tier,
        })
    return rows


def vote_for_word(word: dict, window_rows: list[dict]) -> dict:
    mid = (word["start"] + word["end"]) / 2.0
    candidates = [row for row in window_rows if row["start"] <= mid <= row["end"]]
    if not candidates:
        return {"target_vote": 0.0, "target_count": 0, "other_count": 0, "best_sim": -1.0, "boundary": True}
    target = [row for row in candidates if row["tier"] == "target"]
    other = [row for row in candidates if row["tier"] == "other"]
    best_sim = max(row["target_sim"] for row in candidates)
    target_vote = len(target) / len(candidates)
    boundary = bool(other) or len(target) < 2
    return {
        "target_vote": target_vote,
        "target_count": len(target),
        "other_count": len(other),
        "best_sim": best_sim,
        "boundary": boundary,
    }


def within_hard_exclusion(source_name: str, start: float, end: float, forbidden: list[dict]) -> str | None:
    for row in forbidden:
        if overlap(start, end, row["start"], row["end"]):
            return row["reason"]
    return None


def write_clip(audio: np.ndarray, sr: int, start: float, end: float, path: Path) -> None:
    raw = segment(audio, sr, start, end)
    out = resample_1d(raw, sr, OUTPUT_SR)
    sf.write(str(path), out, OUTPUT_SR, subtype="PCM_16")


def source_other_ratio(audio: np.ndarray, other: np.ndarray | None, sr: int, start: float, end: float) -> float:
    if other is None:
        return 0.0
    a = max(0, int(round(start * sr)))
    b = min(len(audio), len(other), int(round(end * sr)))
    if b <= a:
        return 0.0
    return rms(other[a:b]) / max(rms(audio[a:b]), 1e-6)


def group_words(words: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    current: list[dict] = []
    for word in words:
        if not current or word["start"] - current[-1]["end"] <= WORD_MERGE_GAP_SEC:
            if current and word["end"] - current[0]["start"] > MAX_CLIP_SEC:
                groups.append(current)
                current = [word]
            else:
                current.append(word)
        else:
            groups.append(current)
            current = [word]
    if current:
        groups.append(current)
    return groups


def validate(lines: list[str]) -> dict:
    paths = [Path(line.split("|", 1)[0]) for line in lines]
    missing = [str(p) for p in paths if not p.exists()]
    duplicate = [p for p, count in Counter(str(x) for x in paths).items() if count > 1]
    short = []
    forbidden_names = []
    for path in paths:
        if path.exists():
            duration = sf.info(str(path)).duration
            if duration < MIN_CLIP_SEC:
                short.append({"path": str(path), "duration": duration})
            if path.name in EXCLUDED_BASENAMES:
                forbidden_names.append(str(path))
    return {
        "candidate_count": len(lines),
        "missing_paths": missing,
        "duplicate_paths": duplicate,
        "short_outputs": short,
        "forbidden_names": forbidden_names,
        "passed": not missing and not duplicate and not short and not forbidden_names,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="process only one vocals filename")
    parser.add_argument("--limit", type=int, default=0, help="process at most N sources")
    args = parser.parse_args()

    for path in (VOCALS, OTHER, CAM_MODEL):
        if not path.exists():
            raise FileNotFoundError(path)
    for path in (OUT, ACCEPTED, REVIEW, REJECTED, ASR_CACHE, EMBED_CACHE):
        path.mkdir(parents=True, exist_ok=True)

    source_paths = sorted(VOCALS.glob("*_vocals.wav"))
    if args.source:
        source_paths = [p for p in source_paths if p.name == args.source]
    if args.limit:
        source_paths = source_paths[: args.limit]
    if not source_paths:
        raise RuntimeError("No input vocals found")

    os.chdir(PKG)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cam = AutoModel(model=str(CAM_MODEL), device=device, check_latest=False, disable_pbar=True)
    refs = parse_reference_paths()
    ref_embs = load_reference_embeddings(cam, refs)
    hard_negative_paths = parse_hard_negative_paths()
    hard_negative_embs, hard_negative_labels = load_or_make_hard_negative_embeddings(cam, hard_negative_paths)
    asr = WhisperModel(
        str(PKG / "tools" / "asr" / "models" / "faster-whisper-large-v3-turbo"),
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_index=0 if torch.cuda.is_available() else None,
        compute_type="float16" if torch.cuda.is_available() else "int8",
        cpu_threads=8,
    )

    report = {
        "version": "full_speaker_diarize_new_v3",
        "method": "CAM++ clustering + hard-negative gate + ASR word gate",
        "clearvoice_source": str(ROOT.parent / "ClearerVoice-Studio-main"),
        "training_waveform": "original BS-Roformer vocals; one final 32 kHz export resample",
        "reference_paths": [str(p) for p in refs],
        "reference_count": len(refs),
        "hard_negative_paths": [str(p) for p in hard_negative_paths],
        "hard_negative_reference_count": len(hard_negative_paths),
        "hard_negative_window_count": len(hard_negative_labels),
        "cam_model": str(CAM_MODEL),
        "parameters": {
            "cam_window_sec": CAM_WINDOW_SEC,
            "cam_hop_sec": CAM_HOP_SEC,
            "word_window_sec": WORD_WINDOW_SEC,
            "target_vote_min": TARGET_VOTE_MIN,
            "review_vote_min": REVIEW_VOTE_MIN,
            "target_sim_min": TARGET_SIM_MIN,
            "target_sim_high": TARGET_SIM_HIGH,
            "review_sim_min": REVIEW_SIM_MIN,
            "hard_negative_sim_veto": HARD_NEGATIVE_SIM_VETO,
            "min_clip_sec": MIN_CLIP_SEC,
            "max_clip_sec": MAX_CLIP_SEC,
        },
        "sources": [],
    }
    candidate_lines: list[str] = []
    review_lines: list[str] = []
    clip_index = 0

    for source in source_paths:
        started = time.time()
        print(f"[source] {source.name}", flush=True)
        audio, sr = read_mono(source)
        duration = len(audio) / sr
        other_path = OTHER / source.name.replace("_vocals.wav", "_other.wav")
        other = read_mono(other_path)[0] if other_path.exists() else None
        asr_cache_path = ASR_CACHE / f"{source.stem}.json"
        if asr_cache_path.exists():
            asr_data = json.loads(asr_cache_path.read_text(encoding="utf-8"))
            words = asr_data["words"]
            speech_segments = asr_data["speech_segments"]
            print(f"  [cache] asr words={len(words)}", flush=True)
        else:
            words, speech_segments = make_asr(asr, source)
            asr_cache_path.write_text(json.dumps({"words": words, "speech_segments": speech_segments}, ensure_ascii=False), encoding="utf-8")
            print(f"  [asr] words={len(words)} segments={len(speech_segments)}", flush=True)
        audio16 = resample_1d(audio, sr, CAM_SR)
        specs = window_specs(speech_segments, duration)
        window_embs = load_or_make_embeddings(cam, audio16, source, specs)
        clustering = cluster_windows(window_embs, ref_embs, estimate_k(speech_segments))
        window_rows = windows_to_rows(specs, window_embs, clustering, ref_embs, hard_negative_embs)
        target_cluster = int(clustering["target_cluster"])
        print(f"  [cluster] windows={len(specs)} k={clustering.get('k', 1)} target_cluster={target_cluster} review_clusters={clustering.get('review_clusters', [])}", flush=True)

        forbidden = parse_forbidden_intervals(source.name)
        word_rows = []
        counters = Counter()
        scored_words = [word for word in words if word["end"] - word["start"] >= MIN_WORD_SEC]
        word_waves = [segment(audio16, CAM_SR, word["start"], word["end"], WORD_WINDOW_SEC) for word in scored_words]
        word_embs = cam_embed_batch(cam, word_waves)
        for word, word_emb in zip(scored_words, word_embs):
            exclusion = within_hard_exclusion(source.name, word["start"], word["end"], forbidden)
            vote = vote_for_word(word, window_rows)
            ref_sim = float(torch.max(word_emb @ ref_embs.T).item())
            hard_negative_sim = float(torch.max(word_emb @ hard_negative_embs.T).item()) if len(hard_negative_embs) else -1.0
            row = {**word, **vote, "reference_sim": ref_sim, "hard_exclusion": exclusion}
            row["hard_negative_sim"] = hard_negative_sim
            if exclusion:
                row["decision"] = "rejected_hard_exclusion"
                counters["rejected_hard_exclusion"] += 1
            elif hard_negative_sim >= HARD_NEGATIVE_SIM_VETO:
                row["decision"] = "rejected_hard_negative"
                counters["rejected_hard_negative"] += 1
            elif vote["target_vote"] >= TARGET_VOTE_MIN and vote["target_count"] >= 2 and vote["other_count"] == 0 and ref_sim >= TARGET_SIM_MIN:
                row["decision"] = "accepted_word"
                counters["accepted_word"] += 1
            elif vote["target_vote"] >= REVIEW_VOTE_MIN and ref_sim >= REVIEW_SIM_MIN:
                row["decision"] = "review_word"
                counters["review_word"] += 1
            else:
                row["decision"] = "rejected_non_target_or_boundary"
                counters["rejected_non_target_or_boundary"] += 1
            word_rows.append(row)

        accepted_words = [row for row in word_rows if row["decision"] == "accepted_word"]
        review_words = [row for row in word_rows if row["decision"] == "review_word"]
        source_outputs = []
        source_reviews = []
        for tier_name, tier_words, output_dir, list_target in (("accepted", accepted_words, ACCEPTED, candidate_lines), ("review", review_words, REVIEW, review_lines)):
            for group in group_words(tier_words):
                a = max(0.0, group[0]["start"] - WORD_EDGE_PAD_SEC)
                b = min(duration, group[-1]["end"] + WORD_EDGE_PAD_SEC)
                if b - a < MIN_CLIP_SEC:
                    counters[f"{tier_name}_short_group"] += 1
                    continue
                exclusion = within_hard_exclusion(source.name, a, b, forbidden)
                if exclusion:
                    counters[f"{tier_name}_hard_exclusion"] += 1
                    continue
                ratio = source_other_ratio(audio, other, sr, a, b)
                if ratio > NEGATIVE_OTHER_RATIO:
                    counters[f"{tier_name}_other_dominant"] += 1
                    continue
                name = f"newradio_{clip_index:05d}_{source.stem}_{int(round(a * OUTPUT_SR)):010d}_{int(round(b * OUTPUT_SR)):010d}.wav"
                output_path = output_dir / name
                write_clip(audio, sr, a, b, output_path)
                line = f"{output_path}|full_speaker_diarize_new_v3|JA|{''.join(w['text'] for w in group)}"
                list_target.append(line)
                clip_row = {
                    "file": str(output_path),
                    "source": str(source),
                    "start": a,
                    "end": b,
                    "duration": b - a,
                    "text": "".join(w["text"] for w in group),
                    "other_ratio": ratio,
                    "word_count": len(group),
                    "min_reference_sim": min(w["reference_sim"] for w in group),
                    "min_target_vote": min(w["target_vote"] for w in group),
                    "tier": tier_name,
                }
                if tier_name == "accepted":
                    source_outputs.append(clip_row)
                else:
                    source_reviews.append(clip_row)
                clip_index += 1

        source_report = {
            "source": str(source),
            "other_source": str(other_path) if other_path.exists() else None,
            "duration_sec": duration,
            "sample_rate": sr,
            "asr_word_count": len(words),
            "speech_segment_count": len(speech_segments),
            "window_count": len(window_rows),
            "cluster_count": int(clustering.get("k", 1)),
            "target_cluster": target_cluster,
            "review_clusters": clustering.get("review_clusters", []),
            "cluster_counts": clustering.get("cluster_counts", []).tolist() if hasattr(clustering.get("cluster_counts", []), "tolist") else clustering.get("cluster_counts", []),
            "cluster_target_scores": [float(x) for x in clustering.get("target_scores", [])],
            "hard_exclusions": forbidden,
            "counters": dict(counters),
            "accepted_segments": source_outputs,
            "review_segments": source_reviews,
            "word_rows": word_rows,
            "window_rows": window_rows,
            "elapsed_sec": time.time() - started,
        }
        report["sources"].append(source_report)
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [done] accepted={len(source_outputs)} ({sum(x['duration'] for x in source_outputs):.1f}s) review={len(source_reviews)} ({sum(x['duration'] for x in source_reviews):.1f}s) elapsed={time.time()-started:.1f}s", flush=True)

    CANDIDATE_LIST.write_text("/n".join(candidate_lines) + ("/n" if candidate_lines else ""), encoding="utf-8")
    REVIEW_LIST.write_text("/n".join(review_lines) + ("/n" if review_lines else ""), encoding="utf-8")
    report["candidate_validation"] = validate(candidate_lines)
    report["candidate_total_sec"] = sum(sf.info(line.split("|", 1)[0]).duration for line in candidate_lines if Path(line.split("|", 1)[0]).exists())
    report["review_total_sec"] = sum(sf.info(line.split("|", 1)[0]).duration for line in review_lines if Path(line.split("|", 1)[0]).exists())
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"candidate_list": str(CANDIDATE_LIST), "candidate_count": len(candidate_lines), "candidate_sec": report["candidate_total_sec"], "review_count": len(review_lines), "review_sec": report["review_total_sec"], "validation": report["candidate_validation"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
