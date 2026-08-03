# -*- coding: utf-8 -*-
"""Resumable local full-track target-speaker mining pipeline.

This is intentionally versioned and does not modify any previous list or
start training.  It keeps the original BS-Roformer vocal waveform and uses
ERes2Net, CAM++, ASR, NISQA-TTS, and acoustic vetoes as separate signals.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import gc
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

BOOTSTRAP_DIR = Path(__file__).resolve().parent
if str(BOOTSTRAP_DIR) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_DIR))
from local_resource_policy import apply_cpu_friendly_policy

RESOURCE_POLICY = apply_cpu_friendly_policy(2)

import numpy as np
import soundfile as sf
import torch
import torchaudio


ROOT = Path(__file__).resolve().parent
PKG = ROOT.parent / "GPT-SoVITS-v2pro-20250604-nvidia50" / "GPT-SoVITS-v2pro-20250604-nvidia50"
SPEECHSCORE = ROOT.parent / "ClearerVoice-Studio-main" / "ClearerVoice-Studio-main" / "speechscore"
MANIFEST_DEFAULT = ROOT / "full_track_manifest_v1.json"
OUT = ROOT / "consensus_v1"
SCAN_OUT = ROOT / "full_track_scan_v1"
WORD_OUT = ROOT / "word_target_all_v1"
VERIFIED_LIST = ROOT / "verified_machine.list"

SR = 32000
CAM_SR = 16000
SCAN_WIN = 1.0
SCAN_HOP = 0.25
WORD_WIN = 0.8
WORD_HOP = 0.2
BLOCK_SEC = 30.0
BLOCK_STEP = 25.0
MIN_RUN_WINDOWS = 3
MIN_SEG_SEC = 1.5
MAX_SEG_SEC = 6.0
MERGE_GAP = 0.35
EDGE_PAD = 0.12

SCAN_THRESH = {
    "center": 0.79,
    "median": 0.64,
    "p25": 0.56,
    "margin": 0.08,
    "rms": 0.004,
}
WORD_THRESH = {
    "vote": 0.60,
    "cluster_margin": 0.025,
    "frame_target": 0.60,
    "direct_center": 0.68,
    "direct_p25": 0.43,
    "margin": 0.06,
}


def atomic_write(path: Path, value: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if isinstance(value, bytes):
        tmp.write_bytes(value)
    else:
        tmp.write_text(value, encoding="utf-8")
    tmp.replace(path)


def write_json(path: Path, value: dict) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2))


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(paths: list[Path], params: dict) -> str:
    rows = []
    for path in paths:
        stat = path.stat()
        rows.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    raw = json.dumps({"paths": rows, "params": params}, sort_keys=True).encode()
    return hashlib.sha1(raw).hexdigest()


def overlap(a: float, b: float, c: float, d: float) -> bool:
    return max(a, c) < min(b, d)


def read_mono(path: Path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1).astype(np.float32, copy=False)
    if target_sr and sr != target_sr:
        audio = torchaudio.functional.resample(torch.from_numpy(audio), sr, target_sr).numpy()
        sr = target_sr
    return audio, int(sr)


def save_audio(path: Path, audio: np.ndarray, sr: int) -> None:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    sf.write(str(path), audio, sr, subtype="PCM_16")


def safe_text(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z/u3040-/u30ff/u4e00-/u9fff]", "_", text)[:36] or "speech"


def load_manifest(path: Path) -> dict:
    manifest = load_json(path)
    if not manifest or len(manifest.get("tracks", [])) != 12:
        raise RuntimeError("Manifest must contain exactly 12 tracks")
    for track in manifest["tracks"]:
        vocals = Path(track["vocals"])
        if not vocals.exists():
            raise FileNotFoundError(vocals)
    return manifest


def track_map(manifest: dict) -> dict[str, dict]:
    return {track["id"]: track for track in manifest["tracks"]}


def exclusions_for(manifest: dict, source_id: str) -> list[dict]:
    return [row for row in manifest.get("hard_exclusions", []) if row["source_id"] == source_id]


def find_interval_exclusion(manifest: dict, source_id: str, start: float, end: float) -> str | None:
    for row in exclusions_for(manifest, source_id):
        if overlap(start, end, float(row["start"]), float(row["end"])):
            return str(row["reason"])
    return None


def load_eres2net():
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import word_target_hardcut as e  # loads the validated ERes2Net model once

    return e


def embed_audio_batch(engine, waves: list[torch.Tensor], batch_size: int = 48) -> torch.Tensor:
    if not waves:
        return torch.zeros((0, 192), dtype=torch.float32)
    result = []
    for start in range(0, len(waves), batch_size):
        block = waves[start : start + batch_size]
        max_len = max(int(wave.shape[1]) for wave in block)
        padded = [torch.nn.functional.pad(wave, (0, max_len - int(wave.shape[1]))) for wave in block]
        result.append(engine.embed_batch(torch.cat(padded, dim=0)))
    return torch.cat(result, dim=0)


def fixed_window_wave(wav: torch.Tensor, center_sec: float, length_sec: float) -> torch.Tensor:
    length = int(round(length_sec * SR))
    center = int(round(center_sec * SR))
    start, end = center - length // 2, center + (length - length // 2)
    left, right = max(0, -start), max(0, end - wav.shape[1])
    clip = wav[:, max(0, start) : min(wav.shape[1], end)]
    if left or right:
        clip = torch.nn.functional.pad(clip, (left, right))
    return torch.nn.functional.pad(clip[:, :length], (0, max(0, length - clip.shape[1])))[:, :length]


def load_eres_refs(engine) -> tuple[torch.Tensor, list[Path]]:
    SCAN_OUT.mkdir(parents=True, exist_ok=True)
    cache = SCAN_OUT / "target_refs_eres2net_v1.npz"
    paths = engine.load_refs()
    key = fingerprint(paths, {"model": "ERes2NetV2", "sr": SR})
    if cache.exists():
        data = np.load(str(cache), allow_pickle=True)
        if str(data["fingerprint"].item()) == key:
            return torch.from_numpy(data["embeddings"]).float(), paths
    waves = []
    kept = []
    for path in paths:
        wav, sr = torchaudio.load(str(path))
        wav = wav.mean(0, keepdim=True)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        if wav.shape[1] >= int(0.8 * SR):
            waves.append(wav)
            kept.append(path)
    # The historical references have very different durations.  Padding a
    # long reference to the longest file in a large batch can consume several
    # GiB, so encode them one at a time.
    embeddings = embed_audio_batch(engine, waves, batch_size=1)
    np.savez_compressed(str(cache), embeddings=embeddings.numpy(), paths=np.asarray([str(x) for x in kept]), fingerprint=np.asarray(key))
    return embeddings, kept


def load_eres_negatives(engine, manifest: dict) -> tuple[torch.Tensor, list[dict]]:
    SCAN_OUT.mkdir(parents=True, exist_ok=True)
    cache = SCAN_OUT / "negative_windows_eres2net_v2.npz"
    paths = [Path(x) for x in manifest.get("negative_files", []) if Path(x).exists()]
    key = fingerprint(paths, {"model": "ERes2NetV2", "window": 1.5, "hop": 0.60, "sr": SR, "window_impl": "fixed_v2"})
    if cache.exists():
        data = np.load(str(cache), allow_pickle=True)
        if str(data["fingerprint"].item()) == key:
            rows = json.loads(str(data["rows_json"].item()))
            return torch.from_numpy(data["embeddings"]).float(), rows
    waves = []
    rows = []
    for path in paths:
        wav, sr = read_mono(path, SR)
        duration = len(wav) / SR
        starts = [0.0] if duration <= 1.5 else list(np.arange(0.0, duration - 1.5 + 1e-5, 0.60))
        if duration > 1.5 and starts[-1] + 1.5 < duration:
            starts.append(duration - 1.5)
        for start in starts[:64]:
            tensor = torch.from_numpy(wav).unsqueeze(0)
            center = float(start + 0.75)
            waves.append(fixed_window_wave(tensor, center, 1.5))
            rows.append({"path": str(path), "start": float(start), "end": float(start + 1.5)})
    embeddings = embed_audio_batch(engine, waves)
    np.savez_compressed(str(cache), embeddings=embeddings.numpy(), rows_json=np.asarray(json.dumps(rows)), fingerprint=np.asarray(key))
    return embeddings, rows


def active_from_metrics(center, median, p25, margin, rms):
    return bool(
        center >= SCAN_THRESH["center"]
        and median >= SCAN_THRESH["median"]
        and p25 >= SCAN_THRESH["p25"]
        and margin >= SCAN_THRESH["margin"]
        and rms >= SCAN_THRESH["rms"]
    )


def make_scan_runs(rows: list[dict]) -> list[dict]:
    runs = []
    start = None
    for idx, row in enumerate(rows + [{"active": False}]):
        if row.get("active") and start is None:
            start = idx
        if not row.get("active") and start is not None:
            end = idx
            if end - start >= MIN_RUN_WINDOWS:
                prev_center = rows[start - 1]["center_sec"] if start else None
                first_center = rows[start]["center_sec"]
                last_center = rows[end - 1]["center_sec"]
                next_center = rows[end]["center_sec"] if end < len(rows) else None
                lo = 0.0 if prev_center is None else (prev_center + first_center) / 2.0
                hi = rows[-1]["end_sec"] if next_center is None else (last_center + next_center) / 2.0
                lo += 0.05
                hi -= 0.05
                if hi > lo:
                    runs.append({
                        "start": round(lo, 4),
                        "end": round(hi, 4),
                        "duration": round(hi - lo, 4),
                        "window_start": start,
                        "window_end": end,
                        "window_count": end - start,
                        "min_center": min(x["center"] for x in rows[start:end]),
                        "min_p25": min(x["p25"] for x in rows[start:end]),
                        "min_margin": min(x["margin"] for x in rows[start:end]),
                    })
            start = None
    return runs


def phase_a(manifest: dict, resume: bool) -> dict:
    engine = load_eres2net()
    refs, ref_paths = load_eres_refs(engine)
    negatives, negative_rows = load_eres_negatives(engine, manifest)
    ref_center = torch.nn.functional.normalize(refs.mean(0), dim=-1)
    ref_scores = refs @ refs.T
    positive_rows = []
    for idx, path in enumerate(ref_paths):
        peer = torch.cat([ref_scores[idx, :idx], ref_scores[idx, idx + 1 :]])
        center = float(refs[idx] @ ref_center)
        p25 = float(torch.quantile(peer, 0.25)) if len(peer) else center
        neg_max = float(torch.max(refs[idx] @ negatives.T)) if len(negatives) else -1.0
        positive_rows.append({"path": str(path), "center": center, "p25": p25, "negative_max": neg_max, "margin": center - neg_max, "pass": bool(center >= SCAN_THRESH["center"] and p25 >= SCAN_THRESH["p25"] and center - neg_max >= SCAN_THRESH["margin"])})
    negative_calibration = []
    for idx, emb in enumerate(negatives):
        neg_sim = negatives[idx] @ negatives.T
        peer = torch.cat([neg_sim[:idx], neg_sim[idx + 1 :]])
        neg_max = float(torch.max(peer)) if len(peer) else float(neg_sim[idx])
        target = float(emb @ ref_center)
        negative_calibration.append({"index": idx, "target_center": target, "negative_max": neg_max, "margin": target - neg_max, "hard_rejected": bool(target - neg_max <= 0.0 or neg_max >= target)})
    calibration = {"positive_reference_count": len(positive_rows), "positive_pass_count": sum(x["pass"] for x in positive_rows), "positive_pass_rate": sum(x["pass"] for x in positive_rows) / max(1, len(positive_rows)), "negative_window_count": len(negative_calibration), "negative_hard_reject_count": sum(x["hard_rejected"] for x in negative_calibration), "positive_rows": positive_rows, "negative_rows": negative_calibration}
    report = {"version": "full_track_scan_v1", "parameters": {"window_sec": SCAN_WIN, "hop_sec": SCAN_HOP, **SCAN_THRESH}, "reference_count": len(ref_paths), "negative_window_count": len(negative_rows), "calibration": calibration, "sources": []}
    for track in manifest["tracks"]:
        source_id = track["id"]
        source = Path(track["vocals"])
        out = SCAN_OUT / "source_reports" / f"{source_id}.json"
        params = {"source": str(source), "window": SCAN_WIN, "hop": SCAN_HOP, **SCAN_THRESH}
        fp = fingerprint([source], params)
        old = load_json(out)
        if resume and old and old.get("fingerprint") == fp:
            report["sources"].append(old)
            print(f"[A cache] {source_id} runs={len(old.get('intervals', []))}", flush=True)
            continue
        wav, sr = read_mono(source, SR)
        tensor = torch.from_numpy(wav).unsqueeze(0)
        n = len(wav)
        win = int(round(SCAN_WIN * SR))
        hop = int(round(SCAN_HOP * SR))
        starts = list(range(0, max(1, n - win + 1), hop))
        if n > win and starts[-1] + win < n:
            starts.append(n - win)
        windows = [tensor[:, start : min(n, start + win)] for start in starts]
        windows = [torch.nn.functional.pad(x, (0, max(0, win - x.shape[1]))) for x in windows]
        embeddings = embed_audio_batch(engine, windows)
        sims = embeddings @ refs.T
        neg_sims = embeddings @ negatives.T if len(negatives) else torch.full((len(embeddings), 1), -1.0)
        rows = []
        for idx, start in enumerate(starts):
            vals = sims[idx]
            center = float(embeddings[idx] @ ref_center)
            p25 = float(torch.quantile(vals, 0.25))
            median = float(torch.median(vals))
            neg_max = float(torch.max(neg_sims[idx]))
            rms = float(torch.sqrt(torch.mean(windows[idx] ** 2) + 1e-12))
            row = {"start_sec": start / SR, "end_sec": min(n, start + win) / SR, "center_sec": (start + win / 2) / SR, "center": center, "median": median, "p25": p25, "negative_max": neg_max, "margin": center - neg_max, "rms": rms}
            row["active"] = active_from_metrics(center, median, p25, row["margin"], rms)
            rows.append(row)
        intervals = make_scan_runs(rows)
        item = {"fingerprint": fp, "source_id": source_id, "source": str(source), "duration_sec": len(wav) / SR, "window_count": len(rows), "active_window_count": sum(bool(x["active"]) for x in rows), "intervals": intervals, "windows": rows}
        write_json(out, item)
        report["sources"].append(item)
        print(f"[A done] {source_id} windows={len(rows)} intervals={len(intervals)} active={item['active_window_count']}", flush=True)
        del wav, tensor, windows, embeddings, sims, neg_sims
        gc.collect()
    write_json(SCAN_OUT / "interval_map.json", report)
    return report


def asr_transcribe(model, source: Path) -> list[dict]:
    segments, _ = model.transcribe(str(source), language="ja", beam_size=5, temperature=0.0, word_timestamps=True, vad_filter=True, vad_parameters={"min_silence_duration_ms": 180}, condition_on_previous_text=False, compression_ratio_threshold=2.4, log_prob_threshold=-1.0, no_speech_threshold=0.6)
    words = []
    for seg in segments:
        for word in seg.words or []:
            text = (word.word or "").strip()
            start, end = float(word.start), float(word.end)
            if text and end > start and "/ufffd" not in text:
                words.append({"text": text, "start": start, "end": end, "probability": float(getattr(word, "probability", 0.0))})
    return words


def phase_b_asr(manifest: dict, resume: bool) -> dict:
    from faster_whisper import WhisperModel

    cache_dir = WORD_OUT / "asr_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(str(PKG / "tools" / "asr" / "models" / "faster-whisper-large-v3-turbo"), device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16" if torch.cuda.is_available() else "int8", cpu_threads=2)
    report = {"version": "word_target_all_v1_asr", "model": "faster-whisper-large-v3-turbo", "sources": []}
    for track in manifest["tracks"]:
        source = Path(track["vocals"])
        out = cache_dir / f"{track['id']}.json"
        fp = fingerprint([source], {"model": "large-v3-turbo", "language": "ja", "word_timestamps": True})
        old = load_json(out)
        if resume and old and old.get("fingerprint") == fp and old.get("words"):
            item = old
            print(f"[B ASR cache] {track['id']} words={len(item['words'])}", flush=True)
        else:
            words = asr_transcribe(model, source)
            if any("/ufffd" in str(w["text"]) for w in words):
                raise RuntimeError(f"ASR replacement character in {source}")
            if any(words[i]["start"] > words[i + 1]["start"] for i in range(len(words) - 1)):
                raise RuntimeError(f"ASR timestamps not monotonic in {source}")
            item = {"fingerprint": fp, "source_id": track["id"], "source": str(source), "word_count": len(words), "words": words}
            write_json(out, item)
            print(f"[B ASR done] {track['id']} words={len(words)}", flush=True)
        report["sources"].append(item)
    write_json(WORD_OUT / "asr_report.json", report)
    return report


def spherical_kmeans(x: torch.Tensor, iters: int = 25) -> tuple[torch.Tensor, torch.Tensor]:
    if len(x) < 2:
        return torch.zeros(len(x), dtype=torch.long), x[:1]
    sim = x @ x.T
    i, j = divmod(int(torch.argmin(sim).item()), len(x))
    centers = torch.stack([x[i], x[j]], dim=0)
    for _ in range(iters):
        assignment = (x @ centers.T).argmax(dim=1)
        new = centers.clone()
        for k in range(2):
            part = x[assignment == k]
            if len(part):
                new[k] = torch.nn.functional.normalize(part.mean(0), dim=-1)
        if torch.allclose(new, centers, atol=1e-5):
            break
        centers = new
    return assignment, centers


def run_word_scores(engine, audio: torch.Tensor, words: list[dict], refs: torch.Tensor, negatives: torch.Tensor) -> list[dict]:
    ref_center = torch.nn.functional.normalize(refs.mean(0), dim=-1)
    n = audio.shape[1]
    duration = n / SR
    local_windows = []
    for block_start in np.arange(0.0, max(duration, 0.01), BLOCK_STEP):
        block_end = min(duration, float(block_start + BLOCK_SEC))
        starts = list(np.arange(float(block_start), max(float(block_start), block_end - WORD_WIN) + 1e-5, WORD_HOP))
        if not starts:
            starts = [float(block_start)]
        waves = [engine.window_wave(audio, start + WORD_WIN / 2.0) for start in starts]
        embs = embed_audio_batch(engine, waves)
        voiced = [idx for idx, wave in enumerate(waves) if float(torch.sqrt(torch.mean(wave ** 2) + 1e-12)) >= 0.004]
        labels = torch.full((len(waves),), -1, dtype=torch.long)
        margin = 0.0
        if len(voiced) >= 4:
            assign, centers = spherical_kmeans(embs[voiced])
            scores = centers @ ref_center
            target = int(torch.argmax(scores))
            other = 1 - target if len(scores) > 1 else target
            margin = float(scores[target] - scores[other]) if len(scores) > 1 else 0.0
            for pos, idx in enumerate(voiced):
                labels[idx] = int(assign[pos])
            target_label = target
        else:
            target_label = -1
        for start, emb, label in zip(starts, embs, labels.tolist()):
            vals = emb @ refs.T
            neg_max = float(torch.max(emb @ negatives.T)) if len(negatives) else -1.0
            local_windows.append({"start": float(start), "end": float(min(duration, start + WORD_WIN)), "center": float(start + WORD_WIN / 2.0), "embedding": emb, "label": int(label), "target_label": int(target_label), "cluster_margin": margin, "target_center": float(emb @ ref_center), "target_p25": float(torch.quantile(vals, 0.25)), "negative_max": neg_max})
        if block_end >= duration:
            break
    local_windows.sort(key=lambda x: x["center"])
    centers = [x["center"] for x in local_windows]
    direct_waves = [engine.window_wave(audio, (w["start"] + w["end"]) / 2.0) for w in words]
    direct_embs = embed_audio_batch(engine, direct_waves)
    output = []
    for word, direct in zip(words, direct_embs):
        lo = bisect.bisect_left(centers, word["start"] - 0.28)
        hi = bisect.bisect_right(centers, word["end"] + 0.28)
        neighbors = local_windows[lo:hi]
        labels = [x["label"] for x in neighbors if x["label"] >= 0]
        # target cluster is selected per local block by target score; labels are
        # normalized to target/non-target in the row below using target_center.
        # The scan center threshold (0.79) is intentionally not reused here:
        # Phase B has its own frame_target/direct_center gates.  Reusing the
        # stricter Phase A value made every local vote zero even when the
        # target cluster was correctly identified.
        target_flags = [x["target_center"] >= WORD_THRESH["frame_target"] and x["cluster_margin"] >= WORD_THRESH["cluster_margin"] and x["label"] == x["target_label"] for x in neighbors]
        vote = float(sum(target_flags) / len(neighbors)) if neighbors else 0.0
        target_neighbors = [x for x in neighbors if x["label"] == x["target_label"] and x["target_label"] >= 0]
        frame_target = float(torch.stack([x["embedding"] for x in target_neighbors]).mean(0) @ ref_center) if target_neighbors else 0.0
        vals = direct @ refs.T
        direct_center = float(direct @ ref_center)
        direct_p25 = float(torch.quantile(vals, 0.25))
        neg_max = float(torch.max(direct @ negatives.T)) if len(negatives) else -1.0
        margin = direct_center - neg_max
        cluster_margin = min([x["cluster_margin"] for x in neighbors], default=0.0)
        accepted = bool(vote >= WORD_THRESH["vote"] and cluster_margin >= WORD_THRESH["cluster_margin"] and frame_target >= WORD_THRESH["frame_target"] and direct_center >= WORD_THRESH["direct_center"] and direct_p25 >= WORD_THRESH["direct_p25"] and margin >= WORD_THRESH["margin"])
        hard_veto = bool(margin <= 0.0 or neg_max >= direct_center or not word.get("text") or "/ufffd" in word.get("text", ""))
        output.append({**word, "cluster_vote": vote, "cluster_margin": cluster_margin, "frame_target": frame_target, "direct_center": direct_center, "direct_p25": direct_p25, "negative_max": neg_max, "target_negative_margin": margin, "accepted": accepted, "hard_veto": hard_veto})
    return output


def group_words(words: list[dict]) -> list[list[dict]]:
    groups = []
    for word in words:
        if not groups or word["start"] - groups[-1][-1]["end"] > MERGE_GAP:
            groups.append([word])
        else:
            groups[-1].append(word)
    return groups


def split_group(group: list[dict]) -> list[list[dict]]:
    chunks = []
    current = []
    for word in group:
        if current and word["end"] - current[0]["start"] + 2 * EDGE_PAD > MAX_SEG_SEC:
            chunks.append(current)
            current = []
        current.append(word)
    if current:
        chunks.append(current)
    return chunks


def interval_cover(intervals: list[dict], start: float, end: float) -> bool:
    return any(float(row["start"]) <= start and float(row["end"]) >= end for row in intervals)


def write_segment(source_audio: np.ndarray, source_sr: int, source_id: str, idx: int, group: list[dict], destination: Path) -> dict:
    duration = len(source_audio) / source_sr
    start = max(0.0, group[0]["start"] - EDGE_PAD)
    end = min(duration, group[-1]["end"] + EDGE_PAD)
    audio = source_audio[int(round(start * source_sr)) : int(round(end * source_sr))]
    audio32 = torchaudio.functional.resample(torch.from_numpy(audio), source_sr, SR).numpy() if source_sr != SR else audio
    text = "".join(x["text"] for x in group).strip()
    path = destination / f"{source_id}_{idx:05d}_{int(round(start * SR)):010d}_{int(round(end * SR)):010d}_{safe_text(text)}.wav"
    save_audio(path, audio32, SR)
    return {"file": str(path), "source_id": source_id, "start": start, "end": end, "duration": end - start, "text": text, "word_count": len(group), "words": group}


def phase_b_words(manifest: dict, asr_report: dict, resume: bool) -> dict:
    engine = load_eres2net()
    # word_target_hardcut's helper reads this module global; pin it to the
    # plan's 0.8-second local and direct scoring window.
    engine.WIN_SEC = WORD_WIN
    refs, _ = load_eres_refs(engine)
    negatives, _ = load_eres_negatives(engine, manifest)
    accepted_dir = WORD_OUT / "verified_word_pool"
    review_dir = WORD_OUT / "review"
    accepted_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    scan = load_json(SCAN_OUT / "interval_map.json", {"sources": []})
    scan_by_id = {x["source_id"]: x for x in scan.get("sources", [])}
    report = {"version": "word_target_all_v1", "parameters": {"word_window_sec": WORD_WIN, "word_hop_sec": WORD_HOP, "block_sec": BLOCK_SEC, "block_step_sec": BLOCK_STEP, "merge_gap_sec": MERGE_GAP, "edge_pad_sec": EDGE_PAD, **WORD_THRESH}, "sources": []}
    for source_item in asr_report["sources"]:
        source_id = source_item["source_id"]
        source = Path(source_item["source"])
        out = WORD_OUT / "source_reports" / f"{source_id}.json"
        old = load_json(out)
        fp = fingerprint([source], {"model": "ERes2NetV2", "asr": source_item.get("fingerprint"), "params": WORD_THRESH, "window_sec": WORD_WIN, "cluster_impl": "target_label_v4", "review_min_sec": 0.8})
        if resume and old and old.get("fingerprint") == fp:
            report["sources"].append(old)
            print(f"[B words cache] {source_id} segments={len(old.get('segments', []))}", flush=True)
            continue
        wav, sr = read_mono(source)
        audio = torch.from_numpy(wav).unsqueeze(0)
        scored = run_word_scores(engine, audio, source_item["words"], refs, negatives)
        accepted_words = [x for x in scored if x["accepted"] and not x["hard_veto"]]
        hard_veto_words = [x for x in scored if x["hard_veto"]]
        segments = []
        review_segments = []
        idx = 0
        for merged_group in group_words(accepted_words):
            for group in split_group(merged_group):
                if not group:
                    continue
                duration = group[-1]["end"] - group[0]["start"] + 2 * EDGE_PAD
                if duration < 0.8:
                    # The plan's review pool starts at 0.8 s.  Keep the
                    # diagnostics in word_rows, but do not materialize a
                    # sub-0.8 s clip that cannot be meaningfully audited.
                    continue
                target_dir = accepted_dir if duration >= MIN_SEG_SEC else review_dir
                row = write_segment(wav, sr, source_id, idx, group, target_dir)
                row["duration_before_padding"] = group[-1]["end"] - group[0]["start"]
                row["a_interval_cover"] = interval_cover(scan_by_id.get(source_id, {}).get("intervals", []), row["start"], row["end"])
                (segments if duration >= MIN_SEG_SEC else review_segments).append(row)
                idx += 1
        item = {"fingerprint": fp, "source_id": source_id, "source": str(source), "duration_sec": len(wav) / sr, "asr_word_count": len(source_item["words"]), "scored_word_count": len(scored), "accepted_word_count": len(accepted_words), "hard_veto_word_count": len(hard_veto_words), "word_rows": scored, "segments": segments, "review_segments": review_segments}
        write_json(out, item)
        report["sources"].append(item)
        print(f"[B words done] {source_id} words={len(scored)} accepted_words={len(accepted_words)} segments={len(segments)} review={len(review_segments)}", flush=True)
        del wav, audio, scored
        gc.collect()
    write_json(WORD_OUT / "report.json", report)
    return report


def acoustic_metrics(audio: np.ndarray, sr: int) -> dict:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    frame = 1024
    hop = 256
    count = max(1, 1 + max(0, len(audio) - frame) // hop)
    padded = np.pad(audio, (0, max(0, frame + (count - 1) * hop - len(audio))))
    frames = np.stack([padded[i * hop : i * hop + frame] for i in range(count)])
    spec = np.abs(np.fft.rfft(frames * np.hanning(frame), axis=1)) + 1e-8
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    flat = np.exp(np.mean(np.log(spec), axis=1)) / np.mean(spec, axis=1)
    diff = np.abs(np.diff(audio))
    med = np.median(diff)
    mad = np.median(np.abs(diff - med))
    z = (diff - med) / max(1.4826 * mad, 1e-8)
    positions = np.flatnonzero(z > 40.0)
    click_groups = 0 if len(positions) == 0 else 1 + int(np.sum(np.diff(positions) > max(8, round(sr * 0.004))))
    low = rms <= np.percentile(rms, 20.0)
    speech = max(float(np.percentile(rms, 85.0)), 1e-7)
    floor = max(float(np.percentile(rms[low], 50.0)) if np.any(low) else float(np.percentile(rms, 20.0)), 1e-7)
    return {"click_groups": click_groups, "clicks_per_sec": click_groups / max(len(audio) / sr, 0.1), "clipping_fraction": float(np.mean(np.abs(audio) >= 0.999)), "snr_proxy_db": float(20 * np.log10(speech / floor)), "dynamic_range_db": float(np.percentile(20 * np.log10(rms + 1e-7), 90) - np.percentile(20 * np.log10(rms + 1e-7), 10)), "low_energy_flatness": float(np.mean(flat[low])) if np.any(low) else float(np.mean(flat))}


def load_nisqa_tts():
    import sys
    sys.path.insert(0, str(SPEECHSCORE))
    from scores.nisqa.cal_nisqa import load_nisqa_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_nisqa_model(str(SPEECHSCORE / "scores" / "nisqa" / "weights" / "nisqa_tts.tar"), device=device)


def tts_mos(model, audio: np.ndarray) -> float:
    import pandas as pd
    import scores.nisqa.NISQA_lib as NL
    data = pd.DataFrame(["dummy"], columns=["deg"])
    args = model.args
    dataset = NL.SpeechQualityDataset(data, audio, df_con=None, data_dir="./", filename_column="deg", mos_column="predict_only", seg_length=args["ms_seg_length"], max_length=args["ms_max_segments"], to_memory=None, to_memory_workers=None, seg_hop_length=args["ms_seg_hop_length"], transform=None, ms_n_fft=args["ms_n_fft"], ms_hop_length=args["ms_hop_length"], ms_win_length=args["ms_win_length"], ms_n_mels=args["ms_n_mels"], ms_fmax=args["ms_fmax"], ms_channel=None, double_ended=args["double_ended"], dim=args["dim"], filename_column_ref=None)
    NL.predict_mos(model, dataset, 1, model.device, num_workers=0)
    return float(dataset.df["mos_pred"].squeeze())


def cam_model_and_refs():
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    # The local CAM++ bundle contains a requirements.txt marker.  FunASR
    # attempts to invoke pip for that marker even when the model is already
    # fully present; that is both unnecessary and forbidden in this local
    # mining run.  Dependencies are supplied by the bundled runtime.
    from funasr.utils import install_model_requirements
    install_model_requirements.install_requirements = lambda path: print(f"[CAM++ deps] already bundled; skipped {path}", flush=True)
    import full_speaker_diarize_new_v1 as cam_base
    from funasr import AutoModel

    model = AutoModel(model=str(ROOT / "modelscope_cache" / "models" / "iic" / "speech_campplus_sv_zh-cn_16k-common"), device="cuda:0" if torch.cuda.is_available() else "cpu", check_latest=False, disable_pbar=True)
    ref_data = np.load(str(ROOT / "full_funasr_target_recall_v2" / "embedding_cache" / "all_target_references.npz"), allow_pickle=True)
    neg_data = np.load(str(ROOT / "full_speaker_diarize_new_v3" / "embedding_cache" / "hard_negatives.npz"), allow_pickle=True)
    refs = torch.from_numpy(ref_data["embeddings"]).float()
    negatives = torch.from_numpy(neg_data["embeddings"]).float()
    return model, cam_base, refs, negatives


def source_segment(path: Path, start: float, end: float) -> np.ndarray:
    audio, sr = read_mono(path, SR)
    a, b = int(round(start * SR)), int(round(end * SR))
    return audio[max(0, a) : min(len(audio), b)]


def source_id_from_path(path: Path, tracks: dict[str, dict]) -> str | None:
    """Resolve a historical clip back to one of the 12 manifest sources."""
    name = path.name
    for source_id, track in tracks.items():
        if Path(track["vocals"]).name == name:
            return source_id
    return None


def words_for_interval(source_item: dict, start: float, end: float) -> list[dict]:
    """Use word midpoints so historical boundaries do not borrow neighbors."""
    result = []
    for word in source_item.get("word_rows", []):
        midpoint = (float(word["start"]) + float(word["end"])) / 2.0
        if start <= midpoint < end:
            result.append(word)
    return result


def collect_history_candidates(manifest: dict, word_report: dict, scan_report: dict) -> tuple[list[dict], dict]:
    """Reaudit all old radio_drama_pure and v9 clips through current gates.

    The old files are not whitelisted. Their source interval is mapped onto
    the newly rescored full-track ASR and scan reports, so any old clip that
    now contains a failed word, a speaker switch, or a hard exclusion remains
    rejectable by the same Phase C logic as new candidates.
    """
    tracks = track_map(manifest)
    words_by_source = {x["source_id"]: x for x in word_report.get("sources", [])}
    scan_by_source = {x["source_id"]: x for x in scan_report.get("sources", [])}
    history = []
    counts = {"radio_drama_pure": 0, "full_funasr_gold_v9": 0, "unresolved": 0}
    seen = set()

    pure_dir = ROOT / "radio_drama_pure"
    if pure_dir.exists():
        for path in sorted(pure_dir.glob("*.wav")):
            match = re.search(r"(.+_vocals/.wav)_(/d{10})_(/d{10})/.wav$", path.name)
            if not match:
                counts["unresolved"] += 1
                continue
            source_name, start_samples, end_samples = match.groups()
            source_id = source_id_from_path(Path(source_name), tracks)
            if not source_id:
                counts["unresolved"] += 1
                continue
            start, end = int(start_samples) / SR, int(end_samples) / SR
            if not path.exists() or end <= start or path in seen:
                continue
            source_item = words_by_source.get(source_id, {})
            words = words_for_interval(source_item, start, end)
            scan_item = scan_by_source.get(source_id, {})
            history.append({"file": str(path), "source_id": source_id, "start": start, "end": end, "duration": end - start, "text": "".join(x.get("text", "") for x in words), "word_count": len(words), "words": words, "a_interval_cover": interval_cover(scan_item.get("intervals", []), start, end), "history_origin": "radio_drama_pure"})
            seen.add(path)
            counts["radio_drama_pure"] += 1

    v9 = load_json(ROOT / "full_funasr_gold_v9" / "report.json", {})
    for source_item in v9.get("sources", []):
        source_id = source_id_from_path(Path(source_item.get("source", "")), tracks)
        if not source_id:
            counts["unresolved"] += len(source_item.get("segments", []))
            continue
        current_words = words_by_source.get(source_id, {})
        scan_item = scan_by_source.get(source_id, {})
        for old in source_item.get("segments", source_item.get("accepted_segments", [])):
            path = Path(old.get("file", ""))
            start, end = float(old.get("start", 0.0)), float(old.get("end", 0.0))
            if not path.exists() or end <= start or path in seen:
                continue
            words = words_for_interval(current_words, start, end)
            history.append({"file": str(path), "source_id": source_id, "start": start, "end": end, "duration": end - start, "text": "".join(x.get("text", "") for x in words) or str(old.get("text", "")), "word_count": len(words), "words": words, "a_interval_cover": interval_cover(scan_item.get("intervals", []), start, end), "history_origin": "full_funasr_gold_v9", "historical_text": str(old.get("text", ""))})
            seen.add(path)
            counts["full_funasr_gold_v9"] += 1
    return history, counts


def phase_c(manifest: dict, word_report: dict, scan_report: dict, resume: bool) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    new_rows = []
    for source in word_report["sources"]:
        for row in source.get("segments", []):
            new_rows.append({**row, "candidate_pool": "verified_word_pool"})
        for row in source.get("review_segments", []):
            new_rows.append({**row, "candidate_pool": "review_pool"})
    history_rows, history_counts = collect_history_candidates(manifest, word_report, scan_report)
    candidate_rows = new_rows + history_rows
    tracks = track_map(manifest)
    scan_by_id = {x["source_id"]: x for x in scan_report.get("sources", [])}
    # Compute CAM++ scores in batches, retaining the model-space separation.
    cam, cam_base, cam_refs, cam_negatives = cam_model_and_refs()
    cam_embs = []
    for row in candidate_rows:
        audio, sr = read_mono(Path(row["file"]), CAM_SR)
        cam_embs.append(audio)
    cam_vectors = []
    for start in range(0, len(cam_embs), 64):
        cam_vectors.append(cam_base.cam_embed_batch(cam, cam_embs[start : start + 64]))
    cam_vectors = torch.cat(cam_vectors, dim=0) if cam_vectors else torch.zeros((0, 192))
    nisqa = load_nisqa_tts()
    quality_rows = []
    source_mos = defaultdict(list)
    # Candidates are emitted in source order. Keep only the current full
    # source in memory so the source-relative NISQA comparison does not
    # reread a 200-minute corpus once per candidate.
    current_source_id = None
    current_source_audio = None
    for idx, (row, cam_vec) in enumerate(zip(candidate_rows, cam_vectors)):
        source_id = row["source_id"]
        target_sim = float(torch.max(cam_vec @ cam_refs.T))
        hard_negative_sim = float(torch.max(cam_vec @ cam_negatives.T))
        source = Path(tracks[source_id]["vocals"])
        candidate_audio, _ = read_mono(Path(row["file"]), SR)
        if source_id != current_source_id:
            current_source_audio, _ = read_mono(source, SR)
            current_source_id = source_id
        a, b = int(round(row["start"] * SR)), int(round(row["end"] * SR))
        baseline = current_source_audio[max(0, a) : min(len(current_source_audio), b)]
        mos = tts_mos(nisqa, candidate_audio)
        base_mos = tts_mos(nisqa, baseline)
        det = acoustic_metrics(candidate_audio, SR)
        source_mos[source_id].append(mos)
        quality_rows.append({"candidate_index": idx, "cam_target_sim": target_sim, "cam_hard_negative_sim": hard_negative_sim, "tts_mos": mos, "baseline_tts_mos": base_mos, "tts_delta": mos - base_mos, "acoustic": det})
    p20 = {key: float(np.percentile(values, 20)) if values else 0.0 for key, values in source_mos.items()}
    quality_by_idx = {x["candidate_index"]: x for x in quality_rows}
    verified = []
    guarded = []
    review = []
    rejected = []
    seen_paths = set()
    source_stats = defaultdict(Counter)
    for idx, row in enumerate(candidate_rows):
        q = quality_by_idx[idx]
        source_id = row["source_id"]
        interval_reason = find_interval_exclusion(manifest, source_id, row["start"], row["end"])
        words = row.get("words", [])
        word_vote = bool(words) and all(bool(word.get("accepted")) and not bool(word.get("hard_veto")) for word in words)
        sliding_vote = bool(row.get("a_interval_cover"))
        margin_vote = bool(all(float(word.get("target_negative_margin", -1.0)) >= WORD_THRESH["margin"] for word in words) and q["cam_hard_negative_sim"] < 0.88)
        acoustic = q["acoustic"]
        quality_vote = bool(q["tts_delta"] >= -0.20 and q["tts_mos"] >= p20[source_id] and acoustic["clipping_fraction"] == 0.0 and acoustic["clicks_per_sec"] <= 3.0)
        hard_reasons = []
        if interval_reason:
            hard_reasons.append(interval_reason)
        if any(name in Path(row["file"]).name for name in manifest.get("excluded_basenames", [])):
            hard_reasons.append("excluded_safe_word")
        duration = float(row.get("duration", 0.0))
        if row.get("candidate_pool") == "review_pool":
            if duration < 0.8 or duration > MAX_SEG_SEC:
                hard_reasons.append("duration_out_of_review_range")
        elif duration < MIN_SEG_SEC or duration > MAX_SEG_SEC:
            hard_reasons.append("duration_out_of_training_range")
        if not Path(row["file"]).exists():
            hard_reasons.append("missing_file")
        if row["file"] in seen_paths:
            hard_reasons.append("duplicate_path")
        if any(word.get("hard_veto") for word in words):
            hard_reasons.append("word_non_target_or_negative_veto")
        if q["cam_hard_negative_sim"] >= 0.88:
            hard_reasons.append("cam_hard_negative_veto")
        if any(float(word.get("target_negative_margin", -1.0)) <= 0.0 for word in words):
            hard_reasons.append("eres_margin_nonpositive")
        if acoustic["clipping_fraction"] > 0.0 or acoustic["clicks_per_sec"] > 6.0:
            hard_reasons.append("acoustic_artifact_veto")
        seen_paths.add(row["file"])
        votes = {"word": word_vote, "sliding": sliding_vote, "margin": margin_vote, "quality": quality_vote}
        total = sum(votes.values())
        speaker_votes = sum(votes[k] for k in ("word", "sliding", "margin"))
        is_guarded = bool(tracks[source_id].get("guarded"))
        if hard_reasons:
            tier = "rejected"
        elif row.get("candidate_pool") == "review_pool":
            tier = "review"
        elif is_guarded and total == 4:
            tier = "guarded_acceptance"
        elif not is_guarded and total >= 3 and speaker_votes >= 2:
            tier = "verified_machine"
        elif total >= 2:
            tier = "review"
        else:
            tier = "rejected"
        record = {**row, "source_id": source_id, "votes": votes, "vote_total": total, "speaker_votes": speaker_votes, "cam": {"target_sim": q["cam_target_sim"], "hard_negative_sim": q["cam_hard_negative_sim"]}, "nisqa_tts": {"mos": q["tts_mos"], "baseline_mos": q["baseline_tts_mos"], "delta": q["tts_delta"], "source_p20": p20[source_id]}, "acoustic": acoustic, "hard_reasons": hard_reasons, "tier": tier}
        source_stats[source_id][tier] += 1
        if tier == "verified_machine":
            verified.append(record)
        elif tier == "guarded_acceptance":
            guarded.append(record)
        elif tier == "review":
            review.append(record)
        else:
            rejected.append(record)
    verified.sort(key=lambda x: (x["source_id"], x["start"]))
    lines = [f"{row['file']}|consensus_v1|JA|{row['text']}" for row in verified]
    atomic_write(VERIFIED_LIST, "/n".join(lines) + ("/n" if lines else ""))
    report = {"version": "consensus_v1", "policy": "3/4 votes, at least 2 speaker votes, hard vetoes override; guarded sources require 4/4 and remain acceptance-only", "local_resources": RESOURCE_POLICY, "counts": {"candidate": len(candidate_rows), "new_candidate": len(new_rows), "history_candidate": len(history_rows), "verified_machine": len(verified), "guarded_acceptance": len(guarded), "review": len(review), "rejected": len(rejected), "verified_sec": sum(x["duration"] for x in verified), "guarded_sec": sum(x["duration"] for x in guarded), "review_sec": sum(x["duration"] for x in review)}, "calibration": scan_report.get("calibration", {}), "source_stats": {key: dict(value) for key, value in source_stats.items()}, "verified_list": str(VERIFIED_LIST), "verified": verified, "guarded_acceptance": guarded, "review": review, "rejected": rejected}
    history_records = [x for x in verified + guarded + review + rejected if x.get("history_origin")]
    write_json(OUT / "history_reaudit.json", {"version": "history_reaudit_v1", "input_counts": history_counts, "evaluated_count": len(history_records), "records": history_records})
    report["history_reaudit"] = {"input_counts": history_counts, "evaluated_count": len(history_records), "origins": {key: sum(1 for x in history_records if x.get("history_origin") == key) for key in ("radio_drama_pure", "full_funasr_gold_v9")}}
    auto_sec = sum(x["duration"] for x in verified) + sum(x["duration"] for x in guarded)
    report["decision"] = {"training_ready": bool(auto_sec >= 15.0 and verified), "auto_candidate_sec": auto_sec, "status": "insufficient_data" if auto_sec < 15.0 else "candidate_pool_ready_for_user_acceptance", "note": "Automatic scoring is not a substitute for the user's blind listening acceptance."}
    write_json(OUT / "report.json", report)
    make_acceptance_zip(manifest, report)
    validate_outputs(manifest, word_report, scan_report, report)
    return report


def validate_outputs(manifest: dict, word_report: dict, scan_report: dict, report: dict) -> None:
    final_rows = list(report.get("verified", [])) + list(report.get("guarded_acceptance", []))
    paths = [Path(row["file"]) for row in final_rows]
    checks = {
        "source_count_12": len(scan_report.get("sources", [])) == 12 and len(word_report.get("sources", [])) == 12,
        "total_duration_near_199_76_min": abs(sum(float(x.get("duration_sec", 0.0)) for x in scan_report.get("sources", [])) / 60.0 - 199.76) < 1.0,
        "history_110_plus_52": report.get("history_reaudit", {}).get("input_counts") == {"radio_drama_pure": 110, "full_funasr_gold_v9": 52, "unresolved": 0} and report.get("history_reaudit", {}).get("evaluated_count") == 162,
        "verified_paths_exist": all(path.exists() for path in paths),
        "verified_paths_unique": len({str(path).lower() for path in paths}) == len(paths),
        "verified_duration_1_5_to_6": all(1.5 <= float(row.get("duration", 0.0)) <= 6.0 for row in final_rows),
        "no_forced_exclusions": all(not any(name in path.name for name in manifest.get("excluded_basenames", [])) and find_interval_exclusion(manifest, row["source_id"], row["start"], row["end"]) is None for row, path in zip(final_rows, paths)),
        "no_empty_text": all(str(row.get("text", "")).strip() and "/ufffd" not in str(row.get("text", "")) for row in final_rows),
        "reference_calibration_95pct": float(report.get("calibration", {}).get("positive_pass_rate", 0.0)) >= 0.95,
        "negative_calibration_13_of_13": int(report.get("calibration", {}).get("negative_hard_reject_count", 0)) == 13,
        "training_not_started": True,
    }
    write_json(OUT / "validation.json", {"version": "acceptance_tests_v1", "checks": checks, "all_pass": all(checks.values()), "note": "A failed calibration check is reported as a blocker; it is not overridden to manufacture candidates."})


def make_acceptance_zip(manifest: dict, report: dict) -> None:
    acceptance_dir = OUT / "acceptance_v1"
    audio_dir = acceptance_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    candidates = list(report.get("verified", [])) + list(report.get("guarded_acceptance", []))
    candidates.sort(key=lambda x: (-x.get("vote_total", 0), x.get("source_id", ""), x.get("start", 0.0)))
    chosen = candidates[:20]
    controls = []
    v9_report = load_json(ROOT / "full_funasr_gold_v9" / "report.json", {})
    for source in v9_report.get("sources", []):
        for row in source.get("segments", source.get("accepted_segments", [])):
            if Path(row.get("file", "")).exists():
                controls.append(row)
            if len(controls) >= 4:
                break
        if len(controls) >= 4:
            break
    manifest_rows = []
    for idx, row in enumerate(chosen + controls):
        source_path = Path(row["file"])
        blind = f"sample_{idx + 1:02d}"
        dest = audio_dir / f"{blind}.wav"
        if not dest.exists():
            shutil.copy2(source_path, dest)
        manifest_rows.append({"blind_id": blind, "kind": "machine" if idx < len(chosen) else "gold_control", "audio": f"audio/{dest.name}", "source_id": row.get("source_id", "control"), "text": row.get("text", "")})
    html = ["<!doctype html><meta charset='utf-8'><title>acceptance_v1</title><h1>盲听验收包</h1>", f"<p>machine 样本：{len(chosen)}；gold_control：{len(controls)}。只有报告标记 training_ready 后才允许进入训练。请记录非目标、混合、边界截断或明显杂音。</p>", "<ol>"]
    for row in manifest_rows:
        html.append(f"<li><b>{row['blind_id']}</b> ({row['kind']})<br><audio controls preload='none' src='{row['audio']}'></audio></li>")
    html.append("</ol>")
    (acceptance_dir / "index.html").write_text("/n".join(html), encoding="utf-8")
    write_json(acceptance_dir / "manifest.json", {"samples": manifest_rows})
    with (acceptance_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["blind_id", "kind", "audio", "source_id", "text"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    zip_path = ROOT / "acceptance_v1.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in acceptance_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(acceptance_dir).as_posix())


def phase_all(manifest: dict, resume: bool) -> dict:
    scan = phase_a(manifest, resume)
    asr = phase_b_asr(manifest, resume)
    words = phase_b_words(manifest, asr, resume)
    return phase_c(manifest, words, scan, resume)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument("--phase", choices=["a", "asr", "b", "c", "all"], default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase == "a":
        phase_a(manifest, args.resume)
    elif args.phase == "asr":
        phase_b_asr(manifest, args.resume)
    elif args.phase == "b":
        asr = load_json(WORD_OUT / "asr_report.json") or phase_b_asr(manifest, args.resume)
        phase_b_words(manifest, asr, args.resume)
    elif args.phase == "c":
        scan = load_json(SCAN_OUT / "interval_map.json")
        words = load_json(WORD_OUT / "report.json")
        if not scan or not words:
            raise RuntimeError("Phase C requires completed phase A and B reports")
        phase_c(manifest, words, scan, args.resume)
    else:
        phase_all(manifest, args.resume)


if __name__ == "__main__":
    main()
