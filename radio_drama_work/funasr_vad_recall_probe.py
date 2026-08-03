# -*- coding: utf-8 -*-
"""Probe FunASR FSMN-VAD + CAM++ diarization against the v3 ASR windows.

This is a non-training probe. It reports whether FunASR's VAD/chunking and
spectral speaker clustering recover more target-speaking time before any
audio is written.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from funasr import AutoModel
from funasr.models.campplus.cluster_backend import ClusterBackend

sys.path.insert(0, str(Path(r"radio_drama_work/radio_drama_work")))
import full_speaker_diarize_new_v1 as base  # noqa: E402


ROOT = Path(r"radio_drama_work/radio_drama_work")
PKG = Path(r"radio_drama_work/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50")
VAD_PATH = PKG / "tools" / "asr" / "models" / "speech_fsmn_vad_zh-cn-16k-common-pytorch"
SOURCE = ROOT / "new_radio_batch_v1" / "vocals" / "306672160-1-192_vocals.wav"


def all_reference_paths() -> list[Path]:
    old_dir = PKG / "logs" / "my_voice" / "5-wav32k"
    paths = sorted(old_dir.glob("*.wav"))
    paths.extend(base.parse_reference_paths())
    seen = set()
    return [path for path in paths if not (str(path).lower() in seen or seen.add(str(path).lower()))]


def make_chunks(vad_rows: list[list[int]], audio16: np.ndarray) -> list[tuple[float, float, np.ndarray]]:
    rows = []
    for start_ms, end_ms in vad_rows:
        start = float(start_ms) / 1000.0
        end = float(end_ms) / 1000.0
        a = max(0, int(round(start * base.CAM_SR)))
        b = min(len(audio16), int(round(end * base.CAM_SR)))
        data = audio16[a:b]
        if len(data) <= 0:
            continue
        chunk_len = int(round(base.CAM_WINDOW_SEC * base.CAM_SR))
        shift = int(round(base.CAM_HOP_SEC * base.CAM_SR))
        last = 0
        for chunk_start in range(0, len(data), shift):
            chunk_end = min(chunk_start + chunk_len, len(data))
            if chunk_end <= last:
                break
            last = chunk_end
            chunk_start = max(0, chunk_end - chunk_len)
            chunk = data[chunk_start:chunk_end]
            if len(chunk) < chunk_len:
                chunk = np.pad(chunk, (0, chunk_len - len(chunk)))
            rows.append((start + chunk_start / base.CAM_SR, start + chunk_end / base.CAM_SR, chunk[:chunk_len]))
    return rows


def main() -> None:
    audio, sr = base.read_mono(SOURCE)
    audio16 = base.resample_1d(audio, sr, base.CAM_SR)
    vad = AutoModel(model=str(VAD_PATH), device="cuda:0", disable_update=True, disable_pbar=True)
    vad_result = vad.generate(input=str(SOURCE))[0]
    vad_rows = vad_result["value"]
    chunks = make_chunks(vad_rows, audio16)
    cam = AutoModel(model=str(base.CAM_MODEL), device="cuda:0", check_latest=False, disable_pbar=True)
    refs_paths = all_reference_paths()
    ref_waves = []
    for path in refs_paths:
        ref_audio, ref_sr = base.read_mono(path)
        ref16 = base.resample_1d(ref_audio, ref_sr, base.CAM_SR)
        ref_waves.append(base.segment(ref16, base.CAM_SR, 0.0, len(ref16) / base.CAM_SR, base.CAM_WINDOW_SEC))
    refs = base.cam_embed_batch(cam, ref_waves)
    embs = base.cam_embed_batch(cam, [chunk[2] for chunk in chunks])
    asr_cache = ROOT / "full_speaker_diarize_new_v3" / "asr_cache" / f"{SOURCE.stem}.json"
    words = json.loads(asr_cache.read_text(encoding="utf-8"))["words"] if asr_cache.exists() else []
    fits = []
    for requested_k in (2, 3, 4, 5):
        labels = np.asarray(ClusterBackend()(embs.numpy(), oracle_num=requested_k), dtype=int)
        clusters = sorted(set(labels.tolist()))
        scores = []
        for label in clusters:
            center = base.norm_rows(embs[labels == label].mean(0, keepdim=True))[0]
            sims = center @ refs.T
            scores.append({"cluster": int(label), "count": int(np.sum(labels == label)), "duration_sec": float(sum(chunks[i][1] - chunks[i][0] for i in range(len(chunks)) if labels[i] == label)), "target_sim": float(torch.max(sims).item()), "target_top4": float(torch.topk(sims, k=min(4, len(sims))).values.mean().item())})
        scores.sort(key=lambda row: row["target_top4"], reverse=True)
        target = scores[0]["cluster"]
        target_rows = [chunks[i] for i, label in enumerate(labels) if label == target]
        target_word_count = sum(1 for word in words if any(a <= (word["start"] + word["end"]) / 2.0 <= b for a, b, _ in target_rows))
        target_word_sec = sum(word["end"] - word["start"] for word in words if any(a <= (word["start"] + word["end"]) / 2.0 <= b for a, b, _ in target_rows))
        fits.append({"requested_k": requested_k, "cluster_count": len(scores), "clusters": scores, "target_cluster": target, "target_chunk_sec": round(sum(row[1] - row[0] for row in target_rows), 2), "target_word_count": target_word_count, "target_word_sec": round(target_word_sec, 2)})
    print(json.dumps({"source": str(SOURCE), "vad_segments": len(vad_rows), "cam_chunks": len(chunks), "fits": fits}, ensure_ascii=False))


if __name__ == "__main__":
    main()
