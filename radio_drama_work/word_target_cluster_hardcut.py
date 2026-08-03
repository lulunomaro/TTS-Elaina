# -*- coding: utf-8 -*-
"""Word-level original-waveform hard cut with within-clip speaker clustering."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

ROOT = Path(r"radio_drama_work/radio_drama_work")
sys.path.insert(0, str(ROOT))
import word_target_hardcut as base  # noqa: E402

OUT = ROOT / "word_target_hardcut_cluster_v1"
REPORT = OUT / "word_target_hardcut_cluster_report.json"
FRAME_SEC = 0.80
HOP_SEC = 0.20
VOTE_MIN = 0.60
CLUSTER_MARGIN_MIN = 0.025
DIRECT_CENTER_MIN = 0.68
DIRECT_P25_MIN = 0.43
FRAME_TARGET_MIN = 0.60
EDGE_PAD_SEC = 0.055
FADE_SEC = 0.025
MERGE_GAP_SEC = 0.22


def spherical_kmeans(x: torch.Tensor, iters: int = 40) -> tuple[torch.Tensor, torch.Tensor]:
    if len(x) < 2:
        return torch.zeros(len(x), dtype=torch.long), x[:1]
    sim = x @ x.T
    i, j = divmod(int(torch.argmin(sim).item()), len(x))
    centers = torch.stack([x[i], x[j]], dim=0)
    for _ in range(iters):
        assignment = (x @ centers.T).argmax(dim=1)
        new_centers = centers.clone()
        for k in range(2):
            rows = x[assignment == k]
            if len(rows):
                new_centers[k] = torch.nn.functional.normalize(rows.mean(0), dim=-1)
        if torch.allclose(new_centers, centers, atol=1e-5):
            break
        centers = new_centers
    return assignment, centers


def hard_mask(mask: np.ndarray, start: int, end: int, fade: int) -> None:
    start, end = max(0, start), min(len(mask), end)
    if end <= start:
        return
    mask[start:end] = 1.0
    f = min(fade, (end - start) // 2)
    if f:
        mask[start:start + f] *= np.linspace(0.0, 1.0, f, endpoint=False)
        mask[end - f:end] *= np.linspace(1.0, 0.0, f, endpoint=False)


def merge_words(words: list[dict]) -> list[dict]:
    out = []
    for word in words:
        if not out or word["start"] - out[-1]["end"] > MERGE_GAP_SEC:
            out.append({"start": word["start"], "end": word["end"], "texts": [word["text"]], "words": [word]})
        else:
            out[-1]["end"] = word["end"]
            out[-1]["texts"].append(word["text"])
            out[-1]["words"].append(word)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    word_dir = OUT / "accepted_words"
    seg_dir = OUT / "accepted_segments"
    word_dir.mkdir(exist_ok=True)
    seg_dir.mkdir(exist_ok=True)

    refs = torch.cat([base.embed_file(p) for p in base.load_refs()], dim=0)
    negatives = torch.cat([base.embed_file(p) for p in base.load_negatives()], dim=0)
    center = torch.nn.functional.normalize(refs.mean(0), dim=-1)
    report = {
        "note": "ASR-word gating plus within-clip spherical speaker clustering; retains original 32 kHz waveform.",
        "device": base.device,
        "reference_count": len(refs),
        "negative_count": len(negatives),
        "parameters": {"frame_sec": FRAME_SEC, "hop_sec": HOP_SEC, "vote_min": VOTE_MIN, "cluster_margin_min": CLUSTER_MARGIN_MIN},
        "files": [],
    }

    for source in base.INPUTS:
        wav, sr = torchaudio.load(str(source))
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != base.SR:
            wav = torchaudio.functional.resample(wav, sr, base.SR)
        n = wav.shape[1]
        duration = n / base.SR
        frame_len = int(round(FRAME_SEC * base.SR))
        hop = int(round(HOP_SEC * base.SR))
        starts = list(range(0, max(1, n - frame_len + 1), hop))
        if n > frame_len and starts[-1] + frame_len < n:
            starts.append(n - frame_len)
        centers_sec = [(st + frame_len / 2) / base.SR for st in starts]
        clips = []
        for st in starts:
            clip = wav[:, st:min(n, st + frame_len)]
            if clip.shape[1] < frame_len:
                clip = torch.nn.functional.pad(clip, (0, frame_len - clip.shape[1]))
            clips.append(clip)
        embs = base.embed_batch(torch.cat(clips, dim=0))
        assignment, cluster_centers = spherical_kmeans(embs)
        cluster_target = cluster_centers @ center
        target_cluster = int(torch.argmax(cluster_target).item())
        other_cluster = 1 - target_cluster if len(cluster_centers) > 1 else target_cluster
        cluster_margin = float(cluster_target[target_cluster] - cluster_target[other_cluster]) if len(cluster_centers) > 1 else 0.0
        words = base.transcribe(source)
        scored = []
        accepted = []
        for word in words:
            if word["end"] - word["start"] < base.MIN_WORD_SEC:
                continue
            mid = (word["start"] + word["end"]) / 2
            lo, hi = word["start"] - 0.28, word["end"] + 0.28
            idx = [i for i, c in enumerate(centers_sec) if lo <= c <= hi]
            if not idx:
                idx = [min(range(len(centers_sec)), key=lambda i: abs(centers_sec[i] - mid))]
            votes = float(sum(int(assignment[i]) == target_cluster for i in idx) / len(idx))
            frame_target = float(torch.stack([embs[i] for i in idx]).mean(0) @ center)
            direct = base.embed_batch(base.window_wave(wav, mid))[0]
            direct_vals = direct @ refs.T
            direct_center = float(direct @ center)
            direct_p25 = float(torch.quantile(direct_vals, .25))
            direct_negative = float((direct @ negatives.T).max())
            row = {
                **word,
                "cluster_vote": votes,
                "frame_target_center": frame_target,
                "direct_center": direct_center,
                "direct_p25": direct_p25,
                "direct_negative_max": direct_negative,
                "cluster_margin": cluster_margin,
                "target_cluster": target_cluster,
                "accepted": bool(
                    votes >= VOTE_MIN
                    and cluster_margin >= CLUSTER_MARGIN_MIN
                    and frame_target >= FRAME_TARGET_MIN
                    and direct_center >= DIRECT_CENTER_MIN
                    and direct_p25 >= DIRECT_P25_MIN
                    and direct_center - direct_negative >= base.MARGIN_MIN
                ),
            }
            scored.append(row)
            if row["accepted"]:
                accepted.append(row)

        mask = np.zeros(n, dtype=np.float32)
        pad = int(round(EDGE_PAD_SEC * base.SR))
        fade = int(round(FADE_SEC * base.SR))
        for index, word in enumerate(accepted):
            st = max(0, int(round(word["start"] * base.SR)) - pad)
            en = min(n, int(round(word["end"] * base.SR)) + pad)
            hard_mask(mask, st, en, fade)
            sf.write(str(word_dir / f"{source.stem}_word_{index:03d}.wav"), wav[:, st:en].numpy().T, base.SR, subtype="PCM_16")

        segments = []
        for index, segment in enumerate(merge_words(accepted)):
            st_sec = max(0.0, segment["start"] - EDGE_PAD_SEC)
            en_sec = min(duration, segment["end"] + EDGE_PAD_SEC)
            st, en = int(round(st_sec * base.SR)), int(round(en_sec * base.SR))
            out_path = seg_dir / f"{source.stem}_seg_{index:03d}.wav"
            sf.write(str(out_path), wav[:, st:en].numpy().T, base.SR, subtype="PCM_16")
            segments.append({"file": str(out_path), "start": st_sec, "end": en_sec, "text": "".join(segment["texts"]), "word_count": len(segment["words"])})

        hardcut_path = OUT / f"{source.stem}_target_hardcut.wav"
        sf.write(str(hardcut_path), (wav.numpy() * mask[None, :]).T, base.SR, subtype="PCM_16")
        report["files"].append({
            "source": str(source),
            "duration": duration,
            "cluster_target_scores": [float(x) for x in cluster_target],
            "cluster_target": target_cluster,
            "cluster_margin": cluster_margin,
            "asr_words": words,
            "scored_words": scored,
            "accepted_word_count": len(accepted),
            "accepted_words": accepted,
            "segments": segments,
            "hardcut_file": str(hardcut_path),
        })
        print(f"{source.name}: frames={len(embs)} cluster_target={target_cluster} cluster_margin={cluster_margin:.3f} accepted={len(accepted)} segments={len(segments)}", flush=True)
        for row in scored:
            print(f"  {row['start']:.2f}-{row['end']:.2f} {row['text']} vote={row['cluster_vote']:.2f} frame={row['frame_target_center']:.3f} direct={row['direct_center']:.3f} p25={row['direct_p25']:.3f} margin={row['direct_center'] - row['direct_negative_max']:.3f} {'KEEP' if row['accepted'] else 'CUT'}", flush=True)

    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={REPORT}", flush=True)


if __name__ == "__main__":
    main()
