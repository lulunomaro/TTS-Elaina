"""Reslice silence-based radio slices by target-speaker continuity.

This is a recall/review stage, not an automatic training-list writer.  It
scores short overlapping windows against many known target-speaker references,
then emits only contiguous runs whose target score stays stable.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio


ROOT = Path(r"radio_drama_work/radio_drama_work")
PKG = Path(r"radio_drama_work/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50")
IN_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "new_slices"
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "speaker_aware_candidates"
REPORT = OUT_DIR / "speaker_aware_report.json"
NEGATIVE_LIST = ROOT / "confirmed_non_target_or_mixed.txt"
SR = 32000
# Defaults are conservative.  Environment overrides are used for a fine
# recall pass; that pass still writes review audio only unless continuity is
# confirmed by overlapping windows.
WIN_SEC = float(os.environ.get("SAW_WIN_SEC", "1.5"))
HOP_SEC = float(os.environ.get("SAW_HOP_SEC", "0.5"))
BATCH = 24
CENTER_MIN = float(os.environ.get("SAW_CENTER_MIN", "0.79"))
MEDIAN_MIN = float(os.environ.get("SAW_MEDIAN_MIN", "0.64"))
P25_MIN = float(os.environ.get("SAW_P25_MIN", "0.56"))
MARGIN_MIN = float(os.environ.get("SAW_MARGIN_MIN", "0.08"))
MIN_STABLE_WINDOWS = int(os.environ.get("SAW_MIN_STABLE_WINDOWS", "2"))
MIN_SEG_SEC = float(os.environ.get("SAW_MIN_SEG_SEC", "1.0"))

os.chdir(PKG)
sys.path.insert(0, str(PKG / "GPT_SoVITS" / "eres2net"))
from ERes2NetV2 import ERes2NetV2
import kaldi as Kaldi


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
model.load_state_dict(torch.load(PKG / "GPT_SoVITS" / "pretrained_models" / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt", map_location="cpu"))
model.eval().to(device)
resampler = torchaudio.transforms.Resample(SR, 16000).to(device)


@torch.no_grad()
def embed_batch(waves: torch.Tensor) -> torch.Tensor:
    waves16 = resampler(waves.to(device))
    feats = torch.stack([
        Kaldi.fbank(w.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0)
        for w in waves16
    ])
    return torch.nn.functional.normalize(model.forward(feats), dim=-1).cpu()


@torch.no_grad()
def embed_one(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return embed_batch(wav)


def ref_embeddings() -> tuple[torch.Tensor, torch.Tensor]:
    ref_dir = PKG / "logs" / "my_voice" / "5-wav32k"
    refs = torch.cat([embed_one(p) for p in sorted(ref_dir.glob("*.wav"))], dim=0)
    center = torch.nn.functional.normalize(refs.mean(0), dim=-1)
    return refs, center


def negative_paths() -> list[Path]:
    """Load user-confirmed hard negatives for contrastive screening."""
    paths = [ROOT / "new_candidates_sp75" / "28055832533-1-192_vocals_0017516160_0017573120.wav"]
    if NEGATIVE_LIST.exists():
        for line in NEGATIVE_LIST.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            paths.append(Path(line.split("|", 1)[0].strip()))
    seen = set()
    out = []
    for path in paths:
        key = str(path).lower()
        if path.exists() and key not in seen:
            out.append(path)
            seen.add(key)
    return out


def active_window(row: dict) -> bool:
    # Stricter than the old single-centre threshold.  p25 measures whether a
    # candidate resembles a broad portion of the known target references.
    return (
        row["center"] >= CENTER_MIN
        and row["median"] >= MEDIAN_MIN
        and row["p25"] >= P25_MIN
        and row["rms"] >= 0.004
        and row["target_negative_margin"] >= MARGIN_MIN
    )


def runs(active: list[bool]) -> list[tuple[int, int]]:
    out = []
    start = None
    for i, yes in enumerate(active + [False]):
        if yes and start is None:
            start = i
        elif not yes and start is not None:
            out.append((start, i))
            start = None
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    review_dir = OUT_DIR / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    refs, center = ref_embeddings()
    negative_files = negative_paths()
    negatives = torch.cat([embed_one(path) for path in negative_files], dim=0) if negative_files else None
    files = sorted(IN_DIR.glob("*.wav"))
    all_rows = []
    all_segments = []
    print(f"device={device} refs={len(refs)} negatives={len(negative_files)} slices={len(files)}", flush=True)

    for idx, path in enumerate(files, 1):
        wav, sr = torchaudio.load(str(path))
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        n = int(wav.shape[1])
        win = int(WIN_SEC * SR)
        hop = int(HOP_SEC * SR)
        starts = list(range(0, max(1, n - win + 1), hop))
        if n > win and starts[-1] + win < n:
            starts.append(n - win)
        if n <= win:
            starts = [0]

        windows = []
        for st in starts:
            clip = wav[:, st:min(n, st + win)]
            if clip.shape[1] < win:
                clip = torch.nn.functional.pad(clip, (0, win - clip.shape[1]))
            windows.append(clip)

        embs = []
        for j in range(0, len(windows), BATCH):
            embs.append(embed_batch(torch.cat(windows[j:j + BATCH], dim=0)))
        embs = torch.cat(embs, dim=0)
        sims = embs @ refs.T
        rows = []
        for k, st in enumerate(starts):
            x = windows[k][0].numpy()
            rms = float(np.sqrt(np.mean(x * x) + 1e-12))
            vals = sims[k].numpy()
            row = {
                "start": st / SR,
                "end": min(n, st + win) / SR,
                "center": float(embs[k] @ center),
                "p25": float(np.quantile(vals, .25)),
                "median": float(np.median(vals)),
                "p75": float(np.quantile(vals, .75)),
                "rms": rms,
            }
            if negatives is not None:
                negative_sims = embs[k] @ negatives.T
                row["known_negative_sim"] = float(negative_sims.max())
                row["target_negative_margin"] = row["center"] - row["known_negative_sim"]
                row["negative_count"] = int(negatives.shape[0])
            else:
                row["known_negative_sim"] = float("nan")
                row["target_negative_margin"] = 1.0
                row["negative_count"] = 0
            row["active"] = active_window(row)
            rows.append(row)

        active = [bool(r["active"]) for r in rows]
        for a, b in runs(active):
            start = rows[a]["start"]
            end = rows[b - 1]["end"]
            chosen = rows[a:b]
            duration = end - start
            # A one-window hit is a review-only hint; stable training candidates
            # require two overlapping target windows.
            stable = len(chosen) >= MIN_STABLE_WINDOWS and duration >= max(1.0, MIN_SEG_SEC) and duration <= 8.0
            if duration < MIN_SEG_SEC or duration > 8.0:
                continue
            seg = {
                "source": path.name,
                "start": start,
                "end": end,
                "duration": duration,
                "windows": len(chosen),
                "stable": stable,
                "min_p25": min(r["p25"] for r in chosen),
                "mean_p25": float(np.mean([r["p25"] for r in chosen])),
                "mean_center": float(np.mean([r["center"] for r in chosen])),
                "min_center": min(r["center"] for r in chosen),
                "source_path": str(path),
            }
            all_segments.append(seg)

            if stable:
                clip = wav[:, int(start * SR):min(n, int(end * SR))]
                out_name = f"aware_{len(all_segments):04d}_p25{seg['mean_p25']:.3f}_{path.stem}_{int(start * SR):010d}_{int(end * SR):010d}.wav"
                torchaudio.save(str(OUT_DIR / out_name), clip, SR)
                seg["file"] = str(OUT_DIR / out_name)
            else:
                # Single-window hits are review-only. They are intentionally
                # exported so a human can reject mixed/other-speaker turns;
                # they never enter a training list automatically.
                clip = wav[:, int(start * SR):min(n, int(end * SR))]
                out_name = f"review_{len(all_segments):04d}_p25{seg['mean_p25']:.3f}_{path.stem}_{int(start * SR):010d}_{int(end * SR):010d}.wav"
                torchaudio.save(str(review_dir / out_name), clip, SR)
                seg["review_file"] = str(review_dir / out_name)

        all_rows.append({"file": path.name, "windows": rows})
        if idx % 25 == 0 or idx == len(files):
            print(f"{idx}/{len(files)} segments={len(all_segments)} stable={sum(bool(x.get('stable')) for x in all_segments)}", flush=True)

    REPORT.write_text(json.dumps({
        "note": "speaker-aware recall/review; not an automatic training list",
        "device": device,
        "reference_count": len(refs),
        "negative_files": [str(path) for path in negative_files],
        "window_sec": WIN_SEC,
        "hop_sec": HOP_SEC,
        "thresholds": {
            "center_min": CENTER_MIN,
            "median_min": MEDIAN_MIN,
            "p25_min": P25_MIN,
            "margin_min": MARGIN_MIN,
            "min_stable_windows": MIN_STABLE_WINDOWS,
            "min_segment_sec": MIN_SEG_SEC,
        },
        "files": all_rows,
        "segments": all_segments,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={REPORT} segments={len(all_segments)} stable={sum(bool(x.get('stable')) for x in all_segments)}", flush=True)


if __name__ == "__main__":
    main()
