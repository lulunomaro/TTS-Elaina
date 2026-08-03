# -*- coding: utf-8 -*-
"""Word-level target-speaker hard cut on the original separated vocal track.

This deliberately does not run another source-separation network.  It uses
ASR word timestamps only as time anchors, scores short original-waveform
windows with ERes2NetV2, and keeps only words that pass target and hard-negative
checks.  The output is therefore a cut/masked copy of the input waveform, not
a reconstructed speaker stem.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from faster_whisper import WhisperModel

ROOT = Path(r"radio_drama_work/radio_drama_work")
PKG = Path(r"radio_drama_work/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50")
DEFAULT_INPUTS = [
    ROOT / "new_slices" / "1405440540-1-192_vocals_0012969600_0013081600.wav",
    ROOT / "new_slices" / "28055832533-1-192_vocals_0013268480_0013539200.wav",
    ROOT / "new_slices" / "28055832533-1-192_vocals_0018346880_0018502400.wav",
]
INPUTS = [Path(arg) for arg in sys.argv[1:] if Path(arg).exists()] or DEFAULT_INPUTS
OUT = ROOT / "word_target_hardcut_v1"
REPORT = OUT / "word_target_hardcut_report.json"
REF_DIR = PKG / "logs" / "my_voice" / "5-wav32k"
POSITIVE_LIST = ROOT / "confirmed_target_pure.txt"
NEGATIVE_LIST = ROOT / "confirmed_non_target_or_mixed.txt"
ASR_MODEL = PKG / "tools" / "asr" / "models" / "faster-whisper-large-v3-turbo"

SR = 32000
WIN_SEC = float(os.environ.get("WTH_WIN_SEC", "1.20"))
MIN_WORD_SEC = float(os.environ.get("WTH_MIN_WORD_SEC", "0.20"))
TARGET_CENTER_MIN = float(os.environ.get("WTH_CENTER_MIN", "0.78"))
TARGET_P25_MIN = float(os.environ.get("WTH_P25_MIN", "0.52"))
MARGIN_MIN = float(os.environ.get("WTH_MARGIN_MIN", "0.06"))
EDGE_PAD_SEC = float(os.environ.get("WTH_EDGE_PAD_SEC", "0.055"))
FADE_SEC = float(os.environ.get("WTH_FADE_SEC", "0.025"))
MERGE_GAP_SEC = float(os.environ.get("WTH_MERGE_GAP_SEC", "0.22"))

os.chdir(PKG)
sys.path.insert(0, str(PKG / "GPT_SoVITS" / "eres2net"))
from ERes2NetV2 import ERes2NetV2  # noqa: E402
import kaldi as Kaldi  # noqa: E402


def listed_paths(path: Path) -> list[Path]:
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        item = Path(line.split("|", 1)[0].strip())
        if item.exists():
            result.append(item)
    return result


def load_refs() -> list[Path]:
    paths = sorted(REF_DIR.glob("*.wav"))
    paths.extend(listed_paths(POSITIVE_LIST))
    seen = set()
    result = []
    for path in paths:
        key = str(path).lower()
        if key not in seen:
            result.append(path)
            seen.add(key)
    return result


def load_negatives() -> list[Path]:
    # Mixed clips are useful boundary warnings but are not pure negative
    # speaker references: they may contain target speech themselves.  Only
    # user-confirmed pure non-target clips are used for word-level contrast.
    if not NEGATIVE_LIST.exists():
        return []
    result = []
    for line in NEGATIVE_LIST.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        reason = raw.split("|", 1)[1].lower() if "|" in raw else ""
        if "mixed" in reason or "混" in reason:
            continue
        path = Path(raw.split("|", 1)[0].strip())
        if path.exists():
            result.append(path)
    return result


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


def embed_file(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return embed_batch(wav)


def transcribe(path: Path) -> list[dict]:
    asr = WhisperModel(str(ASR_MODEL), device="cpu", compute_type="int8", cpu_threads=8)
    segments, _info = asr.transcribe(
        str(path), language="ja", beam_size=5, temperature=0.0,
        word_timestamps=True, vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=180),
        condition_on_previous_text=False,
        compression_ratio_threshold=2.4, log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
    )
    words = []
    for seg in segments:
        for word in (seg.words or []):
            text = (word.word or "").strip()
            if text and word.end > word.start:
                words.append({
                    "text": text,
                    "start": float(word.start),
                    "end": float(word.end),
                    "probability": float(getattr(word, "probability", 0.0)),
                })
    return words


def window_wave(wav: torch.Tensor, center_sec: float) -> torch.Tensor:
    n = wav.shape[1]
    length = int(round(WIN_SEC * SR))
    center = int(round(center_sec * SR))
    start = center - length // 2
    end = start + length
    left = max(0, -start)
    right = max(0, end - n)
    start = max(0, start)
    end = min(n, end)
    clip = wav[:, start:end]
    if left or right:
        clip = torch.nn.functional.pad(clip, (left, right))
    if clip.shape[1] < length:
        clip = torch.nn.functional.pad(clip, (0, length - clip.shape[1]))
    return clip[:, :length]


def fade_mask(mask: np.ndarray, start: int, end: int, fade: int) -> None:
    start = max(0, start)
    end = min(len(mask), end)
    if end <= start:
        return
    mask[start:end] = 1.0
    fade = min(fade, (end - start) // 2)
    if fade > 0:
        mask[start:start + fade] *= np.linspace(0.0, 1.0, fade, endpoint=False)
        mask[end - fade:end] *= np.linspace(1.0, 0.0, fade, endpoint=False)


def safe_text(text: str) -> str:
    text = re.sub(r"[^0-9A-Za-z一-龯ぁ-んァ-ンー]", "_", text)
    return text[:18] or "word"


def merge_words(words: list[dict]) -> list[dict]:
    merged = []
    for word in words:
        if not merged or word["start"] - merged[-1]["end"] > MERGE_GAP_SEC:
            merged.append({"start": word["start"], "end": word["end"], "texts": [word["text"]], "words": [word]})
        else:
            merged[-1]["end"] = word["end"]
            merged[-1]["texts"].append(word["text"])
            merged[-1]["words"].append(word)
    return merged


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    word_dir = OUT / "accepted_words"
    seg_dir = OUT / "accepted_segments"
    word_dir.mkdir(exist_ok=True)
    seg_dir.mkdir(exist_ok=True)

    ref_paths = load_refs()
    neg_paths = load_negatives()
    refs = torch.cat([embed_file(path) for path in ref_paths], dim=0)
    negatives = torch.cat([embed_file(path) for path in neg_paths], dim=0)
    center = torch.nn.functional.normalize(refs.mean(0), dim=-1)
    print(f"device={device} refs={len(refs)} negatives={len(negatives)} window={WIN_SEC:.2f}s", flush=True)

    report = {
        "note": "Original-waveform word-level target gate; outputs are hard cuts, not source-separated reconstructions.",
        "device": device,
        "reference_count": len(refs),
        "negative_count": len(negatives),
        "thresholds": {
            "window_sec": WIN_SEC,
            "min_word_sec": MIN_WORD_SEC,
            "center_min": TARGET_CENTER_MIN,
            "p25_min": TARGET_P25_MIN,
            "margin_min": MARGIN_MIN,
            "edge_pad_sec": EDGE_PAD_SEC,
            "fade_sec": FADE_SEC,
            "merge_gap_sec": MERGE_GAP_SEC,
        },
        "files": [],
    }

    for source in INPUTS:
        wav, sr = torchaudio.load(str(source))
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        n = wav.shape[1]
        duration = n / SR
        words = transcribe(source)
        valid = [word for word in words if word["end"] - word["start"] >= MIN_WORD_SEC]
        windows = torch.cat([window_wave(wav, (w["start"] + w["end"]) / 2.0) for w in valid], dim=0) if valid else torch.zeros((0, int(WIN_SEC * SR)))
        embs = embed_batch(windows) if len(valid) else torch.zeros((0, refs.shape[1]))
        accepted = []
        scored_words = []
        for word, emb in zip(valid, embs):
            vals = emb @ refs.T
            neg_vals = emb @ negatives.T
            target_center = float(emb @ center)
            target_p25 = float(torch.quantile(vals, .25))
            target_top20 = float(torch.topk(vals, min(20, len(vals))).values.mean())
            negative_max = float(neg_vals.max())
            margin = target_center - negative_max
            row = {
                **word,
                "target_center": target_center,
                "target_p25": target_p25,
                "target_top20": target_top20,
                "negative_max": negative_max,
                "margin": margin,
                "accepted": bool(
                    target_center >= TARGET_CENTER_MIN
                    and target_p25 >= TARGET_P25_MIN
                    and margin >= MARGIN_MIN
                ),
            }
            scored_words.append(row)
            if row["accepted"]:
                accepted.append(row)

        mask = np.zeros(n, dtype=np.float32)
        pad = int(round(EDGE_PAD_SEC * SR))
        fade = int(round(FADE_SEC * SR))
        for idx, word in enumerate(accepted):
            start = int(round(word["start"] * SR)) - pad
            end = int(round(word["end"] * SR)) + pad
            fade_mask(mask, start, end, fade)
            start = max(0, start)
            end = min(n, end)
            clip = wav[:, start:end].numpy().T
            sf.write(str(word_dir / f"{source.stem}_word_{idx:03d}_{safe_text(word['text'])}.wav"), clip, SR, subtype="PCM_16")

        merged = merge_words(accepted)
        segment_rows = []
        for idx, segment in enumerate(merged):
            start = max(0.0, segment["start"] - EDGE_PAD_SEC)
            end = min(duration, segment["end"] + EDGE_PAD_SEC)
            st, en = int(round(start * SR)), int(round(end * SR))
            clip = wav[:, st:en].numpy().T
            out_path = seg_dir / f"{source.stem}_seg_{idx:03d}.wav"
            sf.write(str(out_path), clip, SR, subtype="PCM_16")
            segment_rows.append({"file": str(out_path), "start": start, "end": end, "text": "".join(segment["texts"]), "word_count": len(segment["words"])})

        hardcut = wav.numpy() * mask[None, :]
        hardcut_path = OUT / f"{source.stem}_target_hardcut.wav"
        sf.write(str(hardcut_path), hardcut.T, SR, subtype="PCM_16")
        file_row = {
            "source": str(source),
            "duration": duration,
            "asr_words": words,
            "scored_words": scored_words,
            "accepted_word_count": len(accepted),
            "accepted_words": accepted,
            "segments": segment_rows,
            "hardcut_file": str(hardcut_path),
        }
        report["files"].append(file_row)
        print(f"{source.name}: asr_words={len(words)} accepted={len(accepted)} segments={len(segment_rows)}", flush=True)
        for row in scored_words:
            print(f"  {row['start']:.2f}-{row['end']:.2f} {row['text']} center={row['target_center']:.3f} p25={row['target_p25']:.3f} margin={row['margin']:.3f} {'KEEP' if row['accepted'] else 'CUT'}", flush=True)

    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={REPORT}", flush=True)


if __name__ == "__main__":
    main()
