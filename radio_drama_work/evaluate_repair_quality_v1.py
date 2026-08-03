"""A/B gate for repaired audio before it can enter TTS training.

This is intentionally a veto-style gate: an enhancement or super-resolution
model may improve apparent brightness while changing the speaker identity or
hallucinating high-frequency content.  Such output is not allowed to replace
the original BS-Roformer waveform automatically.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

ROOT = Path(r"radio_drama_work/radio_drama_work")
PKG = Path(r"radio_drama_work/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "GPT_SoVITS" / "eres2net"))
os.chdir(PKG)

from ERes2NetV2 import ERes2NetV2
import kaldi as Kaldi

REF_DIR = PKG / "logs" / "my_voice" / "5-wav32k"
NEG_LIST = ROOT / "confirmed_non_target_or_mixed.txt"
INPUT = ROOT / "full_funasr_gold_v9" / "accepted_segments" / "gold_00000_243358894-1-192_vocals_0002232960_0002291840.wav"
OUTPUT = ROOT / "clearvoice_sr_smoke_v1" / "clearvoice_MossFormer2_SR_48K.wav"
REPORT = ROOT / "clearvoice_sr_smoke_v1" / "repair_quality_report.json"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SV = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
SV.load_state_dict(torch.load(PKG / "GPT_SoVITS" / "pretrained_models" / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt", map_location="cpu"))
SV.eval().to(DEVICE)


def read(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sr)


@torch.no_grad()
def embed(audio: np.ndarray, sr: int) -> torch.Tensor:
    wav = torch.from_numpy(audio).float().unsqueeze(0)
    if sr != 16000:
        wav = torch.nn.functional.interpolate(wav.unsqueeze(1), size=round(wav.shape[-1] * 16000 / sr), mode="linear", align_corners=False).squeeze(1)
    feat = torch.stack([Kaldi.fbank(w.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0) for w in wav]).to(DEVICE)
    return torch.nn.functional.normalize(SV.forward(feat), dim=-1).squeeze(0).cpu()


def rms_band(audio: np.ndarray, sr: int, lo: float, hi: float) -> float:
    spec = np.abs(librosa.stft(audio, n_fft=2048, hop_length=256, win_length=2048, center=True)) ** 2
    freqs = np.fft.rfftfreq(2048, 1.0 / sr)
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return 0.0
    return float(np.sqrt(np.mean(spec[mask]) + 1e-12))


def crest(audio: np.ndarray) -> float:
    return float(np.max(np.abs(audio)) / max(np.sqrt(np.mean(audio * audio) + 1e-12), 1e-8))


def main() -> None:
    refs = torch.stack([embed(*read(path)) for path in sorted(REF_DIR.glob("*.wav"))])
    negs = []
    for line in NEG_LIST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            path = Path(line.split("|", 1)[0].strip())
            if path.exists():
                negs.append(embed(*read(path)))
    neg = torch.stack(negs) if negs else None
    center = torch.nn.functional.normalize(refs.mean(0), dim=-1)
    rows = []
    for label, path in (("original_bsr_vocals", INPUT), ("clearvoice_sr_48k", OUTPUT)):
        audio, sr = read(path)
        emb = embed(audio, sr)
        target_scores = emb @ refs.T
        negative_max = float((emb @ neg.T).max()) if neg is not None else None
        low_hi = min(12000 if sr <= 32000 else 16000, sr / 2)
        high_lo = 12000 if sr <= 32000 else 16000
        high_hi = min(24000, sr / 2)
        low = rms_band(audio, sr, 0, low_hi)
        high = rms_band(audio, sr, high_lo, high_hi) if high_hi > high_lo else 0.0
        rows.append({
            "label": label,
            "file": str(path),
            "sample_rate": sr,
            "duration_sec": len(audio) / sr,
            "target_center": float(emb @ center),
            "target_top4": float(torch.topk(target_scores, k=min(4, len(target_scores))).values.mean()),
            "target_median": float(target_scores.median()),
            "known_negative_max": negative_max,
            "target_negative_margin": None if negative_max is None else float(emb @ center) - negative_max,
            "lowband_rms": low,
            "upper_band_rms": high,
            "upper_to_low_db": -120.0 if high <= 0 else float(20 * np.log10(high / max(low, 1e-8))),
            "peak": float(np.max(np.abs(audio))),
            "clipping_fraction": float(np.mean(np.abs(audio) >= 0.999)),
            "crest_factor": crest(audio),
        })
    before, after = rows
    criteria = {
        "speaker_identity": "target_center_after >= target_center_before - 0.02 AND target_negative_margin_after >= margin_before - 0.02",
        "no_hf_hallucination": "upper_to_low_db_after <= -10 dB for speech-only repair; if exceeded, manual review veto",
        "no_clipping": "clipping_fraction_after <= 0.001",
        "format": "mono PCM16, 32000 Hz after final downsampling",
        "decision": "automatic pass requires all criteria; otherwise repaired audio stays review-only and original remains training source",
    }
    report = {"device": DEVICE, "input": str(INPUT), "output": str(OUTPUT), "rows": rows, "criteria": criteria, "automatic_decision": "pass" if (after["target_center"] >= before["target_center"] - .02 and after["target_negative_margin"] >= before["target_negative_margin"] - .02 and after["upper_to_low_db"] <= -10 and after["clipping_fraction"] <= .001) else "review_only"}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
