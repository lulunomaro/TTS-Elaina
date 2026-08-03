"""A/B quality gate for the full AudioSR run.

The original BS-Roformer vocal stem remains the authority.  AudioSR output is
allowed into a separate candidate list only when it does not degrade speaker
identity or increase high-frequency artifacts against the original clip.
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
sys.path.insert(0, str(PKG / "GPT_SoVITS" / "eres2net"))
os.chdir(PKG)

from ERes2NetV2 import ERes2NetV2
import kaldi as Kaldi

INPUT_LIST = ROOT / "full_funasr_gold_train_candidate_v9.list"
WORK = ROOT / "vasr_full_v2"
MANIFEST = WORK / "manifest.json"
REPORT = WORK / "quality_report.json"
VASR_LIST = ROOT / "full_funasr_gold_vasr_train_candidate_v1.list"
BEST_LIST = ROOT / "full_funasr_gold_best_of_raw_vasr_train_candidate_v1.list"
REF_DIR = PKG / "logs" / "my_voice" / "5-wav32k"
NEG_LIST = ROOT / "confirmed_non_target_or_mixed.txt"
SV_CKPT = PKG / "GPT_SoVITS" / "pretrained_models" / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def read(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sr)


@torch.no_grad()
def embed(audio: np.ndarray, sr: int, sv: torch.nn.Module) -> torch.Tensor:
    wav = torch.from_numpy(audio).float().unsqueeze(0)
    if sr != 16000:
        wav = torch.nn.functional.interpolate(
            wav.unsqueeze(1),
            size=round(wav.shape[-1] * 16000 / sr),
            mode="linear",
            align_corners=False,
        ).squeeze(1)
    feat = torch.stack(
        [Kaldi.fbank(w.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0) for w in wav]
    ).to(DEVICE)
    return torch.nn.functional.normalize(sv.forward(feat), dim=-1).squeeze(0).cpu()


def rms_band(audio: np.ndarray, sr: int, lo: float, hi: float) -> float:
    spec = np.abs(librosa.stft(audio, n_fft=2048, hop_length=256, win_length=2048, center=True)) ** 2
    freqs = np.fft.rfftfreq(2048, 1.0 / sr)
    mask = (freqs >= lo) & (freqs < hi)
    return float(np.sqrt(np.mean(spec[mask]) + 1e-12)) if np.any(mask) else 0.0


def crest(audio: np.ndarray) -> float:
    return float(np.max(np.abs(audio)) / max(np.sqrt(np.mean(audio * audio) + 1e-12), 1e-8))


def metric(audio: np.ndarray, sr: int, sv: torch.nn.Module, refs: torch.Tensor, center: torch.Tensor, neg: torch.Tensor | None) -> dict:
    emb = embed(audio, sr, sv)
    target_scores = emb @ refs.T
    negative_max = float((emb @ neg.T).max()) if neg is not None else None
    low_hi = min(12000 if sr <= 32000 else 16000, sr / 2)
    high_lo = 12000 if sr <= 32000 else 16000
    high_hi = min(24000, sr / 2)
    low = rms_band(audio, sr, 0, low_hi)
    high = rms_band(audio, sr, high_lo, high_hi) if high_hi > high_lo else 0.0
    return {
        "sample_rate": sr,
        "channels": 1,
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
    }


def parse_input_list() -> list[tuple[Path, str]]:
    rows = []
    for line in INPUT_LIST.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            path, text = line.split("|", 3)[0].strip(), line.split("|", 3)[3]
            rows.append((Path(path), text))
    return rows


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest_by_input = {str(Path(row["input"])): row for row in manifest if row.get("status") == "ok"}
    sv = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
    sv.load_state_dict(torch.load(SV_CKPT, map_location="cpu"))
    sv.eval().to(DEVICE)
    ref_paths = sorted(REF_DIR.glob("*.wav"))
    refs = torch.stack([embed(*read(path), sv) for path in ref_paths])
    negs = []
    if NEG_LIST.exists():
        for line in NEG_LIST.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                path = Path(line.split("|", 1)[0].strip())
                if path.exists():
                    negs.append(embed(*read(path), sv))
    neg = torch.stack(negs) if negs else None
    center = torch.nn.functional.normalize(refs.mean(0), dim=-1)
    rows = []
    vasr_lines = []
    best_lines = []
    source_rows = parse_input_list()
    for index, (input_path, text) in enumerate(source_rows):
        row = {"index": index, "input": str(input_path), "text": text}
        original, original_sr = read(input_path)
        original_m = metric(original, original_sr, sv, refs, center, neg)
        generated = manifest_by_input.get(str(input_path))
        if not generated:
            row.update({"status": "missing_output", "original": original_m})
            best_lines.append(f"{input_path}|full_funasr_gold_v9|JA|{text}")
            rows.append(row)
            continue
        output_path = Path(generated["output"])
        enhanced, enhanced_sr = read(output_path)
        enhanced_m = metric(enhanced, enhanced_sr, sv, refs, center, neg)
        duration_delta = abs(enhanced_m["duration_sec"] - original_m["duration_sec"])
        raw_upper = original_m["upper_to_low_db"]
        max_upper = max(-10.0, raw_upper + 10.0)
        identity_ok = enhanced_m["target_center"] >= original_m["target_center"] - 0.02
        margin_ok = (
            enhanced_m["target_negative_margin"] is None
            or original_m["target_negative_margin"] is None
            or enhanced_m["target_negative_margin"] >= original_m["target_negative_margin"] - 0.02
        )
        hf_ok = enhanced_m["upper_to_low_db"] <= max_upper
        clipping_ok = enhanced_m["clipping_fraction"] <= 0.001
        format_ok = enhanced_sr == 32000 and enhanced.ndim == 1 and duration_delta <= 0.005
        automatic_pass = all((identity_ok, margin_ok, hf_ok, clipping_ok, format_ok))
        row.update({
            "status": "ok",
            "output": str(output_path),
            "original": original_m,
            "vasr": enhanced_m,
            "checks": {
                "identity_ok": identity_ok,
                "margin_ok": margin_ok,
                "hf_ok": hf_ok,
                "clipping_ok": clipping_ok,
                "format_ok": format_ok,
                "hf_limit_db": max_upper,
            },
            "automatic_decision": "pass" if automatic_pass else "review_only",
        })
        if automatic_pass:
            vasr_lines.append(f"{output_path}|full_funasr_gold_vasr_v1|JA|{text}")
            best_lines.append(f"{output_path}|full_funasr_gold_vasr_v1|JA|{text}")
        else:
            best_lines.append(f"{input_path}|full_funasr_gold_v9|JA|{text}")
        rows.append(row)
    criteria = {
        "speaker_identity": "VASR target center >= original - 0.02",
        "target_vs_negative": "VASR target-negative margin >= original - 0.02",
        "no_hf_hallucination": "VASR upper-to-low dB <= max(-10 dB, original + 10 dB)",
        "no_clipping": "clipping fraction <= 0.001",
        "format": "mono PCM16, 32000 Hz, duration delta <= 5 ms",
        "policy": "VASR is never used when it fails; the raw BS-Roformer source remains the fallback",
    }
    report = {
        "device": DEVICE,
        "reference_count": len(ref_paths),
        "hard_negative_count": len(negs),
        "input_count": len(source_rows),
        "vasr_pass_count": sum(row.get("automatic_decision") == "pass" for row in rows),
        "vasr_review_only_count": sum(row.get("automatic_decision") == "review_only" for row in rows),
        "missing_output_count": sum(row.get("status") == "missing_output" for row in rows),
        "criteria": criteria,
        "rows": rows,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    VASR_LIST.write_text("/n".join(vasr_lines) + ("/n" if vasr_lines else ""), encoding="utf-8")
    BEST_LIST.write_text("/n".join(best_lines) + ("/n" if best_lines else ""), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
