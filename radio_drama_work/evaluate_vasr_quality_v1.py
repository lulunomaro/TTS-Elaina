"""Crop and gate the AudioSR speech output before any training use."""
from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

ROOT = Path(r"radio_drama_work/radio_drama_work")
GATE_PATH = ROOT / "evaluate_repair_quality_v1.py"
SPEC = importlib.util.spec_from_file_location("repair_gate", GATE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(GATE_PATH)
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)

INPUT = ROOT / "full_funasr_gold_v9" / "accepted_segments" / "gold_00000_243358894-1-192_vocals_0002232960_0002291840.wav"
VASR = ROOT / "vasr_smoke_v1" / "vasr_speech_gold_00000_243358894-1-192_vocals_0002232960_0002291840.wav"
OUT = ROOT / "vasr_smoke_v1" / "vasr_speech_gold_00000_cropped_32k.wav"
REPORT = ROOT / "vasr_smoke_v1" / "vasr_quality_report.json"


def metric(audio: np.ndarray, sr: int, refs: torch.Tensor, neg: torch.Tensor | None, center: torch.Tensor) -> dict:
    emb = GATE.embed(audio, sr)
    target_scores = emb @ refs.T
    negative_max = float((emb @ neg.T).max()) if neg is not None else None
    low_hi = min(12000 if sr <= 32000 else 16000, sr / 2)
    high_lo = 12000 if sr <= 32000 else 16000
    high_hi = min(24000, sr / 2)
    low = GATE.rms_band(audio, sr, 0, low_hi)
    high = GATE.rms_band(audio, sr, high_lo, high_hi) if high_hi > high_lo else 0.0
    return {
        "sample_rate": sr,
        "duration_sec": len(audio) / sr,
        "target_center": float(emb @ center),
        "target_top4": float(torch.topk(target_scores, k=min(4, len(target_scores))).values.mean()),
        "target_median": float(target_scores.median()),
        "known_negative_max": negative_max,
        "target_negative_margin": None if negative_max is None else float(emb @ center) - negative_max,
        "upper_to_low_db": -120.0 if high <= 0 else float(20 * np.log10(high / max(low, 1e-8))),
        "peak": float(np.max(np.abs(audio))),
        "clipping_fraction": float(np.mean(np.abs(audio) >= .999)),
        "crest_factor": GATE.crest(audio),
    }


def main() -> None:
    raw, raw_sr = GATE.read(INPUT)
    generated, generated_sr = GATE.read(VASR)
    generated = librosa.resample(generated, orig_sr=generated_sr, target_sr=32000).astype(np.float32)
    target_duration = len(raw) / raw_sr
    target_frames = round(target_duration * 32000)
    if len(generated) < target_frames:
        raise RuntimeError(f"VASR output shorter than input after resample: {len(generated)} < {target_frames}")
    generated = generated[:target_frames]
    sf.write(str(OUT), generated, 32000, subtype="PCM_16")

    refs = torch.stack([GATE.embed(*GATE.read(path)) for path in sorted(GATE.REF_DIR.glob("*.wav"))])
    negatives = []
    for line in GATE.NEG_LIST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            path = Path(line.split("|", 1)[0].strip())
            if path.exists():
                negatives.append(GATE.embed(*GATE.read(path)))
    neg = torch.stack(negatives) if negatives else None
    center = torch.nn.functional.normalize(refs.mean(0), dim=-1)
    before = metric(raw, raw_sr, refs, neg, center)
    after = metric(generated, 32000, refs, neg, center)
    identity_ok = after["target_center"] >= before["target_center"] - .02 and after["target_negative_margin"] >= before["target_negative_margin"] - .02
    hf_ok = after["upper_to_low_db"] <= max(-10.0, before["upper_to_low_db"] + 10.0)
    clip_ok = after["clipping_fraction"] <= .001
    report = {
        "input": str(INPUT),
        "vasr_raw_output": str(VASR),
        "final_32k_cropped_output": str(OUT),
        "before_raw": before,
        "after_vasr_cropped_32k": after,
        "criteria": {"speaker_identity": "target center and target-negative margin drop <= 0.02", "high_frequency_safety": "upper-band ratio <= max(-10 dB, raw + 10 dB)", "no_clipping": "clipping_fraction <= 0.001", "duration": "output cropped to exact input duration"},
        "automatic_decision": "pass" if identity_ok and hf_ok and clip_ok else "review_only",
        "reasons": [name for name, ok in (("speaker_identity_changed", identity_ok), ("high_frequency_excess", hf_ok), ("clipping", clip_ok)) if not ok],
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
