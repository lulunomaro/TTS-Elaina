"""Evaluate ClearVoice SR probe outputs against their raw BS-Roformer inputs."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

ROOT = Path(r"radio_drama_work/radio_drama_work")
EVAL_PATH = ROOT / "evaluate_repair_quality_v1.py"
SPEC = importlib.util.spec_from_file_location("repair_gate", EVAL_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(EVAL_PATH)
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)

OUT_DIR = ROOT / os.environ.get("REPAIR_OUT_DIR", "clearvoice_sr_probe_v1")
INPUT_DIR = ROOT / "full_funasr_gold_v9" / "accepted_segments"
REPORT = OUT_DIR / "repair_probe_quality_report.json"


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
        "lowband_rms": GATE.rms_band(audio, sr, 0, low_hi),
        "upper_band_rms": high,
        "upper_to_low_db": -120.0 if high <= 0 else float(20 * np.log10(high / max(low, 1e-8))),
        "peak": float(np.max(np.abs(audio))),
        "clipping_fraction": float(np.mean(np.abs(audio) >= 0.999)),
        "crest_factor": GATE.crest(audio),
    }


def read(path: Path) -> tuple[np.ndarray, int]:
    return GATE.read(path)


def main() -> None:
    refs = torch.stack([GATE.embed(*read(path)) for path in sorted(GATE.REF_DIR.glob("*.wav"))])
    negs = []
    for line in GATE.NEG_LIST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            path = Path(line.split("|", 1)[0].strip())
            if path.exists():
                negs.append(GATE.embed(*read(path)))
    neg = torch.stack(negs) if negs else None
    center = torch.nn.functional.normalize(refs.mean(0), dim=-1)
    rows = []
    sr_outputs = sorted(OUT_DIR.glob("sr_*.wav"))
    for output in sr_outputs:
        name = output.name
        marker = name.find("_gold_")
        if marker < 0:
            continue
        input_name = name[marker + 1:]
        input_path = INPUT_DIR / input_name
        if not input_path.exists():
            continue
        raw, raw_sr = read(input_path)
        repaired, repaired_sr = read(output)
        final = librosa.resample(repaired, orig_sr=repaired_sr, target_sr=32000).astype(np.float32)
        final_path = OUT_DIR / (output.stem + "_32k.wav")
        sf.write(str(final_path), final, 32000, subtype="PCM_16")
        final, final_sr = read(final_path)
        before = metric(raw, raw_sr, refs, neg, center)
        after_48k = metric(repaired, repaired_sr, refs, neg, center)
        after = metric(final, final_sr, refs, neg, center)
        identity_ok = after["target_center"] >= before["target_center"] - 0.02 and after["target_negative_margin"] >= before["target_negative_margin"] - 0.02
        hf_ok = after["upper_to_low_db"] <= max(-10.0, before["upper_to_low_db"] + 10.0)
        clip_ok = after["clipping_fraction"] <= 0.001
        row = {
            "input": str(input_path),
            "sr_output_48k": str(output),
            "final_training_format": str(final_path),
            "before_raw": before,
            "after_sr_48k": after_48k,
            "after_final_32k": after,
            "gate": {
                "speaker_identity": identity_ok,
                "high_frequency_safety": hf_ok,
                "no_clipping": clip_ok,
                "decision": "pass" if identity_ok and hf_ok and clip_ok else "review_only",
                "reasons": [
                    reason for reason, ok in (
                        ("speaker_identity_changed", identity_ok),
                        ("high_frequency_excess", hf_ok),
                        ("clipping", clip_ok),
                    ) if not ok
                ],
            },
        }
        rows.append(row)
        print(json.dumps({"file": output.name, "decision": row["gate"]["decision"], "reasons": row["gate"]["reasons"], "target_center_before": before["target_center"], "target_center_after": after["target_center"]}, ensure_ascii=False), flush=True)
    report = {
        "device": GATE.DEVICE,
        "probe_inputs": len(rows),
        "automatic_pass_count": sum(row["gate"]["decision"] == "pass" for row in rows),
        "review_only_count": sum(row["gate"]["decision"] != "pass" for row in rows),
        "criteria": {
            "speaker_identity": "final 32k target_center and target-negative margin may drop at most 0.02 from raw input",
            "high_frequency_safety": "final 32k upper-band ratio <= max(-10 dB, raw ratio + 10 dB)",
            "no_clipping": "clipping_fraction <= 0.001",
            "training_policy": "only automatic pass may be considered for a later manual-listening stage; this probe never modifies an existing training list",
        },
        "rows": rows,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "pass": report["automatic_pass_count"], "review_only": report["review_only_count"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
