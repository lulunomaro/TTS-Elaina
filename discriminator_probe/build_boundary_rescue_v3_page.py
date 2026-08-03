# -*- coding: utf-8 -*-
"""Build boundary_rescue_v3 spot-check page (Luna-style full page).

27 rescue variants from energy-onset detection. Grouped by origin for easy
comparison. Uses Luna's make_human_review_page_v2.py.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")
ROOT = Path(r"F:\GPTSOVITS\radio_drama_work")
MAKER = ROOT / "make_human_review_page_v2.py"
PY = r"F:\GPTSOVITS\GPT-SoVITS-v2pro-20250604-nvidia50\GPT-SoVITS-v2pro-20250604-nvidia50\runtime\python.exe"
OUT = PROBE / "boundary_rescue_v3"
WAVE_BINS = 32


def waveform(path: Path) -> list[float]:
    audio, _ = sf.read(str(path), dtype="float32", always_2d=True)
    mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
    if not len(mono):
        return [0.0] * WAVE_BINS
    edges = np.linspace(0, len(mono), WAVE_BINS + 1, dtype=int)
    values = []
    for i in range(WAVE_BINS):
        chunk = mono[edges[i]:edges[i + 1]]
        values.append(float(np.sqrt(np.mean(np.square(chunk))) if len(chunk) else 0.0))
    scale = max(max(values), 1e-6)
    return [round(min(1.0, v / scale), 4) for v in values]


def main():
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    samples = report["samples"]
    rows = []
    for r in samples:
        audio_path = Path(r["audio"])
        rows.append({
            "blind_id": r["blind_id"],
            "audio_rel": f"audio/{audio_path.name}",
            "output_duration": round(float(r["duration"]), 3),
            "wave": waveform(audio_path),
        })
    (OUT / "audit_manifest.json").write_text(
        json.dumps({"version": "boundary_rescue_v3_audit", "samples": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    delta_text = "；".join(f"{r['origin']}:{int(r['delta_ms'])}ms" for r in samples)
    cmd = [PY, str(MAKER), "--root", str(OUT), "--title", "边界救援v3·能量起音检测",
           "--round-label", "Boundary rescue v3 · energy onset · 27 variants",
           "--store-key", "ireina_boundary_rescue_v3",
           "--feedback-version", "boundary_rescue_v3_feedback",
           "--download-prefix", "boundary_rescue_v3_feedback",
           "--extra-note", f"每段给出能量检测起音点（偏移：{delta_text}）。选开头完整无切痕的版本；注意别把前一人声也带进来。"]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    print(result.stdout.strip() if result.returncode == 0 else result.stderr[-1000:], flush=True)


if __name__ == "__main__":
    main()
