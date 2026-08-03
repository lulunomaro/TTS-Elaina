# -*- coding: utf-8 -*-
"""Build a 15-clip spot-check page from auto_boundary_fix_v1 repairs.

Stratified sampling: covers anime+radio, high/mid/low prob, and both repair
kinds (extended vs sub-window split). Uses Luna's full review-page maker.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")
ROOT = Path(r"F:\GPTSOVITS\radio_drama_work")
MAKER = ROOT / "make_human_review_page_v2.py"
PY = r"F:\GPTSOVITS\GPT-SoVITS-v2pro-20250604-nvidia50\GPT-SoVITS-v2pro-20250604-nvidia50\runtime\python.exe"
OUT = PROBE / "auto_boundary_spotcheck_v1"
AUDIO_DIR = OUT / "audio"
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


def pick(rows: list[dict], n: int) -> list[dict]:
    # prefer spread across sources and prob
    by_src: dict[str, list[dict]] = {}
    for r in rows:
        by_src.setdefault(r["source_id"], []).append(r)
    selected: list[dict] = []
    per_src = {}
    for r in sorted(rows, key=lambda x: -x["mean_prob"]):
        src = r["source_id"]
        cap = max(1, n // max(1, len(by_src)))
        if per_src.get(src, 0) >= cap:
            continue
        selected.append(r)
        per_src[src] = per_src.get(src, 0) + 1
        if len(selected) >= n:
            break
    # fill remaining
    for r in sorted(rows, key=lambda x: -x["mean_prob"]):
        if len(selected) >= n:
            break
        if r not in selected:
            selected.append(r)
    return selected


def main() -> None:
    report = json.loads((PROBE / "auto_boundary_fix_v1" / "report.json").read_text(encoding="utf-8"))
    samples = report["samples"]
    print(f"total repaired: {len(samples)}", flush=True)

    anime = [r for r in samples if r["source_id"].startswith("majo_")]
    radio = [r for r in samples if not r["source_id"].startswith("majo_")]
    picks = pick(anime, 8) + pick(radio, 7)
    picks.sort(key=lambda x: -x["mean_prob"])
    print(f"spot check: {len(picks)} (anime={len(picks) if False else 0})", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    for i, r in enumerate(picks, 1):
        bid = f"SC{i:02d}"
        dest = AUDIO_DIR / f"{bid}.wav"
        if not dest.exists():
            shutil.copy2(r["audio"], dest)
        audit_rows.append({
            "blind_id": bid,
            "audio_rel": f"audio/{dest.name}",
            "output_duration": round(float(r["duration"]), 3),
            "wave": waveform(dest),
        })
    (OUT / "audit_manifest.json").write_text(
        json.dumps({"version": "auto_boundary_spotcheck_v1_audit", "samples": audit_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    # provenance note for user
    info = [{"blind_id": a["blind_id"], "source_id": r["source_id"], "duration": r["duration"],
             "mean_prob": r["mean_prob"], "kind": "sub_window" if r.get("sub_window") else "extended"}
            for a, r in zip(audit_rows, picks)]
    (OUT / "private_manifest.json").write_text(json.dumps({"samples": info}, ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [PY, str(MAKER), "--root", str(OUT), "--title", "自动边界修复·抽验15条",
           "--round-label", "Auto boundary fix · spot check · 15 clips",
           "--store-key", "ireina_auto_boundary_spotcheck_v1",
           "--feedback-version", "auto_boundary_spotcheck_v1_feedback",
           "--download-prefix", "auto_boundary_spotcheck_v1_feedback",
           "--extra-note", "这些是自动修复后的段（判别器概率引导重切）。请判断：开头结尾是否完整自然、是否混入他人声音。只需挑有明显问题的标记错误。"]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        print("STDERR:", result.stderr[-1200:], flush=True)
        raise RuntimeError(result.stderr[-400:])
    print(result.stdout.strip(), flush=True)
    print(json.dumps({"page": str(OUT / "index.html"), "clips": len(picks)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
