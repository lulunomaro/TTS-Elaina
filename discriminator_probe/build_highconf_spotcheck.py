# -*- coding: utf-8 -*-
"""Build a 15-clip spot-check page from the remaining p>=0.9 pool.

Stratified: anime/radio balance + source spread + prob range. The user's pass
rate on these 15 is extrapolated to the 243-clip pool. Uses Luna's full
review page maker (waveform, autosave, export).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")
ROOT = Path(r"F:\GPTSOVITS\radio_drama_work")
MAKER = ROOT / "make_human_review_page_v2.py"
PY = r"F:\GPTSOVITS\GPT-SoVITS-v2pro-20250604-nvidia50\GPT-SoVITS-v2pro-20250604-nvidia50\runtime\python.exe"
OUT = PROBE / "highconf_spotcheck_v1"
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


def remaining(path, heard, min_p=0.9):
    lines = [l for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return [l for l in lines
            if Path(l.split("|")[0]).name.replace(".wav", "") not in heard
            and float(l.split("|")[1].split("=")[1]) >= min_p]


def pick(rows, n):
    by_src = {}
    for l in rows:
        sid = l.split("|")[2]
        by_src.setdefault(sid, []).append(l)
    selected = []
    per_src = {}
    for l in sorted(rows, key=lambda x: -float(x.split("|")[1].split("=")[1])):
        sid = l.split("|")[2]
        cap = max(1, n // max(1, len(by_src)))
        if per_src.get(sid, 0) >= cap:
            continue
        selected.append(l)
        per_src[sid] = per_src.get(sid, 0) + 1
        if len(selected) >= n:
            break
    return selected


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    heard = set()
    for v in ("v1", "v2", "v3"):
        a = json.loads((PROBE / f"discriminator_priority_review_{v}" / "audit_manifest.json").read_text(encoding="utf-8"))
        for r in a.get("samples", a.get("rows", [])):
            heard.add(str(r.get("origin_blind_id") or r.get("blind_id") or ""))

    anime = remaining(PROBE / "discriminator_rank_v3" / "anime_review_431_scored_v3.list", heard)
    radio = remaining(PROBE / "discriminator_rank_v3" / "radio_review_448_scored_v3.list", heard)
    print(f"remaining: anime={len(anime)} radio={len(radio)}", flush=True)

    picks = pick(anime, 8) + pick(radio, 7)
    print(f"picked: {len(picks)}", flush=True)

    audit_rows = []
    info = []
    for i, l in enumerate(picks, 1):
        parts = l.split("|")
        src_path = Path(parts[0])
        bid = f"HC{i:02d}"
        dest = AUDIO_DIR / f"{bid}.wav"
        if not dest.exists():
            shutil.copy2(src_path, dest)
        audit_rows.append({
            "blind_id": bid,
            "audio_rel": f"audio/{dest.name}",
            "output_duration": round(float(parts[4].replace("s", "")), 3),
            "wave": waveform(dest),
        })
        info.append({"blind_id": bid, "origin": src_path.name.replace(".wav", ""),
                     "source_id": parts[2], "prob": float(parts[1].split("=")[1]),
                     "duration": float(parts[4].replace("s", ""))})
    (OUT / "audit_manifest.json").write_text(
        json.dumps({"version": "highconf_spotcheck_v1_audit", "samples": audit_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (OUT / "private_manifest.json").write_text(
        json.dumps({"version": "highconf_spotcheck_v1_private", "samples": info}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    cmd = [PY, str(MAKER), "--root", str(OUT), "--title", "高置信池·抽验15条",
           "--round-label", "High-confidence pool · spot check · 15 clips",
           "--store-key", "ireina_highconf_spotcheck_v1",
           "--feedback-version", "highconf_spotcheck_v1_feedback",
           "--download-prefix", "highconf_spotcheck_v1_feedback",
           "--extra-note", "这批是判别器 p>=0.9 的机器高置信段。抽验通过率将用于推算剩余 243 条的自动并入比例。请只标记：非目标 / 混合 / 开头结尾切断 / 音质 / 重叠 / 不确定。"]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    print(result.stdout.strip() if result.returncode == 0 else result.stderr[-800:], flush=True)
    print(json.dumps({"page": str(OUT / "index.html"), "clips": len(picks)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
