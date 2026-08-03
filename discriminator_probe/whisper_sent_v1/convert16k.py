# -*- coding: utf-8 -*-
"""Convert all 19+ sources to 16k mono wav via ffmpeg (fast), then exit."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"F:\GPTSOVITS\radio_drama_work")
OUT = Path(r"F:\GPTSOVITS\discriminator_probe\whisper_sent_v1\audio16k")
OUT.mkdir(parents=True, exist_ok=True)
FFMPEG = r"F:\GPTSOVITS\GPT-SoVITS-v2pro-20250604-nvidia50\GPT-SoVITS-v2pro-20250604-nvidia50\runtime\ffmpeg.exe"

sources = []
anime = json.loads((ROOT / "anime_batch_v1" / "manifest.json").read_text(encoding="utf-8"))
radio = json.loads((ROOT / "full_track_manifest_v1.json").read_text(encoding="utf-8"))
for t in anime.get("tracks", []):
    if t["id"].startswith("majo_"):
        sources.append((t["id"], t["vocals"]))
for t in radio.get("tracks", []):
    sources.append((t["id"], t["vocals"]))

t0 = time.time()
for sid, vocals in sources:
    dest = OUT / f"{sid}.wav"
    if dest.exists():
        continue
    r = subprocess.run([FFMPEG, "-y", "-i", str(vocals), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dest)],
                       capture_output=True)
    if r.returncode != 0:
        print("FAIL", sid, r.stderr.decode("utf-8", "ignore")[-200:], flush=True)
    else:
        print(f"ok {sid} elapsed={time.time()-t0:.0f}s", flush=True)
print("ALL DONE", flush=True)
