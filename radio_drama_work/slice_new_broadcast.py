# -*- coding: utf-8 -*-
"""切分新增广播剧分离人声，后续只对切片做 speaker/串音筛选。"""

import os
import sys

import numpy as np
from scipy.io import wavfile


PKG = r"radio_drama_work/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50"
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "tools"))
from tools.my_utils import load_audio  # noqa: E402
from slicer2 import Slicer  # noqa: E402


INPUT_DIR = r"radio_drama_work/new_vocals"
OUTPUT_DIR = r"radio_drama_work/new_slices"
os.makedirs(OUTPUT_DIR, exist_ok=True)

slicer = Slicer(
    sr=32000,
    threshold=-40,
    min_length=1800,
    min_interval=300,
    hop_size=20,
    max_sil_kept=500,
)

total = 0
for name in sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".wav")):
    path = os.path.join(INPUT_DIR, name)
    audio = load_audio(path, 32000)
    count = 0
    for chunk, start, end in slicer.slice(audio):
        peak = float(np.abs(chunk).max())
        if peak <= 0:
            continue
        if peak > 1:
            chunk = chunk / peak
        chunk = (chunk / max(float(np.abs(chunk).max()), 1e-8) * (0.9 * 0.25)) + 0.75 * chunk
        out = os.path.join(OUTPUT_DIR, f"{name[:-4]}_{start:010d}_{end:010d}.wav")
        wavfile.write(out, 32000, (chunk * 32767).astype(np.int16))
        count += 1
    total += count
    print(f"{name}: {count}", flush=True)
print(f"total={total}", flush=True)
