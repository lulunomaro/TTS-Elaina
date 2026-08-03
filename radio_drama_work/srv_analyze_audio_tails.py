#!/usr/bin/env python3
"""Read-only tail residual/BGM screening for prompt and D_old audio."""
import math
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

def load(path):
    y, sr = sf.read(path, always_2d=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr

def stats(path):
    y, sr = load(path)
    n = len(y)
    def rms(a): return float(np.sqrt(np.mean(np.square(a)) + 1e-12))
    def db(x): return 20.0 * math.log10(max(x, 1e-6))
    tail = y[max(0, n-int(0.20*sr)):]
    prev = y[max(0, n-int(0.70*sr)):max(0, n-int(0.20*sr))]
    body = y[:max(1, n-int(0.70*sr))]
    # A short FFT of the last 200 ms: 5-12 kHz activity is a useful BGM hint.
    z = tail * np.hanning(len(tail))
    spec = np.abs(np.fft.rfft(z, n=max(2048, 1 << (len(z)-1).bit_length())))
    freqs = np.fft.rfftfreq(len(spec)*2-2, 1.0/sr)
    band = float(np.mean(spec[(freqs >= 5000) & (freqs <= min(12000, sr/2))]))
    low = float(np.mean(spec[(freqs >= 80) & (freqs <= 1000)]))
    return {
        "file": str(path), "sec": n/sr, "body_db": db(rms(body)),
        "prev_db": db(rms(prev)), "tail_db": db(rms(tail)),
        "tail_minus_prev_db": db(rms(tail))-db(rms(prev)),
        "tail_high_low": band/max(low, 1e-7),
    }

paths = []
if len(sys.argv) > 1 and sys.argv[1] != "--list":
    paths.extend(Path(x) for x in sys.argv[1:])
else:
    paths.append(Path("/root/autodl-tmp/dataset_v1/wavs/old_my_voice.mp3_0014764160_0014986560.wav.refmt_vocals.wav"))
    paths.append(Path("/root/autodl-tmp/dataset_v1/wavs/old_my_voice.mp3_0020053120_0020203520.wav.refmt_vocals.wav"))
    paths.append(Path("/root/autodl-tmp/dataset_v1/wavs/old_my_voice.mp3_0048846080_0049265600.wav.refmt_vocals.wav"))
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        list_path = Path("/root/autodl-tmp/ireina_v2/data/D_old.list")
        paths.extend(Path(line.split("|", 1)[0]) for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip())

for path in paths:
    try:
        r = stats(path)
        print("{file}/tsec={sec:.3f}/tbody_db={body_db:.1f}/tprev_db={prev_db:.1f}/ttail_db={tail_db:.1f}/ttail_minus_prev_db={tail_minus_prev_db:.1f}/ttail_high_low={tail_high_low:.2f}".format(**r), flush=True)
    except Exception as exc:
        print(f"FAIL/t{path}/t{type(exc).__name__}: {exc}", flush=True)
