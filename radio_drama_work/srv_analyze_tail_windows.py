#!/usr/bin/env python3
"""Print 100 ms RMS/high-band windows over the final 1.2 seconds."""
import math
import sys
import numpy as np
import soundfile as sf

for name in sys.argv[1:]:
    y, sr = sf.read(name, always_2d=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1: y = y.mean(axis=1)
    print(f"FILE {name} sec={len(y)/sr:.3f}")
    end = len(y)
    for ms in range(1200, -1, -100):
        a = max(0, end-int((ms+100)*sr/1000))
        b = max(0, end-int(ms*sr/1000))
        z = y[a:b]
        rms = 20*math.log10(max(float(np.sqrt(np.mean(z*z)+1e-12)),1e-6))
        if len(z) > 8:
            sp = np.abs(np.fft.rfft(z*np.hanning(len(z))))
            fr = np.fft.rfftfreq(len(sp)*2-2,1/sr)
            hi = float(np.mean(sp[(fr>=5000)&(fr<=min(12000,sr/2))]))
            lo = float(np.mean(sp[(fr>=80)&(fr<=1000)]))
            ratio = hi/max(lo,1e-7)
        else: ratio = 0.0
        print(f"  end-{ms+100:4d}..end-{ms:4d}ms rms={rms:6.1f}dB hi_low={ratio:.2f}")
