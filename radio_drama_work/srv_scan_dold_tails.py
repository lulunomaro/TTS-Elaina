#!/usr/bin/env python3
"""Rank D_old files by possible tail residual / music contamination."""
import math
from pathlib import Path
import numpy as np
import soundfile as sf

rows=[]
for line in Path('/root/autodl-tmp/ireina_v2/data/D_old.list').read_text(encoding='utf-8').splitlines():
    if not line.strip(): continue
    path=line.split('|',1)[0]
    try:
        y,sr=sf.read(path,always_2d=False)
        y=np.asarray(y,dtype=np.float32)
        if y.ndim>1:y=y.mean(axis=1)
        n=len(y); w=int(.5*sr)
        tail=y[max(0,n-w):]; prev=y[max(0,n-2*w):max(0,n-w)]
        if len(prev)<1000: continue
        rd=lambda z:20*math.log10(max(float(np.sqrt(np.mean(z*z)+1e-12)),1e-6))
        tail_db=rd(tail); prev_db=rd(prev); delta=tail_db-prev_db
        z=tail*np.hanning(len(tail)); sp=np.abs(np.fft.rfft(z,n=max(4096,1<<(len(z)-1).bit_length())))
        fr=np.fft.rfftfreq(len(sp)*2-2,1/sr)
        band=sp[(fr>=5000)&(fr<=min(12000,sr/2))]
        low=sp[(fr>=80)&(fr<=1000)]
        hi_low=float(np.mean(band)/max(np.mean(low),1e-7))
        peak=float(np.max(band)/max(np.median(band),1e-7)) if len(band) else 0.0
        rows.append((delta,tail_db,hi_low,peak,n/sr,path))
    except Exception as exc:
        print(f'FAIL {path} {exc}',flush=True)
print('TOP tail residual')
for r in sorted(rows,reverse=True)[:20]: print('delta=%6.1f tail=%6.1f hi_low=%6.2f peak=%6.1f sec=%5.2f %s'%r)
print('TOP high narrowband')
for r in sorted(rows,key=lambda x:x[3],reverse=True)[:20]: print('delta=%6.1f tail=%6.1f hi_low=%6.2f peak=%6.1f sec=%5.2f %s'%r)
