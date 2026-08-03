# -*- coding: utf-8 -*-
"""Dedup check: compare audio signatures between confirmed 204 list and highconf 239 list."""
import io
import numpy as np
import soundfile as sf
from pathlib import Path

ROOT = Path(r"F:\GPTSOVITS\radio_drama_work")
PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")

confirmed = ROOT / "human_confirmed_radio_anime_v3_train_1p5_6s.list"
conf_wavs = [l.split("|")[0] for l in confirmed.read_text(encoding="utf-8").splitlines() if l.strip()]
# boundary fixes
extra = []
for p, pat in [
    (PROBE / "boundary_fix_v4_clean.list", None),
    (PROBE / "boundary_fix_v4_final.list", None),
    (PROBE / "boundary_all_final.list", None),
]:
    if p.exists():
        for l in p.read_text(encoding="utf-8").splitlines():
            if l.strip():
                extra.append(l.split("|")[0])
high_list = ROOT / "highconf_final_v1" / "highconf_final_v1.list"
high_wavs = [str(ROOT / "highconf_final_v1" / l.split("|")[0]) for l in high_list.read_text(encoding="utf-8").splitlines() if l.strip()]

sig_cache = {}


def sig(path):
    if path in sig_cache:
        return sig_cache[path]
    try:
        a, sr = sf.read(path, dtype="float32", always_2d=True)
        m = np.asarray(a.mean(axis=1), dtype=np.float64)
    except Exception:
        sig_cache[path] = None
        return None
    dur = len(m) / sr
    if len(m) < sr:
        sig_cache[path] = None
        return None
    # signature: rms in 25 bins over first 1.5s + last 1.5s
    def bins(seg):
        n = len(seg)
        k = 25
        idx = (np.linspace(0, n, k + 1)[:-1]).astype(int)
        idx2 = (np.linspace(0, n, k + 1)[1:]).astype(int)
        idx2 = np.maximum(idx2, idx + 1)
        rms = np.sqrt(np.add.reduceat(np.square(seg), idx)[:k] / np.maximum(1, np.diff(np.linspace(0, n, k + 1).astype(int))[:k]) + 1e-12)
        return rms
    b1 = bins(m[: int(1.5 * sr)])
    b2 = bins(m[-int(1.5 * sr):])
    s = np.round(np.concatenate([b1, b2]) / 0.02).astype(np.int16).tobytes()
    sig_cache[path] = (dur, s)
    return sig_cache[path]


def close(a, b, tol=0.15):
    if a is None or b is None:
        return False
    if abs(a[0] - b[0]) > tol:
        return False
    return a[1] == b[1]


all_conf = conf_wavs + extra
n_conf = len(all_conf)
dups = []
for i, cp in enumerate(all_conf):
    cs = sig(cp)
    if cs is None:
        continue
    for hp in high_wavs:
        if close(cs, sig(hp)):
            dups.append((cp, hp))
            break
print(f"confirmed: {n_conf} (incl. {len(extra)} boundary-fix) | highconf: {len(high_wavs)}")
print(f"overlap detected: {len(dups)}")
for d in dups[:20]:
    print("  ", d[0], "<->", d[1])
