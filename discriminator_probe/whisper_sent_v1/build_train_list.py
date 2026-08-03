# -*- coding: utf-8 -*-
"""Final train list: confirmed 204 (minus overlap w/ highconf) + highconf 237.
Paths relativized (F:\\GPTSOVITS\\ -> ../../radio_drama_work/...)."""
import io
import re
from pathlib import Path

import soundfile as sf

ROOT = Path(r"F:\GPTSOVITS\radio_drama_work")
PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")

drop = {"HF0128_AR0149.wav", "HF0205_AR0326.wav"}  # overlap w/ confirmed

confirmed_lines = []
for p, src in [
    (ROOT / "human_confirmed_radio_anime_v3_train_1p5_6s.list", "confirmed"),
    (PROBE / "boundary_fix_v4_clean.list", "bnd_fix"),
    (PROBE / "boundary_fix_v4_final.list", "bnd_fix"),
    (PROBE / "boundary_all_final.list", "bnd_cal"),
]:
    for l in p.read_text(encoding="utf-8").splitlines():
        if l.strip():
            confirmed_lines.append((l, src))

# dedup confirmed itself (by wav path)
seen = set()
conf_unique = []
for l, src in confirmed_lines:
    wav = l.split("|")[0]
    if wav in seen:
        continue
    seen.add(wav)
    conf_unique.append((l, src))

high_lines = [l for l in (ROOT / "highconf_final_v1" / "highconf_final_v1.list").read_text(encoding="utf-8").splitlines()
              if l.strip() and Path(l.split("|")[0]).name not in drop]

out = io.open(ROOT / "ireina_train_final_v1.list", "w", encoding="utf-8")

def dur(wav):
    try:
        info = sf.info(wav)
        return info.frames / info.samplerate
    except Exception:
        return 0.0

total = 0.0
n = 0
for l, src in conf_unique:
    wav = l.split("|")[0]
    d = dur(wav)
    total += d
    n += 1
    rel = wav.replace("F:\\GPTSOVITS\\", "").replace("\\", "/")
    out.write(f"{rel}|ireina|{l.split('|')[2]}|{l.split('|')[3]}\n")

for l in high_lines:
    parts = l.split("|")
    wav = str(ROOT / "highconf_final_v1" / parts[0])
    d = dur(wav)
    total += d
    n += 1
    out.write(f"radio_drama_work/highconf_final_v1/{parts[0].replace(chr(92),'/')}|ireina|{parts[2]}|{parts[3]}\n")

out.close()
print(f"final train list: {n} segments, {total:.1f}s = {total/60:.1f} min")
print(f"  confirmed: {len(conf_unique)} (unique), highconf: {len(high_lines)}")
print("saved -> radio_drama_work/ireina_train_final_v1.list")
