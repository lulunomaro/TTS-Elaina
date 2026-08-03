# -*- coding: utf-8 -*-
"""Cut the 239 final high-conf segments to wavs + texts, build unified train list."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf

PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")
ROOT = Path(r"F:\GPTSOVITS\radio_drama_work")
WS_DIR = PROBE / "whisper_sent_v1"
OUT = ROOT / "highconf_final_v1" / "audio"
OUT.mkdir(parents=True, exist_ok=True)
OUTPUT_SR = 32000
RADIO_MANIFEST = ROOT / "full_track_manifest_v1.json"
ANIME_MANIFEST = ROOT / "anime_batch_v1" / "manifest.json"

data = json.loads((PROBE / "highconf_final_v1.json").read_text(encoding="utf-8"))
samples = data["samples"]


def source_vocals(sid):
    mf = json.loads((ANIME_MANIFEST if sid.startswith("majo_") else RADIO_MANIFEST).read_text(encoding="utf-8"))
    for t in mf.get("tracks", []):
        if t["id"] == sid:
            return Path(t["vocals"])
    return None


def segs_of(sid):
    p = WS_DIR / f"{sid}.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))["segments"]


def text_for(sid, s0, s1):
    mid = (s0 + s1) / 2
    for s in segs_of(sid):
        if s["start"] <= mid <= s["end"]:
            return s["text"]
    return ""


def resample(audio, src, dst):
    if src == dst:
        return audio
    from scipy.signal import resample_poly
    g = np.gcd(src, dst)
    return resample_poly(audio, dst // g, src // g).astype(np.float32)


def micro_fade(audio):
    out = np.asarray(audio, dtype=np.float32).copy()
    count = min(len(out) // 4, max(1, round(OUTPUT_SR * 0.002)))
    if count:
        ramp = np.linspace(0.0, 1.0, count, dtype=np.float32)
        out[:count] *= ramp
        out[-count:] *= ramp[::-1]
    return out


_cache = {}


def get_mono32(sid):
    if sid in _cache:
        return _cache[sid]
    p = WS_DIR / "audio32k" / f"{sid}.wav"
    audio, sr = sf.read(str(p), dtype="float32", always_2d=True)
    mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
    _cache[sid] = mono
    return mono


lines = []
total = 0.0
done = 0
for i, s in enumerate(samples, 1):
    mono = get_mono32(s["source_id"])
    a0 = max(0, round(s["start"] * OUTPUT_SR)); b0 = min(len(mono), round(s["end"] * OUTPUT_SR))
    clip = micro_fade(mono[a0:b0])
    fn = f"HF{i:04d}_{s['origin_blind_id']}.wav"
    sf.write(str(OUT / fn), clip, OUTPUT_SR, subtype="PCM_16")
    txt = s.get("text") or text_for(s["source_id"], s["start"], s["end"])
    txt = txt.strip().replace("|", " ").replace("\n", " ")
    lines.append(f"audio/{fn}|ireina_highconf_v1|JA|{txt}")
    total += s["end"] - s["start"]
    done += 1
    if done % 60 == 0:
        print(f"  {done}/{len(samples)}", flush=True)

list_path = ROOT / "highconf_final_v1" / "highconf_final_v1.list"
list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"cut {len(lines)} wavs, {total:.1f}s = {total/60:.1f} min")
print(f"list -> {list_path}")
