# -*- coding: utf-8 -*-
"""Finalize boundary fix v4: split clean vs cross-segment, build spot check.

- 30 clean segments (verified single-sentence + discriminator prob) -> candidate
- 17 cross-segment -> millisecond variants (Luna-style) for spot check
Reads boundary_fix_v4/report.json, writes:
  boundary_fix_v4_clean.list        (30 clean, usable)
  boundary_spotcheck_v2/            (spot-check page with variants)
"""
from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")
ROOT = Path(r"F:\GPTSOVITS\radio_drama_work")
MAKER = ROOT / "make_human_review_page_v2.py"
PY = r"F:\GPTSOVITS\GPT-SoVITS-v2pro-20250604-nvidia50\GPT-SoVITS-v2pro-20250604-nvidia50\runtime\python.exe"
OUT = PROBE / "boundary_fix_v4"
SPOT = PROBE / "boundary_spotcheck_v2"
AUDIO_DIR = SPOT / "audio"
OUTPUT_SR = 32000
WAVE_BINS = 32
VARIANTS = (0.0, -0.04, -0.08)  # start offsets (s) to try

RADIO_MANIFEST = ROOT / "full_track_manifest_v1.json"
ANIME_MANIFEST = ROOT / "anime_batch_v1" / "manifest.json"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def source_vocals(source_id):
    manifest = json.loads((ANIME_MANIFEST if source_id.startswith("majo_") else RADIO_MANIFEST).read_text(encoding="utf-8"))
    for t in manifest.get("tracks", []):
        if t["id"] == source_id:
            return Path(t["vocals"])
    return None


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


def classify_clean(report: dict) -> tuple[list[dict], list[dict]]:
    """Return (clean, cross) using ASR segment boundaries for anime."""
    clean, cross = [], []
    for x in report.get("samples", []):
        sid = x["source_id"]
        if not sid.startswith("majo_"):
            clean.append(x)
            continue
        rep = ROOT / "anime_mining_v1" / "asr" / "source_reports" / f"{sid}.json"
        d = json.loads(rep.read_text(encoding="utf-8"))
        words = d["words"]
        inwin = [w for w in words if w["end"] > x["new_start"] - 0.05 and w["start"] < x["new_end"] + 0.05]
        if not inwin:
            clean.append(x)
            continue
        si0, si1 = inwin[0].get("segment_index"), inwin[-1].get("segment_index")
        if si0 == si1:
            seg_start = min(w["start"] for w in words if w.get("segment_index") == si0)
            seg_end = max(w["end"] for w in words if w.get("segment_index") == si1)
            if x["new_start"] <= seg_start + 0.20 and x["new_end"] >= seg_end - 0.20:
                clean.append(x)
            else:
                cross.append(x)
        else:
            cross.append(x)
    return clean, cross


def main():
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    clean, cross = classify_clean(report)
    print(f"clean={len(clean)} cross={len(cross)}", flush=True)

    # 1) clean list
    lines = [f"{x['audio']}|boundary_fix_v4_clean|JA|{x.get('text','')}" for x in clean]
    (PROBE / "boundary_fix_v4_clean.list").write_text("\n".join(lines) + "\n", encoding="utf-8")
    clean_sec = sum(x["duration"] for x in clean)
    print(f"clean list: {len(clean)} / {clean_sec:.1f}s -> boundary_fix_v4_clean.list", flush=True)

    # 2) spot check for cross segments: 3 ms variants each, but cap total
    SPOT.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    idx = 0
    for x in cross[:17]:
        sid = x["source_id"]
        vocals = source_vocals(sid)
        if vocals is None or not vocals.exists():
            continue
        audio, sr = sf.read(str(vocals), dtype="float32", always_2d=True)
        mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
        base_start, base_end = float(x["new_start"]), float(x["new_end"])
        for off in VARIANTS:
            s = max(0.0, base_start + off)
            e = min(len(mono) / sr, base_end + 0.02)
            if e - s < 1.0:
                continue
            a = max(0, round(s * sr))
            b = min(len(mono), round(e * sr))
            clip = micro_fade(resample(mono[a:b], sr, OUTPUT_SR))
            bid = f"XV{idx + 1:03d}"
            dest = AUDIO_DIR / f"{bid}.wav"
            sf.write(str(dest), clip, OUTPUT_SR, subtype="PCM_16")
            manifest.append({
                "blind_id": bid,
                "source_id": sid,
                "origin_blind_id": x.get("origin_blind_id", ""),
                "variant_offset": off,
                "start": round(s, 3), "end": round(e, 3),
                "duration": round(e - s, 3),
                "mean_prob": round(float(x.get("mean_prob", 0)), 3),
                "audio": str(dest.resolve()),
            })
            idx += 1
    write_json(SPOT / "audit_manifest.json", {"version": "boundary_spotcheck_v2_audit", "samples": [
        {"blind_id": m["blind_id"], "audio_rel": f"audio/{Path(m['audio']).name}",
         "output_duration": m["duration"], "wave": waveform(Path(m["audio"]))} for m in manifest]})
    write_json(SPOT / "private_manifest.json", {"samples": manifest})
    print(f"spot variants: {len(manifest)}", flush=True)

    cmd = [PY, str(MAKER), "--root", str(SPOT), "--title", "跨句边界·毫秒变体抽验",
           "--round-label", "Cross-segment boundary · ms variants",
           "--store-key", "ireina_boundary_spotcheck_v2",
           "--feedback-version", "boundary_spotcheck_v2_feedback",
           "--download-prefix", "boundary_spotcheck_v2_feedback",
           "--extra-note", "同一段给出起点偏移 0/-40/-80ms 三个版本。选声音完整自然、首尾无切痕的版本；都失败选最接近的。"]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    print(result.stdout.strip() if result.returncode == 0 else result.stderr[-800:], flush=True)
    print(json.dumps({"clean_list": str(PROBE / "boundary_fix_v4_clean.list"), "spot_page": str(SPOT / "index.html")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
