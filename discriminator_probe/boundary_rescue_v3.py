# -*- coding: utf-8 -*-
"""Rescue abandoned boundary segments via energy onset detection.

The abandoned 7 origins all failed at -80ms because whisper sentence starts
lag the true voice onset by 100-300ms. This pass scans backward from the
sentence start for the real energy onset (speech attack), then exports a
rescue variant at the detected onset and at +40ms (safety).

Read-only on radio_drama_work. Output in discriminator_probe/boundary_rescue_v3.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")
ROOT = Path(r"F:\GPTSOVITS\radio_drama_work")
OUT = PROBE / "boundary_rescue_v3"
AUDIO_DIR = OUT / "audio"
OUTPUT_SR = 32000
MAX_SEC = 6.0
MIN_SEC = 1.0
SCAN_BACK = 0.80  # scan up to 800ms before sentence start
FRAME = 0.010
HOP = 0.005

RADIO_MANIFEST = ROOT / "full_track_manifest_v1.json"
ANIME_MANIFEST = ROOT / "anime_batch_v1" / "manifest.json"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def detect_onset(mono, sr, sentence_start):
    """Find the true voice onset before sentence_start using energy attack."""
    lo = max(0, int((sentence_start - SCAN_BACK) * sr))
    hi = min(len(mono), int((sentence_start + 0.10) * sr))
    seg = mono[lo:hi]
    frame = max(1, round(sr * FRAME))
    hop = max(1, round(sr * HOP))
    n = max(1, 1 + (len(seg) - frame) // hop)
    starts = np.arange(n, dtype=np.int64) * hop
    ends = starts + frame
    sq = np.square(seg, dtype=np.float64)
    pref = np.empty(len(sq) + 1)
    pref[0] = 0.0
    np.cumsum(sq, out=pref[1:])
    rms = np.sqrt((pref[ends] - pref[starts]) / frame + 1e-12)
    ftimes = (starts + frame / 2.0) / sr + lo / sr
    if len(rms) < 10:
        return sentence_start - 0.05
    # floor = median of quiet 60% frames
    floor = float(np.percentile(rms, 40))
    if floor < 1e-5:
        floor = float(np.percentile(rms, 60))
    thr = max(floor * 2.5, 1e-4)
    # find onset: first frame after a quiet run where rms crosses thr and stays
    above = rms >= thr
    # scan from sentence_start backward: find the rightmost transition quiet->loud
    t_idx = np.flatnonzero((ftimes >= sentence_start - 0.35) & (ftimes <= sentence_start + 0.05))
    if not len(t_idx):
        t_idx = np.arange(len(rms))
    onset = None
    for k in range(len(t_idx) - 1, -1, -1):
        i = int(t_idx[k])
        if above[i]:
            # check we were quiet just before
            j = i
            while j > 0 and above[j - 1]:
                j -= 1
            onset = ftimes[j]
            break
    if onset is None:
        # fall back: earliest frame above thr in window
        first = int(t_idx[0])
        onset = ftimes[first] if above[first] else sentence_start - 0.03
    return max(lo / sr, onset - 0.005)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    report = json.loads((PROBE / "boundary_fix_v4_final_report.json").read_text(encoding="utf-8"))
    abandoned = report["abandoned"]

    # unique (source, start) pairs to rescue
    seen = set()
    rescue_tasks = []
    for a in abandoned:
        for r in a["results"]:
            key = (r["source_id"], round(r["start"], 2))
            if key in seen:
                continue
            seen.add(key)
            rescue_tasks.append({"source_id": r["source_id"], "sentence_start": float(r["start"]),
                                 "sentence_end": float(r["end"]), "origin": a["origin"],
                                 "note": [x["class"] for x in a["results"]]})
    print(f"rescue tasks: {len(rescue_tasks)}", flush=True)

    manifest = []
    for task in rescue_tasks:
        sid = task["source_id"]
        vocals = source_vocals(sid)
        if vocals is None or not vocals.exists():
            continue
        audio, sr = sf.read(str(vocals), dtype="float32", always_2d=True)
        mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
        onset = detect_onset(mono, sr, task["sentence_start"])
        end = task["sentence_end"] + 0.05
        if end - onset < MIN_SEC or end - onset > MAX_SEC:
            print(f"  SKIP {task['origin']} {sid} onset={onset:.2f} dur={end-onset:.2f}", flush=True)
            continue
        a0 = max(0, round(onset * sr))
        b0 = min(len(mono), round(end * sr))
        clip = micro_fade(resample(mono[a0:b0], sr, OUTPUT_SR))
        bid = f"RV{len(manifest) + 1:03d}"
        dest = AUDIO_DIR / f"{bid}.wav"
        sf.write(str(dest), clip, OUTPUT_SR, subtype="PCM_16")
        manifest.append({
            "blind_id": bid, "source_id": sid, "origin": task["origin"],
            "onset_detected": round(onset, 3), "old_start": round(task["sentence_start"], 3),
            "delta_ms": round((onset - task["sentence_start"]) * 1000, 0),
            "end": round(end, 3), "duration": round(end - onset, 3),
            "audio": str(dest.resolve()),
        })
        print(f"  {bid} {task['origin']} {sid} onset={onset:.2f} (delta {int((onset-task['sentence_start'])*1000)}ms) "
              f"dur={end-onset:.2f}", flush=True)

    write_json(OUT / "report.json", {
        "version": "boundary_rescue_v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tasks": len(rescue_tasks), "rescued": len(manifest),
        "samples": manifest,
    })
    print(json.dumps({"rescued": len(manifest), "total_sec": round(sum(m['duration'] for m in manifest), 1),
                      "saved": str(OUT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
