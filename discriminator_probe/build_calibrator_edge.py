# -*- coding: utf-8 -*-
"""Edge re-calibration page: only the 4 edge-pinned segments, wider window.

BC006/007/008/009 start markers hit the window's left edge. Rebuild those
cards with PAD_LEFT=3.5s so the true onset is reachable. Reuses the same
calibrator HTML template from build_boundary_calibrator.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")
ROOT = Path(r"F:\GPTSOVITS\radio_drama_work")
OUT = PROBE / "boundary_calibrator_edge"
AUDIO_DIR = OUT / "audio"
OUTPUT_SR = 32000
PAD_LEFT = 3.5
PAD_RIGHT = 0.8
RADIO_MANIFEST = ROOT / "full_track_manifest_v1.json"
ANIME_MANIFEST = ROOT / "anime_batch_v1" / "manifest.json"

# 4 edge segments: id, origin_blind_id, source_id, abs_start, abs_end (from finalize report)
EDGE_SEGS = [
    ("BC006", "DP020", "majo_ep12", 1205.688, 1211.201),
    ("BC007", "DP012", "majo_ep12", 330.310, 335.450),
    ("BC008", "DP013", "majo_ep06", 740.963, 744.583),
    ("BC009", "DP014", "majo_ep06", 317.120, 322.600),
]


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


def fine_wave(audio, sr, start, end):
    a = max(0, int(start * sr))
    b = min(len(audio), int(end * sr))
    seg = audio[a:b]
    if len(seg) == 0:
        return []
    hop = max(1, round(sr * 0.010))
    frame = max(1, round(sr * 0.020))
    n = max(1, 1 + (len(seg) - frame) // hop)
    starts = np.arange(n, dtype=np.int64) * hop
    ends = starts + frame
    sq = np.square(seg, dtype=np.float64)
    pref = np.empty(len(sq) + 1)
    pref[0] = 0.0
    np.cumsum(sq, out=pref[1:])
    rms = np.sqrt((pref[ends] - pref[starts]) / frame + 1e-12)
    if len(rms) == 0:
        return []
    scale = float(np.percentile(rms, 90))
    if scale < 1e-6:
        scale = 1.0
    return [round(min(1.0, float(v) / scale), 4) for v in rms]


def micro_fade(audio):
    out = np.asarray(audio, dtype=np.float32).copy()
    count = min(len(out) // 4, max(1, round(OUTPUT_SR * 0.002)))
    if count:
        ramp = np.linspace(0.0, 1.0, count, dtype=np.float32)
        out[:count] *= ramp
        out[-count:] *= ramp[::-1]
    return out


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    cards = []
    for i, (cid, origin, sid, abs_start, abs_end) in enumerate(EDGE_SEGS, 1):
        vocals = source_vocals(sid)
        if vocals is None or not vocals.exists():
            continue
        audio, sr = sf.read(str(vocals), dtype="float32", always_2d=True)
        mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
        win_start = max(0.0, abs_start - PAD_LEFT)
        win_end = min(len(mono) / sr, abs_end + PAD_RIGHT)
        a0 = max(0, round(win_start * sr))
        b0 = min(len(mono), round(win_end * sr))
        clip = micro_fade(resample(mono[a0:b0], sr, OUTPUT_SR))
        new_id = f"EC{i:03d}"
        dest = AUDIO_DIR / f"{new_id}.wav"
        sf.write(str(dest), clip, OUTPUT_SR, subtype="PCM_16")
        wave = fine_wave(clip, OUTPUT_SR, 0.0, len(clip) / OUTPUT_SR)
        cards.append({
            "id": new_id,
            "origin_blind_id": cid,  # keep original BC id for provenance
            "prev_abs_start": abs_start,
            "prev_abs_end": abs_end,
            "source_id": sid,
            "text": f"orig {cid} {origin}",
            "audio": f"audio/{dest.name}",
            "window_start": round(win_start, 3),
            "default_start": round(abs_start - win_start, 3),
            "default_end": round(abs_end - win_start, 3),
            "duration": round(win_end - win_start, 3),
            "wave": wave,
        })
        print(f"  {new_id} ({cid}) {sid} win={win_start:.2f}-{win_end:.2f} def={abs_start-win_start:.2f}-{abs_end-win_start:.2f}", flush=True)

    write_json(OUT / "audit_manifest.json", {"version": "boundary_calibrator_edge_audit", "samples": [
        {"id": c["id"], "audio": c["audio"], "duration": c["duration"]} for c in cards]})

    # reuse the HTML template from the main calibrator build script
    import importlib.util
    spec = importlib.util.spec_from_file_location("bcal", PROBE / "build_boundary_calibrator.py")
    bcal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bcal)
    data_json = json.dumps(cards, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = bcal.HTML_TEMPLATE.replace("__DATA_JSON__", data_json)
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(f"saved={OUT / 'index.html'} cards={len(cards)}", flush=True)


if __name__ == "__main__":
    main()
