# -*- coding: utf-8 -*-
"""Finalize boundary calibrator results: cut audio at user-marked times.

Reads the exported calibrator JSON, converts window-relative marks to absolute
times (window_start + mark), cuts from original vocals, writes final list.
Also flags segments whose start is pinned to the window edge (needs re-window).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")
ROOT = Path(r"F:\GPTSOVITS\radio_drama_work")
OUT = PROBE / "boundary_calibrator_final"
AUDIO_DIR = OUT / "audio"
OUTPUT_SR = 32000
EDGE_FLAG = 0.06  # start within this of window left edge => range too small
RADIO_MANIFEST = ROOT / "full_track_manifest_v1.json"
ANIME_MANIFEST = ROOT / "anime_batch_v1" / "manifest.json"
FEEDBACK = r"C:\Users\lulu\Downloads\boundary_calibrator_v1_2026-08-03T02-58-10-106Z.json"


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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    feedback = json.loads(open(FEEDBACK, encoding="utf-8").read())
    samples = feedback["samples"]
    print(f"calibrated: {len(samples)}", flush=True)

    # audit manifest for window duration check
    audit = json.loads((PROBE / "boundary_calibrator" / "audit_manifest.json").read_text(encoding="utf-8"))["samples"]
    by_id = {a["id"]: a for a in audit}

    rows = []
    needs_rescan = []
    for s in samples:
        cid = s["id"]
        sid = s["source_id"]
        win_start = float(s["window_start"])
        rel_start = float(s["start"])
        rel_end = float(s["end"])
        abs_start = win_start + rel_start
        abs_end = win_start + rel_end
        duration = rel_end - rel_start

        # flag if start pinned to window edge (user said range too small)
        audit_row = by_id.get(cid, {})
        win_dur = float(audit_row.get("duration", 0.0))
        note = s.get("note", "")
        edge = rel_start < EDGE_FLAG or "不够" in note or "范围" in note

        vocals = source_vocals(sid)
        if vocals is None or not vocals.exists():
            print(f"  SKIP {cid} no vocals", flush=True)
            continue
        audio, sr = sf.read(str(vocals), dtype="float32", always_2d=True)
        mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
        a0 = max(0, round(abs_start * sr))
        b0 = min(len(mono), round(abs_end * sr))
        clip = micro_fade(resample(mono[a0:b0], sr, OUTPUT_SR))
        bid = f"CF{len(rows) + 1:03d}"
        dest = AUDIO_DIR / f"{bid}.wav"
        sf.write(str(dest), clip, OUTPUT_SR, subtype="PCM_16")
        rows.append({
            "blind_id": bid, "id": cid, "origin_blind_id": s.get("origin_blind_id", ""),
            "source_id": sid, "abs_start": round(abs_start, 3), "abs_end": round(abs_end, 3),
            "duration": round(duration, 3), "note": note, "audio": str(dest.resolve()),
        })
        status = "EDGE" if edge else "ok"
        print(f"  {status} {bid} ({cid}) {sid} {abs_start:.2f}-{abs_end:.2f} dur={duration:.2f} note={note[:30]}", flush=True)
        if edge:
            needs_rescan.append({"id": cid, "source_id": sid, "abs_start": abs_start, "abs_end": abs_end,
                                 "note": note, "win_duration": win_dur})

    lines = [f"{r['audio']}|boundary_calibrator_final|JA|" for r in rows]
    (OUT / "boundary_calibrator_final.list").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {
        "version": "boundary_calibrator_final",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total": len(rows),
        "duration_sec": round(sum(r["duration"] for r in rows), 2),
        "needs_rescan": needs_rescan,
        "list": str(OUT / "boundary_calibrator_final.list"),
    }
    write_json(OUT / "report.json", report)
    print(json.dumps({"final": len(rows), "sec": report["duration_sec"],
                      "needs_rescan": len(needs_rescan)}, ensure_ascii=False), flush=True)
    print("saved:", OUT / "boundary_calibrator_final.list", flush=True)


if __name__ == "__main__":
    main()
