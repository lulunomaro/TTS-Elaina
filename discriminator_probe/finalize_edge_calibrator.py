# -*- coding: utf-8 -*-
"""Finalize edge calibration (4 segments) + merge into boundary final list.

Cuts the 4 edge segments at user-marked absolute times. If duration > 6s,
splits at the nearest ASR sentence boundary inside the target run (anime
segment_index / radio gap>=0.3s). Writes boundary_calibrator_edge_final.list
and merges with boundary_calibrator_final.list into boundary_all_final.list.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")
ROOT = Path(r"F:\GPTSOVITS\radio_drama_work")
OUT = PROBE / "boundary_calibrator_edge_final"
AUDIO_DIR = OUT / "audio"
OUTPUT_SR = 32000
MAX_SEC = 6.0
MIN_SEC = 1.2
RADIO_MANIFEST = ROOT / "full_track_manifest_v1.json"
ANIME_MANIFEST = ROOT / "anime_batch_v1" / "manifest.json"
FEEDBACK = r"C:\Users\lulu\Downloads\boundary_calibrator_v1_2026-08-03T03-20-57-450Z.json"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_vocals(source_id):
    manifest = json.loads((ANIME_MANIFEST if source_id.startswith("majo_") else RADIO_MANIFEST).read_text(encoding="utf-8"))
    for t in manifest.get("tracks", []):
        if t["id"] == source_id:
            return Path(t["vocals"])
    return None


def load_sentences(source_id):
    """Anime: segment_index groups; Radio: gap>=0.3s groups."""
    if source_id.startswith("majo_"):
        rep = ROOT / "anime_mining_v1" / "asr" / "source_reports" / f"{source_id}.json"
        doc = json.loads(rep.read_text(encoding="utf-8"))
        words = doc.get("words", [])
        groups = {}
        for w in words:
            groups.setdefault(w.get("segment_index", 0), []).append(w)
        return [(float(ws[0]["start"]), float(ws[-1]["end"])) for k in sorted(groups) for ws in [groups[k]]]
    else:
        rep = ROOT / "word_target_all_v1" / "asr_cache" / f"{source_id}.json"
        doc = json.loads(rep.read_text(encoding="utf-8"))
        words = doc.get("words", [])
        if not words:
            return []
        out, cur = [], [words[0]]
        for w in words[1:]:
            if w["start"] - cur[-1]["end"] >= 0.30:
                out.append((cur[0]["start"], cur[-1]["end"]))
                cur = [w]
            else:
                cur.append(w)
        out.append((cur[0]["start"], cur[-1]["end"]))
        return out


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


def export(mono, sr, s, e, idx, prefix="EF"):
    a = max(0, round(s * sr))
    b = min(len(mono), round(e * sr))
    clip = micro_fade(resample(mono[a:b], sr, OUTPUT_SR))
    dest = AUDIO_DIR / f"{prefix}{idx:03d}.wav"
    sf.write(str(dest), clip, OUTPUT_SR, subtype="PCM_16")
    return dest, (e - s)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    feedback = json.loads(open(FEEDBACK, encoding="utf-8").read())
    samples = feedback["samples"]
    print(f"edge calibrated: {len(samples)}", flush=True)

    rows = []
    idx = 0
    for s in samples:
        sid = s["source_id"]
        abs_start = float(s["window_start"]) + float(s["start"])
        abs_end = float(s["window_start"]) + float(s["end"])
        vocals = source_vocals(sid)
        if vocals is None or not vocals.exists():
            print(f"  SKIP {s['id']} no vocals", flush=True)
            continue
        audio, sr = sf.read(str(vocals), dtype="float32", always_2d=True)
        mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
        duration = abs_end - abs_start

        if duration <= MAX_SEC:
            dest, dur = export(mono, sr, abs_start, abs_end, idx)
            rows.append({"blind_id": f"EF{idx + 1:03d}", "id": s["id"],
                         "origin_blind_id": s.get("origin_blind_id", ""),
                         "source_id": sid, "abs_start": round(abs_start, 3),
                         "abs_end": round(abs_end, 3), "duration": round(dur, 3),
                         "audio": str(dest.resolve()), "split": False})
            idx += 1
            print(f"  OK EF{idx:03d} {s['id']} {abs_start:.2f}-{abs_end:.2f} dur={dur:.2f}", flush=True)
        else:
            # split at sentence boundaries inside [abs_start, abs_end]
            sents = load_sentences(sid)
            bounds = [abs_start] + [e for st, e in sents if abs_start < e < abs_end] + [abs_end]
            pieces = []
            for a, b in zip(bounds[:-1], bounds[1:]):
                if b - a >= MIN_SEC and b - a <= MAX_SEC + 0.1:
                    pieces.append((a, b))
            if not pieces:
                # fallback: slide a 5.8s window through
                t = abs_start
                while t + 5.8 <= abs_end:
                    pieces.append((t, t + 5.8))
                    t += 5.8
                if abs_end - t >= MIN_SEC:
                    pieces.append((t, abs_end))
            for a, b in pieces:
                dest, dur = export(mono, sr, a, b, idx)
                rows.append({"blind_id": f"EF{idx + 1:03d}", "id": s["id"],
                             "origin_blind_id": s.get("origin_blind_id", ""),
                             "source_id": sid, "abs_start": round(a, 3),
                             "abs_end": round(b, 3), "duration": round(dur, 3),
                             "audio": str(dest.resolve()), "split": True})
                idx += 1
                print(f"  SPLIT EF{idx:03d} {s['id']} {a:.2f}-{b:.2f} dur={dur:.2f}", flush=True)

    lines = [f"{r['audio']}|boundary_calibrator_edge_final|JA|" for r in rows]
    (OUT / "boundary_calibrator_edge_final.list").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {
        "version": "boundary_calibrator_edge_final",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total": len(rows), "duration_sec": round(sum(r["duration"] for r in rows), 2),
        "samples": rows,
    }
    write_json(OUT / "report.json", report)
    print(json.dumps({"edge_final": len(rows), "sec": report["duration_sec"]}, ensure_ascii=False), flush=True)

    # merge with previous final list
    prev = PROBE / "boundary_calibrator_final" / "boundary_calibrator_final.list"
    merged_lines = []
    if prev.exists():
        merged_lines = [l for l in prev.read_text(encoding="utf-8").splitlines() if l.strip()]
    merged_lines.extend(lines)
    (PROBE / "boundary_all_final.list").write_text("\n".join(merged_lines) + "\n", encoding="utf-8")
    print(f"merged total: {len(merged_lines)} -> boundary_all_final.list", flush=True)


if __name__ == "__main__":
    main()
