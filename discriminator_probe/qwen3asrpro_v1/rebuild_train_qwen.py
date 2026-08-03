# -*- coding: utf-8 -*-
"""Rebuild training set from Qwen3ASRpro full-source results.

1. Speaker anchor: map each old confirmed segment midpoint -> pyannote label;
   majority label across all anchors = Irena's speaker id (per source? global?).
2. New segments: Irena-speaker pyannote spans, merge same-speaker (gap<0.8s),
   cap 12s, snap edges to nearest Qwen3 char boundary (±0.35s).
3. Text: concatenated Qwen3 chars. Output list + cut wavs (32k mono).
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import soundfile as sf

PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")
ROOT = Path(r"F:\GPTSOVITS\radio_drama_work")
QDIR = PROBE / "qwen3asrpro_v1"
NEW_DIR = ROOT / "ireina_qwen_v1"
AUDIO = NEW_DIR / "audio"
AUDIO.mkdir(parents=True, exist_ok=True)
FFMPEG = r"F:\GPTSOVITS\GPT-SoVITS-v2pro-20250604-nvidia50\GPT-SoVITS-v2pro-20250604-nvidia50\runtime\ffmpeg.exe"
ANIME_MANIFEST = ROOT / "anime_batch_v1" / "manifest.json"
RADIO_MANIFEST = ROOT / "full_track_manifest_v1.json"

MERGE_GAP = 0.8
MAX_SEG = 12.0
MIN_SEG = 1.2
SNAP = 0.35


def source_vocals(sid):
    mf = json.loads((ANIME_MANIFEST if sid.startswith("majo_") else RADIO_MANIFEST).read_text(encoding="utf-8"))
    for t in mf.get("tracks", []):
        if t["id"] == sid:
            return Path(t["vocals"])
    return None


def load_old_segments():
    """old confirmed list -> [(source, start, end)] via wav filenames is not possible;
    use highconf_final_v1.json + gold (they have source+times) as anchors."""
    anchors = []
    hf = json.loads((PROBE / "highconf_final_v1.json").read_text(encoding="utf-8"))["samples"]
    for s in hf:
        anchors.append((s["source_id"], s["start"], s["end"]))
    gold = json.loads((PROBE / "highconf_calibrator" / "gold_calibrations.json").read_text(encoding="utf-8"))["samples"]
    for g in gold:
        anchors.append((g["source_id"], g["abs_start"], g["abs_end"]))
    return anchors


def main():
    anchors = load_old_segments()
    print(f"anchors: {len(anchors)}")

    # per-source speaker anchor mapping
    per_src = {}
    for sid, s0, s1 in anchors:
        qj = QDIR / f"{sid}.json"
        if not qj.exists():
            continue
        data = json.loads(qj.read_text(encoding="utf-8"))
        mid = (s0 + s1) / 2
        seg = next((x for x in data.get("diarization", []) if x[0] <= mid < x[1]), None)
        if seg:
            per_src.setdefault(sid, Counter())[seg[2]] += 1
    irena_speaker = {}
    for sid, cnt in per_src.items():
        irena_speaker[sid] = cnt.most_common(1)[0][0]
        print(f"  {sid}: {dict(cnt)} -> IRENA={irena_speaker[sid]}")

    # build segments per source
    all_segs = []
    for sid, label in irena_speaker.items():
        qj = QDIR / f"{sid}.json"
        data = json.loads(qj.read_text(encoding="utf-8"))
        spk = [x for x in data["diarization"] if x[2] == label]
        chars = data.get("chars", [])
        # merge contiguous same-speaker
        blocks = []
        cur = None
        for s in spk:
            if cur is None:
                cur = [s[0], s[1]]
            elif s[0] - cur[1] < MERGE_GAP:
                cur[1] = max(cur[1], s[1])
            else:
                blocks.append(cur)
                cur = [s[0], s[1]]
        if cur:
            blocks.append(cur)

        def snap(t, lo, hi):
            best = t
            bd = abs(lo - t)
            for c in chars:
                if lo <= c["start"] <= hi and abs(c["start"] - t) < SNAP and abs(c["start"] - t) < bd:
                    best, bd = c["start"], abs(c["start"] - t)
                if lo <= c["end"] <= hi and abs(c["end"] - t) < SNAP and abs(c["end"] - t) < bd:
                    best, bd = c["end"], abs(c["end"] - t)
            return best

        for b in blocks:
            dur = b[1] - b[0]
            if dur < MIN_SEG:
                continue
            if dur <= MAX_SEG:
                pieces = [b]
            else:
                pieces = []
                n = int(dur // MAX_SEG) + 1
                step = dur / n
                for k in range(n):
                    pieces.append([b[0] + k * step, b[0] + (k + 1) * step])
            for p in pieces:
                if p[1] - p[0] < MIN_SEG:
                    continue
                ns0 = snap(p[0], p[0], p[1])
                ns1 = snap(p[1], p[0], p[1])
                if ns1 - ns0 < MIN_SEG or ns1 - ns0 > MAX_SEG:
                    continue
                txt = "".join(c["text"] for c in chars if ns0 <= c["start"] < ns1)
                all_segs.append({"source": sid, "start": round(ns0, 3), "end": round(ns1, 3), "text": txt})

    all_segs.sort(key=lambda x: (x["source"], x["start"]))
    total = sum(s["end"] - s["start"] for s in all_segs)
    print(f"\nnew segs: {len(all_segs)}, {total:.1f}s = {total/60:.1f} min")

    # write wavs + list
    lines = []
    for i, s in enumerate(all_segs, 1):
        dest = AUDIO / f"QW{i:04d}_{s['source']}.wav"
        vocals = source_vocals(s["source"])
        r = subprocess.run([FFMPEG, "-y", "-ss", f"{s['start']:.3f}", "-i", str(vocals),
                            "-t", f"{s['end'] - s['start']:.3f}", "-ac", "1", "-ar", "32000",
                            "-c:a", "pcm_s16le", str(dest)], capture_output=True, timeout=60)
        if r.returncode != 0:
            print("FFMPEG FAIL", s)
            continue
        txt = (s["text"] or "").strip().replace("|", " ").replace("\n", " ")
        lines.append(f"radio_drama_work/ireina_qwen_v1/audio/{dest.name}|ireina|JA|{txt}")

    (NEW_DIR / "ireina_qwen_v1.list").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {NEW_DIR / 'ireina_qwen_v1.list'} ({len(lines)} lines)")
    # report vs gold
    gold = json.loads((PROBE / "highconf_calibrator" / "gold_calibrations.json").read_text(encoding="utf-8"))["samples"]
    print("\n=== vs gold ===")
    for g in gold:
        cand = [s for s in all_segs if s["source"] == g["source_id"]
                and s["start"] <= g["abs_start"] and s["end"] >= g["abs_end"]]
        near = min([s for s in all_segs if s["source"] == g["source_id"] and s["start"] < g["abs_end"] and s["end"] > g["abs_start"]],
                   key=lambda s: abs(s["start"] - g["abs_start"]) + abs(s["end"] - g["abs_end"]), default=None)
        if cand:
            c = cand[0]
            print(f"  {g['qc_id']}: COVER {c['start']:.2f}-{c['end']:.2f} vs gold {g['abs_start']:.2f}-{g['abs_end']:.2f} (d {c['start']-g['abs_start']:+.2f}/{c['end']-g['abs_end']:+.2f})")
        elif near:
            print(f"  {g['qc_id']}: NEAR {near['start']:.2f}-{near['end']:.2f} vs gold {g['abs_start']:.2f}-{g['abs_end']:.2f} (d {near['start']-g['abs_start']:+.2f}/{near['end']-g['abs_end']:+.2f})")
        else:
            print(f"  {g['qc_id']}: NONE")


if __name__ == "__main__":
    main()
