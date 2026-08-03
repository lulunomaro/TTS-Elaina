# -*- coding: utf-8 -*-
"""Speaker-aware sentence extension: merge adjacent whisper segments only if
cam++ speaker similarity >= threshold (blocks other-speaker intrusion)."""
from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MODELSCOPE_CACHE", r"F:\GPTSOVITS\model_cache")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from funasr import AutoModel

PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")
WS_DIR = PROBE / "whisper_sent_v1"
ROOT = Path(r"F:\GPTSOVITS\radio_drama_work")
RADIO_MANIFEST = ROOT / "full_track_manifest_v1.json"
ANIME_MANIFEST = ROOT / "anime_batch_v1" / "manifest.json"

MERGE_GAP = 0.6
SIM_THRESHOLD = 0.72
MAX_BLOCK = 12.0


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


def main():
    report = json.loads((PROBE / "highconf_boundary_screen" / "screen_report.json").read_text(encoding="utf-8"))
    gold = json.loads((PROBE / "highconf_calibrator" / "gold_calibrations.json").read_text(encoding="utf-8"))["samples"]

    spk = AutoModel(model="iic/speech_campplus_sv_zh-cn_16k-common", model_revision="v2.0.2",
                    disable_update=True, device="cuda:0")
    kws = {"cache": {}, "disable_pbar": True, "kwargs": {"device": "cuda:0"}}
    emb_cache = {}  # (sid, i) -> emb

    def embed(sid, s0, s1):
        key = (sid, round(s0, 2), round(s1, 2))
        if key in emb_cache:
            return emb_cache[key]
        vocals = source_vocals(sid)
        audio, sr = sf.read(str(vocals), dtype="float32", always_2d=True)
        mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
        if sr != 16000:
            g = np.gcd(sr, 16000)
            mono = resample_poly(mono, 16000 // g, sr // g).astype(np.float32)
        seg = mono[int(s0 * 16000): int(s1 * 16000)]
        if len(seg) < 1600:  # <0.1s guard
            emb_cache[key] = None
            return None
        try:
            r = spk.generate(input=seg, **kws)
            e = np.asarray(r[0]["spk_embedding"].cpu()).ravel()
        except Exception:
            e = None
        emb_cache[key] = e
        return e

    def cos(a, b):
        if a is None or b is None:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    items = report["need_calib"]
    results = []
    t_all = time.time()
    n_emb = 0
    for idx, it in enumerate(items):
        sid = it["source_id"]
        mid = (it["start"] + it["end"]) / 2
        segs = segs_of(sid)
        if not segs:
            results.append({"origin_blind_id": Path(it["audio"]).name.replace(".wav", ""), "source_id": sid,
                            "start": it["start"], "end": it["end"], "extended": False})
            continue
        ci = next((i for i, s in enumerate(segs) if s["start"] <= mid <= s["end"]), None)
        if ci is None:
            results.append({"origin_blind_id": Path(it["audio"]).name.replace(".wav", ""), "source_id": sid,
                            "start": it["start"], "end": it["end"], "extended": False})
            continue
        # candidate neighbors: contiguous with gap < MERGE_GAP
        lo = ci
        while lo > 0 and segs[lo - 1]["end"] + MERGE_GAP > segs[ci]["start"]:
            lo -= 1
        hi = ci
        while hi < len(segs) - 1 and segs[hi + 1]["start"] < segs[ci]["end"] + MERGE_GAP:
            hi += 1
        center_emb = embed(sid, segs[ci]["start"], segs[ci]["end"])
        n_emb += 1
        # extend left while same speaker
        s0, s1 = segs[ci]["start"], segs[ci]["end"]
        text = segs[ci]["text"]
        j = ci - 1
        while j >= lo:
            e = embed(sid, segs[j]["start"], segs[j]["end"]); n_emb += 1
            if cos(center_emb, e) >= SIM_THRESHOLD and (segs[ci]["end"] - segs[j]["start"]) <= MAX_BLOCK:
                s0 = segs[j]["start"]
                text = segs[j]["text"] + text
                j -= 1
            else:
                break
        j = ci + 1
        while j <= hi:
            e = embed(sid, segs[j]["start"], segs[j]["end"]); n_emb += 1
            if cos(center_emb, e) >= SIM_THRESHOLD and (segs[j]["end"] - segs[ci]["start"]) <= MAX_BLOCK:
                s1 = segs[j]["end"]
                text = text + segs[j]["text"]
                j += 1
            else:
                break
        results.append({
            "origin_blind_id": Path(it["audio"]).name.replace(".wav", ""),
            "source_id": sid, "start": round(s0, 3), "end": round(s1, 3),
            "prob": it["prob"], "extended": True,
            "d_start": round(s0 - it["start"], 3), "d_end": round(s1 - it["end"], 3),
            "n_segs": hi - lo + 1, "text": text[:200],
        })
        if (idx + 1) % 40 == 0:
            print(f"  {idx+1}/{len(items)} emb={n_emb} elapsed={time.time()-t_all:.0f}s", flush=True)

    n_ext = sum(1 for r in results if r["extended"])
    print(f"\ntotal {len(results)} | extended {n_ext} | emb computed {n_emb} | elapsed {time.time()-t_all:.0f}s")

    print("\n=== vs gold (10 manual) ===")
    for g in gold:
        cand = next((o for o in results if o["origin_blind_id"] == g["origin_blind_id"]), None)
        if not cand:
            print(f"  {g['qc_id']}: NO CAND"); continue
        print(f"  {g['qc_id']}: auto {cand['start']:.2f}-{cand['end']:.2f} | gold {g['abs_start']:.2f}-{g['abs_end']:.2f} | d {cand['start']-g['abs_start']:+.2f}/{cand['end']-g['abs_end']:+.2f}")

    (PROBE / "highconf_autoextend_v2_speaker.json").write_text(
        json.dumps({"version": "highconf_autoextend_v2_speaker", "sim_threshold": SIM_THRESHOLD,
                    "n": len(results), "samples": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nsaved -> highconf_autoextend_v2_speaker.json")


if __name__ == "__main__":
    main()
