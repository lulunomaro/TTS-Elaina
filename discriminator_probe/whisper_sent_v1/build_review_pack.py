# -*- coding: utf-8 -*-
"""Build A/B acceptance pack: original cut vs whisper-extended cut, 15 items."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf

PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")
ROOT = Path(r"F:\GPTSOVITS\radio_drama_work")
OUT = PROBE / "autoextend_review"
AUDIO = OUT / "audio"
OUTPUT_SR = 32000
RADIO_MANIFEST = ROOT / "full_track_manifest_v1.json"
ANIME_MANIFEST = ROOT / "anime_batch_v1" / "manifest.json"

report = json.loads((PROBE / "highconf_boundary_screen" / "screen_report.json").read_text(encoding="utf-8"))
ext = json.loads((PROBE / "highconf_autoextend_v2_speaker.json").read_text(encoding="utf-8"))["samples"]
gold = json.loads((PROBE / "highconf_calibrator" / "gold_calibrations.json").read_text(encoding="utf-8"))["samples"]

ext_by_blind = {s["origin_blind_id"]: s for s in ext}
orig_by_blind = {Path(i["audio"]).name.replace(".wav", ""): i for i in report["need_calib"]}


def source_vocals(sid):
    mf = json.loads((ANIME_MANIFEST if sid.startswith("majo_") else RADIO_MANIFEST).read_text(encoding="utf-8"))
    for t in mf.get("tracks", []):
        if t["id"] == sid:
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
    AUDIO.mkdir(parents=True, exist_ok=True)

    # 15 items: 10 gold pairs + 5 random (include short-sentence cases)
    gold_blinds = [g["origin_blind_id"] for g in gold]
    others = [b for b in ext_by_blind if b not in gold_blinds]
    rng = np.random.RandomState(42)
    extras = list(rng.choice(others, size=5, replace=False))
    blinds = gold_blinds + extras

    items = []
    for i, blind in enumerate(blinds, 1):
        o = orig_by_blind[blind]
        e = ext_by_blind[blind]
        sid = o["source_id"]
        vocals = source_vocals(sid)
        audio, sr = sf.read(str(vocals), dtype="float32", always_2d=True)
        mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
        pairs = [
            ("A_orig", o["start"], o["end"]),
            ("B_ext", e["start"], e["end"]),
        ]
        rec = {"id": f"AV{i:03d}", "blind": blind, "source": sid,
               "orig": f"{o['start']:.2f}-{o['end']:.2f}", "ext": f"{e['start']:.2f}-{e['end']:.2f}"}
        for tag, s0, s1 in pairs:
            a0 = max(0, round(s0 * sr)); b0 = min(len(mono), round(s1 * sr))
            clip = micro_fade(resample(mono[a0:b0], sr, OUTPUT_SR))
            fn = f"{rec['id']}_{tag}.wav"
            sf.write(str(AUDIO / fn), clip, OUTPUT_SR, subtype="PCM_16")
            rec[tag + "_wav"] = f"audio/{fn}"
        items.append(rec)

    payload = {
        "version": "autoextend_review_v1",
        "n": len(items),
        "instructions": "先听 A(原切) 再听 B(whisper扩展)。B 应包含完整句。判断: OK / 太长 / 还短 / 包含别人说话",
        "items": items,
    }
    (OUT / "review.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"review pack: {len(items)} items -> {OUT}")
    for r in items:
        print(f"  {r['id']} {r['blind']} {r['source']} orig[{r['orig']}] -> ext[{r['ext']}]")


if __name__ == "__main__":
    main()
