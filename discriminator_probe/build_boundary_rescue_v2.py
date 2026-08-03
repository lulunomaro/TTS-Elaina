# -*- coding: utf-8 -*-
"""Build boundary-rescue variants for the 38 v2 boundary-cut clips.

Re-cuts from original BS-Roformer vocals using ASR word boundaries. Produces
two variants per clip: word-expanded (restore adjacent ASR words) and
pad-220ms (pure waveform extension). Writes HTML review page + manifest.
All output lives in discriminator_probe; radio_drama_work is read-only.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")
ROOT = Path(r"F:\GPTSOVITS\radio_drama_work")
OUT = PROBE / "boundary_rescue_v2"
AUDIO_DIR = OUT / "audio"
QUEUE = PROBE / "boundary_rescue_queue_v2.json"
OUTPUT_SR = 32000
PAD_SEC = 0.22
MAX_SEC = 6.0
ANIME_MANIFEST = ROOT / "anime_batch_v1" / "manifest.json"
RADIO_MANIFEST = ROOT / "full_track_manifest_v1.json"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def micro_fade(audio: np.ndarray) -> np.ndarray:
    out = np.asarray(audio, dtype=np.float32).copy()
    count = min(len(out) // 4, max(1, round(OUTPUT_SR * 0.002)))
    if count:
        ramp = np.linspace(0.0, 1.0, count, dtype=np.float32)
        out[:count] *= ramp
        out[-count:] *= ramp[::-1]
    return out


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return np.asarray(audio.mean(axis=1), dtype=np.float32), int(rate)


def resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return audio
    from scipy.signal import resample_poly
    g = np.gcd(src, dst)
    return resample_poly(audio, dst // g, src // g).astype(np.float32)


def source_vocals(source_id: str) -> Path | None:
    if source_id.startswith("majo_"):
        doc = json.loads(ANIME_MANIFEST.read_text(encoding="utf-8"))
        for t in doc["tracks"]:
            if t["id"] == source_id:
                return Path(t["vocals"])
    else:
        doc = json.loads(RADIO_MANIFEST.read_text(encoding="utf-8"))
        for t in doc["tracks"]:
            if t["id"] == source_id:
                return Path(t["vocals"])
    return None


def load_words(source_id: str) -> list[dict]:
    if source_id.startswith("majo_"):
        rep = ROOT / "anime_mining_v1" / "asr" / "source_reports" / f"{source_id}.json"
    else:
        rep = ROOT / "word_target_all_v1" / "source_reports" / f"{source_id}.json"
    if not rep.exists():
        return []
    doc = json.loads(rep.read_text(encoding="utf-8"))
    return doc.get("words", [])


def export_clip(audio: np.ndarray, sr: int, start: float, end: float, path: Path) -> float:
    a = max(0, round(start * sr))
    b = min(len(audio), round(end * sr))
    clip = resample(audio[a:b], sr, OUTPUT_SR)
    clip = micro_fade(clip)
    sf.write(str(path), clip, OUTPUT_SR, subtype="PCM_16")
    return float(len(clip)) / OUTPUT_SR


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    queue = json.loads(QUEUE.read_text(encoding="utf-8"))
    samples = queue.get("samples", [])
    print(f"queue: {len(samples)}", flush=True)

    manifest = []
    idx = 0
    skipped = []
    for row in samples:
        sid = str(row["source_id"])
        start, end = float(row["start"]), float(row["end"])
        vocals = source_vocals(sid)
        words = load_words(sid)
        if vocals is None or not vocals.exists():
            skipped.append({"source_id": sid, "reason": "no_vocals"})
            continue
        audio, sr = read_mono(vocals)

        # word-expanded variant: nearest word boundaries around [start, end]
        inside = [w for w in words if w["end"] > start - 0.3 and w["start"] < end + 0.3]
        if inside:
            w_start = min(float(w["start"]) for w in inside) - 0.05
            w_end = max(float(w["end"]) for w in inside) + 0.05
        else:
            w_start, w_end = start, end
        # pad-220ms variant
        p_start = max(0.0, start - PAD_SEC)
        p_end = end + PAD_SEC

        variants = []
        for name, (s, e) in (("word_expand", (w_start, w_end)), ("pad_220ms", (p_start, p_end))):
            if e - s > MAX_SEC:
                e = s + MAX_SEC
            if e - s < 0.8:
                continue
            bid = f"BR{idx + 1:03d}"
            dest = AUDIO_DIR / f"{bid}_{name}.wav"
            dur = export_clip(audio, sr, s, e, dest)
            variants.append({"blind_id": bid, "variant": name, "audio": str(dest.resolve()),
                             "start": round(s, 3), "end": round(e, 3), "duration": round(dur, 3)})
        if not variants:
            skipped.append({"source_id": sid, "start": start, "end": end, "reason": "too_short"})
            continue
        for v in variants:
            v.update({"source_id": sid, "origin_start": start, "origin_end": end,
                      "text": row.get("text", ""), "note": row.get("note", "")})
            manifest.append(v)
        idx += 1

    print(f"variants: {len(manifest)}, skipped: {len(skipped)}", flush=True)
    write_json(OUT / "audit_manifest.json", {"version": "boundary_rescue_v2_audit", "samples": manifest})
    report = {
        "version": "boundary_rescue_v2_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_queue": len(samples),
        "variant_count": len(manifest),
        "duration_sec": round(sum(r["duration"] for r in manifest), 2),
        "skipped": skipped,
        "per_source": dict(Counter(r["source_id"] for r in manifest)),
        "training_started": False,
    }
    write_json(OUT / "report.json", report)

    items = []
    for i, r in enumerate(manifest, 1):
        src = f"audio/{Path(r['audio']).name}"
        items.append(f"<li><b>{r['blind_id']}·{r['variant']}</b> ({r['source_id']} {r['duration']:.1f}s) "
                     f"<audio controls preload='none' src='{src}'></audio>"
                     f"<br><small>{r['text'][:40]}</small></li>")
    html = ("<!doctype html><meta charset='utf-8'><title>v2边界救援</title>"
            "<h1>判别器v2·38条边界问题救援（52变体）</h1>"
            "<p>每问题两个版本：word_expand（补相邻ASR词）/ pad_220ms（波形外扩）。"
            "判定：目标声音完整自然则通过；只保留通过的版本；都失败则继续裁。</p><ol>"
            + "\n".join(items) + "</ol>")
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(f"saved={OUT / 'index.html'} variants={len(manifest)}", flush=True)


if __name__ == "__main__":
    main()
