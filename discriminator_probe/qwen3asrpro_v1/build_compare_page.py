# -*- coding: utf-8 -*-
"""Build A/B listening page: OUR final cuts vs QWEN3ASRPRO auto cuts, per gold region.
Reads completed qwen3asrpro_v1/*.json; regenerable as sources complete."""
from __future__ import annotations

import io
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")
ROOT = Path(r"F:\GPTSOVITS\radio_drama_work")
QDIR = PROBE / "qwen3asrpro_v1"
OUT = PROBE / "qwen_compare"
OUT.mkdir(parents=True, exist_ok=True)
AUDIO = OUT / "audio"
AUDIO.mkdir(parents=True, exist_ok=True)

FFMPEG = r"F:\GPTSOVITS\GPT-SoVITS-v2pro-20250604-nvidia50\GPT-SoVITS-v2pro-20250604-nvidia50\runtime\ffmpeg.exe"
ANIME_MANIFEST = ROOT / "anime_batch_v1" / "manifest.json"
RADIO_MANIFEST = ROOT / "full_track_manifest_v1.json"

gold = json.loads((PROBE / "highconf_calibrator" / "gold_calibrations.json").read_text(encoding="utf-8"))["samples"]
hf = json.loads((PROBE / "highconf_final_v1.json").read_text(encoding="utf-8"))["samples"]
hf_by_blind = {s["origin_blind_id"]: s for s in hf}


def source_vocals(sid):
    mf = json.loads((ANIME_MANIFEST if sid.startswith("majo_") else RADIO_MANIFEST).read_text(encoding="utf-8"))
    for t in mf.get("tracks", []):
        if t["id"] == sid:
            return Path(t["vocals"])
    return None


def cut(sid, s0, s1, dest):
    vocals = source_vocals(sid)
    r = subprocess.run([FFMPEG, "-y", "-ss", f"{s0:.3f}", "-i", str(vocals), "-t", f"{s1 - s0:.3f}",
                        "-ac", "1", "-ar", "32000", "-c:a", "pcm_s16le", str(dest)], capture_output=True, timeout=60)
    return r.returncode == 0


def qwen_cut_for(g, data):
    """Qwen pipeline cut: pyannote speaker segment containing gold midpoint (merge same-speaker neighbors)."""
    mid = (g["abs_start"] + g["abs_end"]) / 2
    spk = data.get("diarization", [])
    seg = next((s for s in spk if s[0] <= mid < s[1]), None)
    if not seg:
        return None
    label = seg[2]
    s0, s1 = seg[0], seg[1]
    changed = True
    while changed:
        changed = False
        for s in spk:
            if s[2] != label:
                continue
            ns0, ns1 = s0, s1
            if s[1] >= s0 - 0.8 and s[0] <= s0:
                ns0 = min(s0, s[0])
            elif s[0] <= s1 + 0.8 and s[1] >= s1:
                ns1 = max(s1, s[1])
            if ns0 != s0 or ns1 != s1:
                s0, s1, changed = ns0, ns1, True
    txt = "".join(c["text"] for c in data.get("chars", []) if s0 <= c["start"] < s1)
    return {"start": s0, "end": s1, "speaker": label, "text": txt[:100]}


def our_cut_for(g):
    """Our pipeline cut: final list boundary for this blind id."""
    h = hf_by_blind.get(g["origin_blind_id"])
    if h:
        return {"start": h["start"], "end": h["end"], "speaker": "ireina", "text": (h.get("text") or "")[:100]}
    # fallback: gold itself
    return {"start": g["abs_start"], "end": g["abs_end"], "speaker": "ireina", "text": ""}


items = []
for g in gold:
    qj = QDIR / f"{g['source_id']}.json"
    if not qj.exists():
        continue
    data = json.loads(qj.read_text(encoding="utf-8"))
    qc = qwen_cut_for(g, data)
    oc = our_cut_for(g)
    if not qc:
        continue
    rid = g["qc_id"]
    qa = AUDIO / f"{rid}_Q.wav"
    oa = AUDIO / f"{rid}_O.wav"
    cut(g["source_id"], qc["start"], qc["end"], qa)
    cut(g["source_id"], oc["start"], oc["end"], oa)
    items.append({
        "id": rid, "blind": g["origin_blind_id"], "source": g["source_id"],
        "gold": [round(g["abs_start"], 2), round(g["abs_end"], 2)],
        "ours": {**oc, "wav": f"audio/{oa.name}"},
        "qwen": {**qc, "wav": f"audio/{qa.name}"},
    })

cards = []
for it in items:
    def fmt(t):
        return f"{t['start']:.2f}-{t['end']:.2f}s [{t['speaker']}] {t['text'][:40]}"
    cards.append(f"""
    <div class="card">
      <div class="head"><b>{it['id']}</b> {it['blind']} · {it['source']} · 人工金标 {it['gold'][0]}-{it['gold'][1]}s</div>
      <div class="row"><span class="tag">O 我们(whisper+cam++)</span><audio controls preload="none" src="{it['ours']['wav']}"></audio><span class="meta">{fmt(it['ours'])}</span></div>
      <div class="row"><span class="tag">Q Qwen3ASRpro</span><audio controls preload="none" src="{it['qwen']['wav']}"></audio><span class="meta">{fmt(it['qwen'])}</span></div>
    </div>""")

html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"><title>Qwen3ASRpro vs 我们（A/B 试听）</title>
<style>
body{{background:#0d1117;color:#e6edf3;font-family:"Segoe UI",sans-serif;padding:16px;max-width:860px;margin:0 auto}}
h1{{font-size:16px;color:#7ee787}}.note{{color:#8b949e;font-size:13px;margin-bottom:14px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px;margin-bottom:12px}}
.head{{font-size:13px;margin-bottom:8px;color:#79c0ff}}
.row{{display:flex;align-items:center;gap:8px;margin:6px 0;flex-wrap:wrap}}
.tag{{width:110px;font-size:12px;color:#d2a8ff}}
audio{{height:32px;flex:1;min-width:220px}}
.meta{{font-size:11px;color:#8b949e;max-width:340px}}
</style></head><body>
<h1>Qwen3ASRpro vs 我们管道（A/B 试听）</h1>
<div class="note">O = 我们最终训练集切段（whisper+cam++ 管道）｜Q = Qwen3ASRpro 自动（pyannote community-1 说话人段 + Qwen3-ASR）｜金标 = 你人工标的完整句。注意：Q 的段是<b>机器全自动</b>的（含说话人标签与文本）。已处理源：{len(items)}/{len(gold)} 条金标。</div>
{"".join(cards)}
</body></html>"""
(OUT / "index.html").write_text(html, encoding="utf-8")
print(f"page: {OUT / 'index.html'} ({len(items)} items)")
for it in items:
    print(f"  {it['id']} gold {it['gold']} | ours {it['ours']['start']:.2f}-{it['ours']['end']:.2f} | qwen {it['qwen']['start']:.2f}-{it['qwen']['end']:.2f} [{it['qwen']['speaker']}]")
