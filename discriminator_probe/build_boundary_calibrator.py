# -*- coding: utf-8 -*-
"""Rebuild boundary calibrator with reliable click+cursor interaction.

Fix: dragging markers required precise hit (±0.08s) which felt broken.
New interaction:
  - click anywhere on waveform -> moves a cursor (yellow dashed line) + shows time
  - buttons: 「← 设为开头」 「设为结尾 →」 assign cursor to start/end
  - dragging green/red markers still works (threshold relaxed to 0.15s)
  - play buttons always use current mark values
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")
ROOT = Path(r"F:\GPTSOVITS\radio_drama_work")
OUT = PROBE / "boundary_calibrator"
AUDIO_DIR = OUT / "audio"
OUTPUT_SR = 32000
RADIO_MANIFEST = ROOT / "full_track_manifest_v1.json"
ANIME_MANIFEST = ROOT / "anime_batch_v1" / "manifest.json"
WAVE_HOP = 0.010
PAD_LEFT = 1.20
PAD_RIGHT = 0.80


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


def fine_wave(audio, sr, start, end):
    a = max(0, int(start * sr))
    b = min(len(audio), int(end * sr))
    seg = audio[a:b]
    if len(seg) == 0:
        return []
    hop = max(1, round(sr * WAVE_HOP))
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


HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>边界波形标定器</title>
<style>
:root{--bg:#080b12;--panel:#121725;--panel2:#171d2c;--line:#2a3348;--text:#edf2ff;--muted:#9da8c1;--cyan:#68e0dc;--green:#67d99e;--red:#fb7185;--amber:#f5c96d}
*{box-sizing:border-box}body{margin:0;min-height:100vh;background:radial-gradient(circle at 8% 0%,#172444 0,transparent 31%),radial-gradient(circle at 92% 4%,#2a1b3e 0,transparent 27%),var(--bg);color:var(--text);font-family:"Microsoft YaHei","Noto Sans SC",system-ui,sans-serif}
.app{max-width:1100px;margin:auto;padding:24px 20px 72px}
.hero{background:linear-gradient(180deg,rgba(23,29,44,.97),rgba(13,17,28,.97));border:1px solid var(--line);border-radius:20px;padding:22px 26px;margin-bottom:16px}
.eyebrow{color:var(--cyan);font-size:11px;letter-spacing:.18em;text-transform:uppercase}h1{font-size:clamp(24px,4vw,34px);margin:6px 0 10px}
.hero p{color:var(--muted);font-size:13px;line-height:1.7;margin:6px 0}
.hint{margin-top:12px;padding:12px 14px;border:1px solid #31544f;background:#102825;border-radius:12px;color:#d8fff7;font-size:13px;line-height:1.7}
.hint b{color:var(--green)}
.card{background:linear-gradient(180deg,rgba(23,29,44,.97),rgba(13,17,28,.97));border:1px solid var(--line);border-radius:16px;padding:16px 18px;margin-bottom:14px}
.card-head{display:flex;justify-content:space-between;align-items:baseline;gap:10px;margin-bottom:10px;flex-wrap:wrap}
.card-id{font-size:17px;font-weight:800} .card-id small{color:var(--muted);font-weight:400;font-size:11px;margin-left:8px}
.card-text{color:var(--muted);font-size:12px;max-width:520px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.wave-wrap{position:relative;height:150px;background:#0a0e18;border:1px solid var(--line);border-radius:10px;overflow:hidden;cursor:crosshair;user-select:none;touch-action:none}
canvas{display:block;width:100%;height:150px}
.marker{position:absolute;top:0;bottom:0;width:3px;cursor:ew-resize;touch-action:none;z-index:3}
.marker.start{background:var(--green)} .marker.end{background:var(--red)}
.marker::after{content:attr(data-t);position:absolute;top:2px;left:5px;font-size:10px;color:#fff;background:rgba(0,0,0,.6);padding:1px 4px;border-radius:4px;white-space:nowrap}
.cursor-line{position:absolute;top:0;bottom:0;width:2px;background:var(--amber);z-index:2;display:none}
.cursor-line::after{content:attr(data-t);position:absolute;top:2px;left:5px;font-size:10px;color:#fff;background:rgba(0,0,0,.6);padding:1px 4px;border-radius:4px}
.controls{display:flex;gap:8px;align-items:center;margin-top:10px;flex-wrap:wrap}
.btn{border:1px solid var(--line);background:#172033;color:var(--text);padding:7px 12px;border-radius:8px;cursor:pointer;font-size:13px}
.btn:hover{border-color:#62718f} .btn.play{background:#1c5552;border-color:#347d78}
.btn.setStart{background:#143a2a;border-color:#2e7a55;color:#b8f5d2}
.btn.setEnd{background:#3a1420;border-color:#7a2e44;color:#f5c8d4}
.btn.save{background:#7a5c1e;border-color:#a5823a}
.note{height:34px;border:1px solid var(--line);border-radius:8px;background:#101623;color:var(--text);padding:0 8px;flex:1;min-width:140px}
.time-readout{font-size:12px;color:var(--cyan);margin-left:auto;font-variant-numeric:tabular-nums}
.toolbar{display:flex;gap:8px;align-items:center;margin-bottom:14px;flex-wrap:wrap}
.toast{position:fixed;left:50%;bottom:22px;transform:translate(-50%,16px);opacity:0;padding:10px 16px;background:#eefcf8;color:#10221e;border-radius:99px;font-weight:700;transition:.2s;pointer-events:none;z-index:40}
.toast.show{opacity:1;transform:translate(-50%,0)}
</style></head><body><main class="app">
<div class="hero"><div class="eyebrow">Boundary Calibrator · 疑难杂症专用</div><h1>波形边界标定器</h1>
<p><b>点击波形</b>放置<span style="color:var(--amber)">黄色游标</span>，再点「←设为开头」「设为结尾→」确定两条边界线。也可以直接<b>拖</b>绿/红线微调。</p>
<div class="hint"><b>操作步骤：</b>① 点波形听声音（配合 开头0.4s/结尾0.4s 细听边缘）② 点"设为开头"放绿线 ③ 点"设为结尾"放红线 ④ 播区间确认 ⑤ 保存本条 → 下一段。正常段直接"下一段"跳过。全部标完点<b>导出JSON</b>。</div></div>
<div class="toolbar"><button class="btn save" id="exportAll">导出JSON</button><button class="btn" id="prev">上一段</button><button class="btn" id="next">下一段</button><span>跳到: <select id="jumpSel"></select></span></div>
<div id="cards"></div>
</main><div class="toast" id="toast"></div>
<script>
const CARDS = __DATA_JSON__;
const STORE = "ireina_boundary_calibrator_v1";
const marks = {};
for (const c of CARDS) marks[c.id] = {start: c.default_start, end: c.default_end, note: ""};
let current = 0;
let cursor = null;
const $ = id => document.getElementById(id);
function toast(m) { const e = $("toast"); e.textContent = m; e.classList.add("show"); clearTimeout(window._t); window._t = setTimeout(() => e.classList.remove("show"), 1600); }
function draw(card, el, mark) {
  const canvas = el.querySelector("canvas");
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
  const n = card.wave.length;
  if (!n) return;
  const gap = 1, bw = Math.max(1, (w - gap * (n - 1)) / n);
  for (let i = 0; i < n; i++) { const bar = Math.max(2, card.wave[i] * (h - 8)); ctx.fillStyle = "#3b4660"; ctx.fillRect(i * (bw + gap), (h - bar) / 2, bw, bar); }
  const sx = (mark.start / card.duration) * w, ex = (mark.end / card.duration) * w;
  ctx.fillStyle = "rgba(103,217,158,.18)"; ctx.fillRect(sx, 0, ex - sx, h);
}
function placeMarkers(card, el, mark) {
  const wrap = el.querySelector(".wave-wrap");
  const sm = wrap.querySelector(".marker.start"), em = wrap.querySelector(".marker.end");
  const pos = t => Math.max(0, Math.min(1, t / card.duration)) * wrap.clientWidth;
  sm.style.left = pos(mark.start) + "px"; sm.dataset.t = "开头 " + mark.start.toFixed(2) + "s";
  em.style.left = pos(mark.end) + "px"; em.dataset.t = "结尾 " + mark.end.toFixed(2) + "s";
  el.querySelector(".time-readout").textContent = mark.start.toFixed(2) + "s – " + mark.end.toFixed(2) + "s (长 " + (mark.end - mark.start).toFixed(2) + "s)";
}
function placeCursor(card, el, t) {
  const wrap = el.querySelector(".wave-wrap");
  const cl = wrap.querySelector(".cursor-line");
  cursor = Math.max(0, Math.min(card.duration, t));
  cl.style.display = "block";
  cl.style.left = (cursor / card.duration) * wrap.clientWidth + "px";
  cl.dataset.t = "游标 " + cursor.toFixed(2) + "s";
  const audio = el.querySelector("audio");
  audio.currentTime = cursor;
  audio.play();
  clearInterval(audio._iv);
  audio._iv = setInterval(() => {
    if (audio.currentTime >= Math.min(card.duration, cursor + 0.9)) { audio.pause(); clearInterval(audio._iv); }
  }, 40);
}
function bindCard(card) {
  const el = document.querySelector('#cards [data-id="' + card.id + '"]');
  const mark = marks[card.id];
  const audio = el.querySelector("audio"), wrap = el.querySelector(".wave-wrap");
  function xToTime(clientX) { const r = wrap.getBoundingClientRect(); return Math.max(0, Math.min(card.duration, (clientX - r.left) / r.width * card.duration)); }
  let dragging = null;
  wrap.addEventListener("pointerdown", e => {
    const t = xToTime(e.clientX);
    const ds = Math.abs(t - mark.start), de = Math.abs(t - mark.end);
    if (ds < 0.15 && ds <= de) { dragging = "start"; }
    else if (de < 0.15) { dragging = "end"; }
    else { placeCursor(card, el, t); }
  });
  wrap.addEventListener("pointermove", e => {
    if (!dragging) return;
    mark[dragging] = xToTime(e.clientX);
    if (mark.end < mark.start + 0.3) mark.end = mark.start + 0.3;
    placeMarkers(card, el, mark);
  });
  wrap.addEventListener("pointerup", () => { dragging = null; });
  wrap.addEventListener("pointerleave", () => { dragging = null; });
  el.querySelector(".bPlay").onclick = () => {
    audio.currentTime = mark.start;
    audio.play();
    clearInterval(audio._iv);
    audio._iv = setInterval(() => {
      if (audio.currentTime >= mark.end) { audio.pause(); clearInterval(audio._iv); }
    }, 40);
  };
  el.querySelector(".bHead").onclick = () => { audio.currentTime = Math.max(0, mark.start - 0.02); audio.play(); setTimeout(() => audio.pause(), 420); };
  el.querySelector(".bTail").onclick = () => { audio.currentTime = Math.max(0, mark.end - 0.42); audio.play(); setTimeout(() => audio.pause(), 420); };
  el.querySelector(".bStop").onclick = () => { audio.pause(); clearInterval(audio._iv); };
  el.querySelector(".bSetStart").onclick = () => {
    if (cursor === null) { toast("先在波形上点一下放游标"); return; }
    mark.start = Math.min(cursor, mark.end - 0.3);
    placeMarkers(card, el, mark); toast("开头设为 " + mark.start.toFixed(2) + "s");
  };
  el.querySelector(".bSetEnd").onclick = () => {
    if (cursor === null) { toast("先在波形上点一下放游标"); return; }
    mark.end = Math.max(cursor, mark.start + 0.3);
    placeMarkers(card, el, mark); toast("结尾设为 " + mark.end.toFixed(2) + "s");
  };
  el.querySelector(".bSave").onclick = () => { mark.note = el.querySelector(".note").value; localStorage.setItem(STORE, JSON.stringify(marks)); toast("已保存 " + card.id); };
  el.querySelector(".note").oninput = e => { mark.note = e.target.value; };
}
function render() {
  const card = CARDS[current];
  const wrap = $("cards");
  if (!card) { wrap.innerHTML = "<p>全部完成</p>"; return; }
  wrap.innerHTML =
    '<div class="card" data-id="' + card.id + '">' +
    '<div class="card-head"><span class="card-id">' + card.id + '<small>' + card.origin_blind_id + ' · ' + card.source_id + ' · ' + card.duration.toFixed(1) + 's</small></span>' +
    '<span class="card-text">' + (card.text || "") + '</span></div>' +
    '<div class="wave-wrap"><canvas></canvas><div class="cursor-line" data-t=""></div><div class="marker start" data-t=""></div><div class="marker end" data-t=""></div></div>' +
    '<audio src="' + card.audio + '" preload="auto"></audio>' +
    '<div class="controls"><button class="btn setStart bSetStart">← 设为开头</button><button class="btn setEnd bSetEnd">设为结尾 →</button>' +
    '<button class="btn play bPlay">▶ 播区间</button><button class="btn bHead">▶ 开头0.4s</button><button class="btn bTail">▶ 结尾0.4s</button><button class="btn bStop">⏹ 停</button>' +
    '<input class="note" placeholder="备注（可选）"><button class="btn save bSave">保存本条</button><span class="time-readout"></span></div>' +
    '</div>';
  cursor = null;
  bindCard(card);
  const mark = marks[card.id];
  placeMarkers(card, wrap.querySelector('[data-id="' + card.id + '"]'), mark);
  draw(card, wrap.querySelector('[data-id="' + card.id + '"]'), mark);
  $("jumpSel").innerHTML = CARDS.map((c, i) => '<option value="' + i + '">' + c.id + ' (' + c.origin_blind_id + ')</option>').join("");
  $("jumpSel").value = current;
}
$("prev").onclick = () => { if (current > 0) { current--; render(); } };
$("next").onclick = () => { if (current < CARDS.length - 1) { current++; render(); } else toast("已是最后一段"); };
$("jumpSel").onchange = e => { current = +e.target.value; render(); };
$("exportAll").onclick = () => {
  const body = { version: "boundary_calibrator_v1", updated_at: new Date().toISOString(),
    samples: CARDS.map(c => ({ id: c.id, origin_blind_id: c.origin_blind_id, source_id: c.source_id,
      window_start: c.window_start, start: +(marks[c.id].start).toFixed(3), end: +(marks[c.id].end).toFixed(3),
      note: marks[c.id].note })) };
  const blob = new Blob([JSON.stringify(body, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "boundary_calibrator_v1_" + new Date().toISOString().replace(/[:.]/g, "-") + ".json";
  a.click();
  toast("已导出 JSON");
};
window.addEventListener("resize", () => { if (CARDS[current]) draw(CARDS[current], document.querySelector('#cards [data-id="' + CARDS[current].id + '"]'), marks[CARDS[current].id]); });
render();
</script></body></html>"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    r3 = json.loads((PROBE / "boundary_rescue_v3" / "report.json").read_text(encoding="utf-8"))
    hard_origins = set(x["origin"] for x in r3["samples"])

    segments = []
    for v, key in (("v1", "discriminator_priority_review_v1"), ("v2", "discriminator_priority_review_v2"),
                   ("v3", "discriminator_priority_review_v3")):
        rp = PROBE / key / "human_feedback" / "labels_resolved.json"
        if not rp.exists():
            continue
        doc = json.loads(rp.read_text(encoding="utf-8"))
        for row in doc.get("samples", []):
            if row.get("final_class") == "boundary_cut":
                oid = row.get("blind_id", "")
                if oid not in hard_origins:
                    continue
                segments.append({"source_id": row["source_id"], "start": float(row["start"]),
                                 "end": float(row["end"]), "origin": v, "origin_blind_id": oid,
                                 "text": row.get("text", "")})
    seen = set()
    todo = []
    for s in segments:
        key = (s["origin_blind_id"], round(s["start"], 2))
        if key in seen:
            continue
        seen.add(key)
        todo.append(s)
    print(f"to calibrate: {len(todo)}", flush=True)

    cards = []
    for i, seg in enumerate(todo, 1):
        sid = seg["source_id"]
        vocals = source_vocals(sid)
        if vocals is None or not vocals.exists():
            continue
        audio, sr = sf.read(str(vocals), dtype="float32", always_2d=True)
        mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
        win_start = max(0.0, seg["start"] - PAD_LEFT)
        win_end = min(len(mono) / sr, seg["end"] + PAD_RIGHT)
        a0 = max(0, round(win_start * sr))
        b0 = min(len(mono), round(win_end * sr))
        clip = micro_fade(resample(mono[a0:b0], sr, OUTPUT_SR))
        cid = f"BC{i:03d}"
        dest = AUDIO_DIR / f"{cid}.wav"
        sf.write(str(dest), clip, OUTPUT_SR, subtype="PCM_16")
        wave = fine_wave(clip, OUTPUT_SR, 0.0, len(clip) / OUTPUT_SR)
        cards.append({
            "id": cid,
            "origin_blind_id": seg["origin_blind_id"],
            "source_id": sid,
            "text": seg["text"][:80],
            "audio": f"audio/{dest.name}",
            "window_start": round(win_start, 3),
            "default_start": round(seg["start"] - win_start, 3),
            "default_end": round(seg["end"] - win_start, 3),
            "duration": round(win_end - win_start, 3),
            "wave": wave,
        })

    write_json(OUT / "audit_manifest.json", {"version": "boundary_calibrator_audit", "samples": cards})
    write_json(OUT / "report.json", {"version": "boundary_calibrator_v1", "created_at": datetime.now(timezone.utc).isoformat(), "cards": len(cards)})

    data_json = json.dumps(cards, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__DATA_JSON__", data_json)
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(f"saved={OUT / 'index.html'} cards={len(cards)}", flush=True)


if __name__ == "__main__":
    main()
