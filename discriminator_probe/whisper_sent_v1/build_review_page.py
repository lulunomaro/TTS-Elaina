# -*- coding: utf-8 -*-
"""Build a minimal A/B listening review page (15 items, 4-choice)."""
import io
import json
from pathlib import Path

PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")
OUT = PROBE / "autoextend_review"

data = json.loads((OUT / "review.json").read_text(encoding="utf-8"))

cards_html = []
for it in data["items"]:
    cards_html.append(f"""
    <div class="card" data-id="{it['id']}">
      <div class="head"><span class="id">{it['id']}</span><span class="blind">{it['blind']}</span>
      <span class="src">{it['source']} | 原切 {it['orig']}s → 扩展 {it['ext']}s</span></div>
      <div class="row"><span class="tag">A 原切</span><audio controls preload="none" src="{it['A_orig_wav']}"></audio></div>
      <div class="row"><span class="tag">B 扩展</span><audio controls preload="none" src="{it['B_ext_wav']}"></audio></div>
      <div class="choices">
        <button class="c ok" data-v="ok">OK 完整句</button>
        <button class="c long" data-v="long">B太长</button>
        <button class="c short" data-v="short">B还短</button>
        <button class="c other" data-v="other">B含别人/杂音</button>
      </div>
      <div class="verdict"></div>
    </div>""")

html = """<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>自动扩展验收 A/B 听音（15条）</title>
<style>
body{background:#0d1117;color:#e6edf3;font-family:"Segoe UI",sans-serif;padding:16px;max-width:760px;margin:0 auto}
h1{font-size:16px;color:#7ee787}.hint{color:#8b949e;font-size:13px;margin-bottom:14px}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px;margin-bottom:12px}
.head{display:flex;gap:10px;align-items:center;margin-bottom:8px;font-size:13px}
.id{color:#79c0ff;font-weight:700}.blind{color:#d2a8ff}.src{color:#8b949e}
.row{display:flex;align-items:center;gap:8px;margin:4px 0}
.tag{width:64px;font-size:12px;color:#8b949e}
audio{height:32px;flex:1}
.choices{display:flex;gap:8px;margin-top:8px}
.c{padding:5px 10px;border-radius:6px;border:1px solid #30363d;background:#21262d;color:#c9d1d9;cursor:pointer;font-size:13px}
.c.sel{border-color:#7ee787;background:#1f3d2b}
.verdict{margin-top:6px;font-size:12px;color:#7ee787;min-height:14px}
</style></head><body>
<h1>自动扩展验收（whisper 语义句边界）</h1>
<div class="hint">流程：先听 <b>A 原切</b>，再听 <b>B 扩展</b>。B 应该包含完整一句话（可能前后多一句）。判断后点一个按钮（可改）。全部选完点「导出」。回答会进 localStorage，中途关闭不丢。</div>
<div id="cards">__CARDS__</div>
<button id="export" style="position:sticky;bottom:8px;padding:8px 16px;border-radius:8px;background:#238636;color:#fff;border:none;cursor:pointer">导出JSON（标完点我）</button>
<script>
const STORE="ireina_autoextend_review_v1";
const ITEMS = __ITEMS__;
const verdicts = {};
try { const s=JSON.parse(localStorage.getItem(STORE)||"{}"); for(const k in s) verdicts[k]=s[k]; } catch(e){}
document.querySelectorAll(".card").forEach(card=>{
  const id=card.dataset.id;
  const show=()=>{
    const v=verdicts[id];
    card.querySelectorAll(".c").forEach(b=>b.classList.toggle("sel", b.dataset.v===v));
    card.querySelector(".verdict").textContent=v?("你的判断: "+{ok:"OK 完整句",long:"B太长",short:"B还短",other:"B含别人/杂音"}[v]):"";
  };
  card.querySelectorAll(".c").forEach(b=>b.onclick=()=>{verdicts[id]=b.dataset.v;localStorage.setItem(STORE,JSON.stringify(verdicts));show();});
  show();
});
document.getElementById("export").onclick=()=>{
  const body={version:"autoextend_review_v1",updated_at:new Date().toISOString(),
    items:ITEMS.map(it=>({id:it.id,blind:it.blind,source:it.source,orig:it.orig,ext:it.ext,verdict:verdicts[it.id]||"unanswered"}))};
  const blob=new Blob([JSON.stringify(body,null,2)],{type:"application/json"});
  const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="autoextend_review_"+Date.now()+".json";a.click();
};
</script></body></html>"""

cards_joined = "\n".join(cards_html)
html = html.replace("__CARDS__", cards_joined)
html = html.replace("__ITEMS__", json.dumps([{k: it[k] for k in ("id", "blind", "source", "orig", "ext")} for it in data["items"]], ensure_ascii=False))
(OUT / "index.html").write_text(html, encoding="utf-8")
print(f"saved {OUT / 'index.html'}")
