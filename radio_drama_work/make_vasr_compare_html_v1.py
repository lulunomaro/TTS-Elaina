# -*- coding: utf-8 -*-
"""Create a local A/B listening page for the VASR quality-gated candidates."""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(r"radio_drama_work/radio_drama_work")
REPORT = ROOT / "vasr_full_v2" / "quality_report.json"
OUT = ROOT / "vasr_full_v2" / "listen_raw_vs_vasr.html"


def file_uri(path: str) -> str:
    return Path(path).resolve().as_uri()


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    items = []
    for row in report["rows"]:
        original = row["original"]
        vasr = row.get("vasr")
        items.append({
            "index": row["index"] + 1,
            "source": Path(row["input"]).name.split("_vocals_")[0].split("_", 2)[-1],
            "name": Path(row["input"]).name,
            "text": row.get("text", ""),
            "decision": row.get("automatic_decision", "missing_output"),
            "original_uri": file_uri(row["input"]),
            "vasr_uri": file_uri(row["output"]) if row.get("output") else "",
            "duration": round(original["duration_sec"], 3),
            "target_before": round(original["target_center"], 4),
            "target_after": round(vasr["target_center"], 4) if vasr else None,
            "margin_before": round(original["target_negative_margin"], 4) if vasr and original["target_negative_margin"] is not None else None,
            "margin_after": round(vasr["target_negative_margin"], 4) if vasr and vasr["target_negative_margin"] is not None else None,
        })
    data = json.dumps(items, ensure_ascii=False)
    page = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>伊蕾娜候选 A/B 试听 - VASR v2</title>
<style>
:root{{--bg:#10151d;--panel:#1c2633;--line:#344456;--text:#edf4fb;--muted:#a9b7c6;--accent:#77c2ff;--good:#79e2aa;--review:#f0c674}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,-apple-system,"Segoe UI",sans-serif}}
header{{position:sticky;top:0;z-index:2;padding:18px 24px;background:rgba(16,21,29,.97);border-bottom:1px solid var(--line)}}h1{{margin:0 0 6px;font-size:22px}}.summary{{color:var(--muted);line-height:1.6;margin-bottom:12px}}
.toolbar{{display:flex;flex-wrap:wrap;gap:8px}}input,select,button{{padding:9px 11px;border:1px solid var(--line);border-radius:8px;background:var(--panel);color:var(--text);font-size:14px}}input{{min-width:260px;flex:1}}button{{cursor:pointer}}
main{{max-width:1450px;margin:auto;padding:18px 24px 40px}}#count{{color:var(--muted)}}.item{{display:grid;grid-template-columns:42px 210px 100px minmax(260px,1fr) minmax(260px,1fr);gap:12px;align-items:start;margin:8px 0;padding:12px 14px;background:var(--panel);border:1px solid var(--line);border-radius:10px}}.num{{color:var(--muted);text-align:right}}.source{{font-weight:650;word-break:break-all}}.meta{{color:var(--muted);font-size:12px;line-height:1.55;word-break:break-all}}.text{{margin:5px 0;word-break:break-all}}.duration{{white-space:nowrap;color:var(--good)}}.badge{{display:inline-block;padding:2px 6px;border-radius:5px;font-size:11px;margin-left:6px;color:#111;background:var(--good)}}.badge.review_only{{background:var(--review)}}audio{{width:100%;height:38px}}.cell{{min-width:0}}.cell h3{{font-size:13px;margin:0 0 4px}}.raw h3{{color:var(--accent)}}.vasr h3{{color:var(--good)}}.empty{{padding:30px;text-align:center;color:var(--muted)}}
@media(max-width:980px){{header,main{{padding-left:12px;padding-right:12px}}.item{{grid-template-columns:32px 1fr;gap:6px 10px}}.num{{grid-row:span 4}}.duration,.text,.raw,.vasr{{grid-column:2}}}}
</style></head><body>
<header><h1>伊蕾娜候选 A/B 试听：原始 BS-Roformer vs VASR</h1>
<div class="summary">52 条原始高置信度候选；46 条 VASR 通过自动验收，6 条 review-only。试听时重点检查“目标音色是否仍是伊蕾娜、句首是否完整、是否混入接话人声”；训练推荐清单采用通过的 VASR，失败的保留原始版本。</div>
<div class="toolbar"><input id="q" placeholder="搜索来源、文件名或 ASR 文本"><select id="decision"><option value="">全部</option><option value="pass">VASR 通过</option><option value="review_only">VASR review-only</option></select><select id="source"><option value="">全部来源</option></select><button id="stop">停止全部</button></div></header>
<main><p id="count"></p><section id="list"></section></main>
<script>const ITEMS={data};const q=document.getElementById('q'),decision=document.getElementById('decision'),source=document.getElementById('source'),list=document.getElementById('list'),count=document.getElementById('count');
for(const s of [...new Set(ITEMS.map(x=>x.source))].sort()){{const o=document.createElement('option');o.value=s;o.textContent=s;source.appendChild(o)}}
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function rows(){{const t=q.value.trim().toLowerCase();return ITEMS.filter(x=>(!decision.value||x.decision===decision.value)&&(!source.value||x.source===source.value)&&(!t||[x.name,x.source,x.text].join(' ').toLowerCase().includes(t)))}}
function render(){{const r=rows();count.textContent=`显示 ${{r.length}} / ${{ITEMS.length}} 条；每条左侧原始、右侧 VASR`;list.innerHTML=r.length?r.map(x=>`<article class="item"><div class="num">#${{String(x.index).padStart(2,'0')}}</div><div><div class="source">${{esc(x.source)}}<span class="badge ${{x.decision}}">${{x.decision==='pass'?'VASR通过':'VASR复核'}}</span></div><div class="meta">${{esc(x.name)}}</div><div class="text">${{esc(x.text)}}</div></div><div class="duration">${{x.duration.toFixed(2)}} 秒<br><span class="meta">声纹 ${{x.target_before.toFixed(3)}} → ${{x.target_after===null?'—':x.target_after.toFixed(3)}}</span></div><div class="cell raw"><h3>原始 BS-Roformer</h3><audio controls preload="none" src="${{x.original_uri}}"></audio><div class="meta">训练来源基线；VASR 失败时保留它</div></div><div class="cell vasr"><h3>VASR 修复/超分辨率</h3>${{x.vasr_uri?`<audio controls preload="none" src="${{x.vasr_uri}}"></audio><div class="meta">目标/硬负样本间隔：${{x.margin_before===null?'—':x.margin_before.toFixed(3)}} → ${{x.margin_after===null?'—':x.margin_after.toFixed(3)}}</div>`:'<div class="meta">没有输出</div>'}}</div></article>`).join(''):'<div class="empty">没有匹配项目。</div>'}}
q.addEventListener('input',render);decision.addEventListener('change',render);source.addEventListener('change',render);document.getElementById('stop').onclick=()=>document.querySelectorAll('audio').forEach(a=>{{a.pause();a.currentTime=0}});render();</script></body></html>'''
    OUT.write_text(page, encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
