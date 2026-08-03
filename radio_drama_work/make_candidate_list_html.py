# -*- coding: utf-8 -*-
"""Build a local listening page for the new candidate WAV clips."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import soundfile as sf


ROOT = Path(r"radio_drama_work/radio_drama_work")
VERSION = sys.argv[1] if len(sys.argv) > 1 else "v1"
VERSION_DIR = ROOT / f"new_radio_quality_{VERSION}"
LIST_PATH = ROOT / f"new_radio_quality_train_candidate_{VERSION}.list"
REPORT_PATH = VERSION_DIR / "report.json"
CLIP_DIR = VERSION_DIR / "accepted_segments"
HTML_PATH = VERSION_DIR / "listen_candidates.html"


def name_of(value: str) -> str:
    return Path(value.replace("//", "/")).name


def source_of(name: str) -> str:
    return re.sub(r"_vocals_.*$", "", name).replace("_vocals", "")


def interval_of(name: str) -> str:
    match = re.search(r"_(/d{10})_(/d{10})/.wav$", name)
    if not match:
        return ""
    return f"{int(match.group(1)) / 32000:.2f}s–{int(match.group(2)) / 32000:.2f}s"


def load_items() -> list[dict]:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    scores = {}
    for item in report.get("files", []):
        for segment in item.get("segments", []):
            scores[name_of(segment.get("file", ""))] = segment

    items = []
    for index, line in enumerate(LIST_PATH.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        path_text, *_ = line.split("|", 3)
        name = name_of(path_text)
        score = scores.get(name, {})
        items.append({
            "index": index,
            "name": name,
            "source": source_of(name),
            "interval": interval_of(name),
            "src": f"accepted_segments/{name}",
            "duration": round(sf.info(str(CLIP_DIR / name)).duration, 2),
            "text": line.split("|", 3)[-1],
            "center": round(float(score.get("anchor_best_center", 0.0)), 3),
            "p25": round(float(score.get("anchor_min_p25", 0.0)), 3),
            "other": round(float(score.get("other_ratio", 0.0)), 3),
        })
    return items


def build(items: list[dict]) -> str:
    data = json.dumps(items, ensure_ascii=False).replace("</", "<///")
    return """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>伊蕾娜候选训练音频</title>
<style>
:root{color-scheme:dark;--bg:#11151c;--panel:#1c2430;--line:#344052;--text:#edf2f7;--muted:#aab6c5;--accent:#75b9ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
header{position:sticky;top:0;z-index:2;padding:18px 24px;background:rgba(17,21,28,.96);border-bottom:1px solid var(--line)}
h1{margin:0 0 5px;font-size:22px}.summary{color:var(--muted);margin-bottom:12px}
.toolbar{display:flex;flex-wrap:wrap;gap:8px}input,select,button{padding:9px 11px;border:1px solid var(--line);border-radius:8px;background:var(--panel);color:var(--text);font-size:14px}input{min-width:280px;flex:1}button{cursor:pointer}button:hover{border-color:var(--accent)}
main{max-width:1350px;margin:auto;padding:18px 24px 40px}#count{color:var(--muted)}
.item{display:grid;grid-template-columns:42px minmax(180px,1fr) 90px 125px minmax(260px,1.5fr);gap:12px;align-items:center;margin:8px 0;padding:12px 14px;background:var(--panel);border:1px solid var(--line);border-radius:10px}.num{color:var(--muted);text-align:right}.source{font-weight:650}.meta{color:var(--muted);font-size:12px;line-height:1.55}.text{margin-bottom:4px;word-break:break-all}audio{width:100%;height:38px}.duration{color:#7ee2ad;white-space:nowrap}.empty{padding:30px;text-align:center;color:var(--muted)}
@media(max-width:850px){header,main{padding-left:12px;padding-right:12px}.item{grid-template-columns:32px 1fr;gap:6px 10px}.num{grid-row:span 2}.meta,.text,audio{grid-column:2}audio{min-width:0}}
</style></head><body>
<header><h1>伊蕾娜候选训练音频</h1>
<div class="summary">已完成 BGM 分离、滑窗声纹和逐词剔除。自动高置信度候选共 ITEMS_COUNT 段，约 3 分 48 秒。</div>
<div class="toolbar"><input id="q" placeholder="搜索文件名、来源或识别文本…"><select id="source"><option value="">全部来源</option></select><button id="pause">暂停全部</button><button id="continuous">连续播放当前筛选</button></div></header>
<main><p id="count"></p><section id="list"></section></main>
<script>
const ITEMS = ITEMS_DATA;
const q=document.getElementById('q'), source=document.getElementById('source'), list=document.getElementById('list'), count=document.getElementById('count');
const sources=[...new Set(ITEMS.map(x=>x.source))].sort();
for(const value of sources){const o=document.createElement('option');o.value=value;o.textContent=value;source.appendChild(o)}
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function rows(){const term=q.value.trim().toLowerCase();return ITEMS.filter(x=>(!source.value||x.source===source.value)&&(!term||[x.name,x.source,x.text,x.interval].join(' ').toLowerCase().includes(term)))}
function render(){const r=rows();count.textContent=`显示 ${r.length} / ${ITEMS.length} 段`;if(!r.length){list.innerHTML='<div class="empty">没有匹配的候选。</div>';return}list.innerHTML=r.map(x=>`<article class="item"><div class="num">#${String(x.index).padStart(3,'0')}</div><div><div class="source">${esc(x.source)}</div><div class="meta">${esc(x.name)}<br>${esc(x.interval)}</div></div><div class="duration">${x.duration.toFixed(2)} 秒</div><div class="meta">center ${x.center.toFixed(3)} · p25 ${x.p25.toFixed(3)}<br>other ${x.other.toFixed(3)}</div><div><div class="text">${esc(x.text)}</div><audio controls preload="none" data-index="${x.index}" src="${encodeURI(x.src)}"></audio></div></article>`).join('');document.querySelectorAll('audio').forEach(a=>a.addEventListener('ended',next))}
let auto=false;function next(e){if(!auto)return;const r=rows(),i=Number(e.target.dataset.index),n=r.find(x=>x.index>i);if(n){const a=document.querySelector(`audio[data-index="${n.index}"]`);if(a){a.play();a.scrollIntoView({behavior:'smooth',block:'center'})}}}
q.addEventListener('input',render);source.addEventListener('change',render);document.getElementById('pause').onclick=()=>document.querySelectorAll('audio').forEach(a=>{a.pause();a.currentTime=0});document.getElementById('continuous').onclick=()=>{auto=true;const f=rows()[0];if(f){const a=document.querySelector(`audio[data-index="${f.index}"]`);if(a)a.play()}};render();
</script></body></html>""".replace("ITEMS_COUNT", str(len(items))).replace("ITEMS_DATA", data)


def main() -> None:
    items = load_items()
    HTML_PATH.write_text(build(items), encoding="utf-8")
    print(f"created={HTML_PATH}")
    print(f"items={len(items)}")


if __name__ == "__main__":
    main()
