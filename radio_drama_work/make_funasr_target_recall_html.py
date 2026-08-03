"""Build a local listening page for the FunASR v2 candidate sets."""
from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf


ROOT = Path(r"radio_drama_work/radio_drama_work")
VERSION_DIR = ROOT / "full_funasr_target_recall_v2"
REPORT = VERSION_DIR / "report.json"
HTML = VERSION_DIR / "listen_candidates.html"


def short_name(path: str) -> str:
    return Path(path.replace("//", "/")).name.replace("_vocals.wav", "")


def load_items() -> list[dict]:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    items: list[dict] = []
    index = 1
    for source in report.get("sources", []):
        source_name = short_name(source["source"])
        for tier, key, folder in (
            ("accepted", "accepted_segments", "accepted_segments"),
            ("expanded", "expanded_segments", "expanded_review"),
            ("review", "review_segments", "review"),
        ):
            for row in source.get(key, []):
                path = Path(row["file"])
                if not path.exists():
                    continue
                items.append(
                    {
                        "index": index,
                        "tier": tier,
                        "source": source_name,
                        "name": path.name,
                        "src": f"{folder}/{path.name}",
                        "duration": round(float(sf.info(str(path)).duration), 2),
                        "text": row.get("text", ""),
                        "start": round(float(row.get("start", 0.0)), 3),
                        "end": round(float(row.get("end", 0.0)), 3),
                        "reference_sim": round(float(row.get("min_reference_sim", 0.0)), 3),
                        "target_vote": round(float(row.get("min_target_vote", 0.0)), 3),
                        "other_ratio": round(float(row.get("other_ratio", 0.0)), 3),
                    }
                )
                index += 1
    return items


def build(items: list[dict]) -> str:
    data = json.dumps(items, ensure_ascii=False).replace("</", "<///")
    accepted = sum(x["tier"] == "accepted" for x in items)
    expanded = sum(x["tier"] == "expanded" for x in items)
    review = sum(x["tier"] == "review" for x in items)
    html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ireina target voice candidates - FunASR v2</title>
<style>
:root{color-scheme:dark;--bg:#10151d;--panel:#1c2633;--line:#344456;--text:#edf4fb;--muted:#a9b7c6;--accent:#77c2ff;--good:#79e2aa;--review:#f0c674}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:2;padding:18px 24px;background:rgba(16,21,29,.97);border-bottom:1px solid var(--line)}h1{margin:0 0 6px;font-size:22px}.summary{color:var(--muted);margin-bottom:12px;line-height:1.6}
.toolbar{display:flex;flex-wrap:wrap;gap:8px}input,select,button{padding:9px 11px;border:1px solid var(--line);border-radius:8px;background:var(--panel);color:var(--text);font-size:14px}input{min-width:260px;flex:1}button{cursor:pointer}button:hover{border-color:var(--accent)}
main{max-width:1450px;margin:auto;padding:18px 24px 40px}#count{color:var(--muted)}.item{display:grid;grid-template-columns:42px 185px 82px 135px minmax(280px,1.5fr);gap:12px;align-items:center;margin:8px 0;padding:12px 14px;background:var(--panel);border:1px solid var(--line);border-radius:10px}.num{color:var(--muted);text-align:right}.source{font-weight:650;word-break:break-all}.meta{color:var(--muted);font-size:12px;line-height:1.55;word-break:break-all}.text{margin-bottom:5px;word-break:break-all}.duration{white-space:nowrap;color:var(--good)}.badge{display:inline-block;padding:2px 6px;border-radius:5px;font-size:11px;margin-left:6px;color:#111;background:var(--good)}.badge.expanded,.badge.review{background:var(--review)}audio{width:100%;height:38px}.empty{padding:30px;text-align:center;color:var(--muted)}
@media(max-width:900px){header,main{padding-left:12px;padding-right:12px}.item{grid-template-columns:32px 1fr;gap:6px 10px}.num{grid-row:span 3}.duration,.meta,.text,audio{grid-column:2}}
</style></head><body>
<header><h1>Ireina target voice candidates - FunASR v2</h1>
<div class="summary">Original BS-Roformer vocals; FunASR VAD; CAM++ multi-cluster consensus; word-level gate. Train candidates: __ACCEPTED__. Expanded review: __EXPANDED__. Regular review: __REVIEW__. Only train candidates are in the training list.</div>
<div class="toolbar"><input id="q" placeholder="Search source, filename, or ASR text"><select id="tier"><option value="">All tiers</option><option value="accepted">Train candidates</option><option value="expanded">Expanded review</option><option value="review">Regular review</option></select><select id="source"><option value="">All sources</option></select><button id="pause">Stop all</button><button id="continuous">Play filtered items</button></div></header>
<main><p id="count"></p><section id="list"></section></main>
<script>
const ITEMS=__DATA__;const q=document.getElementById('q'),tier=document.getElementById('tier'),source=document.getElementById('source'),list=document.getElementById('list'),count=document.getElementById('count');
for(const s of [...new Set(ITEMS.map(x=>x.source))].sort()){const o=document.createElement('option');o.value=s;o.textContent=s;source.appendChild(o)}
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const label=x=>x.tier==='accepted'?'train candidate':x.tier==='expanded'?'expanded review':'review';
function rows(){const term=q.value.trim().toLowerCase();return ITEMS.filter(x=>(!tier.value||x.tier===tier.value)&&(!source.value||x.source===source.value)&&(!term||[x.name,x.source,x.text].join(' ').toLowerCase().includes(term)))}
function render(){const r=rows();count.textContent=`Showing ${r.length} / ${ITEMS.length} clips`;list.innerHTML=r.length?r.map(x=>`<article class="item"><div class="num">#${String(x.index).padStart(3,'0')}</div><div><div class="source">${esc(x.source)}<span class="badge ${x.tier}">${label(x)}</span></div><div class="meta">${esc(x.name)}<br>${x.start.toFixed(2)}s - ${x.end.toFixed(2)}s</div></div><div class="duration">${x.duration.toFixed(2)} sec</div><div class="meta">ref ${x.reference_sim.toFixed(3)}; vote ${x.target_vote.toFixed(3)}<br>other ${x.other_ratio.toFixed(3)}</div><div><div class="text">${esc(x.text)}</div><audio controls preload="none" data-index="${x.index}" src="${encodeURI(x.src)}"></audio></div></article>`).join(''):'<div class="empty">No matching clips.</div>';document.querySelectorAll('audio').forEach(a=>a.addEventListener('ended',next))}
let auto=false;function next(e){if(!auto)return;const r=rows(),i=Number(e.target.dataset.index),n=r.find(x=>x.index>i);if(n){const a=document.querySelector(`audio[data-index="${n.index}"]`);if(a){a.play();a.scrollIntoView({behavior:'smooth',block:'center'})}}}
q.addEventListener('input',render);tier.addEventListener('change',render);source.addEventListener('change',render);document.getElementById('pause').onclick=()=>document.querySelectorAll('audio').forEach(a=>{a.pause();a.currentTime=0});document.getElementById('continuous').onclick=()=>{auto=true;const f=rows()[0];if(f){const a=document.querySelector(`audio[data-index="${f.index}"]`);if(a)a.play()}};render();
</script></body></html>"""
    return (
        html.replace("__DATA__", data)
        .replace("__ACCEPTED__", str(accepted))
        .replace("__EXPANDED__", str(expanded))
        .replace("__REVIEW__", str(review))
    )


def main() -> None:
    items = load_items()
    HTML.write_text(build(items), encoding="utf-8")
    print(f"created={HTML}")
    print(f"items={len(items)}")


if __name__ == "__main__":
    main()
