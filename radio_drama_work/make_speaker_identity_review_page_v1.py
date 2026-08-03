# -*- coding: utf-8 -*-
"""Generate a direct-open HTML page for speaker-identity calibration."""
from __future__ import annotations

import argparse
import html as html_lib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "anime_speaker_calibration_v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--title", default="伊蕾娜动画说话人校准")
    parser.add_argument("--download-prefix", default="anime_speaker_calibration_feedback_v1")
    parser.add_argument("--store-key", default="ireina_anime_speaker_calibration_v1")
    parser.add_argument("--feedback-version", default="anime_speaker_calibration_feedback_v1")
    args = parser.parse_args()

    root = args.root.resolve()
    audit = json.loads((root / "audit_manifest.json").read_text(encoding="utf-8"))
    samples = [
        {
            "id": str(row["blind_id"]),
            "file": str(row["audio_rel"]),
            "duration": round(float(row["output_duration"]), 3),
            "wave": row["wave"],
        }
        for row in audit["samples"]
    ]
    public_data = json.dumps(samples, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<///"
    )
    title = html_lib.escape(args.title)
    store_key = json.dumps(args.store_key, ensure_ascii=False)
    prefix = json.dumps(args.download_prefix, ensure_ascii=False)
    feedback_version = json.dumps(args.feedback_version, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root{{--bg:#080c13;--panel:#121927;--line:#2a354a;--text:#eef4ff;--muted:#9ca9c2;--target:#57d6a2;--other:#ff7f91;--mixed:#f3c86b;--uncertain:#8aaaf8}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 12% 0,#172949 0,transparent 34%),var(--bg);color:var(--text);font-family:"Microsoft YaHei",system-ui,sans-serif}}
.app{{max-width:1180px;margin:auto;padding:24px 18px 70px}}.panel{{background:linear-gradient(180deg,#151d2d,#0e1420);border:1px solid var(--line);border-radius:17px;box-shadow:0 16px 45px #0005}}
.hero{{padding:23px 25px;margin-bottom:16px}}h1{{margin:0 0 10px;font-size:clamp(27px,4vw,42px)}}p{{color:var(--muted);line-height:1.7;margin:7px 0}}.rule{{margin-top:13px;padding:12px 14px;border:1px solid #3b654f;border-radius:12px;background:#10271e;color:#d9ffeb}}
.sticky{{position:sticky;top:0;z-index:10;padding:9px 0;background:linear-gradient(var(--bg) 72%,transparent)}}.toolbar{{display:flex;gap:9px;align-items:center;padding:11px 12px;background:#101724ee;border:1px solid var(--line);border-radius:14px;backdrop-filter:blur(14px)}}button{{font:inherit;color:var(--text);border:1px solid var(--line);background:#172236;border-radius:10px;padding:9px 13px;cursor:pointer}}button.primary{{background:#1b5a46;border-color:#398866}}.summary{{margin-left:auto;color:var(--muted);font-size:13px}}
.list{{display:grid;gap:11px}}.clip{{display:grid;grid-template-columns:70px minmax(250px,1fr) minmax(440px,1.5fr);gap:14px;align-items:center;padding:15px}}.clip.done{{border-color:#385e50}}.id{{font-size:19px;font-weight:800}}.id small{{display:block;color:var(--muted);font-size:11px;font-weight:400;margin-top:5px}}audio{{width:100%;height:40px}}canvas{{width:100%;height:34px;display:block;background:#090e17;border-radius:7px;margin-bottom:5px}}
.choices{{display:grid;grid-template-columns:repeat(4,1fr);gap:7px}}.choice{{padding:10px 7px;text-align:center;background:#141c2a}}.choice.active[data-value=target]{{border-color:var(--target);color:#cffff0;background:#15392d}}.choice.active[data-value=non_target]{{border-color:var(--other);color:#ffdce1;background:#3d1d28}}.choice.active[data-value=mixed]{{border-color:var(--mixed);color:#fff0c6;background:#392f18}}.choice.active[data-value=uncertain]{{border-color:var(--uncertain);color:#dce7ff;background:#1b2944}}.note{{grid-column:1/-1;width:100%;height:39px;border:1px solid var(--line);border-radius:9px;background:#0e1521;color:var(--text);padding:0 10px;font:inherit;margin-top:7px}}.state{{grid-column:1/-1;text-align:right;color:var(--muted);font-size:11px;min-height:14px}}
.toast{{position:fixed;left:50%;bottom:22px;transform:translate(-50%,20px);opacity:0;background:#e7fff5;color:#10271d;padding:10px 16px;border-radius:99px;font-weight:700;transition:.2s;pointer-events:none}}.toast.show{{opacity:1;transform:translate(-50%,0)}}
@media(max-width:900px){{.clip{{grid-template-columns:55px 1fr}}.judge{{grid-column:1/-1}}}}@media(max-width:620px){{.app{{padding:12px 9px 55px}}.clip{{grid-template-columns:1fr}}.choices{{grid-template-columns:1fr 1fr}}.summary{{width:100%;margin:0}}.toolbar{{flex-wrap:wrap}}}}
</style>
</head><body><main class="app">
<section class="panel hero"><h1>{title}</h1><p>这是动画素材的第一轮“说话人身份校准”，每集 3 条，共 {len(samples)} 条。机器故意混入了部分竞争说话人，用来学习每一集里伊蕾娜与其他角色的区别。</p><div class="rule"><b>本轮只判断谁在说话：</b>整条只有伊蕾娜选“伊蕾娜”；整条都不是选“非伊蕾娜”；两个人无缝衔接、重叠或前后混入选“混合”；实在听不准选“不确定”。<b>这轮暂时忽略切头切尾和音质</b>，它们会在下一轮语义完整切分时单独处理。</div></section>
<div class="sticky"><div class="toolbar"><button class="primary" id="save">保存反馈 JSON</button><button id="next">下一条未判断</button><button id="import">导入反馈</button><input id="file" type="file" accept=".json,application/json" hidden><div class="summary" id="summary"></div></div></div>
<section class="list" id="list"></section></main><div class="toast" id="toast"></div>
<script>
const SAMPLES={public_data}; const STORE={store_key}; const PREFIX={prefix}; const VERSION={feedback_version};
const LABELS={{target:"伊蕾娜",non_target:"非伊蕾娜",mixed:"混合/无缝接话",uncertain:"不确定"}};
let rows={{}}; for(const s of SAMPLES)rows[s.id]={{blind_id:s.id,decision:"",note:"",play_count:0,max_listened_sec:0,duration_sec:s.duration}};
const $=id=>document.getElementById(id); const esc=x=>String(x).replace(/[&<>"']/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));
function payload(){{const classified=Object.values(rows).filter(x=>x.decision).length;return{{version:VERSION,updated_at:new Date().toISOString(),complete:classified===SAMPLES.length,summary:{{classified,total:SAMPLES.length,counts:Object.fromEntries(Object.keys(LABELS).map(k=>[k,Object.values(rows).filter(x=>x.decision===k).length]))}},samples:SAMPLES.map(s=>rows[s.id])}}}}
function persist(){{localStorage.setItem(STORE,JSON.stringify(payload()));updateSummary()}}function updateSummary(){{const p=payload();$("summary").textContent=`已判断 ${{p.summary.classified}} / ${{p.summary.total}} · 伊蕾娜 ${{p.summary.counts.target}} · 非目标 ${{p.summary.counts.non_target}} · 混合 ${{p.summary.counts.mixed}} · 不确定 ${{p.summary.counts.uncertain}}`}}
function toast(t){{const e=$("toast");e.textContent=t;e.classList.add("show");clearTimeout(window.__t);window.__t=setTimeout(()=>e.classList.remove("show"),1400)}}
function wave(canvas,s,ratio=0){{const d=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight,c=canvas.getContext("2d");canvas.width=w*d;canvas.height=h*d;c.setTransform(d,0,0,d,0,0);c.clearRect(0,0,w,h);const bw=w/s.wave.length;s.wave.forEach((v,i)=>{{c.fillStyle=i/s.wave.length<=ratio?"#65ddc0":"#3b4961";const bh=Math.max(2,v*(h-5));c.fillRect(i*bw,(h-bh)/2,Math.max(1,bw-1),bh)}})}}
function render(){{$("list").innerHTML=SAMPLES.map((s,i)=>{{const r=rows[s.id];return `<article class="panel clip ${{r.decision?"done":""}}" id="row-${{s.id}}"><div class="id">${{s.id}}<small>${{i+1}} / ${{SAMPLES.length}}</small></div><div><canvas></canvas><audio controls preload="metadata" src="${{s.file}}"></audio></div><div class="judge"><div class="choices">${{Object.entries(LABELS).map(([k,v])=>`<button class="choice ${{r.decision===k?"active":""}}" data-value="${{k}}">${{v}}</button>`).join("")}}</div><input class="note" value="${{esc(r.note)}}" placeholder="可选备注：第几秒混入、像哪个角色……"><div class="state">${{r.decision?"已记录":"尚未判断"}}</div></div></article>`}}).join("");bind();updateSummary()}}
function bind(){{document.querySelectorAll(".clip").forEach((el,i)=>{{const s=SAMPLES[i],r=rows[s.id],audio=el.querySelector("audio"),canvas=el.querySelector("canvas");wave(canvas,s,Math.min(1,r.max_listened_sec/s.duration));el.querySelectorAll(".choice").forEach(b=>b.onclick=()=>{{r.decision=b.dataset.value;el.classList.add("done");el.querySelectorAll(".choice").forEach(x=>x.classList.toggle("active",x===b));el.querySelector(".state").textContent="已记录："+LABELS[r.decision];persist()}});let timer;el.querySelector(".note").oninput=e=>{{r.note=e.target.value.slice(0,500);clearTimeout(timer);timer=setTimeout(persist,250)}};audio.onplay=()=>{{document.querySelectorAll("audio").forEach(x=>{{if(x!==audio)x.pause()}});r.play_count++;persist()}};audio.ontimeupdate=()=>{{r.max_listened_sec=Math.max(r.max_listened_sec,audio.currentTime);wave(canvas,s,audio.duration?audio.currentTime/audio.duration:0)}};audio.onended=persist;canvas.onclick=e=>{{if(audio.duration){{audio.currentTime=(e.clientX-canvas.getBoundingClientRect().left)/canvas.clientWidth*audio.duration;audio.play().catch(()=>{{}})}}}}}})}}
function save(){{const p=payload(),stamp=new Date().toISOString().replace(/[:.]/g,"-"),blob=new Blob([JSON.stringify(p,null,2)],{{type:"application/json"}}),u=URL.createObjectURL(blob),a=document.createElement("a");a.href=u;a.download=`${{PREFIX}}_${{stamp}}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(u),800);toast(p.complete?"完整反馈已保存到 Downloads":"未全部判断，草稿仍已保存")}}
$("save").onclick=save;$("next").onclick=()=>{{const s=SAMPLES.find(x=>!rows[x.id].decision);if(!s)return toast("全部已判断");$("row-"+s.id).scrollIntoView({{behavior:"smooth",block:"center"}});setTimeout(()=>$("row-"+s.id).querySelector("audio").play().catch(()=>{{}}),350)}};$("import").onclick=()=>$("file").click();$("file").onchange=async e=>{{const d=JSON.parse(await e.target.files[0].text());for(const r of d.samples||[])if(rows[r.blind_id])rows[r.blind_id]=Object.assign(rows[r.blind_id],r);persist();render();toast("反馈已导入")}};
try{{const old=JSON.parse(localStorage.getItem(STORE)||"null");if(old&&Array.isArray(old.samples))for(const r of old.samples)if(rows[r.blind_id])rows[r.blind_id]=Object.assign(rows[r.blind_id],r)}}catch(e){{}}render();
</script></body></html>"""
    output = root / "index.html"
    output.write_text(html, encoding="utf-8")
    print(
        json.dumps(
            {"output": str(output), "sample_count": len(samples)},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
