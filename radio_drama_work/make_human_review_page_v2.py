# -*- coding: utf-8 -*-
"""Generate the simple per-clip checkbox review page requested by the user."""
from __future__ import annotations

import argparse
import html as html_lib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "human_calibration_v1"
AUDIT = ROOT / "audit_manifest.json"
OUTPUT = ROOT / "index.html"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--title", default="伊蕾娜素材逐条试听反馈")
    parser.add_argument("--round-label", default="Human review · round 1")
    parser.add_argument("--store-key", default="ireina_human_feedback_v2")
    parser.add_argument("--feedback-version", default="human_calibration_feedback_v2")
    parser.add_argument("--download-prefix", default="human_calibration_feedback_v2")
    parser.add_argument("--extra-note", default="")
    args = parser.parse_args()
    root = args.root.resolve()
    audit_path = root / "audit_manifest.json"
    output = root / "index.html"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    samples = [
        {
            "id": row["blind_id"],
            "file": row["audio_rel"],
            "duration": round(float(row["output_duration"]), 3),
            "wave": row["wave"],
        }
        for row in audit["samples"]
    ]
    sample_count = len(samples)
    public_data = json.dumps(samples, ensure_ascii=False, separators=(",", ":")).replace("</", "<///")
    js_store = json.dumps(args.store_key, ensure_ascii=False)
    js_feedback_version = json.dumps(args.feedback_version, ensure_ascii=False)
    js_download_prefix = json.dumps(args.download_prefix, ensure_ascii=False)
    extra_note_html = (
        f"<br><b>本轮补充：</b>{html_lib.escape(args.extra_note)}"
        if args.extra_note.strip()
        else ""
    )
    html = f"""<!doctype html>
<html lang=/"zh-CN/">
<head>
  <meta charset=/"utf-8/">
  <meta name=/"viewport/" content=/"width=device-width,initial-scale=1/">
  <title>{args.title}</title>
  <style>
    :root{{--bg:#080b12;--panel:#121725;--panel2:#171d2c;--line:#2a3348;--text:#edf2ff;--muted:#9da8c1;--cyan:#68e0dc;--red:#fb7185;--green:#67d99e;--amber:#f5c96d;--blue:#78aaf8}}
    *{{box-sizing:border-box}} html{{scroll-behavior:smooth}} body{{margin:0;min-height:100vh;background:radial-gradient(circle at 8% 0%,#172444 0,transparent 31%),radial-gradient(circle at 92% 4%,#2a1b3e 0,transparent 27%),var(--bg);color:var(--text);font-family:/"Microsoft YaHei/",/"Noto Sans SC/",system-ui,sans-serif}}
    button,input,select{{font:inherit}} .app{{max-width:1240px;margin:auto;padding:24px 20px 72px}}
    .hero{{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:18px;align-items:stretch;margin-bottom:18px}} .panel{{background:linear-gradient(180deg,rgba(23,29,44,.97),rgba(13,17,28,.97));border:1px solid var(--line);border-radius:20px;box-shadow:0 18px 60px rgba(0,0,0,.3)}}
    .intro{{padding:24px 26px}} .eyebrow{{color:var(--cyan);font-size:11px;letter-spacing:.18em;text-transform:uppercase}} h1{{font-size:clamp(27px,4vw,43px);line-height:1.1;margin:8px 0 12px}} .intro p{{color:var(--muted);font-size:14px;line-height:1.7;margin:7px 0}} .definition{{margin-top:16px;padding:13px 15px;border:1px solid #31544f;background:#102825;border-radius:13px;color:#d8fff7;font-size:13px;line-height:1.65}} .definition b{{color:var(--green)}}
    .summary{{padding:20px;display:flex;flex-direction:column;justify-content:center}} .numbers{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}} .number{{border:1px solid var(--line);background:#111725;border-radius:12px;padding:11px;text-align:center;color:var(--muted);font-size:11px}} .number strong{{display:block;font-size:24px;color:var(--text);margin-bottom:2px}} .progress{{height:8px;background:#252c3e;border-radius:99px;overflow:hidden;margin:14px 0 9px}} .progress i{{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--cyan),#a78bfa);transition:.2s}} .saveState{{font-size:12px;color:var(--muted);min-height:20px}} .saveState.ok{{color:var(--green)}} .saveState.bad{{color:var(--red)}}
    .sticky{{position:sticky;top:0;z-index:20;padding:10px 0;background:linear-gradient(180deg,var(--bg) 70%,transparent)}} .toolbar{{display:flex;align-items:center;gap:9px;padding:10px 12px;border:1px solid var(--line);border-radius:15px;background:rgba(15,20,32,.96);backdrop-filter:blur(16px)}} .tool{{border:1px solid var(--line);color:var(--text);background:#172033;padding:9px 13px;border-radius:10px;cursor:pointer}} .tool:hover{{border-color:#62718f}} .tool.primary{{background:#1c5552;border-color:#347d78}} .tool.danger{{color:#ffc6cf}} .filters{{display:flex;gap:6px;margin-left:auto}} .filter.active{{border-color:var(--cyan);color:var(--cyan)}}
    .list{{display:grid;gap:12px}} .clip{{display:grid;grid-template-columns:78px minmax(260px,1fr) minmax(400px,1.15fr);gap:15px;align-items:center;padding:15px 16px;transition:.16s}} .clip.reviewed{{border-color:#315b4a}} .clip.error{{border-color:#70364a;background:linear-gradient(180deg,rgba(67,25,40,.88),rgba(24,15,24,.96))}} .clip.hidden{{display:none}}
    .id{{font-size:19px;font-weight:800;letter-spacing:.05em}} .id small{{display:block;color:var(--muted);font-size:11px;font-weight:400;margin-top:5px}} .audioCol audio{{width:100%;height:40px}} .wave{{height:36px;width:100%;display:block;margin-bottom:5px;border-radius:7px;background:#0a0e18}}
    .feedback{{display:grid;grid-template-columns:auto auto 150px minmax(130px,1fr);gap:9px;align-items:center}} .check{{display:flex;align-items:center;gap:7px;padding:10px 11px;border:1px solid var(--line);border-radius:10px;background:#151c2b;cursor:pointer;white-space:nowrap;font-size:13px}} .check input{{width:18px;height:18px;accent-color:var(--green)}} .check.wrong input{{accent-color:var(--red)}} .check.wrong.on{{border-color:var(--red);background:#3c1e2a;color:#ffd8df}} select,.note{{height:41px;border:1px solid var(--line);border-radius:10px;background:#101623;color:var(--text);padding:0 10px;min-width:0}} select:disabled{{opacity:.42}} .note::placeholder{{color:#667189}} .note:focus,select:focus{{outline:2px solid #476a94;border-color:transparent}}
    .rowState{{grid-column:1/-1;color:var(--muted);font-size:11px;text-align:right;min-height:15px}} .rowState.saved{{color:var(--green)}} .empty{{display:none;text-align:center;padding:44px;color:var(--muted)}} .empty.show{{display:block}}
    .toast{{position:fixed;left:50%;bottom:22px;transform:translate(-50%,16px);opacity:0;padding:10px 16px;background:#eefcf8;color:#10221e;border-radius:99px;font-weight:700;transition:.2s;pointer-events:none;z-index:40}} .toast.show{{opacity:1;transform:translate(-50%,0)}}
    code{{color:#c8d3f2}} @media(max-width:980px){{.hero{{grid-template-columns:1fr}}.clip{{grid-template-columns:65px 1fr}}.feedback{{grid-column:1/-1}}}} @media(max-width:650px){{.app{{padding:14px 10px 50px}}.intro{{padding:19px}}.toolbar{{flex-wrap:wrap}}.filters{{width:100%;margin-left:0}}.clip{{grid-template-columns:1fr;padding:14px}}.id small{{display:inline;margin-left:8px}}.feedback{{grid-template-columns:1fr 1fr}}select,.note{{grid-column:1/-1}}.numbers strong{{font-size:20px}}}}
  </style>
</head>
<body><main class=/"app/">
  <section class=/"hero/">
    <article class=/"panel intro/"><div class=/"eyebrow/">{args.round_label}</div><h1>逐条试听，勾出错误</h1><p>你需要反馈的是：这条音频能否作为“纯伊蕾娜、可训练”的素材。音频播完会自动勾选“已听”。如果中途已经听出问题，可以直接勾“有错误”。</p><div class=/"definition/"><b>不勾“有错误”代表：</b>整条只有伊蕾娜，没有其他角色哪怕一个字；开头结尾完整；声音清楚自然，没有明显杂音或分离伪影。<br><b>应该勾“有错误”：</b>不是伊蕾娜、混入/无缝接上其他角色、重叠说话、切头切尾、音质不能训练，或你无法确定。<br><b>声音本人但边界有问题：</b>请明确选择“开头被切”或“结尾被切”，不要写成非伊蕾娜；这类会进入自动边界救援而不是直接丢弃。{extra_note_html}</div></article>
    <aside class=/"panel summary/"><div class=/"numbers/"><div class=/"number/"><strong id=/"reviewedCount/">0</strong>已听</div><div class=/"number/"><strong id=/"errorCount/">0</strong>有错误</div><div class=/"number/"><strong>{sample_count}</strong>总数</div></div><div class=/"progress/"><i id=/"progressBar/"></i></div><div class=/"saveState ok/" id=/"saveState/">反馈会自动保存在浏览器；听完后点“保存反馈文件”</div></aside>
  </section>
  <div class=/"sticky/"><div class=/"toolbar/"><button class=/"tool primary/" id=/"saveNow/">保存反馈文件</button><button class=/"tool/" id=/"nextOpen/">跳到下一条未听</button><button class=/"tool/" id=/"importBtn/">导入已有反馈</button><input id=/"importFile/" type=/"file/" accept=/"application/json,.json/" hidden><div class=/"filters/"><button class=/"tool filter active/" data-filter=/"all/">全部</button><button class=/"tool filter/" data-filter=/"open/">未听</button><button class=/"tool filter/" data-filter=/"error/">只看错误</button></div></div></div>
  <section class=/"list/" id=/"list/"></section><div class=/"panel empty/" id=/"empty/">当前筛选下没有样本。</div>
</main><div class=/"toast/" id=/"toast/"></div>
<script>
const SAMPLES={public_data};
const ERROR_TYPES=[/"/",/"非伊蕾娜/",/"混入/无缝接话/",/"重叠说话/",/"开头被切/",/"结尾被切/",/"开头和结尾都被切/",/"切头/切尾/",/"杂音/音质问题/",/"环境音/道具声/",/"文本不对应/",/"无法确定/",/"其他/"];
const STORE={js_store}; const FEEDBACK_VERSION={js_feedback_version}; const DOWNLOAD_PREFIX={js_download_prefix}; let feedback={{}}; let filter=/"all/"; let dirty=false;
const $=id=>document.getElementById(id); const defaultRow=s=>({{blind_id:s.id,reviewed:false,is_error:false,error_type:/"/",note:/"/",play_count:0,max_listened_sec:0,duration_sec:s.duration}});
for(const sample of SAMPLES)feedback[sample.id]=defaultRow(sample);
function escapeHtml(value){{return String(value).replace(/[&<>/"']/g,ch=>({{/"&/":/"&amp;/",/"</":/"&lt;/",/">/":/"&gt;/",'/"':/"&quot;/",/"'/":/"&#39;/"}}[ch]))}}
function toast(message){{const el=$(/"toast/");el.textContent=message;el.classList.add(/"show/");clearTimeout(window.__toast);window.__toast=setTimeout(()=>el.classList.remove(/"show/"),1500)}}
function summary(){{const rows=Object.values(feedback),reviewed=rows.filter(x=>x.reviewed).length,errors=rows.filter(x=>x.is_error).length;return{{reviewed,errors,complete:reviewed===SAMPLES.length}}}}
function payload(){{const sum=summary();return{{version:FEEDBACK_VERSION,updated_at:new Date().toISOString(),complete:sum.complete,summary:sum,samples:SAMPLES.map(s=>feedback[s.id])}}}}
function persistLocal(){{localStorage.setItem(STORE,JSON.stringify(payload()))}}
function markDirty(){{dirty=true;persistLocal();updateSummary();const state=$(/"saveState/");const sum=summary();state.className=/"saveState ok/";state.textContent=`浏览器已自动保存草稿 · ${{sum.reviewed}}/${{SAMPLES.length}} 已听 · ${{sum.errors}} 条错误`;document.querySelectorAll(/".rowState/").forEach(el=>{{if(el.dataset.dirty===/"1/"){{el.textContent=/"已保存到浏览器/";el.className=/"rowState saved/";el.dataset.dirty=/"0/"}}}})}}
function updateSummary(){{const s=summary();$(/"reviewedCount/").textContent=s.reviewed;$(/"errorCount/").textContent=s.errors;$(/"progressBar/").style.width=`${{s.reviewed/SAMPLES.length*100}}%`;applyFilter()}}
function drawWave(canvas,sample,ratio=0){{const ctx=canvas.getContext(/"2d/"),dpr=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);const gap=1,bw=Math.max(1,(w-gap*(sample.wave.length-1))/sample.wave.length);sample.wave.forEach((v,i)=>{{const bar=Math.max(2,v*(h-6));ctx.fillStyle=i/sample.wave.length<=ratio?/"#68e0dc/":/"#3b4660/";ctx.fillRect(i*(bw+gap),(h-bar)/2,bw,bar)}})}}
function render(){{$(/"list/").innerHTML=SAMPLES.map((sample,index)=>{{const row=feedback[sample.id];return `<article class=/"panel clip ${{row.reviewed?/"reviewed/":/"/"}} ${{row.is_error?/"error/":/"/"}}/" id=/"row-${{sample.id}}/" data-id=/"${{sample.id}}/"><div class=/"id/">${{sample.id}}<small>${{index+1}} / ${{SAMPLES.length}}</small></div><div class=/"audioCol/"><canvas class=/"wave/" data-wave=/"${{sample.id}}/"></canvas><audio controls preload=/"metadata/" src=/"${{sample.file}}/"></audio></div><div class=/"feedback/"><label class=/"check/"><input class=/"reviewedCheck/" type=/"checkbox/" ${{row.reviewed?/"checked/":/"/"}}> 已听完成</label><label class=/"check wrong ${{row.is_error?/"on/":/"/"}}/"><input class=/"errorCheck/" type=/"checkbox/" ${{row.is_error?/"checked/":/"/"}}> 有错误</label><select class=/"errorType/" ${{row.is_error?/"/":/"disabled/"}} aria-label=/"错误类型/">${{ERROR_TYPES.map(x=>`<option value=/"${{x}}/" ${{row.error_type===x?/"selected/":/"/"}}>${{x||/"选择错误类型/"}}</option>`).join(/"/")}}</select><input class=/"note/" value=/"${{escapeHtml(row.note)}}/" placeholder=/"可输入：第 1.2 秒开始是别人…/"><div class=/"rowState/" data-dirty=/"0/">${{row.reviewed?/"已记录/":/"/"}}</div></div></article>`}}).join(/"/");bindRows();updateSummary()}}
function setRowClass(id){{const el=$(/"row-/"+id),row=feedback[id];el.classList.toggle(/"reviewed/",row.reviewed);el.classList.toggle(/"error/",row.is_error);const wrong=el.querySelector(/".check.wrong/");wrong.classList.toggle(/"on/",row.is_error);el.querySelector(/".errorType/").disabled=!row.is_error}}
function rowChanged(id){{const state=$(/"row-/"+id).querySelector(/".rowState/");state.textContent=/"等待自动保存…/";state.className=/"rowState/";state.dataset.dirty=/"1/";setRowClass(id);markDirty()}}
function bindRows(){{document.querySelectorAll(/".clip/").forEach((clip,index)=>{{const id=clip.dataset.id,row=feedback[id],sample=SAMPLES[index],audio=clip.querySelector(/"audio/"),canvas=clip.querySelector(/"canvas/");drawWave(canvas,sample,Math.min(1,row.max_listened_sec/sample.duration));clip.querySelector(/".reviewedCheck/").onchange=event=>{{row.reviewed=event.target.checked;rowChanged(id)}};clip.querySelector(/".errorCheck/").onchange=event=>{{row.is_error=event.target.checked;if(row.is_error)row.reviewed=true;else row.error_type=/"/";clip.querySelector(/".reviewedCheck/").checked=row.reviewed;clip.querySelector(/".errorType/").value=row.error_type;rowChanged(id)}};clip.querySelector(/".errorType/").onchange=event=>{{row.error_type=event.target.value;row.is_error=Boolean(row.error_type)||row.is_error;row.reviewed=true;clip.querySelector(/".errorCheck/").checked=row.is_error;clip.querySelector(/".reviewedCheck/").checked=true;rowChanged(id)}};let noteTimer=null;clip.querySelector(/".note/").oninput=event=>{{row.note=event.target.value.slice(0,500);if(row.note.trim()){{row.is_error=true;row.reviewed=true;clip.querySelector(/".errorCheck/").checked=true;clip.querySelector(/".reviewedCheck/").checked=true}}setRowClass(id);clearTimeout(noteTimer);noteTimer=setTimeout(()=>rowChanged(id),350)}};audio.addEventListener(/"play/",()=>{{document.querySelectorAll(/"audio/").forEach(other=>{{if(other!==audio)other.pause()}});row.play_count++;markDirty()}});audio.addEventListener(/"timeupdate/",()=>{{row.max_listened_sec=Math.max(row.max_listened_sec,audio.currentTime);drawWave(canvas,sample,audio.duration?audio.currentTime/audio.duration:0);if(audio.duration&&audio.currentTime/audio.duration>=.88&&!row.reviewed){{row.reviewed=true;clip.querySelector(/".reviewedCheck/").checked=true;rowChanged(id)}}}});audio.addEventListener(/"ended/",()=>{{if(!row.reviewed){{row.reviewed=true;clip.querySelector(/".reviewedCheck/").checked=true;rowChanged(id)}}}});canvas.onclick=event=>{{if(audio.duration){{const rect=canvas.getBoundingClientRect();audio.currentTime=(event.clientX-rect.left)/rect.width*audio.duration;audio.play().catch(()=>{{}})}}}}}})}}
function applyFilter(){{let visible=0;document.querySelectorAll(/".clip/").forEach(el=>{{const row=feedback[el.dataset.id],show=filter===/"all/"||(filter===/"open/"&&!row.reviewed)||(filter===/"error/"&&row.is_error);el.classList.toggle(/"hidden/",!show);if(show)visible++}});$(/"empty/").classList.toggle(/"show/",visible===0)}}
function jumpOpen(){{const sample=SAMPLES.find(x=>!feedback[x.id].reviewed);if(!sample){{toast(`${{SAMPLES.length}} 条都已听完`);return}}filter=/"all/";document.querySelectorAll(/".filter/").forEach(x=>x.classList.toggle(/"active/",x.dataset.filter===/"all/"));applyFilter();$(/"row-/"+sample.id).scrollIntoView({{behavior:/"smooth/",block:/"center/"}});setTimeout(()=>$(/"row-/"+sample.id).querySelector(/"audio/").play().catch(()=>{{}}),450)}}
function saveFeedbackFile(){{const body=payload(),stamp=new Date().toISOString().replace(/[:.]/g,/"-/"),blob=new Blob([JSON.stringify(body,null,2)],{{type:/"application/json/"}}),url=URL.createObjectURL(blob),a=document.createElement(/"a/");a.href=url;a.download=`${{DOWNLOAD_PREFIX}}_${{stamp}}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);dirty=false;const state=$(/"saveState/");state.className=/"saveState ok/";state.textContent=`反馈文件已下载 · ${{body.summary.reviewed}}/${{SAMPLES.length}} 已听 · ${{body.summary.errors}} 条错误`;toast(/"反馈文件已保存到 Downloads/")}}
function importFeedback(data){{if(!data||!Array.isArray(data.samples))throw new Error(/"反馈文件格式不正确/");for(const row of data.samples)if(feedback[row.blind_id])feedback[row.blind_id]=Object.assign(feedback[row.blind_id],row);persistLocal();dirty=true;render();toast(/"反馈已导入/")}}
function loadFeedback(){{let loaded=null;try{{const local=JSON.parse(localStorage.getItem(STORE)||/"null/");if(local&&Array.isArray(local.samples))loaded=local}}catch(_e){{}}if(loaded)for(const row of loaded.samples)if(feedback[row.blind_id])feedback[row.blind_id]=Object.assign(feedback[row.blind_id],row);render();const sum=summary();$(/"saveState/").textContent=`浏览器已自动保存草稿 · ${{sum.reviewed}}/${{SAMPLES.length}} 已听 · ${{sum.errors}} 条错误`}}
$(/"saveNow/").onclick=saveFeedbackFile;$(/"nextOpen/").onclick=jumpOpen;$(/"importBtn/").onclick=()=>$(/"importFile/").click();$(/"importFile/").onchange=async event=>{{const file=event.target.files[0];if(!file)return;try{{importFeedback(JSON.parse(await file.text()))}}catch(error){{alert(/"无法导入：/"+error.message)}}event.target.value=/"/"}};document.querySelectorAll(/".filter/").forEach(btn=>btn.onclick=()=>{{filter=btn.dataset.filter;document.querySelectorAll(/".filter/").forEach(x=>x.classList.toggle(/"active/",x===btn));applyFilter()}});window.addEventListener(/"resize/",()=>document.querySelectorAll(/"canvas.wave/").forEach((canvas,index)=>drawWave(canvas,SAMPLES[index],0)));document.addEventListener(/"keydown/",event=>{{if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()===/"s/"){{event.preventDefault();saveFeedbackFile()}}}});loadFeedback();
</script></body></html>"""
    output.write_text(html, encoding="utf-8")
    print(json.dumps({"output": str(output), "sample_count": len(samples)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
