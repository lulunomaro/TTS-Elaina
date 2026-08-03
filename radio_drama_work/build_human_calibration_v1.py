# -*- coding: utf-8 -*-
"""Build the first blind human speaker-calibration package.

Outputs 60 source-stratified probes (five per full radio track) plus twelve
hidden controls.  The web page embeds no source name, ASR text, model score, or
expected label.  Private provenance stays in audit_manifest.json.
"""
from __future__ import annotations

import json
import math
import random
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "human_calibration_v1"
AUDIO_DIR = OUT / "audio"
MANIFEST_PATH = ROOT / "full_track_manifest_v1.json"
WORD_REPORT_DIR = ROOT / "word_target_all_v1" / "source_reports"
TARGET_LIST = ROOT / "confirmed_target_pure.txt"
UNSAFE_LIST = ROOT / "confirmed_non_target_or_mixed.txt"
OUTPUT_SR = 32000
RANDOM_SEED = 20260802
MIN_SEPARATION_SEC = 12.0
JAPANESE_RE = re.compile(r"[/u3040-/u30ff/u3400-/u9fff]")

EXPLICIT_HARD_NEGATIVES = [
    ROOT / "new_radio_quality_v1" / "accepted_segments" / "newradio_00001_243358894-1-192_vocals_0003417760_0003505760.wav",
    ROOT / "new_radio_quality_v1" / "accepted_segments" / "newradio_00003_243358894-1-192_vocals_0007518240_0007613280.wav",
    ROOT / "new_radio_quality_v1" / "accepted_segments" / "newradio_00004_243358894-1-192_vocals_0007968000_0008032000.wav",
]

LABELS = {
    "A": "纯伊蕾娜，可训练",
    "B": "是伊蕾娜，但边界或音质差",
    "C": "纯非目标人声",
    "D": "混合、重叠或无缝接话",
    "E": "无法确定",
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def listed_rows(path: Path) -> list[tuple[Path, str]]:
    result = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        left, reason = (raw.split("|", 1) + [""])[:2]
        audio = Path(left.strip())
        if audio.exists():
            result.append((audio.resolve(), reason.strip()))
    return result


def percentile_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / max(1, len(values) - 1)


def finite(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def infer_source_interval(path: Path, source_ids: list[str]) -> tuple[str, float, float] | None:
    name = path.name
    source_id = next((item for item in sorted(source_ids, key=len, reverse=True) if item in name), None)
    if source_id is None:
        return None
    tail = name.split(source_id, 1)[1]
    numbers = [int(value) for value in re.findall(r"(?<!/d)(/d{7,12})(?!/d)", tail)]
    if len(numbers) < 2:
        return None
    outer_start = numbers[0] / 32000.0
    if len(numbers) >= 4:
        start = outer_start + numbers[-2] / 32000.0
        end = outer_start + numbers[-1] / 32000.0
    else:
        start = outer_start
        end = numbers[1] / 32000.0
    return source_id, float(start), float(end)


def overlaps(start: float, end: float, masks: list[tuple[float, float]]) -> bool:
    return any(max(start, left) < min(end, right) for left, right in masks)


def build_interval(rows: list[dict], index: int, second_index: int | None, duration: float) -> tuple[float, float, list[int]]:
    left = index
    right = second_index if second_index is not None else index
    if right < left:
        left, right = right, left

    def current_span(a: int, b: int) -> float:
        return finite(rows[b], "end") - finite(rows[a], "start")

    while current_span(left, right) < 2.15:
        choices: list[tuple[float, str]] = []
        if left > 0 and finite(rows[left], "start") - finite(rows[left - 1], "end") <= 0.48:
            span = finite(rows[right], "end") - finite(rows[left - 1], "start")
            if span <= 3.05:
                choices.append((span, "left"))
        if right + 1 < len(rows) and finite(rows[right + 1], "start") - finite(rows[right], "end") <= 0.48:
            span = finite(rows[right + 1], "end") - finite(rows[left], "start")
            if span <= 3.05:
                choices.append((span, "right"))
        if not choices:
            break
        _, side = min(choices)
        if side == "left":
            left -= 1
        else:
            right += 1

    start = max(0.0, finite(rows[left], "start") - 0.10)
    end = min(duration, finite(rows[right], "end") + 0.10)
    if end - start < 1.55:
        missing = 1.55 - (end - start)
        start = max(0.0, start - missing / 2.0)
        end = min(duration, end + missing / 2.0)
        if end - start < 1.55:
            start = max(0.0, end - 1.55)
    if end - start > 3.20:
        center = (finite(rows[index], "start") + finite(rows[index], "end")) / 2.0
        start = max(0.0, center - 1.60)
        end = min(duration, start + 3.20)
    return float(start), float(end), list(range(left, right + 1))


def select_source_probes(source: dict, masks: list[tuple[float, float]]) -> list[dict]:
    report_path = WORD_REPORT_DIR / f"{source['id']}.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows: list[dict] = report["word_rows"]
    duration = float(report["duration_sec"])
    keys = ("direct_center", "direct_p25", "target_negative_margin", "frame_target", "cluster_vote")
    arrays = {key: np.asarray([finite(row, key) for row in rows], dtype=np.float64) for key in keys}
    ranks = {key: percentile_rank(values) for key, values in arrays.items()}
    score = (
        0.30 * ranks["direct_center"]
        + 0.22 * ranks["direct_p25"]
        + 0.18 * ranks["target_negative_margin"]
        + 0.18 * ranks["frame_target"]
        + 0.12 * ranks["cluster_vote"]
    )
    direct_profile = 0.60 * ranks["direct_center"] + 0.40 * ranks["direct_p25"]
    disagreement = np.abs(direct_profile - (0.65 * ranks["frame_target"] + 0.35 * ranks["cluster_vote"]))

    valid = []
    for idx, row in enumerate(rows):
        text = str(row.get("text", "")).strip()
        word_duration = finite(row, "end") - finite(row, "start")
        valid.append(
            bool(text)
            and bool(JAPANESE_RE.search(text))
            and finite(row, "probability", 0.0) >= 0.72
            and 0.08 <= word_duration <= 1.35
            and finite(row, "start") >= 0.8
            and finite(row, "end") <= duration - 0.8
        )
    valid_mask = np.asarray(valid, dtype=bool)

    profiles: list[tuple[str, list[tuple[int, int | None]]]] = []
    high_consensus = [int(idx) for idx in np.argsort(-score) if valid_mask[idx]]
    high_direct = [int(idx) for idx in np.argsort(-direct_profile) if valid_mask[idx]]
    disagreement_order = [
        int(idx)
        for idx in np.argsort(-(disagreement + 0.20 * np.maximum(direct_profile, ranks["frame_target"])))
        if valid_mask[idx] and max(direct_profile[idx], ranks["frame_target"][idx]) >= 0.70
    ]
    low_speech = [
        int(idx)
        for idx in np.argsort(score)
        if valid_mask[idx] and finite(rows[idx], "probability") >= 0.90
    ]
    boundary_pairs: list[tuple[float, int]] = []
    for idx in range(len(rows) - 1):
        if not (valid_mask[idx] and valid_mask[idx + 1]):
            continue
        gap = finite(rows[idx + 1], "start") - finite(rows[idx], "end")
        if gap > 0.25:
            continue
        veto_change = 0.45 if bool(rows[idx].get("hard_veto")) != bool(rows[idx + 1].get("hard_veto")) else 0.0
        transition = abs(float(score[idx]) - float(score[idx + 1])) + veto_change
        if transition >= 0.28:
            boundary_pairs.append((transition, idx))
    boundary_pairs.sort(reverse=True)

    profiles.append(("high_consensus", [(idx, None) for idx in high_consensus]))
    profiles.append(("high_direct", [(idx, None) for idx in high_direct]))
    profiles.append(("model_disagreement", [(idx, None) for idx in disagreement_order]))
    profiles.append(("speaker_transition", [(idx, idx + 1) for _, idx in boundary_pairs]))
    profiles.append(("low_target_clear_speech", [(idx, None) for idx in low_speech]))

    selected: list[dict] = []
    selected_centers: list[float] = []
    for profile, candidates in profiles:
        chosen = None
        for index, second in candidates:
            start, end, included = build_interval(rows, index, second, duration)
            center = (start + end) / 2.0
            if overlaps(start, end, masks):
                continue
            if any(abs(center - old_center) < MIN_SEPARATION_SEC for old_center in selected_centers):
                continue
            chosen = {
                "kind": "probe",
                "source_id": source["id"],
                "source": str(Path(source["vocals"]).resolve()),
                "start": start,
                "end": end,
                "duration": end - start,
                "selection_profile": profile,
                "anchor_index": index,
                "included_word_indices": included,
                "private_text": "".join(str(rows[item].get("text", "")) for item in included),
                "private_scores": {
                    "combined_rank": float(score[index]),
                    "direct_center": finite(rows[index], "direct_center"),
                    "direct_p25": finite(rows[index], "direct_p25"),
                    "target_negative_margin": finite(rows[index], "target_negative_margin"),
                    "frame_target": finite(rows[index], "frame_target"),
                    "cluster_vote": finite(rows[index], "cluster_vote"),
                    "hard_veto": bool(rows[index].get("hard_veto")),
                },
            }
            break
        if chosen is None:
            raise RuntimeError(f"Could not select {profile} for {source['id']}")
        selected.append(chosen)
        selected_centers.append((chosen["start"] + chosen["end"]) / 2.0)
    return selected


def read_interval(path: Path, start: float | None = None, end: float | None = None) -> tuple[np.ndarray, int]:
    with sf.SoundFile(str(path), "r") as handle:
        sr = int(handle.samplerate)
        if start is None:
            start_frame = 0
            frame_count = len(handle)
        else:
            start_frame = max(0, min(len(handle), int(round(start * sr))))
            end_frame = max(start_frame, min(len(handle), int(round(float(end) * sr))))
            frame_count = end_frame - start_frame
        handle.seek(start_frame)
        audio = handle.read(frame_count, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32, copy=False)
    return mono, sr


def resample(audio: np.ndarray, source_sr: int) -> np.ndarray:
    if source_sr == OUTPUT_SR:
        return audio.astype(np.float32, copy=False)
    tensor = torch.from_numpy(audio)
    return torchaudio.functional.resample(tensor, source_sr, OUTPUT_SR).numpy().astype(np.float32, copy=False)


def waveform(audio: np.ndarray, bins: int = 120) -> list[float]:
    if not len(audio):
        return [0.0] * bins
    edges = np.linspace(0, len(audio), bins + 1, dtype=int)
    values = []
    for idx in range(bins):
        chunk = audio[edges[idx] : edges[idx + 1]]
        values.append(float(np.sqrt(np.mean(np.square(chunk))) if len(chunk) else 0.0))
    scale = max(max(values), 1e-6)
    return [round(min(1.0, value / scale), 4) for value in values]


def build_controls(source_ids: list[str]) -> list[dict]:
    controls = []
    for path, reason in listed_rows(TARGET_LIST):
        controls.append(
            {
                "kind": "control",
                "control_type": "confirmed_target",
                "expected_label": "A",
                "source": str(path),
                "reason": reason,
                "source_interval": infer_source_interval(path, source_ids),
            }
        )
    seen = set()
    for path, reason in listed_rows(UNSAFE_LIST):
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        expected = "D" if "mixed" in reason.lower() or "混" in reason else "C"
        controls.append(
            {
                "kind": "control",
                "control_type": "confirmed_mixed" if expected == "D" else "confirmed_non_target",
                "expected_label": expected,
                "source": str(path),
                "reason": reason,
                "source_interval": infer_source_interval(path, source_ids),
            }
        )
    for path in EXPLICIT_HARD_NEGATIVES:
        resolved = path.resolve()
        key = str(resolved).lower()
        if not resolved.exists() or key in seen:
            continue
        seen.add(key)
        controls.append(
            {
                "kind": "control",
                "control_type": "explicit_hard_negative",
                "expected_label": "C",
                "source": str(resolved),
                "reason": "user-confirmed non-target",
                "source_interval": infer_source_interval(resolved, source_ids),
            }
        )
    if len(controls) != 12:
        raise RuntimeError(f"Expected 12 hidden controls, found {len(controls)}")
    return controls


def page_html(blind_samples: list[dict]) -> str:
    public = [
        {
            "id": item["blind_id"],
            "file": item["audio_rel"],
            "duration": round(float(item["output_duration"]), 3),
            "wave": item["wave"],
        }
        for item in blind_samples
    ]
    data = json.dumps(public, ensure_ascii=False, separators=(",", ":")).replace("</", "<///")
    labels = json.dumps(LABELS, ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html>
<html lang=/"zh-CN/">
<head>
  <meta charset=/"utf-8/">
  <meta name=/"viewport/" content=/"width=device-width,initial-scale=1/">
  <title>伊蕾娜声纹校准 · 第一轮盲听</title>
  <style>
    :root{{--bg:#090b12;--panel:#111522;--panel2:#171c2b;--line:#2a3145;--text:#eef2ff;--muted:#98a2bd;--cyan:#6ee7e7;--violet:#a78bfa;--green:#65d59a;--amber:#f6c76e;--red:#fb7185;--blue:#76a9fa}}
    *{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at 18% 0%,#17223e 0,transparent 35%),radial-gradient(circle at 85% 12%,#241b3c 0,transparent 32%),var(--bg);color:var(--text);font-family:/"Microsoft YaHei/",/"Noto Sans SC/",system-ui,sans-serif;min-height:100vh}}
    button,input{{font:inherit}} .app{{max-width:1240px;margin:auto;padding:28px 22px 56px}}
    .top{{display:flex;gap:22px;align-items:flex-end;justify-content:space-between;margin-bottom:20px}} .eyebrow{{font-size:12px;letter-spacing:.18em;color:var(--cyan);text-transform:uppercase}} h1{{font-size:clamp(26px,4vw,44px);margin:8px 0 5px;line-height:1.08}} .lead{{color:var(--muted);font-size:14px;margin:0}}
    .progressBox{{min-width:260px;text-align:right}} .progressText{{font-size:28px;font-weight:750}} .bar{{height:8px;background:#22283a;border-radius:99px;overflow:hidden;margin-top:8px}} .bar>i{{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--cyan),var(--violet));transition:width .25s}}
    .layout{{display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:18px}} .card{{background:linear-gradient(180deg,rgba(23,28,43,.96),rgba(13,17,28,.96));border:1px solid var(--line);border-radius:20px;box-shadow:0 18px 55px rgba(0,0,0,.3)}}
    .player{{padding:25px}} .sampleHead{{display:flex;justify-content:space-between;align-items:center;gap:16px}} .sampleId{{font-size:30px;font-weight:800;letter-spacing:.04em}} .duration{{color:var(--muted)}}
    .waveWrap{{margin:24px 0 14px;padding:12px;border:1px solid #273047;border-radius:14px;background:#090d17}} canvas{{display:block;width:100%;height:128px;cursor:pointer}} audio{{width:100%;height:44px}}
    .listenMeta{{display:flex;justify-content:space-between;color:var(--muted);font-size:12px;margin-top:8px}} .hint{{margin:22px 0 12px;color:#c5cce0;font-size:14px}}
    .labels{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}} .labelBtn{{border:1px solid var(--line);border-radius:13px;background:#151b2a;color:var(--text);padding:13px 14px;text-align:left;cursor:pointer;transition:.15s;min-height:58px}} .labelBtn:hover{{transform:translateY(-1px);border-color:#65718e;background:#1b2335}} .labelBtn.active{{outline:2px solid var(--cyan);background:#172d35}} .labelBtn b{{display:inline-grid;place-items:center;width:27px;height:27px;border-radius:8px;background:#252e43;margin-right:8px}} .labelBtn small{{display:block;color:var(--muted);margin:5px 0 0 36px}}
    .labelBtn[data-label=/"E/"]{{grid-column:1/-1}} .nav{{display:flex;align-items:center;gap:10px;margin-top:18px}} .nav button,.toolBtn{{border:1px solid var(--line);color:var(--text);background:#151b2a;padding:10px 14px;border-radius:11px;cursor:pointer}} .nav .next{{margin-left:auto;background:linear-gradient(135deg,#29435d,#3e315e)}}
    .side{{padding:18px;display:flex;flex-direction:column;max-height:740px}} .side h2{{font-size:15px;margin:0 0 12px}} .stats{{display:grid;grid-template-columns:repeat(5,1fr);gap:6px;margin-bottom:13px}} .stat{{border:1px solid var(--line);border-radius:9px;text-align:center;padding:7px 3px;color:var(--muted);font-size:11px}} .stat strong{{display:block;color:var(--text);font-size:16px}} .grid{{display:grid;grid-template-columns:repeat(6,1fr);gap:6px;overflow:auto;padding-right:4px}} .chip{{aspect-ratio:1;border:1px solid var(--line);border-radius:8px;background:#121827;color:#75809b;cursor:pointer;font-size:11px}} .chip.current{{border-color:var(--cyan);color:var(--cyan);box-shadow:0 0 0 1px var(--cyan)}} .chip.done{{color:white;background:#25304a}} .chip.A{{background:#1e4b42}} .chip.B{{background:#59471f}} .chip.C{{background:#542b39}} .chip.D{{background:#493461}} .chip.E{{background:#30435e}}
    .tools{{display:flex;flex-wrap:wrap;gap:8px;margin-top:16px;padding-top:15px;border-top:1px solid var(--line)}} .toolBtn.primary{{background:#225252;border-color:#3e8585}} .auto{{display:flex;align-items:center;gap:7px;color:var(--muted);font-size:12px;margin-top:13px}}
    .instructions{{margin-top:18px;padding:18px 20px}} .instructions h2{{font-size:16px;margin:0 0 8px}} .instructions p{{margin:5px 0;color:var(--muted);font-size:13px;line-height:1.6}} kbd{{background:#222a3d;border:1px solid #39435d;border-radius:5px;padding:2px 6px;color:#dce4fb}}
    .toast{{position:fixed;left:50%;bottom:24px;transform:translate(-50%,20px);opacity:0;background:#edf9f7;color:#10201f;padding:10px 16px;border-radius:99px;font-weight:650;transition:.2s;pointer-events:none}} .toast.show{{opacity:1;transform:translate(-50%,0)}}
    @media(max-width:860px){{.top{{align-items:flex-start;flex-direction:column}}.progressBox{{width:100%;text-align:left}}.layout{{grid-template-columns:1fr}}.side{{max-height:none}}.grid{{grid-template-columns:repeat(9,1fr)}}}}
    @media(max-width:520px){{.app{{padding:20px 12px 40px}}.player{{padding:17px}}.labels{{grid-template-columns:1fr}}.labelBtn[data-label=/"E/"]{{grid-column:auto}}.grid{{grid-template-columns:repeat(8,1fr)}}}}
  </style>
</head>
<body>
<main class=/"app/">
  <header class=/"top/"><div><div class=/"eyebrow/">Blind speaker calibration · round 1</div><h1>伊蕾娜声纹校准</h1><p class=/"lead/">共 72 条盲听样本。只按你实际听到的声音判断，不用猜机器意图。</p></div><div class=/"progressBox/"><div class=/"progressText/"><span id=/"doneCount/">0</span> / 72</div><div class=/"bar/"><i id=/"progressBar/"></i></div></div></header>
  <section class=/"layout/">
    <article class=/"card player/">
      <div class=/"sampleHead/"><div><div class=/"eyebrow/">当前盲样本</div><div class=/"sampleId/" id=/"sampleId/">S001</div></div><div class=/"duration/" id=/"duration/">0:00</div></div>
      <div class=/"waveWrap/"><canvas id=/"wave/" aria-label=/"音频波形；点击可跳转播放位置/"></canvas></div>
      <audio id=/"audio/" controls preload=/"metadata/"></audio>
      <div class=/"listenMeta/"><span id=/"listenRatio/">已听 0%</span><span>点击波形可定位</span></div>
      <div class=/"hint/">请选择最符合整条片段的标签。只要出现另一角色，即使只有一个字，也选 D。</div>
      <div class=/"labels/" id=/"labelButtons/"></div>
      <div class=/"nav/"><button id=/"prev/">← 上一条</button><button id=/"replay/">从头播放</button><button class=/"next/" id=/"next/">下一条 →</button></div>
    </article>
    <aside class=/"card side/"><h2>标注进度</h2><div class=/"stats/" id=/"stats/"></div><div class=/"grid/" id=/"sampleGrid/"></div><label class=/"auto/"><input type=/"checkbox/" id=/"autoPlay/"> 标注后自动播放下一条</label><div class=/"tools/"><button class=/"toolBtn primary/" id=/"export/">导出标签 JSON</button><button class=/"toolBtn/" id=/"importBtn/">导入标签</button><button class=/"toolBtn/" id=/"nextOpen/">跳到未标</button><button class=/"toolBtn/" id=/"reset/">清空</button><input id=/"importFile/" type=/"file/" accept=/"application/json,.json/" hidden></div></aside>
  </section>
  <section class=/"card instructions/"><h2>快捷键与交付</h2><p><kbd>空格</kbd> 播放/暂停，<kbd>1</kbd>–<kbd>5</kbd> 对应 A–E，<kbd>←</kbd>/<kbd>→</kbd> 切换，<kbd>R</kbd> 从头播放。页面会自动保存在本机浏览器。</p><p>全部标完后点击“导出标签 JSON”，把下载的文件交给我。不要打开 <code>audit_manifest.json</code>，其中包含盲测答案和来源信息。</p></section>
</main><div class=/"toast/" id=/"toast/"></div>
<script>
const SAMPLES={data}; const LABEL_NAMES={labels}; const ORDER=[/"A/",/"B/",/"C/",/"D/",/"E/"]; const STORE=/"ireina_human_calibration_v1/";
let saved={{labels:{{}},maxTime:{{}},playCount:{{}},current:0,updatedAt:null}}; try{{saved=Object.assign(saved,JSON.parse(localStorage.getItem(STORE)||/"{{}}/"))}}catch(_e){{}}
let index=Math.max(0,Math.min(SAMPLES.length-1,Number(saved.current)||0)); const audio=document.getElementById(/"audio/"); const canvas=document.getElementById(/"wave/"); const ctx=canvas.getContext(/"2d/");
const $=id=>document.getElementById(id); function persist(){{saved.current=index;saved.updatedAt=new Date().toISOString();localStorage.setItem(STORE,JSON.stringify(saved))}}
function formatTime(v){{const s=Math.round(v);return `${{Math.floor(s/60)}}:${{String(s%60).padStart(2,/"0/")}}`}}
function toast(message){{const el=$(/"toast/");el.textContent=message;el.classList.add(/"show/");clearTimeout(window.__toast);window.__toast=setTimeout(()=>el.classList.remove(/"show/"),1500)}}
function drawWave(){{const sample=SAMPLES[index],ratio=audio.duration?audio.currentTime/audio.duration:0,dpr=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);const gap=2,bw=Math.max(1,(w-(sample.wave.length-1)*gap)/sample.wave.length);sample.wave.forEach((v,i)=>{{const x=i*(bw+gap),bar=Math.max(3,v*(h-18)),played=i/sample.wave.length<=ratio;ctx.fillStyle=played?/"#6ee7e7/":/"#3c4763/";ctx.fillRect(x,(h-bar)/2,bw,bar)}})}}
function stats(){{const counts=Object.fromEntries(ORDER.map(x=>[x,0]));Object.values(saved.labels).forEach(x=>{{if(counts[x]!==undefined)counts[x]++}});$(/"stats/").innerHTML=ORDER.map(x=>`<div class=/"stat/"><strong>${{counts[x]}}</strong>${{x}}</div>`).join(/"/");const done=Object.keys(saved.labels).length;$(/"doneCount/").textContent=done;$(/"progressBar/").style.width=`${{done/SAMPLES.length*100}}%`;return counts}}
function renderGrid(){{$ (/"sampleGrid/").innerHTML=SAMPLES.map((s,i)=>{{const label=saved.labels[s.id]||/"/";return `<button class=/"chip ${{i===index?/"current/":/"/"}} ${{label?/"done /"+label:/"/"}}/" data-index=/"${{i}}/" title=/"${{s.id}}${{label?/" · /"+label:/"/"}}/">${{i+1}}</button>`}}).join(/"/");document.querySelectorAll(/".chip/").forEach(el=>el.onclick=()=>go(Number(el.dataset.index),false))}}
function renderButtons(){{$ (/"labelButtons/").innerHTML=ORDER.map((label,i)=>`<button class=/"labelBtn ${{saved.labels[SAMPLES[index].id]===label?/"active/":/"/"}}/" data-label=/"${{label}}/"><b>${{i+1}}</b>${{label}} · ${{LABEL_NAMES[label]}}<small>${{label===/"D/"?/"出现另一角色一个字也选此项/":label===/"B/"?/"身份正确，但不能直接训练/":/"/"}}</small></button>`).join(/"/");document.querySelectorAll(/".labelBtn/").forEach(el=>el.onclick=()=>setLabel(el.dataset.label))}}
function render(){{const sample=SAMPLES[index];$(/"sampleId/").textContent=sample.id;$(/"duration/").textContent=formatTime(sample.duration);audio.src=sample.file;audio.load();$(/"listenRatio/").textContent=`已听 ${{Math.round(Math.min(1,(saved.maxTime[sample.id]||0)/sample.duration)*100)}}%`;renderButtons();stats();renderGrid();drawWave();persist()}}
function go(next,play){{index=(next+SAMPLES.length)%SAMPLES.length;render();if(play)audio.play().catch(()=>{{}})}}
function nextUnlabelled(){{const start=index;for(let n=1;n<=SAMPLES.length;n++){{const i=(start+n)%SAMPLES.length;if(!saved.labels[SAMPLES[i].id]){{go(i,$(/"autoPlay/").checked);return}}}}toast(/"全部样本已标注/")}}
function setLabel(label){{const id=SAMPLES[index].id;saved.labels[id]=label;persist();renderButtons();stats();renderGrid();if(Object.keys(saved.labels).length===SAMPLES.length)toast(/"72 条已全部完成，可以导出/");else nextUnlabelled()}}
audio.addEventListener(/"play/",()=>{{const id=SAMPLES[index].id;saved.playCount[id]=(saved.playCount[id]||0)+1;persist()}});audio.addEventListener(/"timeupdate/",()=>{{const id=SAMPLES[index].id;saved.maxTime[id]=Math.max(saved.maxTime[id]||0,audio.currentTime);$(/"listenRatio/").textContent=`已听 ${{Math.round(Math.min(1,(saved.maxTime[id]||0)/SAMPLES[index].duration)*100)}}%`;drawWave();persist()}});audio.addEventListener(/"loadedmetadata/",drawWave);audio.addEventListener(/"ended/",drawWave);window.addEventListener(/"resize/",drawWave);
canvas.addEventListener(/"click/",event=>{{if(audio.duration){{const rect=canvas.getBoundingClientRect();audio.currentTime=Math.max(0,Math.min(audio.duration,(event.clientX-rect.left)/rect.width*audio.duration));audio.play().catch(()=>{{}})}}}});
$(/"prev/").onclick=()=>go(index-1,false);$(/"next/").onclick=()=>go(index+1,false);$(/"nextOpen/").onclick=nextUnlabelled;$(/"replay/").onclick=()=>{{audio.currentTime=0;audio.play().catch(()=>{{}})}};
$(/"export/").onclick=()=>{{const counts=stats(),rows=SAMPLES.map(s=>({{blind_id:s.id,label:saved.labels[s.id]||null,label_name:LABEL_NAMES[saved.labels[s.id]]||null,play_count:saved.playCount[s.id]||0,max_listened_sec:Number((saved.maxTime[s.id]||0).toFixed(3)),duration_sec:s.duration}}));const payload={{version:/"human_calibration_v1_labels/",exported_at:new Date().toISOString(),completed:rows.every(x=>x.label),label_counts:counts,samples:rows}};const blob=new Blob([JSON.stringify(payload,null,2)],{{type:/"application/json/"}}),url=URL.createObjectURL(blob),a=document.createElement(/"a/");a.href=url;a.download=`human_calibration_v1_labels_${{new Date().toISOString().slice(0,10)}}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);toast(/"标签 JSON 已导出/")}};
$(/"importBtn/").onclick=()=>$(/"importFile/").click();$(/"importFile/").onchange=async event=>{{const file=event.target.files[0];if(!file)return;try{{const payload=JSON.parse(await file.text());for(const row of payload.samples||[])if(SAMPLES.some(s=>s.id===row.blind_id)&&ORDER.includes(row.label))saved.labels[row.blind_id]=row.label;persist();render();toast(/"标签已导入/")}}catch(e){{alert(/"无法导入：/"+e.message)}}event.target.value=/"/"}};
$(/"reset/").onclick=()=>{{if(confirm(/"确定清空全部标签和试听进度？此操作不可撤销。/")&&confirm(/"再次确认：真的清空？/")){{saved={{labels:{{}},maxTime:{{}},playCount:{{}},current:0,updatedAt:null}};index=0;persist();render()}}}};
document.addEventListener(/"keydown/",event=>{{if(event.target.matches(/"input,button/"))return;if(event.code===/"Space/"){{event.preventDefault();audio.paused?audio.play().catch(()=>{{}}):audio.pause()}}else if(event.key>=/"1/"&&event.key<=/"5/")setLabel(ORDER[Number(event.key)-1]);else if(event.key===/"ArrowLeft/")go(index-1,false);else if(event.key===/"ArrowRight/")go(index+1,false);else if(event.key.toLowerCase()===/"r/"){{audio.currentTime=0;audio.play().catch(()=>{{}})}}}});render();
</script></body></html>"""


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    tracks = manifest["tracks"]
    source_ids = [track["id"] for track in tracks]
    for track in tracks:
        if not Path(track["vocals"]).exists():
            raise FileNotFoundError(track["vocals"])
        if not (WORD_REPORT_DIR / f"{track['id']}.json").exists():
            raise FileNotFoundError(WORD_REPORT_DIR / f"{track['id']}.json")

    controls = build_controls(source_ids)
    masks: dict[str, list[tuple[float, float]]] = {source_id: [] for source_id in source_ids}
    for item in manifest.get("hard_exclusions", []):
        masks[item["source_id"]].append((float(item["start"]) - 0.25, float(item["end"]) + 0.25))
    for control in controls:
        interval = control.get("source_interval")
        if interval:
            source_id, start, end = interval
            masks[source_id].append((start - 0.30, end + 0.30))

    probes = []
    for track in tracks:
        selected = select_source_probes(track, masks[track["id"]])
        probes.extend(selected)
        print(f"[select] {track['id']} -> {len(selected)}", flush=True)
    if len(probes) != 60:
        raise RuntimeError(f"Expected 60 probes, got {len(probes)}")

    items = probes + controls
    random.Random(RANDOM_SEED).shuffle(items)
    if OUT.exists():
        # Version v1 is deterministic.  Refuse to overwrite completed labels,
        # but refresh generated media/manifests when no labels file is present.
        protected = OUT / "human_calibration_v1_labels.json"
        if protected.exists():
            raise RuntimeError(f"Refusing to overwrite existing user labels: {protected}")
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    blind_samples = []
    for idx, item in enumerate(items, 1):
        blind_id = f"S{idx:03d}"
        output = AUDIO_DIR / f"{blind_id}.wav"
        source = Path(item["source"])
        if item["kind"] == "probe":
            audio, sr = read_interval(source, float(item["start"]), float(item["end"]))
        else:
            audio, sr = read_interval(source)
        audio32 = resample(audio, sr)
        sf.write(str(output), audio32, OUTPUT_SR, subtype="PCM_16")
        output_duration = len(audio32) / OUTPUT_SR
        item["blind_id"] = blind_id
        item["audio"] = str(output.resolve())
        item["audio_rel"] = f"audio/{output.name}"
        item["output_duration"] = output_duration
        item["wave"] = waveform(audio32)
        blind_samples.append(item)
        if idx % 12 == 0 or idx == len(items):
            print(f"[audio] {idx}/{len(items)}", flush=True)

    public_manifest = {
        "version": "human_calibration_v1_blind",
        "sample_count": len(blind_samples),
        "labels": LABELS,
        "samples": [
            {
                "blind_id": item["blind_id"],
                "audio": item["audio_rel"],
                "duration_sec": round(float(item["output_duration"]), 3),
            }
            for item in blind_samples
        ],
    }
    audit_manifest = {
        "version": "human_calibration_v1_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "policy": {
            "blind": "HTML and blind_manifest omit source, ASR, model score, selection profile, and expected control label",
            "probe_count": 60,
            "hidden_control_count": 12,
            "mixed_controls": "validation-only; never pure-negative embeddings",
        },
        "samples": blind_samples,
    }
    write_json(OUT / "blind_manifest.json", public_manifest)
    write_json(OUT / "audit_manifest.json", audit_manifest)
    (OUT / "index.html").write_text(page_html(blind_samples), encoding="utf-8")
    (OUT / "README_试听说明.txt").write_text(
        "双击 index.html 开始试听。/n"
        "按 1-5 标注，空格播放/暂停，左右方向键切换。/n"
        "完成后点击“导出标签 JSON”，把下载的 JSON 文件交给 Codex。/n"
        "请不要查看 audit_manifest.json；它包含盲样本来源与隐藏控制答案。/n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUT), "samples": len(blind_samples), "probes": len(probes), "controls": len(controls), "total_sec": round(sum(float(item["output_duration"]) for item in blind_samples), 3)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
