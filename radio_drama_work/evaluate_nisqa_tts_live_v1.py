# -*- coding: utf-8 -*-
"""Independent NISQA and NISQA-TTS gate for the overnight candidates."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

ROOT = Path(r"radio_drama_work/radio_drama_work")
WORK = ROOT / "vasr_overnight_search_v1"
STATE_PATH = WORK / "state.json"
ACOUSTIC_REPORT = WORK / "training_noise_quality_live.json"
REPORT_PATH = WORK / "nisqa_tts_quality_live.json"
NISQA_LIST = ROOT / "full_funasr_gold_overnight_nisqa_gated_score90_v1.list"
LOG_PATH = WORK / "nisqa_tts_gate.log"
SPEECHSCORE = Path(r"radio_drama_work/ClearerVoice-Studio-main/ClearerVoice-Studio-main/speechscore")
NISQA_WEIGHTS = SPEECHSCORE / "scores" / "nisqa" / "weights" / "nisqa.tar"
NISQA_TTS_WEIGHTS = SPEECHSCORE / "scores" / "nisqa" / "weights" / "nisqa_tts.tar"

import sys

sys.path.insert(0, str(SPEECHSCORE))
from scores.nisqa.cal_nisqa import cal_NISQA, load_nisqa_model
import scores.nisqa.NISQA_lib as NL


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "/n")


def read(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def tts_nisqa(model, audio: np.ndarray) -> float:
    df_val = pd.DataFrame(["dummy_file"], columns=["deg"])
    args = model.args
    dataset = NL.SpeechQualityDataset(
        df_val,
        audio,
        df_con=None,
        data_dir="./",
        filename_column="deg",
        mos_column="predict_only",
        seg_length=args["ms_seg_length"],
        max_length=args["ms_max_segments"],
        to_memory=None,
        to_memory_workers=None,
        seg_hop_length=args["ms_seg_hop_length"],
        transform=None,
        ms_n_fft=args["ms_n_fft"],
        ms_hop_length=args["ms_hop_length"],
        ms_win_length=args["ms_win_length"],
        ms_n_mels=args["ms_n_mels"],
        ms_fmax=args["ms_fmax"],
        ms_channel=None,
        double_ended=args["double_ended"],
        dim=args["dim"],
        filename_column_ref=None,
    )
    NL.predict_mos(model, dataset, 1, "cpu", num_workers=0)
    return float(dataset.df["mos_pred"].squeeze())


def relative_ok(after: dict, before: dict) -> tuple[bool, list[str]]:
    reasons = []
    for key in ("mos_pred", "noi_pred", "dis_pred", "col_pred", "loud_pred"):
        if key in after and key in before and float(after[key]) < float(before[key]) - 0.20:
            reasons.append(f"{key}_drop")
    if float(after.get("mos_pred", 0.0)) < 2.0:
        reasons.append("absolute_mos_low")
    if "noi_pred" in after and float(after.get("noi_pred", 0.0)) < 2.5:
        reasons.append("absolute_noise_low")
    return not reasons, reasons


def main_once() -> dict:
    payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    rows = payload.get("rows", {})
    input_map = {}
    for line in (ROOT / "full_funasr_gold_train_candidate_v9.list").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            input_map[len(input_map)] = Path(line.split("|", 1)[0].strip())
    acoustic = json.loads(ACOUSTIC_REPORT.read_text(encoding="utf-8")) if ACOUSTIC_REPORT.exists() else {"rows": []}
    acoustic_by_index = {str(row["index"]): row for row in acoustic.get("rows", [])}
    cache = {}
    if REPORT_PATH.exists():
        try:
            old = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
            cache = {str(row["index"]): row for row in old.get("rows", []) if row.get("path")}
        except Exception:
            cache = {}
    general = load_nisqa_model(str(NISQA_WEIGHTS), device="cpu")
    tts = load_nisqa_model(str(NISQA_TTS_WEIGHTS), device="cpu")
    evaluated = []
    gated = []
    for key in sorted(rows, key=lambda value: int(value)):
        row = rows[key]
        path = Path(row["best_path"])
        item = {"index": int(key), "path": str(path), "status": "missing"}
        if not path.exists():
            evaluated.append(item)
            continue
        cached = cache.get(key)
        if cached and cached.get("path") == str(path) and cached.get("general") and cached.get("tts"):
            after_general = cached["general"]
            after_tts = cached["tts"]
        else:
            audio, _ = read(path)
            raw_path = input_map.get(int(key))
            raw_audio, _ = read(raw_path) if raw_path and raw_path.exists() else (audio, 32000)
            after_general = cal_NISQA(general, audio)
            after_tts = {"mos_pred": tts_nisqa(tts, audio)}
            before_general = cal_NISQA(general, raw_audio)
            before_tts = {"mos_pred": tts_nisqa(tts, raw_audio)}
            item["baseline_general"] = before_general
            item["baseline_tts"] = before_tts
        if "baseline_general" not in item:
            raw_path = input_map.get(int(key))
            raw_audio, _ = read(raw_path) if raw_path and raw_path.exists() else read(path)
            item["baseline_general"] = cal_NISQA(general, raw_audio)
            item["baseline_tts"] = {"mos_pred": tts_nisqa(tts, raw_audio)}
        general_ok, general_reasons = relative_ok(after_general, item["baseline_general"])
        tts_ok, tts_reasons = relative_ok(after_tts, item["baseline_tts"])
        acoustic_item = acoustic_by_index.get(key, {})
        acoustic_ok = bool(acoustic_item.get("quality", {}).get("hard_gate_score90", False))
        ok = general_ok and tts_ok and acoustic_ok
        item.update({
            "status": "ok",
            "general": after_general,
            "tts": after_tts,
            "general_ok": general_ok,
            "tts_ok": tts_ok,
            "acoustic_gate_ok": acoustic_ok,
            "decision": "pass" if ok else "review_only",
            "reasons": general_reasons + tts_reasons + ([] if acoustic_ok else ["acoustic_gate_not_90"]),
        })
        if ok:
            gated.append(f"{path}|full_funasr_gold_overnight_nisqa_gated_score90_v1|JA|{row.get('text','')}")
        evaluated.append(item)
    report = {
        "updated_at": time.time(),
        "search_deadline": payload.get("deadline"),
        "policy": {
            "general_nisqa": "all available dimensions must not drop more than 0.20 from the original clean BS-Roformer stem",
            "nisqa_tts": "MOS must not drop more than 0.20 and must be >= 2.0",
            "combined": "also requires the independent acoustic score90 gate",
            "note": "NISQA is an automatic quality proxy; it is not a guarantee of a human MOS or final TTS benchmark score.",
        },
        "count": len(evaluated),
        "ok_count": sum(item["status"] == "ok" for item in evaluated),
        "score90_count": sum(item.get("decision") == "pass" for item in evaluated),
        "score90_sec": sum(float(acoustic_by_index.get(str(item["index"]), {}).get("duration_sec", 0.0)) for item in evaluated if item.get("decision") == "pass"),
        "list": str(NISQA_LIST),
        "rows": evaluated,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    NISQA_LIST.write_text("/n".join(gated) + ("/n" if gated else ""), encoding="utf-8")
    log(f"evaluated={len(evaluated)} nisqa_score90={report['score90_count']} sec={report['score90_sec']:.2f}")
    return report


def main() -> None:
    if os.environ.get("NISQA_ONCE") == "1":
        main_once()
        return
    interval = int(os.environ.get("NISQA_INTERVAL_SEC", "900"))
    while True:
        try:
            report = main_once()
        except Exception as exc:
            log(f"error={exc!r}")
            report = {}
        deadline = report.get("search_deadline")
        if deadline and time.time() >= float(deadline):
            log("deadline_reached")
            return
        time.sleep(interval)


if __name__ == "__main__":
    main()
