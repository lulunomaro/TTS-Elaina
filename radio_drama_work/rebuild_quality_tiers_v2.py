# -*- coding: utf-8 -*-
"""Build transparent quality tiers from the completed overnight reports.

The original strict NISQA gate used an absolute general-MOS floor.  That floor
is poorly calibrated for BGM-separated stems, so this script keeps the strict
gold list and creates a separate expanded list using relative quality checks.
It never changes the original 90+ list or starts training.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(r"radio_drama_work/radio_drama_work")
WORK = ROOT / "vasr_overnight_search_v1"
STATE = json.loads((WORK / "state.json").read_text(encoding="utf-8"))
NISQA = json.loads((WORK / "nisqa_tts_quality_live.json").read_text(encoding="utf-8"))
ACOUSTIC = json.loads((WORK / "training_noise_quality_live.json").read_text(encoding="utf-8"))

OUT_REPORT = WORK / "quality_tiers_v2.json"
STRICT_LIST = ROOT / "full_funasr_gold_overnight_gold_strict_v2.list"
EXPANDED_LIST = ROOT / "full_funasr_gold_overnight_high_confidence_v2.list"
REVIEW_LIST = ROOT / "full_funasr_gold_overnight_review_v2.list"

GENERAL_KEYS = ("mos_pred", "noi_pred", "dis_pred", "col_pred", "loud_pred")
HARD_EXCLUDE = ("safe_word_000", "safe_word_001")


def line(path: str, tier: str, text: str) -> str:
    return f"{path}|{tier}|JA|{text}"


def main() -> None:
    state_rows = {str(k): v for k, v in STATE.get("rows", {}).items()}
    acoustic_rows = {str(v["index"]): v for v in ACOUSTIC.get("rows", [])}
    evaluated = []
    tiers = {"gold_strict": [], "high_confidence": [], "review": []}

    for nrow in NISQA.get("rows", []):
        idx = str(nrow["index"])
        srow = state_rows.get(idx, {})
        arow = acoustic_rows.get(idx, {})
        path = str(nrow.get("path", srow.get("best_path", "")))
        if not path or any(token in path for token in HARD_EXCLUDE):
            continue

        acoustic_quality = arow.get("quality", {})
        acoustic_ok = bool(acoustic_quality.get("hard_gate_score90", False))
        tts = nrow.get("tts", {})
        baseline_tts = nrow.get("baseline_tts", {})
        tts_mos = float(tts.get("mos_pred", 0.0))
        tts_delta = tts_mos - float(baseline_tts.get("mos_pred", 0.0))

        general = nrow.get("general", {})
        baseline_general = nrow.get("baseline_general", {})
        general_deltas = {
            key: float(general[key]) - float(baseline_general[key])
            for key in GENERAL_KEYS
            if key in general and key in baseline_general
        }

        # Strict is the original intersection, retained as a tiny gold set.
        strict = nrow.get("decision") == "pass"
        # Expanded removes only the absolute general-MOS floor.  It still
        # requires acoustic score90, TTS MOS >=3, relative TTS stability, and
        # no large relative degradation in any general NISQA dimension.
        expanded = (
            acoustic_ok
            and tts_mos >= 3.0
            and tts_delta >= -0.30
            and general_deltas
            and min(general_deltas.values()) >= -0.50
        )
        review = (
            acoustic_ok
            and tts_mos >= 2.0
            and tts_delta >= -0.40
            and not expanded
        )

        if strict:
            tier = "gold_strict"
        elif expanded:
            tier = "high_confidence"
        elif review:
            tier = "review"
        else:
            tier = "rejected_or_low_confidence"

        if tier in tiers:
            tiers[tier].append(line(path, f"full_funasr_gold_overnight_{tier}_v2", srow.get("text", "")))

        evaluated.append({
            "index": int(idx),
            "path": path,
            "duration_sec": float(arow.get("duration_sec", 0.0)),
            "acoustic_score": float(acoustic_quality.get("score_100", 0.0)),
            "acoustic_gate_ok": acoustic_ok,
            "tts_mos": tts_mos,
            "tts_delta": tts_delta,
            "general_deltas": general_deltas,
            "original_strict_decision": nrow.get("decision"),
            "original_reasons": nrow.get("reasons", []),
            "tier": tier,
        })

    for path, rows in (
        (STRICT_LIST, tiers["gold_strict"]),
        (EXPANDED_LIST, tiers["high_confidence"]),
        (REVIEW_LIST, tiers["review"]),
    ):
        path.write_text("/n".join(rows) + ("/n" if rows else ""), encoding="utf-8")

    def summary(name: str) -> dict:
        rows = [row for row in evaluated if row["tier"] == name]
        return {
            "count": len(rows),
            "duration_sec": round(sum(row["duration_sec"] for row in rows), 3),
            "indices": [row["index"] for row in rows],
        }

    report = {
        "version": "quality_tiers_v2",
        "source_report": str(WORK / "nisqa_tts_quality_live.json"),
        "source_acoustic_report": str(WORK / "training_noise_quality_live.json"),
        "policy": {
            "gold_strict": "Original strict NISQA intersection; absolute general MOS floor retained.",
            "high_confidence": "Acoustic score90; NISQA-TTS MOS >=3.0; TTS delta >=-0.30; every available general NISQA dimension delta >=-0.50. Absolute general MOS floor intentionally removed because separated stems are miscalibrated by that floor.",
            "review": "Acoustic score90; NISQA-TTS MOS >=2.0; TTS delta >=-0.40; not high-confidence. Must be manually auditioned before training.",
            "note": "All tiers are automatic screening results, not a guarantee of human purity or a final TTS benchmark score.",
        },
        "lists": {
            "gold_strict": str(STRICT_LIST),
            "high_confidence": str(EXPANDED_LIST),
            "review": str(REVIEW_LIST),
        },
        "summary": {name: summary(name) for name in ("gold_strict", "high_confidence", "review")},
        "evaluated": evaluated,
    }
    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
