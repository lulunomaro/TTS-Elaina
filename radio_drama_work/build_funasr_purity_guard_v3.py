"""Create a conservative, non-destructive candidate list from FunASR v2.

The v2 run selected the best target cluster even when the best and second
best cluster profiles were nearly tied.  This guard keeps those files for
listening review but removes them from the automatic training list.
"""
from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf


ROOT = Path(r"radio_drama_work/radio_drama_work")
V2 = ROOT / "full_funasr_target_recall_v2"
V3 = ROOT / "full_funasr_purity_guard_v3"
REPORT_V2 = V2 / "report.json"
REPORT_V3 = V3 / "report.json"
CANDIDATE_LIST = ROOT / "full_funasr_purity_guard_train_candidate_v3.list"
REVIEW_LIST = ROOT / "full_funasr_purity_guard_review_v3.list"
MARGIN_MIN = 0.04


def source_name(path: str) -> str:
    return Path(path).name.replace("_vocals.wav", "")


def min_cluster_margin(source: dict) -> float:
    margins: list[float] = []
    for run in source.get("cluster_runs", []):
        values = sorted((float(row["target_top4"]) for row in run.get("clusters", [])), reverse=True)
        if len(values) >= 2:
            margins.append(values[0] - values[1])
    return min(margins) if margins else 0.0


def row_line(row: dict, label: str, reason: str = "") -> str:
    path = Path(row["file"])
    text = str(row.get("text", "")).replace("/n", " ").strip()
    suffix = f";{reason}" if reason else ""
    return f"{path}|{label}|JA|{text}{suffix}"


def main() -> None:
    report_v2 = json.loads(REPORT_V2.read_text(encoding="utf-8"))
    V3.mkdir(parents=True, exist_ok=True)
    accepted_lines: list[str] = []
    review_lines: list[str] = []
    sources: list[dict] = []
    for source in report_v2.get("sources", []):
        name = source_name(source["source"])
        margin = min_cluster_margin(source)
        guarded = margin < MARGIN_MIN
        safe_rows = [] if guarded else list(source.get("accepted_segments", []))
        guarded_rows = list(source.get("accepted_segments", [])) if guarded else []
        expanded_rows = list(source.get("expanded_segments", []))
        accepted_lines.extend(row_line(row, "full_funasr_purity_guard_v3") for row in safe_rows)
        review_lines.extend(row_line(row, "full_funasr_purity_guard_v3_review", "expanded_review") for row in expanded_rows)
        review_lines.extend(row_line(row, "full_funasr_purity_guard_v3_review", "ambiguous_source") for row in guarded_rows)
        sources.append(
            {
                "source": source["source"],
                "cluster_margin_min": margin,
                "guarded_from_training": guarded,
                "guard_reason": "best/second-best target-cluster profiles too close" if guarded else None,
                "accepted_segments": safe_rows,
                "guarded_segments": guarded_rows,
                "expanded_segments": expanded_rows,
                "accepted_sec": sum(float(row["duration"]) for row in safe_rows),
                "guarded_sec": sum(float(row["duration"]) for row in guarded_rows),
                "expanded_sec": sum(float(row["duration"]) for row in expanded_rows),
            }
        )

    CANDIDATE_LIST.write_text("/n".join(accepted_lines) + ("/n" if accepted_lines else ""), encoding="utf-8")
    REVIEW_LIST.write_text("/n".join(review_lines) + ("/n" if review_lines else ""), encoding="utf-8")
    candidate_sec = sum(sf.info(str(Path(line.split("|", 1)[0]))).duration for line in accepted_lines)
    review_sec = sum(sf.info(str(Path(line.split("|", 1)[0]))).duration for line in review_lines)
    report_v3 = {
        "version": "full_funasr_purity_guard_v3",
        "input_report": str(REPORT_V2),
        "method": "v2 FunASR candidates plus source-level cluster-margin purity guard",
        "margin_min": MARGIN_MIN,
        "training_candidate_policy": "exclude any source whose minimum best-vs-second cluster target_top4 margin is below margin_min",
        "candidate_list": str(CANDIDATE_LIST),
        "review_list": str(REVIEW_LIST),
        "candidate_count": len(accepted_lines),
        "candidate_total_sec": candidate_sec,
        "review_count": len(review_lines),
        "review_total_sec": review_sec,
        "sources": sources,
        "note": "Audio is not deleted or reconstructed; guarded v2 files remain available for manual review.",
    }
    REPORT_V3.write_text(json.dumps(report_v3, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"candidate_count": len(accepted_lines), "candidate_sec": candidate_sec, "review_count": len(review_lines), "review_sec": review_sec, "candidate_list": str(CANDIDATE_LIST), "review_list": str(REVIEW_LIST)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
