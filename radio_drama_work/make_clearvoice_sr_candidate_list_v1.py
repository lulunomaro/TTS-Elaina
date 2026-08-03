"""Export only ClearVoice SR outputs that pass the repair gate."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"radio_drama_work/radio_drama_work")
REPORT = ROOT / "clearvoice_sr_full_v1" / "repair_probe_quality_report.json"
RAW_LIST = ROOT / "full_funasr_gold_train_candidate_v9.list"
PASS_LIST = ROOT / "full_funasr_gold_clearvoice_sr_train_candidate_v1.list"
REVIEW_LIST = ROOT / "full_funasr_gold_clearvoice_sr_review_v1.list"


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    metadata = {}
    for line in RAW_LIST.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            fields = line.split("|", 3)
            metadata[fields[0].strip()] = fields[3].strip() if len(fields) > 3 else ""
    passed, review = [], []
    for row in report["rows"]:
        output = Path(row["final_training_format"])
        text = metadata.get(row["input"], "")
        line = f"{output}|clearvoice_sr_auto_pass_v1|JA|{text}"
        if row["gate"]["decision"] == "pass" and output.exists():
            passed.append(line)
        else:
            review.append(f"{row['sr_output_48k']}|clearvoice_sr_review_only_v1|JA|{text}")
    PASS_LIST.write_text("/n".join(passed) + ("/n" if passed else ""), encoding="utf-8")
    REVIEW_LIST.write_text("/n".join(review) + ("/n" if review else ""), encoding="utf-8")
    summary = {
        "pass_count": len(passed),
        "pass_duration_sec": sum(float(row["after_final_32k"]["duration_sec"]) for row in report["rows"] if row["gate"]["decision"] == "pass"),
        "review_only_count": len(review),
        "pass_list": str(PASS_LIST),
        "review_list": str(REVIEW_LIST),
        "policy": "automatic gate only; no human listening claim; raw v9 list remains unchanged",
    }
    (ROOT / "clearvoice_sr_full_v1" / "candidate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
