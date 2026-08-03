from __future__ import annotations

import json
from pathlib import Path
import sys

import soundfile as sf

ROOT = Path(r"radio_drama_work/radio_drama_work")
VERSION = ROOT / "full_funasr_gold_v9"
REPORT = VERSION / "report.json"
HTML = VERSION / "listen_gold_train_candidates_only.html"
sys.path.insert(0, str(ROOT))
from make_funasr_target_recall_html import build


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    items = []
    index = 1
    for source in report.get("sources", []):
        source_name = Path(source["source"]).name.replace("_vocals.wav", "")
        for row in source.get("segments", []):
            path = Path(row["file"])
            if not path.exists():
                continue
            items.append({
                "index": index,
                "tier": "gold_train_candidate_v9",
                "source": source_name,
                "name": path.name,
                "src": path.resolve().as_uri(),
                "duration": round(float(sf.info(str(path)).duration), 2),
                "text": row.get("text", ""),
                "start": round(float(row.get("start", 0.0)), 3),
                "end": round(float(row.get("end", 0.0)), 3),
                "reference_sim": round(float(row.get("min_word_mid_top4", 0.0)), 3),
                "target_vote": 1.0,
                "other_ratio": 0.0,
            })
            index += 1
    HTML.write_text(build(items).replace("FunASR v2", "GOLD TRAIN CANDIDATES ONLY - FunASR v9"), encoding="utf-8")
    print(f"created={HTML}")
    print(f"items={len(items)}")


if __name__ == "__main__":
    main()
