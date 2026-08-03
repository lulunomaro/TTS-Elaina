from __future__ import annotations

import json
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from make_funasr_target_recall_html import build


ROOT = Path(r"radio_drama_work/radio_drama_work")
VERSION = ROOT / "full_funasr_local_boundary_v7"
REPORT = VERSION / "report.json"
HTML = VERSION / "listen_train_candidates_only.html"


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    items: list[dict] = []
    index = 1
    for source in report.get("sources", []):
        source_name = Path(source["source"]).name.replace("_vocals.wav", "")
        for row in source.get("accepted_segments", []):
            path = Path(row["file"])
            if not path.exists():
                continue
            items.append({
                "index": index,
                "tier": "accepted",
                "source": source_name,
                "name": path.name,
                "src": path.resolve().as_uri(),
                "duration": round(float(sf.info(str(path)).duration), 2),
                "text": row.get("text", ""),
                "start": round(float(row.get("start", 0.0)), 3),
                "end": round(float(row.get("end", 0.0)), 3),
                "reference_sim": round(float(row.get("local_mid_top4", 0.0)), 3),
                "target_vote": 1.0,
                "other_ratio": 0.0,
            })
            index += 1
    HTML.write_text(build(items).replace("FunASR v2", "TRAIN CANDIDATES ONLY - FunASR v7"), encoding="utf-8")
    print(f"created={HTML}")
    print(f"items={len(items)}")


if __name__ == "__main__":
    main()
