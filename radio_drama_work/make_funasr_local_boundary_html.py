from __future__ import annotations

import json
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from make_funasr_target_recall_html import build


ROOT = Path(r"radio_drama_work/radio_drama_work")
V6 = ROOT / "full_funasr_local_boundary_v7"
REPORT = V6 / "report.json"
HTML = V6 / "listen_candidates.html"


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    items: list[dict] = []
    index = 1
    for source in report.get("sources", []):
        source_name = Path(source["source"]).name.replace("_vocals.wav", "")
        for tier, rows in (
            ("accepted", source.get("accepted_segments", [])),
            ("expanded", source.get("review_segments", [])),
            ("review", source.get("boundary_review_segments", [])),
        ):
            for row in rows:
                path = Path(row["file"])
                if not path.exists():
                    continue
                items.append(
                    {
                        "index": index,
                        "tier": tier,
                        "source": source_name,
                        "name": path.name,
                        "src": path.resolve().as_uri(),
                        "duration": round(float(sf.info(str(path)).duration), 2),
                        "text": row.get("text", ""),
                        "start": round(float(row.get("start", 0.0)), 3),
                        "end": round(float(row.get("end", 0.0)), 3),
                        "reference_sim": round(float(row.get("local_mid_top4", row.get("min_reference_sim", 0.0))), 3),
                        "target_vote": 1.0,
                        "other_ratio": 0.0,
                    }
                )
                index += 1
    HTML.write_text(build(items).replace("FunASR v2", "FunASR local boundary v7"), encoding="utf-8")
    print(f"created={HTML}")
    print(f"items={len(items)}")


if __name__ == "__main__":
    main()
