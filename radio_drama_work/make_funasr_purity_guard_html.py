from __future__ import annotations

import json
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from make_funasr_target_recall_html import build


ROOT = Path(r"radio_drama_work/radio_drama_work")
V3 = ROOT / "full_funasr_purity_guard_v3"
REPORT = V3 / "report.json"
HTML = V3 / "listen_candidates.html"


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    items: list[dict] = []
    index = 1
    for source in report.get("sources", []):
        source_name = Path(source["source"]).name.replace("_vocals.wav", "")
        for tier, rows in (
            ("accepted", source.get("accepted_segments", [])),
            ("expanded", source.get("expanded_segments", [])),
            ("review", source.get("guarded_segments", [])),
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
                        "reference_sim": round(float(row.get("min_reference_sim", 0.0)), 3),
                        "target_vote": round(float(row.get("min_target_vote", 0.0)), 3),
                        "other_ratio": round(float(row.get("other_ratio", 0.0)), 3),
                    }
                )
                index += 1
    page = build(items).replace("FunASR v2", "FunASR purity guard v3")
    HTML.write_text(page, encoding="utf-8")
    print(f"created={HTML}")
    print(f"items={len(items)}")


if __name__ == "__main__":
    main()
