# -*- coding: utf-8 -*-
"""Ingest cross-segment spot-check feedback; pick best variant per origin.

Groups variants by origin (3 variants each: 0/-40/-80ms). For each origin:
  - if any variant passed -> keep the earliest offset that passed
  - if all failed -> mark as abandoned (record reason)
Writes boundary_fix_v4_final.list (passed variants) + report.
"""
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")
SPOT = PROBE / "boundary_spotcheck_v2"
AUDIT = SPOT / "audit_manifest.json"
PRIV = SPOT / "private_manifest.json"
OUT = PROBE / "boundary_fix_v4_final.list"


def classify(human: dict) -> str:
    if not bool(human.get("is_error")):
        return "pass"
    text = f"{human.get('error_type','')} {human.get('note','')}"
    if any(t in text for t in ("非目标", "非伊蕾娜")):
        return "non_target"
    if any(t in text for t in ("混合", "无缝", "接话")):
        return "mixed"
    if any(t in text for t in ("开头", "结尾", "切")):
        return "boundary_cut"
    if any(t in text for t in ("音质", "杂音", "环境")):
        return "quality"
    return "boundary_cut"


def main() -> None:
    feedback = json.loads(open(r"C:\Users\lulu\Downloads\boundary_spotcheck_v2_feedback_2026-08-03T02-27-06-633Z.json", encoding="utf-8").read())
    priv = json.loads(PRIV.read_text(encoding="utf-8"))["samples"]
    by_bid = {str(m["blind_id"]): m for m in priv}
    human_by = {str(s["blind_id"]): s for s in feedback["samples"]}

    # group by origin segment (variant groups of 3)
    groups: dict[str, list[dict]] = defaultdict(list)
    for m in priv:
        groups[m["origin_blind_id"]].append(m)

    final_rows = []
    abandoned = []
    for origin in sorted(groups):
        variants = sorted(groups[origin], key=lambda x: x["variant_offset"])
        results = []
        for v in variants:
            h = human_by.get(v["blind_id"])
            if not h:
                continue
            results.append({"blind_id": v["blind_id"], "offset": v["variant_offset"],
                            "class": classify(h), "audio": v["audio"],
                            "source_id": v["source_id"], "start": v["start"], "end": v["end"],
                            "duration": v["duration"], "mean_prob": v["mean_prob"]})
        passed = [r for r in results if r["class"] == "pass"]
        if passed:
            # keep earliest offset that passed
            best = min(passed, key=lambda r: r["offset"])
            final_rows.append(best)
            print(f"PASS {origin} -> {best['blind_id']} off={best['offset']}ms", flush=True)
        else:
            abandoned.append({"origin": origin, "results": results})
            print(f"FAIL {origin} (all {len(results)} variants)", flush=True)

    lines = [f"{r['audio']}|boundary_fix_v4_final|JA|" for r in final_rows]
    OUT.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    report = {
        "version": "boundary_fix_v4_final",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "origins": len(groups),
        "passed_origins": len(final_rows),
        "abandoned_origins": len(abandoned),
        "passed_duration_sec": round(sum(r["duration"] for r in final_rows), 2),
        "abandoned": abandoned,
        "list": str(OUT),
    }
    (PROBE / "boundary_fix_v4_final_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": len(final_rows), "abandoned": len(abandoned),
                      "sec": report["passed_duration_sec"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
