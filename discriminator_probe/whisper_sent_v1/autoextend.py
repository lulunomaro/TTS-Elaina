# -*- coding: utf-8 -*-
"""Auto-extend short-cut segments to whisper sentence boundaries.

Pipeline (run AFTER whisper_sent_v1 batch transcribe completes):
  1. load whisper sentence segments per source
  2. merge adjacent whisper segments separated by < 0.6s into utterance blocks
     (only while block < 12s) to avoid over-merge (long dialogue chains)
  3. for each candidate segment (149 energy-flagged + 94 clean/ok high-conf),
     find the block containing its midpoint -> new boundaries
  4. verify against the 10 gold calibrations; write report + new .list
"""
from __future__ import annotations

import io
import json
from pathlib import Path

PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")
WS_DIR = PROBE / "whisper_sent_v1"
REPORT = json.loads((PROBE / "highconf_boundary_screen" / "screen_report.json").read_text(encoding="utf-8"))
GOLD = json.loads((PROBE / "highconf_calibrator" / "gold_calibrations.json").read_text(encoding="utf-8"))["samples"]

MERGE_GAP = 0.6
MAX_BLOCK = 12.0


def blocks_for(source_id):
    p = WS_DIR / f"{source_id}.json"
    if not p.exists():
        return []
    segs = json.loads(p.read_text(encoding="utf-8"))["segments"]
    if not segs:
        return []
    blocks = []
    cur = [segs[0]]
    for s in segs[1:]:
        if s["start"] - cur[-1]["end"] < MERGE_GAP and (s["end"] - cur[0]["start"]) <= MAX_BLOCK:
            cur.append(s)
        else:
            blocks.append(cur)
            cur = [s]
    blocks.append(cur)
    return [{"start": b[0]["start"], "end": b[-1]["end"],
             "text": "".join(x["text"] for x in b)} for b in blocks]


def main():
    items = REPORT["need_calib"]
    out = []
    no_ws = 0
    for it in items:
        sid = it["source_id"]
        mid = (it["start"] + it["end"]) / 2
        blocks = blocks_for(sid)
        hit = next((b for b in blocks if b["start"] <= mid <= b["end"]), None)
        if hit is None:
            no_ws += 1
            out.append({"origin_blind_id": Path(it["audio"]).name.replace(".wav", ""),
                        "source_id": sid, "start": it["start"], "end": it["end"],
                        "prob": it["prob"], "extended": False, "text": ""})
            continue
        out.append({"origin_blind_id": Path(it["audio"]).name.replace(".wav", ""),
                    "source_id": sid, "start": round(hit["start"], 3), "end": round(hit["end"], 3),
                    "prob": it["prob"], "extended": True,
                    "d_start": round(hit["start"] - it["start"], 3),
                    "d_end": round(hit["end"] - it["end"], 3),
                    "text": hit["text"][:200]})
    n_ext = sum(1 for o in out if o["extended"])
    print(f"total {len(out)} | extended {n_ext} | no-whisper {no_ws}")
    ds = [o["d_start"] for o in out if o["extended"]]
    de = [o["d_end"] for o in out if o["extended"]]
    import statistics
    print(f"d_start: median {statistics.median(ds):.2f}s (range {min(ds):.2f}..{max(ds):.2f})")
    print(f"d_end:   median {statistics.median(de):.2f}s (range {min(de):.2f}..{max(de):.2f})")

    # verify vs gold
    print("\n=== vs gold (10 manual) ===")
    for g in GOLD:
        cand = next((o for o in out if o["origin_blind_id"] == g["origin_blind_id"]), None)
        if not cand:
            print(f"  {g['qc_id']} {g['origin_blind_id']}: NO CAND")
            continue
        print(f"  {g['qc_id']} {g['origin_blind_id']}: auto {cand['start']:.2f}-{cand['end']:.2f} "
              f"| gold {g['abs_start']:.2f}-{g['abs_end']:.2f} | d {cand['start']-g['abs_start']:+.2f}/{cand['end']-g['abs_end']:+.2f}")

    (PROBE / "highconf_autoextend_v1.json").write_text(
        json.dumps({"version": "highconf_autoextend_v1", "n": len(out), "samples": out},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nsaved -> highconf_autoextend_v1.json")


if __name__ == "__main__":
    main()
