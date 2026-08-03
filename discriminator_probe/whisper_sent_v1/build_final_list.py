# -*- coding: utf-8 -*-
"""Final high-conf list: reviewed 15 use verdict (ok->ext, other->orig),
remaining 224 use ext only if meaningful (>0.5s change), else keep orig."""
import io
import json
from pathlib import Path

PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")

review = json.loads(io.open(r"C:\Users\lulu\Downloads\autoextend_review_1785741390909.json", encoding="utf-8").read())["items"]
ext_all = json.loads((PROBE / "highconf_autoextend_v2_speaker.json").read_text(encoding="utf-8"))["samples"]
orig_all = json.loads((PROBE / "highconf_boundary_screen" / "screen_report.json").read_text(encoding="utf-8"))["need_calib"]

ext_by = {s["origin_blind_id"]: s for s in ext_all}
orig_by = {Path(i["audio"]).name.replace(".wav", ""): i for i in orig_all}

MIN_CHANGE = 0.5  # only adopt extension if boundary moved >0.5s

final = []
stats = {"reviewed_ok_ext": 0, "reviewed_other_orig": 0, "auto_ext": 0, "auto_orig": 0}
for it in review:
    blind, verdict = it["blind"], it["verdict"]
    o = orig_by[blind]
    e = ext_by[blind]
    if verdict == "ok":
        final.append({"origin_blind_id": blind, "source_id": o["source_id"],
                      "start": e["start"], "end": e["end"], "prob": o["prob"], "src": "review_ok_ext"})
        stats["reviewed_ok_ext"] += 1
    else:
        final.append({"origin_blind_id": blind, "source_id": o["source_id"],
                      "start": o["start"], "end": o["end"], "prob": o["prob"], "src": "review_other_orig"})
        stats["reviewed_other_orig"] += 1

reviewed = {it["blind"] for it in review}
for o in orig_all:
    blind = Path(o["audio"]).name.replace(".wav", "")
    if blind in reviewed:
        continue
    e = ext_by.get(blind)
    moved = e and (abs(e["start"] - o["start"]) + abs(e["end"] - o["end"])) > MIN_CHANGE
    if moved:
        final.append({"origin_blind_id": blind, "source_id": o["source_id"],
                      "start": e["start"], "end": e["end"], "prob": o["prob"], "src": "auto_ext"})
        stats["auto_ext"] += 1
    else:
        final.append({"origin_blind_id": blind, "source_id": o["source_id"],
                      "start": o["start"], "end": o["end"], "prob": o["prob"], "src": "auto_orig"})
        stats["auto_orig"] += 1

final.sort(key=lambda x: (x["source_id"], x["start"]))
total_dur = sum(x["end"] - x["start"] for x in final)
print("stats:", json.dumps(stats, ensure_ascii=False))
print(f"final: {len(final)} segments, {total_dur:.1f}s = {total_dur/60:.1f} min")

(PROBE / "highconf_final_v1.json").write_text(
    json.dumps({"version": "highconf_final_v1", "stats": stats, "samples": final},
               ensure_ascii=False, indent=1), encoding="utf-8")
print("saved -> highconf_final_v1.json")
