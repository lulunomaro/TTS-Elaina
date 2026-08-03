# -*- coding: utf-8 -*-
"""Build v3 discriminator-priority review page (isolated, hidden scores).

Selects 24 anime + 24 radio clips from v3 ranking, excluding all previously
reviewed origin_blind_ids (v1+v2). Writes a self-contained HTML page.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")
RANK = PROBE / "discriminator_rank_v3"
OUT = PROBE / "discriminator_priority_review_v3"
AUDIO = OUT / "audio"
SEED = 2026080301


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_heard() -> set[str]:
    heard = set()
    for v in ("v1", "v2"):
        a = json.loads((PROBE / f"discriminator_priority_review_{v}" / "audit_manifest.json").read_text(encoding="utf-8"))
        for r in a.get("samples", a.get("rows", [])):
            heard.add(str(r.get("origin_blind_id") or r.get("blind_id") or ""))
    return heard


def parse_scored(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 6:
            continue
        rows.append({
            "audio": parts[0],
            "prob": float(parts[1].split("=")[1]),
            "source_id": parts[2],
            "start": float(parts[3].split("-")[0]),
            "end": float(parts[3].split("-")[1]),
            "duration": float(parts[4].replace("s", "")),
            "text": parts[5],
        })
    return rows


def select(rows: list[dict], heard: set[str], n: int) -> list[dict]:
    cands = [r for r in rows if r["audio"].split("\\")[-1] not in heard
             and Path(r["audio"]).exists()]
    # source-balanced greedy: per-source cap, prefer higher prob
    by_src: dict[str, list[dict]] = {}
    for r in cands:
        by_src.setdefault(r["source_id"], []).append(r)
    selected: list[dict] = []
    per_src = {}
    for r in sorted(cands, key=lambda x: -x["prob"]):
        src = r["source_id"]
        cap = max(2, n // max(1, len(by_src)))
        if per_src.get(src, 0) >= cap:
            continue
        selected.append(r)
        per_src[src] = per_src.get(src, 0) + 1
        if len(selected) >= n:
            break
    return selected


def build_html(rows: list[dict], title: str) -> str:
    items = []
    for i, r in enumerate(rows, 1):
        bid = f"DP{i:03d}"
        src_rel = f"audio/{Path(r['audio']).name}"
        items.append(f"<li><b>{bid}</b> <audio controls preload='none' src='{src_rel}'></audio><br><small>时长 {r['duration']:.1f}s</small></li>")
    return (
        "<!doctype html><meta charset='utf-8'><title>" + title + "</title>"
        "<h1>" + title + "</h1>"
        "<p>机器分数已隐藏。判定：完整伊蕾娜 / 非目标 / 混合/无缝接话 / 开头或结尾切断 / 杂音/音质 / 重叠 / 不确定。"
        "边界被切的只标边界，不标非目标。</p><ol>"
        + "\n".join(items) +
        "</ol>"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    AUDIO.mkdir(parents=True, exist_ok=True)
    heard = load_heard()
    print(f"heard origins: {len(heard)}", flush=True)

    anime = select(parse_scored(RANK / "anime_review_431_scored_v3.list"), heard, 24)
    radio = select(parse_scored(RANK / "radio_review_448_scored_v3.list"), heard, 24)
    all_rows = anime + radio
    print(f"selected: anime={len(anime)} radio={len(radio)} total={len(all_rows)}", flush=True)

    manifest_rows = []
    for i, r in enumerate(all_rows, 1):
        bid = f"DP{i:03d}"
        dest = AUDIO / f"{bid}.wav"
        if not dest.exists():
            shutil.copy2(r["audio"], dest)
        manifest_rows.append({
            "blind_id": bid,
            "origin_blind_id": Path(r["audio"]).name,
            "pool": "anime" if r["source_id"].startswith("majo_") else "radio",
            "prob": r["prob"],
            "source_id": r["source_id"],
            "start": r["start"],
            "end": r["end"],
            "duration": r["duration"],
            "text": r["text"],
            "audio": str(dest.resolve()),
        })

    write_json(OUT / "audit_manifest.json", {"version": "discriminator_priority_review_v3_audit", "samples": manifest_rows})
    write_json(OUT / "private_manifest.json", {"version": "v3_private", "samples": manifest_rows})
    report = {
        "version": "discriminator_priority_review_v3_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {"anime": len(parse_scored(RANK / "anime_review_431_scored_v3.list")),
                  "radio": len(parse_scored(RANK / "radio_review_448_scored_v3.list"))},
        "excluded_heard": len(heard),
        "selected_count": len(all_rows),
        "selected_duration_sec": round(sum(r["duration"] for r in all_rows), 2),
        "per_pool": dict(Counter(r["pool"] for r in manifest_rows)),
        "per_source": dict(Counter(r["source_id"] for r in manifest_rows)),
        "machine_scores_hidden_in_html": True,
        "training_started": False,
        "existing_training_lists_modified": False,
    }
    write_json(OUT / "report.json", report)
    (OUT / "index.html").write_text(build_html(all_rows, "判别器v3·第三批48条试听"), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"saved={OUT / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
