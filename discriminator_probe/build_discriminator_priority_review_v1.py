# -*- coding: utf-8 -*-
"""Build an isolated discriminator-priority human review batch.

This script only reads the existing radio_drama_work artifacts and the two
scored lists. It writes under discriminator_probe/discriminator_priority_review_v1
and never writes a training list or changes the existing mining pipeline.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf


PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")
WORK = Path(r"F:\GPTSOVITS\radio_drama_work")
OUT = PROBE / "discriminator_priority_review_v1"
AUDIO_DIR = OUT / "audio"
VERSION = "discriminator_priority_review_v1"
BATCH_SIZE = 48
PER_POOL = 24
OVERLAP_PAD = 0.80

SCORED_LISTS = {
    "anime": PROBE / "anime_review_431_scored.list",
    "radio": PROBE / "radio_review_448_scored.list",
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def waveform(path: Path, points: int = 120) -> list[float]:
    audio, _rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
    if not len(mono):
        return [0.0] * points
    edges = np.linspace(0, len(mono), points + 1, dtype=np.int64)
    values = []
    for left, right in zip(edges[:-1], edges[1:]):
        block = mono[left:max(left + 1, right)]
        values.append(float(np.max(np.abs(block))) if len(block) else 0.0)
    peak = max(values) if values else 0.0
    return [round(min(1.0, value / max(peak, 1e-8)), 4) for value in values]


def parse_scored(path: Path, pool: str) -> list[dict]:
    rows = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split("|", 5)
        if len(parts) != 6 or not parts[1].startswith("prob="):
            raise ValueError(f"{path}:{line_number}: invalid scored-list row")
        audio, prob_text, source_id, interval, duration_text, text = parts
        try:
            start_text, end_text = interval.split("-", 1)
            start = float(start_text)
            end = float(end_text)
            prob = float(prob_text.split("=", 1)[1])
            duration = float(duration_text.removesuffix("s"))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid numeric fields") from exc
        audio_path = Path(audio).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"missing scored audio: {audio_path}")
        rows.append(
            {
                "pool": pool,
                "audio_source": str(audio_path),
                "prob": prob,
                "source_id": source_id,
                "start": start,
                "end": end,
                "duration": duration,
                "text": text,
                "origin_blind_id": audio_path.stem,
            }
        )
    return rows


def extract_intervals(value: object) -> list[dict]:
    """Recursively collect source/start/end records from old JSON artifacts."""
    found: list[dict] = []
    if isinstance(value, dict):
        if {"source_id", "start", "end"}.issubset(value):
            try:
                found.append(
                    {
                        "source_id": str(value["source_id"]),
                        "start": float(value["start"]),
                        "end": float(value["end"]),
                        "reason": str(value.get("final_class") or value.get("reason") or "prior_artifact"),
                    }
                )
            except (TypeError, ValueError):
                pass
        for child in value.values():
            found.extend(extract_intervals(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(extract_intervals(child))
    return found


def load_prior_intervals() -> tuple[list[dict], list[dict]]:
    prior: list[dict] = []
    hard: list[dict] = []
    for root in (WORK, PROBE):
        for path in root.rglob("labels_resolved.json"):
            try:
                prior.extend(extract_intervals(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, json.JSONDecodeError):
                continue
        for path in root.rglob("audit_manifest.json"):
            try:
                prior.extend(extract_intervals(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, json.JSONDecodeError):
                continue
    for path in WORK.rglob("hard_exclusion*.json"):
        try:
            hard.extend(extract_intervals(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError):
            continue
    return prior, hard


def overlaps(row: dict, old: dict, padding: float = OVERLAP_PAD) -> bool:
    return str(row["source_id"]) == str(old["source_id"]) and max(
        float(row["start"]), float(old["start"]) - padding
    ) < min(float(row["end"]), float(old["end"]) + padding)


def exclusion_reason(row: dict, prior: list[dict], hard: list[dict]) -> str | None:
    if any(overlaps(row, item) for item in hard):
        return "hard_mask"
    if any(overlaps(row, item) for item in prior):
        return "already_reviewed_or_calibrated"
    return None


def tier(prob: float) -> str:
    if prob >= 0.9:
        return "p>=0.9"
    if prob >= 0.7:
        return "0.7-0.9"
    if prob >= 0.5:
        return "0.5-0.7"
    if prob >= 0.3:
        return "0.3-0.5"
    return "<0.3"


def rank(row: dict) -> tuple:
    return (float(row["prob"]), float(row["duration"]), -float(row["start"]))


def select_pool(rows: list[dict], target: int) -> list[dict]:
    eligible = [row for row in rows if row.get("eligible")]
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in eligible:
        by_source[str(row["source_id"])].append(row)
    for source_rows in by_source.values():
        source_rows.sort(key=rank, reverse=True)

    selected: list[dict] = []
    selected_ids: set[str] = set()
    # First cover every source using the highest available tier, then fill the
    # remaining slots in tier order. This preserves score priority without
    # allowing one episode/drama to dominate the first listening round.
    for source_id in sorted(by_source):
        options = by_source[source_id]
        candidate = next(
            (row for row in options if not any(overlaps(row, old, 0.0) for old in selected)),
            None,
        )
        if candidate is not None:
            selected.append(candidate)
            selected_ids.add(str(candidate["row_id"]))

    for minimum_prob in (0.9, 0.7, 0.5, 0.3):
        pool = sorted(
            [row for row in eligible if float(row["prob"]) >= minimum_prob and str(row["row_id"]) not in selected_ids],
            key=rank,
            reverse=True,
        )
        progress = True
        while progress and len(selected) < target:
            progress = False
            for row in pool:
                if str(row["row_id"]) in selected_ids:
                    continue
                if any(overlaps(row, old, OVERLAP_PAD) for old in selected):
                    continue
                selected.append(row)
                selected_ids.add(str(row["row_id"]))
                progress = True
                if len(selected) >= target:
                    break
    if len(selected) < target:
        fallback = sorted(
            [row for row in eligible if str(row["row_id"]) not in selected_ids],
            key=rank,
            reverse=True,
        )
        for row in fallback:
            if any(overlaps(row, old, 0.0) for old in selected):
                continue
            selected.append(row)
            selected_ids.add(str(row["row_id"]))
            if len(selected) >= target:
                break
    if len(selected) != target:
        raise RuntimeError(f"Only selected {len(selected)} of requested {target}")
    return sorted(selected, key=lambda row: (-float(row["prob"]), row["pool"], row["source_id"], row["start"]))


def main() -> None:
    all_rows = []
    for pool, path in SCORED_LISTS.items():
        all_rows.extend(parse_scored(path, pool))
    for index, row in enumerate(all_rows):
        row["row_id"] = f"{row['pool']}_{index:04d}"

    prior, hard = load_prior_intervals()
    low_score = []
    excluded = []
    eligible = []
    for row in all_rows:
        row["tier"] = tier(float(row["prob"]))
        reason = exclusion_reason(row, prior, hard)
        if reason:
            row["eligible"] = False
            row["exclusion_reason"] = reason
            excluded.append(row)
        elif float(row["prob"]) < 0.3:
            row["eligible"] = False
            row["exclusion_reason"] = "discriminator_low_score"
            low_score.append(row)
        else:
            row["eligible"] = True
            eligible.append(row)

    selected = []
    per_pool = {}
    for pool in ("anime", "radio"):
        pool_rows = [row for row in eligible if row["pool"] == pool]
        chosen = select_pool(pool_rows, PER_POOL)
        selected.extend(chosen)
        per_pool[pool] = {
            "available_after_exclusion": len(pool_rows),
            "selected": len(chosen),
            "selected_tiers": dict(Counter(row["tier"] for row in chosen)),
            "selected_sources": dict(Counter(row["source_id"] for row in chosen)),
        }

    OUT.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    audit_samples = []
    for index, source_row in enumerate(selected, 1):
        blind_id = f"DP{index:03d}"
        source_audio = Path(source_row["audio_source"])
        output = AUDIO_DIR / f"{blind_id}.wav"
        shutil.copy2(source_audio, output)
        info = sf.info(str(output))
        if info.channels != 1 or info.subtype != "PCM_16" or info.samplerate != 32000:
            raise RuntimeError(f"unexpected review audio format: {output} {info}")
        audit_samples.append(
            {
                **source_row,
                "blind_id": blind_id,
                "audio": str(output.resolve()),
                "audio_rel": f"audio/{output.name}",
                "output_duration": float(info.duration),
                "wave": waveform(output),
                "review_policy": "discriminator priority only; human identity, mixed speech, semantic boundary and quality review required",
            }
        )

    audit = {
        "version": f"{VERSION}_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "source": "anime_review_431_scored.list + radio_review_448_scored.list",
            "batch_size": BATCH_SIZE,
            "allocation": "24 anime + 24 radio",
            "priority": "p>=0.9, then 0.7-0.9, then lower eligible tiers",
            "p_lt_0_3": "excluded from listening and recorded as low-score candidates",
            "discriminator_only_orders": True,
            "quality_semantic_boundary_gates_preserved": True,
            "training_list_written": False,
            "training_started": False,
            "existing_radio_drama_work_modified": False,
        },
        "sample_count": len(audit_samples),
        "samples": audit_samples,
    }
    report = {
        "version": f"{VERSION}_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_counts": {pool: sum(1 for row in all_rows if row["pool"] == pool) for pool in SCORED_LISTS},
        "input_tiers": {pool: dict(Counter(row["tier"] for row in all_rows if row["pool"] == pool)) for pool in SCORED_LISTS},
        "excluded_counts": dict(Counter(row.get("exclusion_reason", "") for row in excluded)),
        "low_score_count": len(low_score),
        "eligible_count": len(eligible),
        "per_pool": per_pool,
        "selected_count": len(audit_samples),
        "selected_duration_sec": sum(row["output_duration"] for row in audit_samples),
        "selected_tiers": dict(Counter(row["tier"] for row in audit_samples)),
        "selected_sources": dict(Counter(row["source_id"] for row in audit_samples)),
        "machine_scores_hidden_in_html": True,
        "training_started": False,
        "existing_training_lists_modified": False,
        "score_90_claimed": False,
    }
    write_json(OUT / "audit_manifest.json", audit)
    write_json(OUT / "private_manifest.json", audit)
    write_json(OUT / "report.json", report)
    write_json(
        OUT / "low_score_candidates.json",
        {"version": f"{VERSION}_low_score_candidates", "threshold": 0.3, "samples": low_score},
    )
    write_json(
        OUT / "excluded_candidates.json",
        {"version": f"{VERSION}_exclusions", "samples": excluded},
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
