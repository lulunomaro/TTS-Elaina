# -*- coding: utf-8 -*-
"""Calibrate v2 soft/hard word states from all current human labels."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "human_calibration_combined_v3" / "labels_resolved.json"
BOUNDARY = ROOT / "human_verified_boundary_rescues_v1" / "audit_manifest.json"
ACCEPTANCE = ROOT / "human_full_semantic_acceptance_v1" / "labels_resolved.json"
SEMANTIC = ROOT / "human_verified_semantic_rescues_v3" / "audit_manifest.json"
ACCEPTANCE_HARD = ROOT / "human_full_semantic_acceptance_v1" / "hard_exclusion_intervals.json"
RESCUE_HARD = ROOT / "human_full_semantic_boundary_rescue_v1" / "hard_exclusion_intervals.json"
WORDS = ROOT / "full_source_local_word_scores_v1" / "source_reports"
OUT = ROOT / "full_semantic_word_state_calibration_v2.json"

FEATURES = (
    "cluster_margin",
    "context_target_fraction",
    "anchor_margin",
    "target_center_delta",
    "target_center_similarity",
    "non_target_center_similarity",
    "positive_anchor_similarity",
    "negative_anchor_similarity",
    "probability",
)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8"
    )


def add_interval(target: list[dict], row: dict, provenance: str) -> None:
    target.append(
        {
            "source_id": str(row["source_id"]),
            "start": float(row["start"]),
            "end": float(row["end"]),
            "provenance": provenance,
        }
    )


def normalized_word(row: dict) -> dict:
    result = dict(row)
    result["target_center_delta"] = float(row["target_center_similarity"]) - float(
        row["non_target_center_similarity"]
    )
    for key in FEATURES:
        result[key] = float(result.get(key, 0.0) or 0.0)
    return result


def map_words(intervals: list[dict], words_by_source: dict[str, list[dict]]) -> list[dict]:
    selected = {}
    by_source: dict[str, list[dict]] = defaultdict(list)
    for interval in intervals:
        by_source[interval["source_id"]].append(interval)
    for source_id, source_intervals in by_source.items():
        for index, raw in enumerate(words_by_source[source_id]):
            start = float(raw["start"])
            end = float(raw["end"])
            center = 0.5 * (start + end)
            hits = [row for row in source_intervals if row["start"] <= center <= row["end"]]
            if not hits:
                continue
            row = normalized_word(raw)
            row["source_id"] = source_id
            row["word_index"] = index
            row["provenance"] = sorted({hit["provenance"] for hit in hits})
            selected[(source_id, index)] = row
    return list(selected.values())


def quantiles(rows: list[dict]) -> dict:
    result = {}
    for feature in FEATURES:
        values = np.asarray([float(row[feature]) for row in rows], dtype=np.float64)
        result[feature] = {
            "p01": float(np.percentile(values, 1)),
            "p05": float(np.percentile(values, 5)),
            "p10": float(np.percentile(values, 10)),
            "p25": float(np.percentile(values, 25)),
            "p50": float(np.percentile(values, 50)),
            "p75": float(np.percentile(values, 75)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
        }
    result["strict_speaker_pass_rate"] = float(
        np.mean([bool(row["strict_speaker_pass"]) for row in rows])
    )
    return result


def conditions(row: dict) -> dict[str, bool]:
    context = float(row["context_target_fraction"])
    cluster = float(row["cluster_margin"])
    anchor = float(row["anchor_margin"])
    delta = float(row["target_center_delta"])
    probability = float(row["probability"])
    return {
        "strict": bool(row["strict_speaker_pass"] and probability >= 0.20),
        "soft_balanced": bool(
            probability >= 0.10
            and context >= 0.50
            and cluster >= -0.060
            and anchor >= 0.020
            and delta >= -0.030
        ),
        "soft_context_anchor": bool(
            probability >= 0.10
            and context >= 0.50
            and cluster >= -0.085
            and anchor >= 0.000
            and delta >= -0.060
        ),
        "soft_context_only": bool(
            probability >= 0.10 and context >= 0.50 and cluster >= -0.100
        ),
        "edge_soft_anchor": bool(
            probability >= 0.10
            and cluster >= -0.040
            and anchor >= 0.030
            and delta >= -0.040
            and float(row["target_center_similarity"]) >= 0.580
        ),
        "hard_negative_balanced": bool(
            context <= 0.00
            and cluster <= -0.040
            and anchor <= 0.030
            and delta <= 0.000
        ),
        "hard_negative_strong": bool(
            context <= 0.00
            and cluster <= -0.080
            and anchor <= 0.000
            and delta <= -0.030
        ),
    }


def rates(rows: list[dict]) -> dict:
    names = list(conditions(rows[0]))
    return {
        name: float(np.mean([conditions(row)[name] for row in rows])) for name in names
    }


def main() -> None:
    base = json.loads(BASE.read_text(encoding="utf-8"))["samples"]
    boundary = json.loads(BOUNDARY.read_text(encoding="utf-8"))["samples"]
    acceptance = json.loads(ACCEPTANCE.read_text(encoding="utf-8"))["samples"]
    semantic = json.loads(SEMANTIC.read_text(encoding="utf-8"))["samples"]

    positive: list[dict] = []
    pure_negative: list[dict] = []
    unsafe: list[dict] = []
    for row in base:
        if row["final_class"] == "target":
            add_interval(positive, row, f"base_target:{row['human_key']}")
        elif row["final_class"] == "pure_non_target":
            add_interval(pure_negative, row, f"base_negative:{row['human_key']}")
        elif row["final_class"] in {"mixed", "target_variant_exclude"}:
            add_interval(unsafe, row, f"base_unsafe:{row['human_key']}")
    for row in boundary:
        positive.append(
            {
                "source_id": str(row["source_id"]),
                "start": float(row["source_start"]),
                "end": float(row["source_end"]),
                "provenance": f"boundary_rescue:{row['final_id']}",
            }
        )
    for row in acceptance:
        if row["final_class"] == "target":
            add_interval(positive, row, f"acceptance_target:{row['blind_id']}")
    for row in semantic:
        add_interval(positive, row, f"semantic_rescue:{row['final_id']}")
    for path in (ACCEPTANCE_HARD, RESCUE_HARD):
        for row in json.loads(path.read_text(encoding="utf-8"))["intervals"]:
            unsafe.append(
                {
                    "source_id": str(row["source_id"]),
                    "start": float(row["start"]),
                    "end": float(row["end"]),
                    "provenance": str(row["reason"]),
                }
            )

    words_by_source = {}
    for path in sorted(WORDS.glob("*.json")):
        words_by_source[path.stem] = json.loads(path.read_text(encoding="utf-8"))["word_rows"]
    positive_words = map_words(positive, words_by_source)
    negative_words = map_words(pure_negative, words_by_source)
    unsafe_words = map_words(unsafe, words_by_source)

    report = {
        "version": "full_semantic_word_state_calibration_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "interval_counts": {
            "positive": len(positive),
            "pure_negative": len(pure_negative),
            "unsafe_mixed": len(unsafe),
        },
        "word_counts": {
            "positive": len(positive_words),
            "pure_negative": len(negative_words),
            "unsafe_mixed": len(unsafe_words),
        },
        "distributions": {
            "positive": quantiles(positive_words),
            "pure_negative": quantiles(negative_words),
            "unsafe_mixed": quantiles(unsafe_words),
        },
        "condition_rates": {
            "positive": rates(positive_words),
            "pure_negative": rates(negative_words),
            "unsafe_mixed": rates(unsafe_words),
        },
        "policy": {
            "positive": "all current human target intervals including direct acceptance and final semantic rescues",
            "negative": "pure non-target only",
            "unsafe": "mixed/variant/hard intervals used for rejection analysis, never pure-negative speaker training",
        },
    }
    write_json(OUT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
