# -*- coding: utf-8 -*-
"""Fit source-local speaker clusters for all 12 tracks with rescued anchors.

The v3 model still used four original boundary-cut clips as target anchors and
covered only seven sources.  V4 removes every dirty boundary clip from model
fitting, appends the 12 user-audited rescued versions as positive anchors, and
fits the same conservative source-local clustering policy over all 12 CAM++
caches.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import source_local_cluster_anchor_v2 as pipeline


OUT = ROOT / "source_local_cluster_anchor_v4"
CACHE = ROOT / "full_source_local_campp_v1" / "embedding_cache"
BASE_LABELS = ROOT / "human_calibration_combined_v3" / "labels_resolved.json"
RESCUES = ROOT / "human_verified_boundary_rescues_v1" / "audit_manifest.json"


def main() -> None:
    base_doc = json.loads(BASE_LABELS.read_text(encoding="utf-8"))
    rescue_doc = json.loads(RESCUES.read_text(encoding="utf-8"))

    cluster_rows = []
    for row in base_doc["samples"]:
        item = dict(row)
        item["cluster_original_class"] = item["final_class"]
        if item["final_class"] == "boundary_cut":
            item["final_class"] = "boundary_replaced_or_ignored"
        cluster_rows.append(item)

    for rescue in rescue_doc["samples"]:
        cluster_rows.append(
            {
                "blind_id": rescue["final_id"],
                "kind": "human_verified_boundary_rescue",
                "source_id": rescue["source_id"],
                "start": rescue["source_start"],
                "end": rescue["source_end"],
                "duration": rescue["duration"],
                "audio": rescue["audio"],
                "private_text": rescue["text"],
                "final_class": "target",
                "speaker_training_role": "positive",
                "hard_exclusion": False,
                "human_round": "boundary_rescue_v1",
                "human_key": f"boundary_rescue_v1:{rescue['final_id']}",
                "root_origin_human_key": rescue["root_origin_human_key"],
                "cluster_original_class": "human_verified_rescue_target",
            }
        )

    cluster_label_path = OUT / "cluster_labels_resolved.json"
    cluster_label_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_label_path.write_text(
        json.dumps(
            {
                "version": "source_local_cluster_anchor_v4_labels",
                "policy": {
                    "base_target": "human-confirmed clean target clips",
                    "rescued_target": "12 final boundary rescues explicitly accepted by the user",
                    "boundary_original": "all original boundary-cut clips ignored",
                    "negative": "pure non-target only",
                    "mixed": "validation-only and required to reject",
                },
                "samples": cluster_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "/n",
        encoding="utf-8",
    )

    pipeline.OUT = OUT
    pipeline.CACHE = CACHE
    pipeline.TARGET_REFS = CACHE / "all_target_references.npz"
    pipeline.LABELS = cluster_label_path
    pipeline.TARGET_CLASSES = {"target"}
    pipeline.NEGATIVE_CLASSES = {"pure_non_target"}
    pipeline.MIXED_CLASSES = {"mixed", "overlap", "target_variant_exclude"}
    pipeline.main()

    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["version"] = "source_local_cluster_anchor_v4"
    report["policy"]["coverage"] = "all 12 manifest sources"
    report["policy"]["target_anchor"] = (
        "clean human target clips plus all 12 user-accepted final boundary rescues"
    )
    report["policy"]["boundary_anchor"] = (
        "original boundary-cut audio is never used; only its accepted inward-trim rescue may anchor"
    )
    report["policy"]["negative_anchor"] = "human-confirmed pure non-target clips"
    report["policy"]["mixed"] = (
        "mixed/overlap/target-variant clips are validation-only and must be rejected"
    )
    report["policy"]["quality_issue"] = "ignored for speaker fitting and hard-excluded later"
    report["human_verified_rescue_anchor_count"] = len(rescue_doc["samples"])
    report["expected_source_count"] = 12
    report["actual_source_count"] = len(report["sources"])
    if report["actual_source_count"] != report["expected_source_count"]:
        raise RuntimeError(
            f"Expected 12 source models, got {report['actual_source_count']}"
        )
    report["deployable"] = False
    report["recommendation"] = (
        "mine complete semantic phrases, apply inward transition trimming, then human-review a fresh batch"
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "/n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
