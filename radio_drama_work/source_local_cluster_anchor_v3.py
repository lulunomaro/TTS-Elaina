# -*- coding: utf-8 -*-
"""Re-fit source-local speaker clusters with all three human-review rounds."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import source_local_cluster_anchor_v2 as pipeline


OUT = ROOT / "source_local_cluster_anchor_v3"


def main() -> None:
    pipeline.OUT = OUT
    source_doc = json.loads(
        (ROOT / "human_calibration_combined_v3" / "labels_resolved.json").read_text(encoding="utf-8")
    )
    cluster_rows = []
    for row in source_doc["samples"]:
        item = dict(row)
        item["cluster_original_class"] = item["final_class"]
        if item["final_class"] == "boundary_cut":
            # Only round 3 has the user's explicit statement that the voice is
            # correct and only the boundary is broken.  Older boundary labels
            # remain ignored by the speaker fit until their rescues are heard.
            item["final_class"] = "target" if item["human_round"] == "round3" else "boundary_ignored"
        cluster_rows.append(item)
    cluster_label_path = OUT / "cluster_labels_resolved.json"
    cluster_label_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_label_path.write_text(
        json.dumps({"version": "source_local_cluster_anchor_v3_labels", "samples": cluster_rows}, ensure_ascii=False, indent=2)
        + "/n",
        encoding="utf-8",
    )
    pipeline.LABELS = cluster_label_path
    pipeline.TARGET_CLASSES = {"target"}
    pipeline.NEGATIVE_CLASSES = {"pure_non_target"}
    pipeline.MIXED_CLASSES = {"mixed", "overlap", "target_variant_exclude"}
    pipeline.main()

    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["version"] = "source_local_cluster_anchor_v3"
    report["policy"]["target_anchor"] = "human target plus the four explicitly speaker-safe round-3 boundary clips"
    report["policy"]["negative_anchor"] = "human pure non-target clips from all three rounds"
    report["policy"]["mixed"] = "mixed/overlap/target-variant are validation-only and must be rejected"
    report["policy"]["quality_issue"] = "ignored for speaker clustering; remains hard-excluded from training"
    report["policy"]["older_boundary"] = "ignored by speaker fitting until rescued audio is heard"
    report["deployable"] = False
    report["recommendation"] = "validate repaired boundaries and another fresh source-balanced batch"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "/n", encoding="utf-8")


if __name__ == "__main__":
    main()
