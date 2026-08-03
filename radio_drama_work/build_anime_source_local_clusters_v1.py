# -*- coding: utf-8 -*-
"""Fit anime episode-local speaker clusters from the user's identity labels."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import source_local_cluster_anchor_v2 as pipeline


CAM_ROOT = ROOT / "anime_mining_v1" / "source_local_campp"
OUT = ROOT / "anime_mining_v1" / "source_local_cluster_anchor_v1"

pipeline.OUT = OUT
pipeline.CACHE = CAM_ROOT / "embedding_cache"
pipeline.TARGET_REFS = pipeline.CACHE / "all_target_references.npz"
pipeline.LABELS = ROOT / "human_anime_speaker_calibration_v3" / "labels_resolved.json"
pipeline.TARGET_CLASSES = {"target"}
pipeline.NEGATIVE_CLASSES = {"pure_non_target"}
pipeline.MIXED_CLASSES = {"mixed"}
pipeline.K_GRID = tuple(range(4, 25))


def main() -> None:
    labels = json.loads(pipeline.LABELS.read_text(encoding="utf-8"))["samples"]
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in labels:
        counts[str(row["source_id"])][str(row["final_class"])] += 1
    expected = {f"majo_ep{episode:02d}" for episode in range(1, 13)}
    missing_target = sorted(
        source_id for source_id in expected if not counts[source_id]["target"]
    )
    missing_negative = sorted(
        source_id for source_id in expected if not counts[source_id]["pure_non_target"]
    )
    if missing_target or missing_negative:
        raise RuntimeError(
            "Episode-local fitting requires at least one human target and one "
            f"pure non-target per episode; missing_target={missing_target}, "
            f"missing_negative={missing_negative}. Build a targeted calibration "
            "round instead of fitting an unanchored source."
        )
    pipeline.main()


if __name__ == "__main__":
    main()
