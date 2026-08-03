# -*- coding: utf-8 -*-
"""Build original-vocals boundary rescue variants for semantic review batch 2."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_anime_boundary_rescue_batch_v1 as base

base.BATCH_LABELS = ROOT / "human_anime_semantic_review_batch_v2" / "labels_resolved.json"
base.OUT = ROOT / "anime_boundary_rescue_batch_v2"
base.AUDIO_DIR = base.OUT / "audio"
base.RESCUE_VERSION = "anime_boundary_rescue_batch_v2"
base.EXPECTED_BOUNDARY_COUNT = 22

if __name__ == "__main__":
    base.main()
