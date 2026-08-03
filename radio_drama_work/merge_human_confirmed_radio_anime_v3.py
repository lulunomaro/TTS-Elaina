# -*- coding: utf-8 -*-
"""Merge the current human-approved set, including semantic review batch 2."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import merge_human_confirmed_radio_anime_v2 as base

base.INPUTS = [
    ("radio_v5", ROOT / "human_confirmed_broadcast_v5_train_1p5_6s.list"),
    ("anime_direct_v1", ROOT / "human_verified_anime_semantic_v1_direct.list"),
    ("anime_review_v1", ROOT / "human_verified_anime_review_batch_v1.list"),
    ("anime_boundary_rescue_v1", ROOT / "human_verified_anime_boundary_rescue_v1.list"),
    ("anime_review_v2", ROOT / "human_verified_anime_review_batch_v2.list"),
]
base.ANIME_LIST = ROOT / "human_verified_anime_v3_train_1p5_6s.list"
base.COMBINED_LIST = ROOT / "human_confirmed_radio_anime_v3_train_1p5_6s.list"
base.REPORT = ROOT / "human_confirmed_radio_anime_v3_report.json"

if __name__ == "__main__":
    base.main()
