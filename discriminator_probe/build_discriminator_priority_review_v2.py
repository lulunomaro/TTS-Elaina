# -*- coding: utf-8 -*-
"""Build the next isolated priority review batch from discriminator v2 scores."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_discriminator_priority_review_v1 as base

base.OUT = ROOT / "discriminator_priority_review_v2"
base.AUDIO_DIR = base.OUT / "audio"
base.VERSION = "discriminator_priority_review_v2"
base.SCORED_LISTS = {
    "anime": ROOT / "discriminator_rank_v2" / "anime_review_431_scored_v2.list",
    "radio": ROOT / "discriminator_rank_v2" / "radio_review_448_scored_v2.list",
}

if __name__ == "__main__":
    base.main()
