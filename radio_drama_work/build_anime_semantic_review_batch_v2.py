# -*- coding: utf-8 -*-
"""Build the second stratified anime review batch without repeating batch 1."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_anime_semantic_review_batch_v1 as base

base.OUT = ROOT / "anime_semantic_review_batch_v2"
base.AUDIO_DIR = base.OUT / "audio"
base.REVIEW_VERSION = "anime_semantic_review_batch_v2"
base.EXCLUDED_ORIGIN_IDS_PATH = ROOT / "anime_semantic_review_batch_v1" / "audit_manifest.json"
base.MIN_PER_SOURCE = 1
base.MAX_PER_SOURCE = 6

if __name__ == "__main__":
    base.main()
