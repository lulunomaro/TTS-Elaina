# -*- coding: utf-8 -*-
"""Ingest the second anime review batch into its own frozen namespace."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ingest_anime_semantic_review_batch_v1 as base

base.AUDIT = ROOT / "anime_semantic_review_batch_v2" / "audit_manifest.json"
base.OUT = ROOT / "human_anime_semantic_review_batch_v2"
base.FROZEN = ROOT / "human_verified_anime_review_batch_v2"
base.LIST = ROOT / "human_verified_anime_review_batch_v2.list"
base.BATCH_VERSION = "anime_semantic_review_batch_v2"
base.FINAL_PREFIX = "HAVB2"

if __name__ == "__main__":
    base.main()
