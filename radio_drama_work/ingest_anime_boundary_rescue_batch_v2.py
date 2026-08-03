# -*- coding: utf-8 -*-
"""Ingest batch-2 boundary rescue feedback and freeze one best variant per origin."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ingest_anime_boundary_rescue_batch_v1 as base

base.AUDIT = ROOT / "anime_boundary_rescue_batch_v2" / "audit_manifest.json"
base.OUT = ROOT / "human_anime_boundary_rescue_batch_v2"
base.FROZEN = ROOT / "human_verified_anime_boundary_rescue_v2"
base.LIST = ROOT / "human_verified_anime_boundary_rescue_v2.list"
base.BATCH_VERSION = "anime_boundary_rescue_batch_v2"
base.FINAL_PREFIX = "HABR2"

if __name__ == "__main__":
    base.main()
