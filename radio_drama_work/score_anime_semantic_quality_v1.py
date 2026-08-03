# -*- coding: utf-8 -*-
"""Apply the validated GPU relative-quality gate to anime strict candidates."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import score_full_semantic_quality_v2 as pipeline


OUT = ROOT / "anime_semantic_quality_v1"

pipeline.INPUT = ROOT / "anime_semantic_mining_v1" / "audit_manifest.json"
pipeline.OUT = OUT
pipeline.REPORT = OUT / "report.json"
pipeline.AUDIT = OUT / "audit_manifest.json"
pipeline.QUALITY_LIST = ROOT / "anime_semantic_quality_strict_review_v1.list"
pipeline.QUALITY_REVIEW_LIST = ROOT / "anime_semantic_quality_review_v1.list"
pipeline.VERSION = "anime_nisqa_tts_v1_source_delta_boundary"


if __name__ == "__main__":
    pipeline.main()
