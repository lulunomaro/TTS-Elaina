# -*- coding: utf-8 -*-
"""Score all anime ASR words in the human-anchored episode-local models."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import score_full_source_local_words_v1 as pipeline


MODEL_ROOT = ROOT / "anime_mining_v1" / "source_local_cluster_anchor_v1"
OUT = ROOT / "anime_mining_v1" / "source_local_word_scores_v1"

pipeline.MANIFEST = ROOT / "anime_batch_v1" / "manifest.json"
pipeline.WORDS_DIR = ROOT / "anime_mining_v1" / "asr" / "source_reports"
pipeline.MODEL_DIR = MODEL_ROOT / "models"
pipeline.MODEL_REPORT = MODEL_ROOT / "report.json"
pipeline.CLUSTER_LABELS = ROOT / "human_anime_speaker_calibration_v3" / "labels_resolved.json"
pipeline.OUT = OUT
pipeline.EMBED_CACHE = OUT / "embedding_cache"
pipeline.SOURCE_REPORTS = OUT / "source_reports"
pipeline.REPORT = OUT / "report.json"
# Until a full semantic acceptance round succeeds, every anime episode stays
# guarded and uses the stricter word thresholds.
pipeline.GUARDED_SOURCES = {f"majo_ep{episode:02d}" for episode in range(1, 13)}


if __name__ == "__main__":
    pipeline.main()
