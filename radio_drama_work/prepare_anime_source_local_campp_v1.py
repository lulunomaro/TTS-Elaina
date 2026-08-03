# -*- coding: utf-8 -*-
"""Run the validated CAM++/FunASR cache builder in an anime-only namespace."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import prepare_full_source_local_campp_v1 as pipeline


OUT = ROOT / "anime_mining_v1" / "source_local_campp"

# Keep all anime products isolated. The old cache directory is used only as
# the immutable source of the 168 CAM++ target reference embeddings; no radio
# source cache has an anime filename, so source audio is always freshly encoded.
pipeline.MANIFEST = ROOT / "anime_batch_v1" / "manifest.json"
pipeline.LEGACY = ROOT / "full_source_local_campp_v1"
pipeline.OUT = OUT
pipeline.EMBED_CACHE = OUT / "embedding_cache"
pipeline.VAD_CACHE = OUT / "vad_cache"
pipeline.REPORT = OUT / "report.json"
pipeline.TARGET_REFS = pipeline.EMBED_CACHE / "all_target_references.npz"


if __name__ == "__main__":
    pipeline.main()
