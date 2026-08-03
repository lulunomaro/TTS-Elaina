# -*- coding: utf-8 -*-
"""Run the validated ERes2Net scan in an anime-only output namespace."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_full_local_mining_v1 as pipeline


ANIME_MANIFEST = ROOT / "anime_batch_v1" / "manifest.json"
RADIO_MANIFEST = ROOT / "full_track_manifest_v1.json"
OUT = ROOT / "anime_mining_v1" / "eres_scan"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(ANIME_MANIFEST.read_text(encoding="utf-8"))
    radio = json.loads(RADIO_MANIFEST.read_text(encoding="utf-8"))
    manifest["negative_files"] = radio.get("negative_files", [])
    manifest["hard_exclusions"] = []
    manifest["excluded_basenames"] = []

    pipeline.SCAN_OUT = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    # Same model space, paths, and parameters: reuse the immutable reference
    # caches rather than encoding the 168 clean references and 13 negatives again.
    for name in ("target_refs_eres2net_v1.npz", "negative_windows_eres2net_v2.npz"):
        source = ROOT / "full_track_scan_v1" / name
        destination = OUT / name
        if source.is_file() and not destination.is_file():
            shutil.copy2(source, destination)
    pipeline.phase_a(manifest, args.resume)


if __name__ == "__main__":
    main()
