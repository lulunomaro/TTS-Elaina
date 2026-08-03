# -*- coding: utf-8 -*-
"""Recoverable post-separation runner through the first anime review page."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STATUS = ROOT / "anime_mining_v1" / "precalibration_status.json"
PHASES = ("asr", "campp", "eres", "calibration", "page")


def write_status(completed: list[str], current: str, state: str) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(
        json.dumps(
            {
                "version": "anime_precalibration_v1_status",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "completed": completed,
                "current": current,
                "state": state,
                "training_started": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "/n",
        encoding="utf-8",
    )


def run(script: str, *arguments: str) -> None:
    command = [sys.executable, str(ROOT / script), *arguments]
    print(f"[precalibration] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=str(ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("all",) + PHASES, default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    wanted = list(PHASES) if args.phase == "all" else [args.phase]
    completed: list[str] = []
    try:
        for phase in wanted:
            write_status(completed, phase, "running")
            if phase == "asr":
                run("run_anime_asr_v1.py", *( ["--resume"] if args.resume else []))
            elif phase == "campp":
                run(
                    "prepare_anime_source_local_campp_v1.py",
                    *( ["--resume"] if args.resume else []),
                )
            elif phase == "eres":
                run("run_anime_eres_scan_v1.py", *( ["--resume"] if args.resume else []))
            elif phase == "calibration":
                run("build_anime_cluster_calibration_v1.py")
            elif phase == "page":
                run("make_speaker_identity_review_page_v1.py")
            completed.append(phase)
            write_status(completed, "", "running")
    except Exception:
        write_status(completed, phase, "failed")
        raise
    write_status(completed, "", "complete")
    print(
        json.dumps(
            {
                "completed": completed,
                "review_page": str(
                    ROOT / "anime_speaker_calibration_v1" / "index.html"
                ),
                "training_started": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
