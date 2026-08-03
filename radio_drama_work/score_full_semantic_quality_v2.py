# -*- coding: utf-8 -*-
"""GPU NISQA-TTS and acoustic gates for the v2 semantic review pool."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_resource_policy import apply_cpu_friendly_policy


RESOURCE_POLICY = apply_cpu_friendly_policy(2)

import numpy as np
import soundfile as sf
import torch

import full_speaker_diarize_new_v1 as audio_base
import run_full_local_mining_v1 as quality_base


INPUT = ROOT / "full_semantic_mining_v2" / "audit_manifest.json"
OUT = ROOT / "full_semantic_quality_v2"
REPORT = OUT / "report.json"
AUDIT = OUT / "audit_manifest.json"
QUALITY_LIST = ROOT / "full_semantic_quality_strict_review_v2.list"
QUALITY_REVIEW_LIST = ROOT / "full_semantic_quality_review_v2.list"

VERSION = "nisqa_tts_v2_source_delta_boundary"
BOUNDARY_HARD_RATIO = 0.85
BOUNDARY_RISK_RATIO = 0.55


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8"
    )


def file_fingerprint(row: dict) -> str:
    output = Path(row["audio"])
    source = Path(row["source"])
    output_stat = output.stat()
    source_stat = source.stat()
    payload = "|".join(
        (
            str(output.resolve()),
            str(output_stat.st_size),
            str(output_stat.st_mtime_ns),
            str(source.resolve()),
            str(source_stat.st_size),
            str(source_stat.st_mtime_ns),
            f"{float(row['start']):.9f}",
            f"{float(row['end']):.9f}",
            VERSION,
        )
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


class SourceReader:
    """Keep at most one full radio track in RAM while rows stay source-sorted."""

    def __init__(self) -> None:
        self.path: Path | None = None
        self.audio: np.ndarray | None = None
        self.sample_rate = 0

    def interval(self, path: Path, start: float, end: float, output_rate: int) -> np.ndarray:
        if self.path != path:
            audio, sample_rate = sf.read(
                str(path), dtype="float32", always_2d=True
            )
            self.path = path
            self.audio = np.asarray(audio.mean(axis=1), dtype=np.float32)
            self.sample_rate = int(sample_rate)
        assert self.audio is not None
        left = max(0, round(start * self.sample_rate))
        right = min(len(self.audio), round(end * self.sample_rate))
        raw = self.audio[left:right]
        return audio_base.resample_1d(raw, self.sample_rate, output_rate)


def boundary_metrics(candidate: dict) -> dict:
    left = float(candidate["left_boundary"]["energy_ratio"])
    right = float(candidate["right_boundary"]["energy_ratio"])
    left_mode = str(candidate["boundary_modes"]["left"])
    right_mode = str(candidate["boundary_modes"]["right"])
    hard_left = left >= BOUNDARY_HARD_RATIO
    hard_right = right >= BOUNDARY_HARD_RATIO
    return {
        "left_energy_ratio": left,
        "right_energy_ratio": right,
        "left_mode": left_mode,
        "right_mode": right_mode,
        "hard_cutoff": bool(hard_left or hard_right),
        "hard_cutoff_sides": [
            side for side, value in (("left", hard_left), ("right", hard_right)) if value
        ],
        "review_risk": bool(left >= BOUNDARY_RISK_RATIO or right >= BOUNDARY_RISK_RATIO),
    }


def partial_report(rows: list[dict]) -> dict:
    return {
        "version": "full_semantic_quality_v2_partial",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    source_audit = json.loads(INPUT.read_text(encoding="utf-8"))
    source_rows = source_audit["samples"]
    cached: dict[str, dict] = {}
    if args.resume and REPORT.is_file():
        try:
            cached = {
                row["fingerprint"]: row
                for row in json.loads(REPORT.read_text(encoding="utf-8")).get("rows", [])
                if row.get("fingerprint")
            }
        except Exception:
            cached = {}

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this GPU-only quality pass")
    model = quality_base.load_nisqa_tts()
    source_reader = SourceReader()
    scored: list[dict] = []
    for index, candidate in enumerate(source_rows, start=1):
        fingerprint = file_fingerprint(candidate)
        if fingerprint in cached:
            item = dict(cached[fingerprint])
            item["candidate"] = candidate
            scored.append(item)
            print(f"[{index}/{len(source_rows)}] resume {candidate['blind_id']}", flush=True)
            continue

        output_path = Path(candidate["audio"])
        output_audio, output_rate = sf.read(
            str(output_path), dtype="float32", always_2d=True
        )
        output_audio = np.asarray(output_audio.mean(axis=1), dtype=np.float32)
        source_audio = source_reader.interval(
            Path(candidate["source"]),
            float(candidate["start"]),
            float(candidate["end"]),
            int(output_rate),
        )
        output_acoustic = quality_base.acoustic_metrics(output_audio, int(output_rate))
        source_acoustic = quality_base.acoustic_metrics(source_audio, int(output_rate))
        output_mos = quality_base.tts_mos(model, output_audio)
        source_mos = quality_base.tts_mos(model, source_audio)
        boundary = boundary_metrics(candidate)
        item = {
            "blind_id": candidate["blind_id"],
            "path": str(output_path),
            "source_id": candidate["source_id"],
            "duration": float(candidate["output_duration"]),
            "fingerprint": fingerprint,
            "nisqa_tts_mos": float(output_mos),
            "source_interval_nisqa_tts_mos": float(source_mos),
            "nisqa_tts_delta": float(output_mos - source_mos),
            "acoustic": output_acoustic,
            "source_interval_acoustic": source_acoustic,
            "boundary": boundary,
            "candidate": candidate,
        }
        scored.append(item)
        print(
            f"[{index}/{len(source_rows)}] {candidate['blind_id']} "
            f"mos={output_mos:.3f} delta={output_mos - source_mos:+.3f} "
            f"clicks={output_acoustic['clicks_per_sec']:.2f}/s "
            f"boundary={'HARD' if boundary['hard_cutoff'] else 'ok'}",
            flush=True,
        )
        if index % 5 == 0:
            write_json(REPORT, partial_report(scored))

    del model
    source_reader.audio = None
    gc.collect()
    torch.cuda.empty_cache()

    by_source: dict[str, list[float]] = defaultdict(list)
    for row in scored:
        by_source[row["source_id"]].append(float(row["nisqa_tts_mos"]))
    source_p20 = {
        source_id: float(np.percentile(values, 20.0))
        for source_id, values in by_source.items()
    }

    accepted: list[dict] = []
    quality_review: list[dict] = []
    reason_counts = Counter()
    for row in scored:
        acoustic = row["acoustic"]
        reasons = []
        if float(acoustic["clipping_fraction"]) > 0.0:
            reasons.append("clipping")
        if float(acoustic["clicks_per_sec"]) > 3.0:
            reasons.append("clicks")
        if float(acoustic["snr_proxy_db"]) < 5.0:
            reasons.append("low_snr_proxy")
        if float(row["nisqa_tts_delta"]) < -0.20:
            reasons.append("nisqa_source_delta_below_minus_0p20")
        if float(row["nisqa_tts_mos"]) < source_p20[row["source_id"]]:
            reasons.append("below_source_p20")
        if row["boundary"]["hard_cutoff"]:
            reasons.append("severe_boundary_cutoff")
        row["source_nisqa_p20"] = source_p20[row["source_id"]]
        row["quality_gate_reasons"] = reasons
        row["quality_gate_pass"] = not reasons
        reason_counts.update(reasons)
        if row["quality_gate_pass"]:
            accepted.append(row)
        else:
            quality_review.append(row)

    audit_rows = []
    for row in accepted:
        candidate = dict(row["candidate"])
        candidate["quality_v2"] = {
            "nisqa_tts_mos": row["nisqa_tts_mos"],
            "source_interval_nisqa_tts_mos": row["source_interval_nisqa_tts_mos"],
            "nisqa_tts_delta": row["nisqa_tts_delta"],
            "source_p20": row["source_nisqa_p20"],
            "acoustic": row["acoustic"],
            "boundary": row["boundary"],
            "gate_pass": True,
        }
        candidate["audio_rel"] = (
            "../full_semantic_mining_v2/" + row["candidate"]["audio_rel"]
        )
        audit_rows.append(candidate)

    write_json(
        AUDIT,
        {
            "version": "full_semantic_quality_v2_audit",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "policy": {
                "speaker_boundary_input": "full_semantic_mining_v2 machine-strict review pool",
                "nisqa": "GPU NISQA-TTS source-local 20th percentile plus same-source-interval delta >= -0.20",
                "acoustic_hard_veto": "no clipping, <=3 click groups/sec, SNR proxy >=5 dB",
                "boundary_hard_veto": f"source-edge energy ratio < {BOUNDARY_HARD_RATIO:.2f}",
                "status": "quality-strong machine review pool; still requires human listening",
            },
            "sample_count": len(audit_rows),
            "samples": audit_rows,
        },
    )
    QUALITY_LIST.write_text(
        "/n".join(
            f"{row['candidate']['audio']}|ireina_quality_strict_review_v2|JA|{row['candidate']['text']}"
            for row in accepted
        )
        + ("/n" if accepted else ""),
        encoding="utf-8",
    )
    QUALITY_REVIEW_LIST.write_text(
        "/n".join(
            f"{row['candidate']['audio']}|ireina_quality_review_v2|JA|{row['candidate']['text']}"
            for row in quality_review
        )
        + ("/n" if quality_review else ""),
        encoding="utf-8",
    )

    report = {
        "version": "full_semantic_quality_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime_sec": time.time() - started,
        "device": "cuda:0",
        "local_resources": RESOURCE_POLICY,
        "input_count": len(source_rows),
        "input_duration_sec": sum(float(row["output_duration"]) for row in source_rows),
        "quality_strict_count": len(accepted),
        "quality_strict_duration_sec": sum(float(row["duration"]) for row in accepted),
        "quality_review_count": len(quality_review),
        "quality_review_duration_sec": sum(float(row["duration"]) for row in quality_review),
        "source_p20": source_p20,
        "rejection_reason_counts": dict(reason_counts),
        "policy_note": (
            "NISQA-TTS is a relative automatic proxy, not a human MOS and not a 90-point claim. "
            "Failed clips are moved to review, not deleted. Severe source-edge energy is treated "
            "as a likely cut-head/cut-tail veto learned from the prior human boundary rounds."
        ),
        "rows": scored,
        "quality_review": quality_review,
        "outputs": {
            "audit": str(AUDIT),
            "list": str(QUALITY_LIST),
            "review_list": str(QUALITY_REVIEW_LIST),
        },
        "training_started": False,
        "score_90_claimed": False,
    }
    write_json(REPORT, report)
    print(
        json.dumps(
            {
                "input": len(source_rows),
                "quality_strict": len(accepted),
                "quality_strict_sec": report["quality_strict_duration_sec"],
                "quality_review": len(quality_review),
                "rejection_reason_counts": dict(reason_counts),
                "report": str(REPORT),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
