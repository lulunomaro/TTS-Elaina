# -*- coding: utf-8 -*-
"""GPU NISQA-TTS and independent acoustic scoring for semantic candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
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

import run_full_local_mining_v1 as quality_base


INPUT = ROOT / "full_semantic_mining_v1" / "audit_manifest.json"
OUT = ROOT / "full_semantic_quality_v1"
REPORT = OUT / "report.json"
AUDIT = OUT / "audit_manifest.json"
QUALITY_LIST = ROOT / "full_semantic_quality_strict_review_v1.list"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8"
    )


def file_fingerprint(path: Path) -> str:
    stat = path.stat()
    payload = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|nisqa_tts_v1"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    source_audit = json.loads(INPUT.read_text(encoding="utf-8"))
    source_rows = source_audit["samples"]
    cached = {}
    if args.resume and REPORT.is_file():
        try:
            cached = {
                row["fingerprint"]: row
                for row in json.loads(REPORT.read_text(encoding="utf-8")).get("rows", [])
                if row.get("fingerprint")
            }
        except Exception:
            cached = {}

    model = quality_base.load_nisqa_tts()
    scored = []
    for index, row in enumerate(source_rows, start=1):
        path = Path(row["audio"])
        fingerprint = file_fingerprint(path)
        if fingerprint in cached:
            item = dict(cached[fingerprint])
            item["candidate"] = row
            scored.append(item)
            print(f"[{index}/{len(source_rows)}] resume {row['blind_id']}", flush=True)
            continue
        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        acoustic = quality_base.acoustic_metrics(audio, int(sample_rate))
        mos = quality_base.tts_mos(model, np.asarray(audio, dtype=np.float32))
        item = {
            "blind_id": row["blind_id"],
            "path": str(path),
            "source_id": row["source_id"],
            "duration": row["output_duration"],
            "fingerprint": fingerprint,
            "nisqa_tts_mos": float(mos),
            "acoustic": acoustic,
            "candidate": row,
        }
        scored.append(item)
        print(
            f"[{index}/{len(source_rows)}] {row['blind_id']} mos={mos:.3f} "
            f"clicks={acoustic['clicks_per_sec']:.2f}/s",
            flush=True,
        )
        if index % 10 == 0:
            write_json(
                REPORT,
                {
                    "version": "full_semantic_quality_v1_partial",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "rows": scored,
                },
            )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    by_source: dict[str, list[float]] = defaultdict(list)
    for row in scored:
        by_source[row["source_id"]].append(float(row["nisqa_tts_mos"]))
    source_p20 = {
        source_id: float(np.percentile(values, 20.0))
        for source_id, values in by_source.items()
    }

    accepted = []
    quality_review = []
    for row in scored:
        acoustic = row["acoustic"]
        reasons = []
        if float(acoustic["clipping_fraction"]) > 0.0:
            reasons.append("clipping")
        if float(acoustic["clicks_per_sec"]) > 3.0:
            reasons.append("clicks")
        if float(acoustic["snr_proxy_db"]) < 5.0:
            reasons.append("low_snr_proxy")
        if float(row["nisqa_tts_mos"]) < source_p20[row["source_id"]]:
            reasons.append("below_source_p20")
        row["source_nisqa_p20"] = source_p20[row["source_id"]]
        row["quality_gate_reasons"] = reasons
        row["quality_gate_pass"] = not reasons
        if row["quality_gate_pass"]:
            accepted.append(row)
        else:
            quality_review.append(row)

    audit_rows = []
    for row in accepted:
        candidate = dict(row["candidate"])
        candidate["quality_v1"] = {
            "nisqa_tts_mos": row["nisqa_tts_mos"],
            "source_p20": row["source_nisqa_p20"],
            "acoustic": row["acoustic"],
            "gate_pass": True,
        }
        # The quality page is one directory beside full_semantic_mining_v1.
        candidate["audio_rel"] = (
            "../full_semantic_mining_v1/" + row["candidate"]["audio_rel"]
        )
        audit_rows.append(candidate)

    write_json(
        AUDIT,
        {
            "version": "full_semantic_quality_v1_audit",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "policy": {
                "speaker_boundary_input": "full_semantic_mining_v1 machine-strict pool",
                "nisqa": "NISQA-TTS source-local 20th percentile; relative ranking only",
                "acoustic_hard_veto": "no clipping, <=3 click groups/sec, SNR proxy >=5 dB",
                "status": "quality-strong machine review pool; still requires human listening",
            },
            "sample_count": len(audit_rows),
            "samples": audit_rows,
        },
    )
    QUALITY_LIST.write_text(
        "/n".join(
            f"{row['candidate']['audio']}|ireina_quality_strict_review_v1|JA|{row['candidate']['text']}"
            for row in accepted
        )
        + ("/n" if accepted else ""),
        encoding="utf-8",
    )

    report = {
        "version": "full_semantic_quality_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime_sec": time.time() - started,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "local_resources": RESOURCE_POLICY,
        "input_count": len(source_rows),
        "input_duration_sec": sum(float(row["output_duration"]) for row in source_rows),
        "quality_strict_count": len(accepted),
        "quality_strict_duration_sec": sum(float(row["duration"]) for row in accepted),
        "quality_review_count": len(quality_review),
        "quality_review_duration_sec": sum(float(row["duration"]) for row in quality_review),
        "source_p20": source_p20,
        "policy_note": (
            "NISQA-TTS is a relative automatic proxy, not a human MOS and not a 90-point claim. "
            "Bottom-source-quintile clips are moved to review, not deleted."
        ),
        "rows": scored,
        "quality_review": quality_review,
        "outputs": {
            "audit": str(AUDIT),
            "list": str(QUALITY_LIST),
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
                "report": str(REPORT),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
