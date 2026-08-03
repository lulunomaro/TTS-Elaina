# -*- coding: utf-8 -*-
"""Prepare direct-file HTML manifest roots for the two salvage review pools."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SEMANTIC = ROOT / "full_semantic_mining_v1"
QUALITY = ROOT / "full_semantic_quality_v1"
BOUNDARY_OUT = ROOT / "full_semantic_boundary_review_v1"
QUALITY_REVIEW_OUT = ROOT / "full_semantic_quality_review_v1"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8"
    )


def main() -> None:
    boundary = json.loads(
        (SEMANTIC / "boundary_review_manifest.json").read_text(encoding="utf-8")
    )
    boundary_rows = []
    for row in boundary["samples"]:
        item = dict(row)
        item["audio_rel"] = "../full_semantic_mining_v1/" + row["audio_rel"]
        boundary_rows.append(item)
    write_json(
        BOUNDARY_OUT / "audit_manifest.json",
        {
            "version": "full_semantic_boundary_review_v1_audit",
            "policy": {
                "reason": "speaker words passed, but duration/boundary/acoustic completeness needs human rescue review",
                "status": "review only; not training-approved",
            },
            "sample_count": len(boundary_rows),
            "samples": boundary_rows,
        },
    )

    quality = json.loads((QUALITY / "report.json").read_text(encoding="utf-8"))
    quality_rows = []
    for scored in quality["quality_review"]:
        item = dict(scored["candidate"])
        item["audio_rel"] = "../full_semantic_mining_v1/" + item["audio_rel"]
        item["quality_review_reasons"] = scored["quality_gate_reasons"]
        item["nisqa_tts_mos"] = scored["nisqa_tts_mos"]
        item["source_nisqa_p20"] = scored["source_nisqa_p20"]
        item["acoustic_v1"] = scored["acoustic"]
        quality_rows.append(item)
    write_json(
        QUALITY_REVIEW_OUT / "audit_manifest.json",
        {
            "version": "full_semantic_quality_review_v1_audit",
            "policy": {
                "reason": "speaker and boundary passed, but source-relative NISQA/acoustic gate requested review",
                "status": "review only; not deleted and not training-approved",
            },
            "sample_count": len(quality_rows),
            "samples": quality_rows,
        },
    )
    print(
        json.dumps(
            {
                "boundary_review": len(boundary_rows),
                "quality_review": len(quality_rows),
                "boundary_root": str(BOUNDARY_OUT),
                "quality_review_root": str(QUALITY_REVIEW_OUT),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
