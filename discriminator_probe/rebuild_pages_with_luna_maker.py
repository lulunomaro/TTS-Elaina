# -*- coding: utf-8 -*-
"""Regenerate v3 priority review + boundary rescue pages using Luna's full
review-page generator (waveform, progress, autosave, export JSON).

Reads discriminator_probe manifests, computes 32-bin waveforms, then calls
radio_drama_work/make_human_review_page_v2.py for each page. read-only on
radio_drama_work (import only, no writes there).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PROBE = Path(r"F:\GPTSOVITS\discriminator_probe")
ROOT = Path(r"F:\GPTSOVITS\radio_drama_work")
MAKER = ROOT / "make_human_review_page_v2.py"
PY = r"F:\GPTSOVITS\GPT-SoVITS-v2pro-20250604-nvidia50\GPT-SoVITS-v2pro-20250604-nvidia50\runtime\python.exe"
WAVE_BINS = 32


def waveform(path: Path) -> list[float]:
    audio, _ = sf.read(str(path), dtype="float32", always_2d=True)
    mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
    if not len(mono):
        return [0.0] * WAVE_BINS
    edges = np.linspace(0, len(mono), WAVE_BINS + 1, dtype=int)
    values = []
    for i in range(WAVE_BINS):
        chunk = mono[edges[i]:edges[i + 1]]
        values.append(float(np.sqrt(np.mean(np.square(chunk))) if len(chunk) else 0.0))
    scale = max(max(values), 1e-6)
    return [round(min(1.0, v / scale), 4) for v in values]


def build_audit(root: Path, samples: list[dict]) -> Path:
    rows = []
    for r in samples:
        audio_path = Path(r["audio"])
        rows.append({
            "blind_id": r["blind_id"],
            "audio_rel": f"audio/{audio_path.name}",
            "output_duration": round(float(r.get("duration", 0.0)), 3),
            "wave": waveform(audio_path),
        })
    audit = {"version": "audit", "samples": rows}
    (root / "audit_manifest.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return root / "audit_manifest.json"


def run_maker(root: Path, title: str, round_label: str, store_key: str,
              feedback_version: str, download_prefix: str, extra_note: str = "") -> None:
    cmd = [
        PY, str(MAKER),
        "--root", str(root),
        "--title", title,
        "--round-label", round_label,
        "--store-key", store_key,
        "--feedback-version", feedback_version,
        "--download-prefix", download_prefix,
        "--extra-note", extra_note,
    ]
    print("running:", " ".join(cmd[-10:]), flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        print("STDERR:", result.stderr[-1500:], flush=True)
        raise RuntimeError(result.stderr[-500:])
    print(result.stdout.strip(), flush=True)


def main() -> None:
    # --- page 1: v3 priority review ---
    v3 = PROBE / "discriminator_priority_review_v3"
    manifest = json.loads((v3 / "private_manifest.json").read_text(encoding="utf-8"))
    samples = manifest.get("samples") or manifest.get("rows") or []
    build_audit(v3, samples)
    run_maker(
        v3,
        "判别器v3·第三批48条试听",
        "Discriminator v3 · round 3 · 48 clips",
        "ireina_discriminator_v3_review",
        "discriminator_priority_review_v3_feedback",
        "discriminator_priority_review_v3_feedback",
        "本批判别器已用全部 249 条人工标签重训（AUC 0.9857）。判定：完整伊蕾娜 / 非目标 / 混合 / 切头切尾 / 音质 / 重叠 / 不确定。",
    )

    # --- page 2: boundary rescue v2 ---
    br = PROBE / "boundary_rescue_v2"
    audit = json.loads((br / "audit_manifest.json").read_text(encoding="utf-8"))
    br_samples = audit.get("samples") or []
    build_audit(br, br_samples)
    run_maker(
        br,
        "v2边界救援·38问题76变体",
        "Boundary rescue v2 · 38 origins · 76 variants",
        "ireina_boundary_rescue_v2_review",
        "boundary_rescue_v2_feedback",
        "boundary_rescue_v2_feedback",
        "每个原问题两个版本：word_expand（补相邻ASR词）/ pad_220ms（波形外扩）。只保留声音完整自然的版本；同一问题两个都过则保留更完整者。",
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
