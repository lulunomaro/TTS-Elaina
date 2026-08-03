# -*- coding: utf-8 -*-
"""Export only stable target-speaker runs from the speaker-aware report."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import soundfile as sf
import torch
import torchaudio


ROOT = Path(r"radio_drama_work/radio_drama_work")
V1_REPORT = ROOT / "new_radio_quality_v1" / "report.json"
SA_REPORT = ROOT / "new_radio_speaker_aware_v1" / "speaker_aware_report.json"
V3 = ROOT / "new_radio_quality_v3"
ACCEPTED = V3 / "accepted_segments"
REVIEW = V3 / "review"
REPORT_OUT = V3 / "report.json"
VALIDATION_OUT = V3 / "validation.json"
LIST_OUT = ROOT / "new_radio_quality_train_candidate_v3.list"
SR = 32000
MIN_SEC = 1.5
EDGE = 0.03
OTHER_RATIO_MAX = 0.80

BALANCED = (0.76, 0.62, 0.52, 0.05)
RECALL_REVIEW = (0.74, 0.60, 0.50, 0.03)
HARD_EXCLUDE = {
    "newradio_00001_243358894-1-192_vocals_0003417760_0003505760.wav",
    "newradio_00003_243358894-1-192_vocals_0007518240_0007613280.wav",
    "newradio_00004_243358894-1-192_vocals_0007968000_0008032000.wav",
}


def load_audio(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.contiguous()


def rms(wav: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(wav.float() ** 2) + 1e-12)) if wav.numel() else 0.0


def other_ratio(vocal: torch.Tensor, other: torch.Tensor | None, start: float, end: float) -> float:
    if other is None:
        return 0.0
    a = max(0, int(round(start * SR)))
    b = min(vocal.shape[1], other.shape[1], int(round(end * SR)))
    if b <= a:
        return 0.0
    return rms(other[:, a:b]) / max(rms(vocal[:, a:b]), 1e-6)


def active_runs(rows: list[dict], config: tuple[float, float, float, float]) -> list[dict]:
    center_min, median_min, p25_min, margin_min = config
    active = [
        row["center"] >= center_min
        and row["median"] >= median_min
        and row["p25"] >= p25_min
        and row.get("target_negative_margin", 1.0) >= margin_min
        for row in rows
    ]
    result = []
    start = None
    for index, yes in enumerate(active + [False]):
        if yes and start is None:
            start = index
        elif not yes and start is not None:
            chosen = rows[start:index]
            run = {
                "start": chosen[0]["start"],
                "end": chosen[-1]["end"],
                "duration": chosen[-1]["end"] - chosen[0]["start"],
                "windows": len(chosen),
                "min_center": min(row["center"] for row in chosen),
                "min_median": min(row["median"] for row in chosen),
                "min_p25": min(row["p25"] for row in chosen),
                "min_margin": min(row.get("target_negative_margin", 1.0) for row in chosen),
            }
            if run["windows"] >= 2 and run["duration"] >= MIN_SEC:
                result.append(run)
            start = None
    return result


def trim_to_words(run: dict, words: list[dict]) -> tuple[float, float, str] | None:
    inside = [word for word in words if word["end"] > run["start"] and word["start"] < run["end"]]
    if not inside:
        return None
    # A clear negative word breaks the stable run instead of being silently
    # included.  Borderline words remain inside the already stable speaker run.
    if any(word.get("clear_non_target", False) for word in inside):
        return None
    start = max(run["start"], inside[0]["start"] - EDGE)
    end = min(run["end"], inside[-1]["end"] + EDGE)
    if end - start < MIN_SEC:
        return None
    return start, end, "".join(word["text"] for word in inside)


def validate(lines: list[str]) -> dict:
    paths = [Path(line.split("|", 1)[0]) for line in lines]
    missing = [str(path) for path in paths if not path.exists()]
    duplicates = [name for name, count in Counter(str(path) for path in paths).items() if count > 1]
    short = []
    for path in paths:
        if path.exists() and sf.info(str(path)).duration < MIN_SEC:
            short.append(str(path))
    result = {"candidate_count": len(lines), "missing_paths": missing, "duplicate_paths": duplicates, "short_outputs": short, "hard_exclude_hits": [line for line in lines if any(name in line for name in HARD_EXCLUDE)], "passed": not missing and not duplicates and not short and not any(name in line for line in lines for name in HARD_EXCLUDE)}
    VALIDATION_OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    v1 = json.loads(V1_REPORT.read_text(encoding="utf-8"))
    speaker = json.loads(SA_REPORT.read_text(encoding="utf-8"))
    word_by_source = {Path(item["source"]).name: item.get("word_scores", []) for item in v1["files"]}
    for path in (ACCEPTED, REVIEW):
        path.mkdir(parents=True, exist_ok=True)

    report = {"version": "new_radio_quality_v3", "method": "BS-Roformer vocals + 1.5s/0.5s stable speaker windows + ASR boundary trim", "balanced_thresholds": dict(zip(("center", "median", "p25", "margin"), BALANCED)), "review_thresholds": dict(zip(("center", "median", "p25", "margin"), RECALL_REVIEW)), "v1_status": "invalidated after user-confirmed false positives and word-level audit", "files": []}
    lines = []
    index = 0
    for item in speaker["files"]:
        source_name = item["file"]
        source_text = next((row.get("source_path") for row in item["windows"] if row.get("source_path")), None)
        source = Path(source_text) if source_text else None
        if source is None or not source.is_file():
            source = ROOT / "new_radio_batch_v1" / "vocals" / source_name
        other = source.parent.parent / "other" / source.name.replace("_vocals.wav", "_other.wav")
        wav = load_audio(source)
        other_wav = load_audio(other) if other.exists() else None
        words = word_by_source.get(source.name, [])
        balanced_runs = active_runs(item["windows"], BALANCED)
        review_runs = active_runs(item["windows"], RECALL_REVIEW)
        accepted = []
        rejected = Counter()
        for run in balanced_runs:
            trimmed = trim_to_words(run, words)
            if trimmed is None:
                rejected["no_clean_word_boundary"] += 1
                continue
            start, end, text = trimmed
            ratio = other_ratio(wav, other_wav, start, end)
            if ratio > OTHER_RATIO_MAX:
                rejected["other_stem_dominant"] += 1
                continue
            output = ACCEPTED / f"stable_{index:05d}_{source.stem}_{int(round(start * SR)):010d}_{int(round(end * SR)):010d}.wav"
            if output.name in HARD_EXCLUDE:
                rejected["hard_excluded"] += 1
                continue
            a, b = int(round(start * SR)), int(round(end * SR))
            sf.write(str(output), wav[:, a:b].numpy().T, SR, subtype="PCM_16")
            row = {"file": str(output), "start": start, "end": end, "duration": end - start, "text": text, "other_ratio": ratio, "stable_windows": run["windows"], "min_center": run["min_center"], "min_median": run["min_median"], "min_p25": run["min_p25"], "min_margin": run["min_margin"]}
            accepted.append(row)
            lines.append(f"{output.resolve()}|new_radio_quality_v3|JA|{text}")
            index += 1

        # Review-only recall intervals are exported for listening, but never
        # enter the training list.  Keep the three user-confirmed false v1
        # intervals out even from this review convenience pack.
        review_written = 0
        for run in review_runs:
            if review_written >= 80:
                break
            trimmed = trim_to_words(run, words)
            if trimmed is None:
                continue
            start, end, text = trimmed
            name = f"review_{review_written:04d}_{source.stem}_{int(round(start * SR)):010d}_{int(round(end * SR)):010d}.wav"
            if any(token in name for token in HARD_EXCLUDE):
                continue
            output = REVIEW / name
            a, b = int(round(start * SR)), int(round(end * SR))
            sf.write(str(output), wav[:, a:b].numpy().T, SR, subtype="PCM_16")
            review_written += 1

        report["files"].append({"source": str(source), "duration": wav.shape[1] / SR, "balanced_runs": len(balanced_runs), "accepted_count": len(accepted), "accepted_duration": sum(row["duration"] for row in accepted), "review_runs": len(review_runs), "review_written": review_written, "reject_counts": dict(rejected), "segments": accepted})
        print(f"{source.name} balanced_runs={len(balanced_runs)} accepted={len(accepted)} sec={sum(row['duration'] for row in accepted):.1f} review_runs={len(review_runs)}", flush=True)

    LIST_OUT.write_text("/n".join(lines) + ("/n" if lines else ""), encoding="utf-8")
    report["candidate_count"] = len(lines)
    report["candidate_duration"] = sum(item["accepted_duration"] for item in report["files"])
    REPORT_OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    validation = validate(lines)
    print(f"saved_report={REPORT_OUT}", flush=True)
    print(f"saved_list={LIST_OUT} count={len(lines)} sec={report['candidate_duration']:.1f}", flush=True)
    print(f"validation_passed={validation['passed']}", flush=True)


if __name__ == "__main__":
    main()
