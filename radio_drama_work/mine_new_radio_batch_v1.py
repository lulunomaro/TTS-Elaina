# -*- coding: utf-8 -*-
"""Mine target-speaker training clips from the new BS-Roformer vocals.

The order is deliberate:
    MP4 mixed audio -> BS-Roformer vocals -> sliding speaker score -> ASR
    word boundaries -> clear non-target word cuts -> other-stem veto.

This script creates a new candidate list only.  It does not modify any
existing training list and it never reconstructs audio with ClearVoice.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from faster_whisper import WhisperModel


ROOT = Path(r"radio_drama_work/radio_drama_work")
PKG = Path(r"radio_drama_work/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50")
BATCH = ROOT / "new_radio_batch_v1"
VOCALS = BATCH / "vocals"
OTHER = BATCH / "other"
OUT = ROOT / "new_radio_quality_v1"
ACCEPTED = OUT / "accepted_segments"
REVIEW = OUT / "review"
REJECTED = OUT / "rejected"
REPORT = OUT / "report.json"
VALIDATION = OUT / "validation.json"
CANDIDATE_LIST = ROOT / "new_radio_quality_train_candidate_v1.list"

SAMPLE_RATE = 32000
WORD_WINDOW_SEC = 1.20
SLIDE_WINDOW_SEC = 2.00
SLIDE_HOP_SEC = 1.00
MIN_SEC = 1.50
MAX_SEC = 8.00
WORD_GAP_SEC = 0.45
ANCHOR_MERGE_GAP_SEC = 1.15
EDGE_PAD_SEC = 0.055
OTHER_RATIO_MAX = 0.80

# These are calibrated against the existing 168-reference bank.  The
# sliding window is the anchor; word scores are used to cut clear negatives,
# not to require every short ASR token to independently pass.
HIGH_CENTER_MIN = 0.74
HIGH_P25_MIN = 0.50
HIGH_MARGIN_MIN = 0.01
REVIEW_CENTER_MIN = 0.68
REVIEW_P25_MIN = 0.44


def load_audio(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav.contiguous()


def rms(wav: torch.Tensor) -> float:
    if wav.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(wav.float() ** 2) + 1e-12))


def other_ratio(vocal: torch.Tensor, other: torch.Tensor, start: float, end: float) -> float:
    a = max(0, int(round(start * SAMPLE_RATE)))
    b = min(vocal.shape[1], other.shape[1], int(round(end * SAMPLE_RATE)))
    if b <= a:
        return 0.0
    return rms(other[:, a:b]) / max(rms(vocal[:, a:b]), 1e-6)


def safe_text(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z一-龥ぁ-ゖァ-ヺー]", "_", text)[:28] or "speech"


def overlap(a: float, b: float, c: float, d: float) -> bool:
    return max(a, c) < min(b, d)


def window_wave(wav: torch.Tensor, start: float, length: float) -> torch.Tensor:
    n = int(round(length * SAMPLE_RATE))
    a = int(round(start * SAMPLE_RATE))
    b = a + n
    left = max(0, -a)
    right = max(0, b - wav.shape[1])
    a = max(0, a)
    b = min(wav.shape[1], b)
    clip = wav[:, a:b]
    if left or right:
        clip = torch.nn.functional.pad(clip, (left, right))
    if clip.shape[1] < n:
        clip = torch.nn.functional.pad(clip, (0, n - clip.shape[1]))
    return clip[:, :n]


def score_embeddings(embs: torch.Tensor, refs: torch.Tensor, negatives: torch.Tensor, center: torch.Tensor) -> dict[str, np.ndarray]:
    sims = embs @ refs.T
    target_center = (embs @ center).numpy()
    target_p25 = torch.quantile(sims, 0.25, dim=1).numpy()
    negative_max = (embs @ negatives.T).max(dim=1).values.numpy() if len(negatives) else np.full(len(embs), -1.0)
    return {
        "center": target_center,
        "p25": target_p25,
        "negative_max": negative_max,
        "margin": target_center - negative_max,
    }


def batched_score(waves: list[torch.Tensor], refs: torch.Tensor, negatives: torch.Tensor, center: torch.Tensor, embed_batch) -> dict[str, np.ndarray]:
    chunks = []
    for start in range(0, len(waves), 64):
        batch = torch.cat(waves[start : start + 64], dim=0)
        chunks.append(embed_batch(batch))
    embs = torch.cat(chunks, dim=0) if chunks else torch.zeros((0, refs.shape[1]))
    return score_embeddings(embs, refs, negatives, center)


def transcribe(asr: WhisperModel, path: Path) -> list[dict]:
    segments, _ = asr.transcribe(
        str(path), language="ja", beam_size=5, temperature=0.0,
        word_timestamps=True, vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=180),
        condition_on_previous_text=False, compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0, no_speech_threshold=0.6,
    )
    words = []
    for segment in segments:
        for word in segment.words or []:
            text = (word.word or "").strip()
            if text and word.end > word.start:
                words.append({
                    "text": text,
                    "start": float(word.start),
                    "end": float(word.end),
                    "probability": float(getattr(word, "probability", 0.0)),
                })
    return words


def score_words(wav: torch.Tensor, words: list[dict], refs: torch.Tensor, negatives: torch.Tensor, center: torch.Tensor, embed_batch) -> list[dict]:
    valid = [row for row in words if row["end"] - row["start"] >= 0.20]
    waves = [window_wave(wav, (row["start"] + row["end"]) / 2 - WORD_WINDOW_SEC / 2, WORD_WINDOW_SEC) for row in valid]
    stats = batched_score(waves, refs, negatives, center, embed_batch)
    output = []
    for i, word in enumerate(valid):
        row = dict(word)
        row.update({key: float(value[i]) for key, value in stats.items()})
        row["clear_non_target"] = bool(
            (row["center"] < 0.55 and row["margin"] < 0.0)
            or (row["center"] < 0.48 and row["p25"] < 0.38)
        )
        output.append(row)
    return output


def anchors(wav: torch.Tensor, refs: torch.Tensor, negatives: torch.Tensor, center: torch.Tensor, embed_batch) -> tuple[list[dict], list[dict]]:
    duration = wav.shape[1] / SAMPLE_RATE
    starts = list(np.arange(0.0, max(0.0, duration - SLIDE_WINDOW_SEC) + 1e-5, SLIDE_HOP_SEC))
    if not starts:
        starts = [0.0]
    if duration > SLIDE_WINDOW_SEC and starts[-1] + SLIDE_WINDOW_SEC < duration:
        starts.append(duration - SLIDE_WINDOW_SEC)
    waves = [window_wave(wav, start, SLIDE_WINDOW_SEC) for start in starts]
    stats = batched_score(waves, refs, negatives, center, embed_batch)
    rows = []
    high = []
    review = []
    for i, start in enumerate(starts):
        row = {
            "start": float(start),
            "end": float(min(duration, start + SLIDE_WINDOW_SEC)),
            "center": float(stats["center"][i]),
            "p25": float(stats["p25"][i]),
            "negative_max": float(stats["negative_max"][i]),
            "margin": float(stats["margin"][i]),
        }
        row["high_pass"] = bool(row["center"] >= HIGH_CENTER_MIN and row["p25"] >= HIGH_P25_MIN and row["margin"] >= HIGH_MARGIN_MIN)
        row["review_pass"] = bool(row["center"] >= REVIEW_CENTER_MIN and row["p25"] >= REVIEW_P25_MIN)
        rows.append(row)
        if row["high_pass"]:
            high.append(row)
        elif row["review_pass"]:
            review.append(row)
    return rows, high + review


def merge_anchor_rows(rows: list[dict]) -> list[tuple[float, float, list[dict]]]:
    if not rows:
        return []
    rows = sorted(rows, key=lambda row: row["start"])
    merged = [{"start": rows[0]["start"], "end": rows[0]["end"], "rows": [rows[0]]}]
    for row in rows[1:]:
        previous = merged[-1]
        if row["start"] - previous["end"] <= ANCHOR_MERGE_GAP_SEC:
            previous["end"] = max(previous["end"], row["end"])
            previous["rows"].append(row)
        else:
            merged.append({"start": row["start"], "end": row["end"], "rows": [row]})
    return [(item["start"], item["end"], item["rows"]) for item in merged]


def words_to_segments(words: list[dict], start: float, end: float) -> list[tuple[float, float, str, list[dict]]]:
    in_window = [word for word in words if overlap(word["start"], word["end"], start, end)]
    kept = [word for word in in_window if not word["clear_non_target"]]
    groups = []
    current = []
    for word in kept:
        if not current or (word["start"] - current[-1]["end"] <= WORD_GAP_SEC and word["end"] - current[0]["start"] <= MAX_SEC):
            current.append(word)
        else:
            groups.append(current)
            current = [word]
    if current:
        groups.append(current)
    output = []
    for group in groups:
        a = max(start, group[0]["start"] - EDGE_PAD_SEC)
        b = min(end, group[-1]["end"] + EDGE_PAD_SEC)
        if b - a >= MIN_SEC:
            output.append((a, b, "".join(word["text"] for word in group), group))
    return output


def write_clip(wav: torch.Tensor, start: float, end: float, path: Path) -> None:
    a = max(0, int(round(start * SAMPLE_RATE)))
    b = min(wav.shape[1], int(round(end * SAMPLE_RATE)))
    sf.write(str(path), wav[:, a:b].numpy().T, SAMPLE_RATE, subtype="PCM_16")


def validate(lines: list[str]) -> dict:
    paths = [Path(line.split("|", 1)[0]) for line in lines]
    missing = [str(path) for path in paths if not path.exists()]
    duplicates = [path for path, count in Counter(str(path) for path in paths).items() if count > 1]
    short = []
    for path in paths:
        if path.exists() and sf.info(str(path)).duration < MIN_SEC:
            short.append({"path": str(path), "duration": sf.info(str(path)).duration})
    result = {
        "candidate_count": len(lines),
        "missing_paths": missing,
        "duplicate_paths": duplicates,
        "short_outputs": short,
        "passed": not missing and not duplicates and not short,
    }
    VALIDATION.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    for path in (VOCALS, OTHER):
        if not path.exists():
            raise FileNotFoundError(f"Missing separated directory: {path}")
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (ACCEPTED, REVIEW, REJECTED):
        path.mkdir(parents=True, exist_ok=True)

    os.chdir(PKG)
    sys.path.insert(0, str(ROOT))
    import word_target_hardcut as sv  # noqa: E402

    refs_paths = sv.load_refs()
    negative_paths = sv.load_negatives()
    refs = torch.cat([sv.embed_file(path) for path in refs_paths], dim=0)
    negatives = torch.cat([sv.embed_file(path) for path in negative_paths], dim=0) if negative_paths else torch.zeros((1, refs.shape[1]))
    center = torch.nn.functional.normalize(refs.mean(0), dim=-1)
    if torch.cuda.is_available():
        asr = WhisperModel(str(sv.ASR_MODEL), device="cuda", compute_type="float16", device_index=0)
        asr_runtime = {"device": "cuda", "compute_type": "float16"}
    else:
        asr = WhisperModel(str(sv.ASR_MODEL), device="cpu", compute_type="int8", cpu_threads=8)
        asr_runtime = {"device": "cpu", "compute_type": "int8", "cpu_threads": 8}

    report = {
        "version": "new_radio_quality_v1",
        "inputs": [],
        "reference_count": len(refs_paths),
        "negative_count": len(negative_paths),
        "asr_runtime": asr_runtime,
        "parameters": {
            "high_center_min": HIGH_CENTER_MIN,
            "high_p25_min": HIGH_P25_MIN,
            "high_margin_min": HIGH_MARGIN_MIN,
            "review_center_min": REVIEW_CENTER_MIN,
            "review_p25_min": REVIEW_P25_MIN,
            "word_window_sec": WORD_WINDOW_SEC,
            "slide_window_sec": SLIDE_WINDOW_SEC,
            "slide_hop_sec": SLIDE_HOP_SEC,
            "other_ratio_max": OTHER_RATIO_MAX,
        },
        "files": [],
    }
    list_lines = []
    file_index = 0

    source_paths = sorted(VOCALS.glob("*_vocals.wav"))
    for source in source_paths:
        other_path = OTHER / source.name.replace("_vocals.wav", "_other.wav")
        print(f"[source] {source.name}", flush=True)
        wav = load_audio(source)
        other = load_audio(other_path) if other_path.exists() else None
        words = transcribe(asr, source)
        word_rows = score_words(wav, words, refs, negatives, center, sv.embed_batch)
        slide_rows, review_rows = anchors(wav, refs, negatives, center, sv.embed_batch)
        high_rows = [row for row in slide_rows if row["high_pass"]]
        review_only = [row for row in review_rows if not row["high_pass"]]
        accepted_windows = merge_anchor_rows(high_rows)
        review_windows = merge_anchor_rows(review_only)
        counters = Counter()
        outputs = []
        seen_intervals = []

        for start, end, anchor_rows in accepted_windows:
            for a, b, text, group in words_to_segments(word_rows, start, end):
                if any(overlap(a, b, x, y) for x, y in seen_intervals):
                    continue
                ratio = other_ratio(wav, other, a, b) if other is not None else 0.0
                if ratio > OTHER_RATIO_MAX:
                    counters["other_stem_dominant"] += 1
                    continue
                seen_intervals.append((a, b))
                path = ACCEPTED / f"newradio_{file_index:05d}_{source.stem}_{int(round(a * SAMPLE_RATE)):010d}_{int(round(b * SAMPLE_RATE)):010d}.wav"
                write_clip(wav, a, b, path)
                row = {
                    "file": str(path), "start": a, "end": b, "duration": b - a,
                    "text": text, "other_ratio": ratio,
                    "anchor_count": len(anchor_rows),
                    "anchor_best_center": max(item["center"] for item in anchor_rows),
                    "anchor_min_p25": min(item["p25"] for item in anchor_rows),
                }
                outputs.append(row)
                list_lines.append(f"{path.resolve()}|new_radio_quality_v1|JA|{text}")
                file_index += 1

        # Keep a bounded review pack from borderline windows.  Review is not
        # included in the candidate list and is explicitly marked as such.
        review_written = 0
        for row in sorted(review_rows, key=lambda item: (item["center"], item["p25"]), reverse=True):
            if review_written >= 80:
                break
            path = REVIEW / f"{source.stem}_review_{review_written:04d}_{int(round(row['start'] * SAMPLE_RATE)):010d}.wav"
            write_clip(wav, row["start"], row["end"], path)
            row["file"] = str(path)
            review_written += 1

        source_report = {
            "source": str(source),
            "other_source": str(other_path) if other_path.exists() else None,
            "duration": wav.shape[1] / SAMPLE_RATE,
            "asr_word_count": len(words),
            "scored_word_count": len(word_rows),
            "sliding_window_count": len(slide_rows),
            "high_anchor_count": len(high_rows),
            "high_anchor_duration": sum(row["end"] - row["start"] for row in high_rows),
            "accepted_segment_count": len(outputs),
            "accepted_duration": sum(row["duration"] for row in outputs),
            "review_window_count": len(review_rows),
            "review_written": review_written,
            "reject_counts": dict(counters),
            "segments": outputs,
            "sliding_scores": slide_rows,
            "word_scores": word_rows,
        }
        report["inputs"].append(str(source))
        report["files"].append(source_report)
        print(
            f"  duration={source_report['duration']:.1f}s words={len(words)} high_anchors={len(high_rows)} accepted={len(outputs)} sec={source_report['accepted_duration']:.1f}",
            flush=True,
        )

    CANDIDATE_LIST.write_text("/n".join(list_lines) + ("/n" if list_lines else ""), encoding="utf-8")
    report["candidate_count"] = len(list_lines)
    report["candidate_duration"] = sum(item["accepted_duration"] for item in report["files"])
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    validation = validate(list_lines)
    print(f"saved_report={REPORT}", flush=True)
    print(f"saved_list={CANDIDATE_LIST} count={len(list_lines)} sec={report['candidate_duration']:.1f}", flush=True)
    print(f"validation_passed={validation['passed']}", flush=True)
    if not validation["passed"]:
        raise RuntimeError(f"Validation failed: {validation}")


if __name__ == "__main__":
    main()
