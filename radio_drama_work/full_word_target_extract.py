# -*- coding: utf-8 -*-
"""Full-radio word-level target-speaker extraction.

This is a versioned, high-precision extraction pass.  It reads the already
separated vocal stems, scores Japanese ASR words against the target reference
bank, rejects known contaminated intervals, and writes clips from the
original vocal waveform.  It never runs source reconstruction and never edits
the existing training lists.
"""
from __future__ import annotations

import json
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
sys.path.insert(0, str(ROOT))
import word_target_hardcut as base  # noqa: E402


INPUTS = [
    ROOT / "new_vocals" / "1311215796_u1-1-192_vocals.wav",
    ROOT / "new_vocals" / "1405440540-1-192_vocals.wav",
    ROOT / "new_vocals" / "28055832533-1-192_vocals.wav",
    ROOT / "vocals" / "280479210-1-192_vocals.wav",
    ROOT / "vocals" / "113814583-1-160_vocals.wav",
]
OTHER_INPUTS = {
    "1311215796_u1-1-192_vocals.wav": ROOT / "new_ins" / "1311215796_u1-1-192_other.wav",
    "1405440540-1-192_vocals.wav": ROOT / "new_ins" / "1405440540-1-192_other.wav",
    "28055832533-1-192_vocals.wav": ROOT / "new_ins" / "28055832533-1-192_other.wav",
    "280479210-1-192_vocals.wav": ROOT / "ins" / "280479210-1-192_other.wav",
    "113814583-1-160_vocals.wav": ROOT / "ins" / "113814583-1-160_other.wav",
}

OUT = ROOT / "full_word_target_v1"
ACCEPTED_DIR = OUT / "accepted_segments"
REVIEW_DIR = OUT / "review"
REJECTED_DIR = OUT / "rejected"
REPORT_PATH = OUT / "report.json"
VALIDATION_PATH = OUT / "validation.json"
CANDIDATE_LIST = ROOT / "full_word_target_train_candidate.list"

POSITIVE_LIST = ROOT / "confirmed_target_pure.txt"
NEGATIVE_LIST = ROOT / "confirmed_non_target_or_mixed.txt"
SAMPLE_RATE = 32000
INDEX_RATE = 32000

# Conservative gates.  The direct word gate is the primary acceptance gate;
# local frames and clustering can only veto a word, never relax it.
WORD_WINDOW_SEC = 1.20
FRAME_SEC = 0.80
FRAME_HOP_SEC = 0.20
MIN_WORD_SEC = 0.20
MIN_FINAL_SEC = 1.50
MERGE_GAP_SEC = 0.22
EDGE_PAD_SEC = 0.055
FADE_SEC = 0.025
WORD_CENTER_MIN = 0.72
WORD_P25_MIN = 0.50
WORD_MARGIN_MIN = 0.02
FRAME_CENTER_MIN = 0.68
FRAME_P25_MIN = 0.43
FRAME_MARGIN_MIN = 0.06
FRAME_VOTE_MIN = 0.40
CLUSTER_VOTE_MIN = 0.40
CLUSTER_MARGIN_MIN = 0.025
# The other stem is a diagnostic for residual BGM/non-vocal energy.  A
# clearly dominant other stem vetoes a clip; ordinary low-level residual is
# retained in the report for audit.
OTHER_RATIO_MAX = 0.80
REVIEW_LIMIT_PER_SOURCE = 60
BATCH_SIZE = 96

EXCLUDED_BASENAMES = {
    "28055832533-1-192_vocals_0013268480_0013539200_safe_word_000.wav",
    "28055832533-1-192_vocals_0013268480_0013539200_safe_word_001.wav",
}


def load_audio_32(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav.contiguous()


def safe_text(text: str) -> str:
    text = re.sub(r"[^0-9A-Za-zぁ-んァ-ヶ一-龯々ー]", "_", text)
    return text[:24] or "word"


def rms(wav: torch.Tensor) -> float:
    if wav.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(wav.float() ** 2) + 1e-12))


def other_ratio(vocal: torch.Tensor, other: torch.Tensor | None, start: int, end: int) -> float:
    if other is None:
        return 0.0
    start = max(0, start)
    end = min(vocal.shape[1], end, other.shape[1])
    if end <= start:
        return 0.0
    return rms(other[:, start:end]) / max(rms(vocal[:, start:end]), 1e-6)


def load_asr() -> WhisperModel:
    return WhisperModel(
        str(base.ASR_MODEL),
        device="cpu",
        compute_type="int8",
        cpu_threads=8,
    )


def transcribe(asr: WhisperModel, path: Path) -> list[dict]:
    segments, _info = asr.transcribe(
        str(path),
        language="ja",
        beam_size=5,
        temperature=0.0,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=180),
        condition_on_previous_text=False,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
    )
    words: list[dict] = []
    for segment in segments:
        for word in segment.words or []:
            text = (word.word or "").strip()
            if text and word.end > word.start:
                words.append(
                    {
                        "text": text,
                        "start": float(word.start),
                        "end": float(word.end),
                        "probability": float(getattr(word, "probability", 0.0)),
                    }
                )
    return words


def batched_embeddings(waves: list[torch.Tensor]) -> torch.Tensor:
    if not waves:
        return torch.zeros((0, 256), dtype=torch.float32)
    chunks = []
    for start in range(0, len(waves), BATCH_SIZE):
        batch = torch.cat(waves[start : start + BATCH_SIZE], dim=0)
        chunks.append(base.embed_batch(batch))
    return torch.cat(chunks, dim=0)


def score_embeddings(embs: torch.Tensor, refs: torch.Tensor, negatives: torch.Tensor, center: torch.Tensor) -> dict[str, np.ndarray]:
    vals = embs @ refs.T
    target_center = (embs @ center).numpy()
    target_p25 = torch.quantile(vals, 0.25, dim=1).numpy()
    negative_max = (embs @ negatives.T).max(dim=1).values.numpy() if len(negatives) else np.full(len(embs), -1.0)
    return {
        "target_center": target_center,
        "target_p25": target_p25,
        "negative_max": negative_max,
        "margin": target_center - negative_max,
    }


def spherical_kmeans(x: torch.Tensor, iters: int = 30) -> tuple[torch.Tensor, torch.Tensor]:
    if len(x) < 2:
        return torch.zeros(len(x), dtype=torch.long), x[:1]
    first = 0
    second = int(torch.argmin(x[first] @ x.T).item())
    centers = torch.stack([x[first], x[second]], dim=0)
    for _ in range(iters):
        assignment = (x @ centers.T).argmax(dim=1)
        new_centers = centers.clone()
        for k in range(2):
            rows = x[assignment == k]
            if len(rows):
                new_centers[k] = torch.nn.functional.normalize(rows.mean(0), dim=-1)
        if torch.allclose(new_centers, centers, atol=1e-5):
            break
        centers = new_centers
    return assignment, centers


def source_for_name(name: str) -> str | None:
    for path in INPUTS:
        if path.name in name:
            return path.name
    return None


def parse_forbidden_intervals() -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {path.name: [] for path in INPUTS}
    if not NEGATIVE_LIST.exists():
        return result
    for raw in NEGATIVE_LIST.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#") or "|" not in raw:
            continue
        path_text, reason = [part.strip() for part in raw.split("|", 1)]
        source_name = source_for_name(Path(path_text).name)
        if source_name is None:
            continue
        numbers = [int(value) for value in re.findall(r"/d{10}", Path(path_text).name)]
        if len(numbers) >= 4:
            start = (numbers[-4] + numbers[-2]) / INDEX_RATE
            end = (numbers[-4] + numbers[-1]) / INDEX_RATE
        elif len(numbers) >= 2:
            start = numbers[-2] / INDEX_RATE
            end = numbers[-1] / INDEX_RATE
        else:
            continue
        result[source_name].append({"start": start, "end": end, "reason": reason, "source": path_text})
    return result


def overlaps(start: float, end: float, intervals: list[dict]) -> bool:
    return any(max(start, item["start"]) < min(end, item["end"]) for item in intervals)


def window_wave(wav: torch.Tensor, center_sec: float, length_sec: float) -> torch.Tensor:
    length = int(round(length_sec * SAMPLE_RATE))
    center = int(round(center_sec * SAMPLE_RATE))
    start = center - length // 2
    end = start + length
    left = max(0, -start)
    right = max(0, end - wav.shape[1])
    start = max(0, start)
    end = min(wav.shape[1], end)
    clip = wav[:, start:end]
    if left or right:
        clip = torch.nn.functional.pad(clip, (left, right))
    if clip.shape[1] < length:
        clip = torch.nn.functional.pad(clip, (0, length - clip.shape[1]))
    return clip[:, :length]


def score_words(wav: torch.Tensor, words: list[dict], refs: torch.Tensor, negatives: torch.Tensor, center: torch.Tensor) -> list[dict]:
    valid = [word for word in words if word["end"] - word["start"] >= MIN_WORD_SEC]
    windows = [window_wave(wav, (word["start"] + word["end"]) / 2.0, WORD_WINDOW_SEC) for word in valid]
    scores = batched_embeddings(windows)
    stats = score_embeddings(scores, refs, negatives, center)
    result = []
    for index, word in enumerate(valid):
        row = dict(word)
        row.update({key: float(values[index]) for key, values in stats.items()})
        row["direct_pass"] = bool(
            row["target_center"] >= WORD_CENTER_MIN
            and row["target_p25"] >= WORD_P25_MIN
            and row["margin"] >= WORD_MARGIN_MIN
        )
        result.append(row)
    return result


def score_frames(wav: torch.Tensor, refs: torch.Tensor, negatives: torch.Tensor, center: torch.Tensor) -> dict:
    frame_len = int(round(FRAME_SEC * SAMPLE_RATE))
    hop = int(round(FRAME_HOP_SEC * SAMPLE_RATE))
    starts = list(range(0, max(1, wav.shape[1] - frame_len + 1), hop))
    if wav.shape[1] > frame_len and starts[-1] + frame_len < wav.shape[1]:
        starts.append(wav.shape[1] - frame_len)
    clips = []
    centers_sec = []
    for start in starts:
        clip = wav[:, start : min(wav.shape[1], start + frame_len)]
        if clip.shape[1] < frame_len:
            clip = torch.nn.functional.pad(clip, (0, frame_len - clip.shape[1]))
        clips.append(clip)
        centers_sec.append((start + frame_len / 2) / SAMPLE_RATE)
    embs = batched_embeddings(clips)
    stats = score_embeddings(embs, refs, negatives, center)
    frame_like = (
        (stats["target_center"] >= FRAME_CENTER_MIN)
        & (stats["target_p25"] >= FRAME_P25_MIN)
        & (stats["margin"] >= FRAME_MARGIN_MIN)
    )
    assignments, cluster_centers = spherical_kmeans(embs)
    cluster_target_scores = cluster_centers @ center if len(cluster_centers) else torch.zeros(1)
    target_cluster = int(torch.argmax(cluster_target_scores).item()) if len(cluster_centers) else 0
    cluster_margin = (
        float(cluster_target_scores[target_cluster] - cluster_target_scores[1 - target_cluster])
        if len(cluster_centers) == 2
        else 0.0
    )
    return {
        "centers": np.asarray(centers_sec, dtype=np.float32),
        "assignments": assignments.numpy(),
        "frame_like": frame_like,
        "frame_center": stats["target_center"],
        "cluster_target": target_cluster,
        "cluster_margin": cluster_margin,
        "cluster_scores": [float(value) for value in cluster_target_scores],
    }


def add_frame_gate(rows: list[dict], frame_data: dict) -> None:
    centers = frame_data["centers"]
    assignments = frame_data["assignments"]
    frame_like = frame_data["frame_like"]
    target_cluster = frame_data["cluster_target"]
    for row in rows:
        mid = (row["start"] + row["end"]) / 2
        indices = np.where((centers >= row["start"] - 0.28) & (centers <= row["end"] + 0.28))[0]
        if len(indices) == 0:
            indices = np.asarray([int(np.argmin(np.abs(centers - mid)))])
        row["frame_vote"] = float(np.mean(frame_like[indices]))
        row["cluster_vote"] = float(np.mean(assignments[indices] == target_cluster))
        row["frame_center"] = float(np.mean(frame_data["frame_center"][indices]))
        # The word-level target gate is primary.  Frame target_like can be
        # zero on a confirmed pure clip because the 0.8 s frame also contains
        # silence, BGM residue, or an ASR boundary.  Use the local cluster as
        # a veto for a clearly different speaker instead of requiring every
        # frame to independently pass the target gate.
        row["frame_pass"] = bool(
            row["cluster_vote"] >= CLUSTER_VOTE_MIN
            and frame_data["cluster_margin"] >= CLUSTER_MARGIN_MIN
        )


def accepted_words(rows: list[dict], forbidden: list[dict], vocal: torch.Tensor, other: torch.Tensor | None) -> tuple[list[dict], Counter]:
    reasons = Counter()
    for row in rows:
        row["forbidden"] = overlaps(row["start"], row["end"], forbidden)
        st = int(round(max(0.0, row["start"] - EDGE_PAD_SEC) * SAMPLE_RATE))
        en = int(round(min(vocal.shape[1] / SAMPLE_RATE, row["end"] + EDGE_PAD_SEC) * SAMPLE_RATE))
        row["other_ratio"] = float(other_ratio(vocal, other, st, en))
        row["other_pass"] = bool(row["other_ratio"] <= OTHER_RATIO_MAX)
        row["accepted"] = bool(row["direct_pass"] and row["frame_pass"] and not row["forbidden"] and row["other_pass"])
        if row["accepted"]:
            continue
        if row["forbidden"]:
            reasons["confirmed_forbidden_interval"] += 1
        elif not row["direct_pass"]:
            reasons["target_similarity_gate"] += 1
        elif not row["frame_pass"]:
            reasons["local_frame_or_cluster_gate"] += 1
        elif not row["other_pass"]:
            reasons["other_stem_dominant"] += 1
        else:
            reasons["other"] += 1
    return [row for row in rows if row["accepted"]], reasons


def merge_accepted(accepted: list[dict], all_rows: list[dict], forbidden: list[dict]) -> list[dict]:
    merged: list[dict] = []
    accepted = sorted(accepted, key=lambda row: row["start"])
    for row in accepted:
        if not merged:
            merged.append({"start": row["start"], "end": row["end"], "words": [row]})
            continue
        previous = merged[-1]
        gap = row["start"] - previous["end"]
        between_rejected = any(
            not candidate["accepted"]
            and candidate["start"] < row["start"]
            and candidate["end"] > previous["end"]
            for candidate in all_rows
        )
        if gap <= MERGE_GAP_SEC and not between_rejected and not overlaps(previous["end"], row["start"], forbidden):
            previous["end"] = row["end"]
            previous["words"].append(row)
        else:
            merged.append({"start": row["start"], "end": row["end"], "words": [row]})
    return merged


def clip_from_source(wav: torch.Tensor, start: float, end: float) -> tuple[torch.Tensor, float, float]:
    start = max(0.0, start - EDGE_PAD_SEC)
    end = min(wav.shape[1] / SAMPLE_RATE, end + EDGE_PAD_SEC)
    st = int(round(start * SAMPLE_RATE))
    en = int(round(end * SAMPLE_RATE))
    return wav[:, st:en], start, end


def write_review(wav: torch.Tensor, rows: list[dict], source_name: str) -> None:
    candidates = [
        row for row in rows
        if not row["accepted"]
        and not row["forbidden"]
        and row.get("target_center", -1.0) >= 0.70
        and row.get("target_p25", -1.0) >= 0.43
        and row["end"] - row["start"] >= 0.80
    ]
    candidates.sort(key=lambda row: (row.get("target_center", 0.0), row.get("margin", 0.0)), reverse=True)
    for index, row in enumerate(candidates[:REVIEW_LIMIT_PER_SOURCE]):
        clip, start, end = clip_from_source(wav, row["start"], row["end"])
        path = REVIEW_DIR / f"{Path(source_name).stem}_review_{index:03d}_{safe_text(row['text'])}.wav"
        sf.write(str(path), clip.numpy().T, SAMPLE_RATE, subtype="PCM_16")
        row["review_file"] = str(path)


def validate() -> dict:
    lines = [line.strip() for line in CANDIDATE_LIST.read_text(encoding="utf-8").splitlines() if line.strip()] if CANDIDATE_LIST.exists() else []
    paths = [Path(line.split("|", 1)[0]) for line in lines]
    safe_hits = [line for line in lines if "safe_word_000.wav" in line or "safe_word_001.wav" in line]
    missing = [str(path) for path in paths if not path.exists()]
    duplicate_paths = [str(path) for path, count in Counter(str(path) for path in paths).items() if count > 1]
    short = []
    for path in paths:
        try:
            info = sf.info(str(path))
            if info.duration < MIN_FINAL_SEC:
                short.append({"path": str(path), "duration": info.duration})
        except Exception as exc:  # pragma: no cover - validation report
            missing.append(f"{path}: {exc}")
    validation = {
        "candidate_count": len(lines),
        "missing_paths": missing,
        "duplicate_paths": duplicate_paths,
        "safe_word_hits": safe_hits,
        "short_outputs": short,
        "passed": not missing and not duplicate_paths and not safe_hits and not short,
    }
    VALIDATION_PATH.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    return validation


def main() -> None:
    missing_inputs = [str(path) for path in INPUTS if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError("Missing input audio: " + ", ".join(missing_inputs))
    OUT.mkdir(parents=True, exist_ok=True)
    ACCEPTED_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    REJECTED_DIR.mkdir(parents=True, exist_ok=True)

    forbidden_by_source = parse_forbidden_intervals()
    refs_paths = base.load_refs()
    negative_paths = base.load_negatives()
    refs = torch.cat([base.embed_file(path) for path in refs_paths], dim=0)
    negatives = torch.cat([base.embed_file(path) for path in negative_paths], dim=0) if negative_paths else torch.zeros((1, refs.shape[1]))
    center = torch.nn.functional.normalize(refs.mean(0), dim=-1)
    asr = load_asr()
    all_list_lines: list[str] = []
    report = {
        "version": "full_word_target_v1",
        "note": "High-precision word-level target gate on original separated vocal waveforms; not source reconstruction.",
        "inputs": [str(path) for path in INPUTS],
        "excluded_basenames": sorted(EXCLUDED_BASENAMES),
        "reference_files": len(refs_paths),
        "reference_embeddings": len(refs),
        "negative_files": len(negative_paths),
        "parameters": {
            "word_window_sec": WORD_WINDOW_SEC,
            "frame_sec": FRAME_SEC,
            "frame_hop_sec": FRAME_HOP_SEC,
            "min_final_sec": MIN_FINAL_SEC,
            "merge_gap_sec": MERGE_GAP_SEC,
            "word_center_min": WORD_CENTER_MIN,
            "word_p25_min": WORD_P25_MIN,
            "word_margin_min": WORD_MARGIN_MIN,
            "frame_vote_min": FRAME_VOTE_MIN,
            "cluster_margin_min": CLUSTER_MARGIN_MIN,
            "other_ratio_max": OTHER_RATIO_MAX,
        },
        "forbidden_intervals": forbidden_by_source,
        "files": [],
    }

    print(f"device={base.device} refs={len(refs)} negatives={len(negative_paths)}", flush=True)
    print(f"inputs={len(INPUTS)} output={OUT}", flush=True)
    for source in INPUTS:
        print(f"[source] {source.name}", flush=True)
        vocal = load_audio_32(source)
        other_path = OTHER_INPUTS.get(source.name)
        other = load_audio_32(other_path) if other_path and other_path.exists() else None
        words = transcribe(asr, source)
        print(f"  asr_words={len(words)} duration={vocal.shape[1] / SAMPLE_RATE:.1f}s", flush=True)
        rows = score_words(vocal, words, refs, negatives, center)
        frame_data = score_frames(vocal, refs, negatives, center)
        add_frame_gate(rows, frame_data)
        accepted, reject_counts = accepted_words(rows, forbidden_by_source[source.name], vocal, other)
        merged = merge_accepted(accepted, rows, forbidden_by_source[source.name])
        final_segments = []
        short_count = 0
        for index, segment in enumerate(merged):
            if segment["end"] - segment["start"] < MIN_FINAL_SEC:
                short_count += 1
                continue
            if overlaps(segment["start"], segment["end"], forbidden_by_source[source.name]):
                reject_counts["merged_forbidden_interval"] += 1
                continue
            clip, start, end = clip_from_source(vocal, segment["start"], segment["end"])
            text = "".join(word["text"] for word in segment["words"])
            output = ACCEPTED_DIR / f"{source.stem}_target_{len(final_segments):04d}.wav"
            if output.name in EXCLUDED_BASENAMES:
                raise RuntimeError(f"Forbidden output name generated: {output.name}")
            sf.write(str(output), clip.numpy().T, SAMPLE_RATE, subtype="PCM_16")
            final_segments.append(
                {
                    "file": str(output),
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "text": text,
                    "word_count": len(segment["words"]),
                    "words": segment["words"],
                }
            )
            all_list_lines.append(f"{output.resolve()}|full_word_target|JA|{text}")
        if short_count:
            reject_counts["merged_segment_too_short"] += short_count
        write_review(vocal, rows, source.name)
        REJECTED_DIR.joinpath(f"{source.stem}_rejected.json").write_text(
            json.dumps([row for row in rows if not row["accepted"]], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        file_report = {
            "source": str(source),
            "other_source": str(other_path) if other_path and other_path.exists() else None,
            "duration": vocal.shape[1] / SAMPLE_RATE,
            "asr_word_count": len(words),
            "scored_word_count": len(rows),
            "accepted_word_count": len(accepted),
            "final_segment_count": len(final_segments),
            "final_duration": sum(segment["duration"] for segment in final_segments),
            "reject_counts": dict(reject_counts),
            "cluster_margin": frame_data["cluster_margin"],
            "cluster_scores": frame_data["cluster_scores"],
            "segments": final_segments,
            "scored_words": rows,
        }
        report["files"].append(file_report)
        print(
            f"  scored={len(rows)} accepted_words={len(accepted)} final_segments={len(final_segments)} final_sec={file_report['final_duration']:.1f}",
            flush=True,
        )

    CANDIDATE_LIST.write_text("/n".join(all_list_lines) + ("/n" if all_list_lines else ""), encoding="utf-8")
    report["candidate_count"] = len(all_list_lines)
    report["candidate_duration"] = sum(segment["final_duration"] for item in report["files"] for segment in [item])
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    validation = validate()
    print(f"saved_report={REPORT_PATH}", flush=True)
    print(f"saved_list={CANDIDATE_LIST} count={len(all_list_lines)}", flush=True)
    print(f"validation_passed={validation['passed']}", flush=True)
    if not validation["passed"]:
        raise RuntimeError("Validation failed; see " + str(VALIDATION_PATH))


if __name__ == "__main__":
    main()
