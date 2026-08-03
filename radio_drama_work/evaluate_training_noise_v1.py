# -*- coding: utf-8 -*-
"""Live multi-detector quality gate for the overnight training candidate set."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(r"radio_drama_work/radio_drama_work")
INPUT_LIST = ROOT / "full_funasr_gold_train_candidate_v9.list"
WORK = ROOT / "vasr_overnight_search_v1"
STATE_PATH = WORK / "state.json"
REPORT_PATH = WORK / "training_noise_quality_live.json"
STATUS_PATH = WORK / "status.json"
NOISE_LIST = ROOT / "full_funasr_gold_overnight_noise_gated_score90_v1.list"
ALL_LIST = ROOT / "full_funasr_gold_overnight_best_noise_checked_v1.list"
LOG_PATH = WORK / "noise_gate.log"


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "/n")


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def read(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def robust_z(values: np.ndarray) -> np.ndarray:
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return (values - med) / max(1.4826 * mad, 1e-8)


def frame_matrix(audio: np.ndarray, frame: int = 1024, hop: int = 256) -> np.ndarray:
    count = max(1, 1 + max(0, len(audio) - frame) // hop)
    padded = np.pad(audio, (0, max(0, frame + (count - 1) * hop - len(audio))))
    return np.stack([padded[i * hop : i * hop + frame] for i in range(count)])


def detector_metrics(audio: np.ndarray, sr: int) -> dict:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    frames = frame_matrix(audio)
    window = np.hanning(frames.shape[1]).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)) + 1e-8
    power = spectrum * spectrum
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    rms_db = 20.0 * np.log10(rms + 1e-7)
    flatness = np.exp(np.mean(np.log(spectrum), axis=1)) / np.mean(spectrum, axis=1)
    flux = np.sqrt(np.mean(np.diff(np.log(spectrum), axis=0) ** 2, axis=1)) if len(spectrum) > 1 else np.zeros(1)
    diff = np.abs(np.diff(audio))
    diff_z = robust_z(diff)
    # A normal voiced consonant can have many large first differences.  A
    # possible click is therefore counted only at a much more extreme robust
    # threshold; later we also compare it with the original clean stem.
    spike_mask = diff_z > 40.0
    spike_positions = np.flatnonzero(spike_mask)
    if len(spike_positions):
        groups = 1 + int(np.sum(np.diff(spike_positions) > max(8, round(sr * 0.004))))
    else:
        groups = 0
    duration = len(audio) / sr
    spikes_per_sec = groups / max(duration, 0.1)
    flux_z = robust_z(flux)
    flux_outlier_fraction = float(np.mean(flux_z > 8.0)) if len(flux_z) else 0.0
    frequencies = np.fft.rfftfreq(frames.shape[1], 1.0 / sr)
    high_mask = frequencies >= min(8000.0, sr * 0.25)
    high_ratio = np.sum(power[:, high_mask], axis=1) / np.maximum(np.sum(power, axis=1), 1e-8)
    high_burst_fraction = float(np.mean(robust_z(high_ratio) > 8.0)) if len(high_ratio) else 0.0
    low_energy = rms <= np.percentile(rms, 20.0)
    speech_energy = max(float(np.percentile(rms, 85.0)), 1e-7)
    floor_energy = max(float(np.percentile(rms[low_energy], 50.0)) if np.any(low_energy) else float(np.percentile(rms, 20.0)), 1e-7)
    snr_db = float(20.0 * np.log10(speech_energy / floor_energy))
    floor_flatness = float(np.mean(flatness[low_energy])) if np.any(low_energy) else float(np.mean(flatness))
    q10, q90 = np.percentile(rms_db, [10.0, 90.0])
    dynamic_db = float(q90 - q10)
    env = np.convolve(np.abs(audio), np.ones(max(3, round(sr * 0.04))) / max(3, round(sr * 0.04)), mode="same")[::256]
    env_delta = float(np.median(np.abs(np.diff(env))) / max(np.percentile(env, 75.0), 1e-7)) if len(env) > 1 else 0.0
    spectral_delta = float(np.median(np.abs(np.diff(np.log(spectrum), axis=0)))) if len(spectrum) > 1 else 0.0
    clipping_fraction = float(np.mean(np.abs(audio) >= 0.999))
    return {
        "click_groups": groups,
        "spikes_per_sec": spikes_per_sec,
        "spectral_flux_outlier_fraction": flux_outlier_fraction,
        "high_frequency_burst_fraction": high_burst_fraction,
        "snr_proxy_db": snr_db,
        "low_energy_spectral_flatness": floor_flatness,
        "dynamic_range_db": dynamic_db,
        "envelope_delta": env_delta,
        "spectral_delta": spectral_delta,
        "clipping_fraction": clipping_fraction,
    }


def natural_similarity(raw: np.ndarray, candidate: np.ndarray, sr: int, raw_det: dict, cand_det: dict) -> dict:
    length = min(len(raw), len(candidate))
    raw = raw[:length].astype(np.float32)
    candidate = candidate[:length].astype(np.float32)
    kernel = max(3, round(sr * 0.04))
    window = np.ones(kernel, dtype=np.float32) / kernel
    raw_env = np.convolve(np.abs(raw), window, mode="same")[::256]
    cand_env = np.convolve(np.abs(candidate), window, mode="same")[::256]
    if raw_env.std() < 1e-8 or cand_env.std() < 1e-8:
        envelope_similarity = 0.0
    else:
        envelope_similarity = float(np.clip((np.corrcoef(raw_env, cand_env)[0, 1] + 1.0) / 2.0, 0.0, 1.0))
    raw_spec = np.log1p(np.abs(librosa.stft(raw, n_fft=1024, hop_length=256))).mean(axis=1)
    cand_spec = np.log1p(np.abs(librosa.stft(candidate, n_fft=1024, hop_length=256))).mean(axis=1)
    raw_spec = raw_spec / max(np.linalg.norm(raw_spec), 1e-8)
    cand_spec = cand_spec / max(np.linalg.norm(cand_spec), 1e-8)
    spectral_similarity = float(np.clip(np.dot(raw_spec, cand_spec), 0.0, 1.0))
    dynamic_similarity = float(np.clip(math.exp(-abs(cand_det["dynamic_range_db"] - raw_det["dynamic_range_db"]) / 18.0), 0.0, 1.0))
    return {
        "envelope_similarity": envelope_similarity,
        "spectral_similarity": spectral_similarity,
        "dynamic_similarity": dynamic_similarity,
        "naturalness_proxy": 0.45 * envelope_similarity + 0.35 * spectral_similarity + 0.20 * dynamic_similarity,
    }


def score_audio(audio: np.ndarray, sr: int, state_metrics: dict, raw_audio: np.ndarray | None = None, raw_sr: int | None = None) -> dict:
    detectors = detector_metrics(audio, sr)
    if raw_audio is None:
        raw_audio = audio
        raw_sr = sr
    if raw_sr != sr:
        raw_audio = librosa.resample(raw_audio, orig_sr=raw_sr, target_sr=sr).astype(np.float32)
    baseline_det = detector_metrics(raw_audio, sr)
    click_excess = max(0.0, detectors["spikes_per_sec"] - baseline_det["spikes_per_sec"] - 2.0)
    flux_excess = max(0.0, detectors["spectral_flux_outlier_fraction"] - baseline_det["spectral_flux_outlier_fraction"] - 0.08)
    high_excess = max(0.0, detectors["high_frequency_burst_fraction"] - baseline_det["high_frequency_burst_fraction"] - 0.12)
    click_score = float(np.clip(math.exp(-click_excess / 2.0), 0.0, 1.0))
    impulse_score = float(np.clip(math.exp(-6.0 * flux_excess - 2.0 * high_excess), 0.0, 1.0))
    absolute_floor = float(np.clip((detectors["snr_proxy_db"] - 10.0) / 18.0, 0.0, 1.0))
    relative_floor = float(np.clip((detectors["snr_proxy_db"] - baseline_det["snr_proxy_db"] + 8.0) / 16.0, 0.0, 1.0))
    floor_score = 0.65 * absolute_floor + 0.35 * relative_floor
    clipping_score = float(np.clip(1.0 - detectors["clipping_fraction"] / 0.001, 0.0, 1.0))
    natural = natural_similarity(raw_audio, audio, sr, baseline_det, detectors)
    target_center = float(state_metrics.get("target_center", 0.0))
    margin = state_metrics.get("target_negative_margin")
    target_component = float(np.clip((target_center - 0.40) / 0.40, 0.0, 1.0))
    margin_component = float(np.clip((float(margin) + 0.05) / 0.15, 0.0, 1.0)) if margin is not None else 0.5
    speaker_score = 22.0 * target_component + 18.0 * margin_component
    noise_score = 30.0 * (0.40 * click_score + 0.30 * impulse_score + 0.20 * floor_score + 0.10 * clipping_score)
    natural_score = 20.0 * natural["naturalness_proxy"]
    format_score = 10.0 if sr == 32000 else 0.0
    total = speaker_score + noise_score + natural_score + format_score
    components = {
        "speaker_identity": speaker_score,
        "noise_artifact_control": noise_score,
        "natural_continuity": natural_score,
        "format": format_score,
    }
    hard_gate = (
        total >= 90.0
        and noise_score >= 25.0
        and natural_score >= 17.0
        and detectors["clipping_fraction"] <= 0.001
        and detectors["spikes_per_sec"] <= max(6.0, baseline_det["spikes_per_sec"] + 3.0)
        and detectors["spectral_flux_outlier_fraction"] <= max(0.15, baseline_det["spectral_flux_outlier_fraction"] + 0.10)
        and detectors["high_frequency_burst_fraction"] <= max(0.55, baseline_det["high_frequency_burst_fraction"] + 0.20)
        and detectors["snr_proxy_db"] >= max(10.0, baseline_det["snr_proxy_db"] - 10.0)
    )
    return {
        "score_100": float(total),
        "components": components,
        "hard_gate_score90": hard_gate,
        "detectors": {
            **detectors,
            "baseline_spikes_per_sec": baseline_det["spikes_per_sec"],
            "baseline_spectral_flux_outlier_fraction": baseline_det["spectral_flux_outlier_fraction"],
            "baseline_high_frequency_burst_fraction": baseline_det["high_frequency_burst_fraction"],
            "baseline_snr_proxy_db": baseline_det["snr_proxy_db"],
            "click_excess": click_excess,
            "flux_excess": flux_excess,
            "high_frequency_excess": high_excess,
            "click_score": click_score,
            "impulse_score": impulse_score,
            "floor_score": floor_score,
            **natural,
        },
    }


def evaluate_once() -> dict:
    if not STATE_PATH.exists():
        return {"status": "waiting_for_state"}
    payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    rows = payload.get("rows", {})
    source_map = {}
    for line in INPUT_LIST.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            source_map[len(source_map)] = Path(line.split("|", 1)[0].strip())
    evaluated = []
    all_lines = []
    gated_lines = []
    for key in sorted(rows, key=lambda value: int(value)):
        row = rows[key]
        path = Path(row["best_path"])
        item = {"index": int(key), "path": str(path), "text": row.get("text", ""), "search_score": row.get("best_score"), "status": "missing"}
        if path.exists():
            try:
                audio, sr = read(path)
                raw_path = source_map.get(int(key))
                raw_audio, raw_sr = read(raw_path) if raw_path and raw_path.exists() else (audio, sr)
                quality = score_audio(audio, sr, row.get("metrics", {}), raw_audio, raw_sr)
                item.update({"status": "ok", "duration_sec": len(audio) / sr, "quality": quality})
                all_lines.append(f"{path}|full_funasr_gold_overnight_noise_checked_v1|JA|{row.get('text','')}")
                if quality["hard_gate_score90"]:
                    gated_lines.append(f"{path}|full_funasr_gold_overnight_noise_gated_score90_v1|JA|{row.get('text','')}")
            except Exception as exc:
                item["error"] = repr(exc)
        evaluated.append(item)
    report = {
        "updated_at": time.time(),
        "search_deadline": payload.get("deadline"),
        "detector_policy": {
            "score": "speaker identity 40 + noise/artifact 30 + natural continuity 20 + format 10",
            "score90_gate": "total >= 90, noise/artifact >= 25, natural continuity >= 17, no clipping, click rate no more than clean-stem baseline + 3/sec, spectral-flux outliers no more than clean-stem baseline + 0.10",
            "note": "This is an automatic acoustic gate, not a claim of human listening perfection or a guaranteed TTS benchmark score.",
        },
        "count": len(evaluated),
        "ok_count": sum(item["status"] == "ok" for item in evaluated),
        "score90_count": sum(item.get("quality", {}).get("hard_gate_score90", False) for item in evaluated),
        "total_sec": sum(item.get("duration_sec", 0.0) for item in evaluated),
        "score90_sec": sum(item.get("duration_sec", 0.0) for item in evaluated if item.get("quality", {}).get("hard_gate_score90", False)),
        "noise_checked_list": str(ALL_LIST),
        "noise_gated_score90_list": str(NOISE_LIST),
        "rows": evaluated,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    ALL_LIST.write_text("/n".join(all_lines) + ("/n" if all_lines else ""), encoding="utf-8")
    NOISE_LIST.write_text("/n".join(gated_lines) + ("/n" if gated_lines else ""), encoding="utf-8")
    log(f"evaluated={len(evaluated)} score90={report['score90_count']} score90_sec={report['score90_sec']:.2f}")
    return report


def main() -> None:
    if os.environ.get("NOISE_GATE_ONCE") == "1":
        evaluate_once()
        return
    while True:
        deadline = None
        if STATE_PATH.exists():
            try:
                deadline = float(json.loads(STATE_PATH.read_text(encoding="utf-8")).get("deadline", 0.0))
            except Exception:
                deadline = None
        evaluate_once()
        if deadline and time.time() >= deadline:
            log("deadline_reached")
            break
        time.sleep(300)


if __name__ == "__main__":
    main()
