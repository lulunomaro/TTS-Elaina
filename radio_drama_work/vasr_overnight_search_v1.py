# -*- coding: utf-8 -*-
"""Eight-hour AudioSR parameter search with a veto-style quality gate.

This process never replaces the source material.  It keeps the raw
BS-Roformer clip as a baseline and only promotes a generated waveform when
speaker identity, naturalness proxies, and artifact checks all improve or
remain within the allowed tolerance.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

os.environ.setdefault("HF_HOME", r"radio_drama_work/hf_cache")
sys.path.insert(0, r"radio_drama_work/vasr_light_shims")
sys.path.append(r"radio_drama_work/vasr_deps")
sys.path.append(r"radio_drama_work/versatile_audio_super_resolution-main/versatile_audio_super_resolution-main")

import transformers


class DummyRobertaTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


transformers.RobertaTokenizer = DummyRobertaTokenizer
transformers.RobertaConfig.from_pretrained = classmethod(
    lambda cls, *args, **kwargs: cls(max_position_embeddings=514, type_vocab_size=1)
)

PKG = Path(r"radio_drama_work/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50")
sys.path.insert(0, str(PKG / "GPT_SoVITS" / "eres2net"))
os.chdir(PKG)

from audiosr import build_model, super_resolution
import audiosr.pipeline as audiosr_pipeline
from ERes2NetV2 import ERes2NetV2
import kaldi as Kaldi

ROOT = Path(r"radio_drama_work/radio_drama_work")
INPUT_LIST = ROOT / "full_funasr_gold_train_candidate_v9.list"
WORK = ROOT / "vasr_overnight_search_v1"
PREPARED = WORK / "prepared_inputs"
WINNERS = WORK / "winners_32k"
STATE_PATH = WORK / "state.json"
REPORT_PATH = WORK / "final_report.json"
STATUS_PATH = WORK / "status.json"
LOG_PATH = WORK / "search.log"
ALL_LIST = ROOT / "full_funasr_gold_overnight_best_v1.list"
SCORE90_LIST = ROOT / "full_funasr_gold_overnight_score90_v1.list"
CKPT = ROOT / "hf_cache" / "hub" / "models--haoheliu--audiosr_speech" / "blobs" / "59fd60a9b93346c698788bac9b6077977a21733e7a78db9d3c3af3f1bdebc640"
REF_DIR = PKG / "logs" / "my_voice" / "5-wav32k"
NEG_LIST = ROOT / "confirmed_non_target_or_mixed.txt"
SV_CKPT = PKG / "GPT_SoVITS" / "pretrained_models" / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


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


def parse_inputs() -> list[tuple[Path, str]]:
    rows = []
    for line in INPUT_LIST.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            parts = line.split("|", 3)
            rows.append((Path(parts[0].strip()), parts[3] if len(parts) == 4 else ""))
    return rows


@torch.no_grad()
def embed(audio: np.ndarray, sr: int, sv: torch.nn.Module) -> torch.Tensor:
    wav = torch.from_numpy(audio).float().unsqueeze(0)
    if sr != 16000:
        wav = torch.nn.functional.interpolate(
            wav.unsqueeze(1),
            size=round(wav.shape[-1] * 16000 / sr),
            mode="linear",
            align_corners=False,
        ).squeeze(1)
    feat = torch.stack(
        [Kaldi.fbank(w.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0) for w in wav]
    ).to(DEVICE)
    return torch.nn.functional.normalize(sv.forward(feat), dim=-1).squeeze(0).cpu()


def rms_band(audio: np.ndarray, sr: int, lo: float, hi: float) -> float:
    spec = np.abs(librosa.stft(audio, n_fft=2048, hop_length=256, win_length=2048, center=True)) ** 2
    freqs = np.fft.rfftfreq(2048, 1.0 / sr)
    mask = (freqs >= lo) & (freqs < hi)
    return float(np.sqrt(np.mean(spec[mask]) + 1e-12)) if np.any(mask) else 0.0


def crest(audio: np.ndarray) -> float:
    return float(np.max(np.abs(audio)) / max(np.sqrt(np.mean(audio * audio) + 1e-12), 1e-8))


def artifact_metrics(audio: np.ndarray, sr: int) -> dict:
    low_hi = min(12000 if sr <= 32000 else 16000, sr / 2)
    high_lo = 12000 if sr <= 32000 else 16000
    high_hi = min(24000, sr / 2)
    low = rms_band(audio, sr, 0, low_hi)
    high = rms_band(audio, sr, high_lo, high_hi) if high_hi > high_lo else 0.0
    return {
        "upper_to_low_db": -120.0 if high <= 0 else float(20 * np.log10(high / max(low, 1e-8))),
        "peak": float(np.max(np.abs(audio))),
        "clipping_fraction": float(np.mean(np.abs(audio) >= 0.999)),
        "crest_factor": crest(audio),
    }


def naturalness_metrics(raw: np.ndarray, candidate: np.ndarray, sr: int) -> dict:
    length = min(len(raw), len(candidate))
    raw = raw[:length].astype(np.float32)
    candidate = candidate[:length].astype(np.float32)
    raw_abs = np.abs(raw)
    cand_abs = np.abs(candidate)
    kernel = max(3, round(sr * 0.04))
    kernel = kernel if kernel % 2 else kernel + 1
    window = np.ones(kernel, dtype=np.float32) / kernel
    raw_env = np.convolve(raw_abs, window, mode="same")[::256]
    cand_env = np.convolve(cand_abs, window, mode="same")[::256]
    if raw_env.std() < 1e-8 or cand_env.std() < 1e-8:
        envelope_corr = 0.0
    else:
        envelope_corr = float(np.corrcoef(raw_env, cand_env)[0, 1])
    raw_mfcc = librosa.feature.mfcc(y=raw, sr=sr, n_mfcc=20, n_fft=1024, hop_length=256, n_mels=40)
    cand_mfcc = librosa.feature.mfcc(y=candidate, sr=sr, n_mfcc=20, n_fft=1024, hop_length=256, n_mels=40)
    raw_vec = np.concatenate([raw_mfcc.mean(axis=1), raw_mfcc.std(axis=1)])
    cand_vec = np.concatenate([cand_mfcc.mean(axis=1), cand_mfcc.std(axis=1)])
    mfcc_cos = float(np.dot(raw_vec, cand_vec) / max(np.linalg.norm(raw_vec) * np.linalg.norm(cand_vec), 1e-8))
    raw_spec = np.log1p(np.abs(librosa.stft(raw, n_fft=1024, hop_length=256))[:385]).mean(axis=1)
    cand_spec = np.log1p(np.abs(librosa.stft(candidate, n_fft=1024, hop_length=256))[:385]).mean(axis=1)
    raw_spec = raw_spec / max(np.linalg.norm(raw_spec), 1e-8)
    cand_spec = cand_spec / max(np.linalg.norm(cand_spec), 1e-8)
    spectral_similarity = float(np.dot(raw_spec, cand_spec))
    envelope_corr = float(np.clip((envelope_corr + 1.0) / 2.0, 0.0, 1.0))
    mfcc_cos = float(np.clip((mfcc_cos + 1.0) / 2.0, 0.0, 1.0))
    spectral_similarity = float(np.clip(spectral_similarity, 0.0, 1.0))
    naturalness = 0.45 * envelope_corr + 0.30 * mfcc_cos + 0.25 * spectral_similarity
    return {
        "envelope_similarity": envelope_corr,
        "mfcc_similarity": mfcc_cos,
        "spectral_similarity": spectral_similarity,
        "naturalness_proxy": naturalness,
    }


def measure(audio: np.ndarray, sr: int, raw: np.ndarray, sv: torch.nn.Module, refs: torch.Tensor, center: torch.Tensor, neg: torch.Tensor | None) -> dict:
    emb = embed(audio, sr, sv)
    target_scores = emb @ refs.T
    negative_max = float((emb @ neg.T).max()) if neg is not None else None
    metrics = {
        "sample_rate": sr,
        "channels": 1,
        "duration_sec": len(audio) / sr,
        "target_center": float(emb @ center),
        "target_top4": float(torch.topk(target_scores, k=min(4, len(target_scores))).values.mean()),
        "target_median": float(target_scores.median()),
        "known_negative_max": negative_max,
        "target_negative_margin": None if negative_max is None else float(emb @ center) - negative_max,
    }
    metrics.update(artifact_metrics(audio, sr))
    metrics.update(naturalness_metrics(raw, audio, sr))
    return metrics


def score(metrics: dict) -> dict:
    target = float(np.clip((metrics["target_center"] - 0.40) / 0.40, 0.0, 1.0))
    margin = float(np.clip((metrics["target_negative_margin"] + 0.05) / 0.15, 0.0, 1.0)) if metrics["target_negative_margin"] is not None else 0.5
    natural = float(np.clip(metrics["naturalness_proxy"], 0.0, 1.0))
    hf = metrics["upper_to_low_db"]
    hf_score = 1.0 if hf <= -10.0 else float(np.clip(1.0 - (hf + 10.0) / 20.0, 0.0, 1.0))
    clip_score = 1.0 if metrics["clipping_fraction"] <= 0.001 else 0.0
    format_score = 1.0 if metrics["sample_rate"] == 32000 and metrics["channels"] == 1 else 0.0
    parts = {
        "speaker_identity": 22.0 * target + 18.0 * margin,
        "naturalness": 30.0 * natural,
        "artifact_control": 16.0 * hf_score + 4.0 * clip_score,
        "format": 10.0 * format_score,
    }
    return {"score": float(sum(parts.values())), "parts": parts}


def allowed(candidate: dict, raw: dict) -> tuple[bool, list[str]]:
    reasons = []
    if candidate["target_center"] < raw["target_center"] - 0.02:
        reasons.append("speaker_identity_drop")
    if candidate["target_negative_margin"] is not None and raw["target_negative_margin"] is not None and candidate["target_negative_margin"] < raw["target_negative_margin"] - 0.02:
        reasons.append("target_negative_margin_drop")
    if candidate["upper_to_low_db"] > max(-10.0, raw["upper_to_low_db"] + 10.0):
        reasons.append("high_frequency_artifact")
    if candidate["clipping_fraction"] > 0.001:
        reasons.append("clipping")
    if candidate["naturalness_proxy"] < raw["naturalness_proxy"] - 0.06:
        reasons.append("naturalness_proxy_drop")
    return not reasons, reasons


def build_prepared(inputs: list[tuple[Path, str]], pad_sec: float) -> dict[int, Path]:
    PREPARED.mkdir(parents=True, exist_ok=True)
    paths = {}
    for index, (path, _) in enumerate(inputs):
        target = PREPARED / f"{index:05d}_pad{round(pad_sec * 1000):03d}.wav"
        if not target.exists():
            raw, sr = read(path)
            prepared = librosa.resample(raw, orig_sr=sr, target_sr=48000).astype(np.float32) if sr != 48000 else raw
            prepared = np.pad(prepared, (0, round(pad_sec * 48000)))
            sf.write(str(target), prepared, 48000, subtype="PCM_16")
        paths[index] = target
    return paths


def save_state(state: dict, inputs: list[tuple[Path, str]], started: float, deadline: float, config_count: int, clip_count: int) -> None:
    state_payload = {
        "started_at": started,
        "deadline": deadline,
        "device": DEVICE,
        "config_count": config_count,
        "clip_count": clip_count,
        "rows": state,
    }
    atomic_json(STATE_PATH, state_payload)
    all_lines = []
    score_lines = []
    for index, (source_path, text) in enumerate(inputs):
        row = state.get(str(index))
        if not row:
            continue
        path = row["best_path"]
        all_lines.append(f"{path}|full_funasr_gold_overnight_best_v1|JA|{text}")
        if row["best_score"] >= 90.0:
            score_lines.append(f"{path}|full_funasr_gold_overnight_score90_v1|JA|{text}")
    ALL_LIST.write_text("/n".join(all_lines) + ("/n" if all_lines else ""), encoding="utf-8")
    SCORE90_LIST.write_text("/n".join(score_lines) + ("/n" if score_lines else ""), encoding="utf-8")
    summary = {
        "updated_at": time.time(),
        "deadline": deadline,
        "device": DEVICE,
        "config_count": config_count,
        "clip_count": clip_count,
        "best_count": len(state),
        "score90_count": sum(row["best_score"] >= 90.0 for row in state.values()),
        "best_total_sec": sum(row.get("duration_sec", 0.0) for row in state.values()),
        "score90_total_sec": sum(row.get("duration_sec", 0.0) for row in state.values() if row["best_score"] >= 90.0),
        "all_list": str(ALL_LIST),
        "score90_list": str(SCORE90_LIST),
    }
    atomic_json(STATUS_PATH, summary)


def main() -> None:
    if not CKPT.exists() or CKPT.stat().st_size != 6178193436:
        raise FileNotFoundError(CKPT)
    WORK.mkdir(parents=True, exist_ok=True)
    WINNERS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    hours = float(os.environ.get("OVERNIGHT_HOURS", "8"))
    deadline = started + hours * 3600.0
    inputs = parse_inputs()
    log(f"start hours={hours} inputs={len(inputs)} device={DEVICE}")
    sv = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
    sv.load_state_dict(torch.load(SV_CKPT, map_location="cpu"))
    sv.eval().to(DEVICE)
    ref_paths = sorted(REF_DIR.glob("*.wav"))
    refs = torch.stack([embed(*read(path), sv) for path in ref_paths])
    negs = []
    if NEG_LIST.exists():
        for line in NEG_LIST.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                path = Path(line.split("|", 1)[0].strip())
                if path.exists():
                    negs.append(embed(*read(path), sv))
    neg = torch.stack(negs) if negs else None
    center = torch.nn.functional.normalize(refs.mean(0), dim=-1)
    state = {}
    if STATE_PATH.exists():
        try:
            state = json.loads(STATE_PATH.read_text(encoding="utf-8")).get("rows", {})
            log(f"resumed rows={len(state)}")
        except Exception as exc:
            log(f"state_resume_failed={exc!r}")
    raw_cache = {}
    for index, (path, text) in enumerate(inputs):
        raw, sr = read(path)
        raw_cache[index] = (raw, sr)
        if str(index) not in state:
            metrics = measure(raw, sr, raw, sv, refs, center, neg)
            scored = score(metrics)
            state[str(index)] = {
                "best_path": str(path),
                "best_kind": "raw_bsr",
                "best_score": scored["score"],
                "score_parts": scored["parts"],
                "metrics": metrics,
                "duration_sec": len(raw) / sr,
                "text": text,
                "config": None,
            }
    save_state(state, inputs, started, deadline, 0, 0)
    audiosr_pipeline.download_checkpoint = lambda checkpoint_name="speech": str(CKPT)

    def stable_lowpass(batch):
        return {"waveform_lowpass": batch["waveform"].clone()}

    audiosr_pipeline.lowpass_filtering_prepare_inference = stable_lowpass
    model = build_model(model_name="speech", device=DEVICE)
    base_configs = list(itertools.product([6, 8, 10, 15], [2.5, 3.5, 4.5], [0.03, 0.05]))
    seeds = [11, 42, 77, 121, 2026, 2718]
    configs = [{"ddim_steps": a, "guidance_scale": b, "pad_sec": c, "seed": s} for s in seeds for a, b, c in base_configs]
    config_count = 0
    clip_count = 0
    while time.time() < deadline:
        for config in configs:
            if time.time() >= deadline:
                break
            config_count += 1
            prepared = build_prepared(inputs, config["pad_sec"])
            log(f"config={config_count} {config}")
            for index, (source_path, text) in enumerate(inputs):
                if time.time() >= deadline:
                    break
                raw, sr = raw_cache[index]
                try:
                    result = super_resolution(
                        model,
                        str(prepared[index]),
                        seed=config["seed"],
                        guidance_scale=config["guidance_scale"],
                        ddim_steps=config["ddim_steps"],
                        latent_t_per_second=12.8,
                    )
                    array = result[0] if hasattr(result, "__len__") else result
                    if torch.is_tensor(array):
                        array = array.detach().cpu().numpy()
                    array = np.asarray(array, dtype=np.float32).squeeze()
                    duration = len(raw) / sr
                    target_frames = round(duration * 48000)
                    if len(array) < target_frames:
                        raise RuntimeError(f"short_output={len(array)} target={target_frames}")
                    array = array[:target_frames]
                    final = librosa.resample(array, orig_sr=48000, target_sr=32000).astype(np.float32)
                    final = final[:round(duration * 32000)]
                    metrics = measure(final, 32000, raw, sv, refs, center, neg)
                    scored = score(metrics)
                    current = state[str(index)]
                    raw_metrics = current["metrics"] if current["best_kind"] == "raw_bsr" else measure(raw, sr, raw, sv, refs, center, neg)
                    ok, veto_reasons = allowed(metrics, raw_metrics)
                    improved = scored["score"] > current["best_score"] + 0.05
                    if ok and improved:
                        winner_path = WINNERS / f"best_{index:05d}_{source_path.stem}.wav"
                        sf.write(str(winner_path), final, 32000, subtype="PCM_16")
                        state[str(index)] = {
                            "best_path": str(winner_path),
                            "best_kind": "vasr_search",
                            "best_score": scored["score"],
                            "score_parts": scored["parts"],
                            "metrics": metrics,
                            "duration_sec": len(final) / 32000,
                            "text": text,
                            "config": config,
                            "veto_reasons": [],
                        }
                        log(f"promote index={index} score={scored['score']:.2f} source={source_path.name}")
                    clip_count += 1
                    if clip_count % 4 == 0:
                        save_state(state, inputs, started, deadline, config_count, clip_count)
                except Exception as exc:
                    log(f"clip_error index={index} error={exc!r}")
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
            save_state(state, inputs, started, deadline, config_count, clip_count)
            log(f"checkpoint config={config_count} elapsed_hours={(time.time()-started)/3600:.2f} score90={sum(x['best_score'] >= 90 for x in state.values())}")
    save_state(state, inputs, started, deadline, config_count, clip_count)
    final_report = {
        "started_at": started,
        "finished_at": time.time(),
        "planned_hours": hours,
        "actual_hours": (time.time() - started) / 3600.0,
        "device": DEVICE,
        "reference_count": len(ref_paths),
        "hard_negative_count": len(negs),
        "configs_completed": config_count,
        "clip_evaluations": clip_count,
        "best_count": len(state),
        "score90_count": sum(x["best_score"] >= 90.0 for x in state.values()),
        "best_total_sec": sum(x.get("duration_sec", 0.0) for x in state.values()),
        "score90_total_sec": sum(x.get("duration_sec", 0.0) for x in state.values() if x["best_score"] >= 90.0),
        "all_list": str(ALL_LIST),
        "score90_list": str(SCORE90_LIST),
        "state": str(STATE_PATH),
        "policy": "90 is a measured threshold, not a guarantee; raw BS-Roformer is retained whenever repair is vetoed",
    }
    atomic_json(REPORT_PATH, {**final_report, "rows": state})
    log(json.dumps(final_report, ensure_ascii=False))


if __name__ == "__main__":
    main()
