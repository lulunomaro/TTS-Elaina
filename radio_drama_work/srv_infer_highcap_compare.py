#!/usr/bin/env python3
"""Evaluate high-capacity D_old GPT candidates against acoustic refinements."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import soundfile as sf

REPO = "/root/GPT-SoVITS"
DATASET = "/root/autodl-tmp/dataset_v1"
MANIFEST = "/root/autodl-tmp/paired_emo_eval.jsonl"
OUT = Path(os.environ.get("INFER_OUT", "/root/autodl-tmp/infer_highcap_paired_v1"))
GPT_DIR = Path("/root/autodl-tmp/weights/GPT_ireina_s1_dold_hicap_b8s28_v1")

REF = os.path.join(
    DATASET, "wavs",
    "old_my_voice.mp3_0014764160_0014986560.wav.refmt_vocals.wav",
)
PROMPT = (
    "/u738b/u7acb/u30bb/u30ec/u30b9/u30c6/u30ea/u30a2/u3068/u3044/u3046/u56fd/u304c"
    "/u3069/u3053/u304b/u79c1/u306e/u65c5/u305d/u306e/u3082/u306e/u3068/u91cd/u306a/u3063/u3066"
    "/u3044/u308b/u304b/u3089/u306a/u306e/u304b/u3082/u3057/u308c/u307e/u305b/u3093/u3002"
)

SOVITS = [
    ("old_e4", "/root/autodl-tmp/weights/SoVITS_ireina_s2_old_v2/ireina_s2_old_v2_e4_s100.pth"),
    ("old_e6", "/root/autodl-tmp/weights/SoVITS_ireina_s2_old_v2/ireina_s2_old_v2_e6_s150.pth"),
    ("balanced48_e2", "/root/autodl-tmp/weights/SoVITS_ireina_s2_balanced48_v3/ireina_s2_balanced48_v3_e2_s62.pth"),
    ("curated_e4", "/root/autodl-tmp/weights/SoVITS_ireina_s2_curated_v2/ireina_s2_curated_v2_e4_s152.pth"),
    ("emo_balanced_e2", "/root/autodl-tmp/weights/SoVITS_ireina_s2_balanced48_emo20_dlr2e5_b16_v1/ireina_s2_balanced48_emo20_dlr2e5_b16_v1_e2_s40.pth"),
]

PARAMS = {
    "expr": dict(top_k=30, top_p=0.95, temperature=0.95, repetition_penalty=1.15),
    "stable": dict(top_k=20, top_p=0.90, temperature=0.85, repetition_penalty=1.20),
}
PROFILE = os.environ.get("PARAM_PROFILE", "expr")
if PROFILE not in PARAMS:
    raise SystemExit(f"unknown PARAM_PROFILE={PROFILE}")

epoch_filter = {
    int(x.strip()) for x in os.environ.get("GPT_EPOCHS", "").split(",")
    if x.strip()
}
sovits_filter = {
    x.strip() for x in os.environ.get("SOVITS_LABELS", "").split(",")
    if x.strip()
}

rows = [json.loads(line) for line in Path(MANIFEST).read_text(encoding="utf-8").splitlines() if line.strip()]
gpt_paths = []
for epoch in (1, 2, 4, 6, 8, 10, 12):
    if epoch_filter and epoch not in epoch_filter:
        continue
    path = GPT_DIR / f"ireina_s1_dold_hicap_b8s28_v1-e{epoch}.ckpt"
    if path.exists():
        gpt_paths.append((f"hicap_e{epoch}", str(path)))
if not gpt_paths:
    raise SystemExit(f"no GPT weights under {GPT_DIR}")

sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))
os.chdir(REPO)
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
cfg = TTS_Config(os.path.join(REPO, "GPT_SoVITS/configs/tts_infer.yaml"))
cfg.device = "cuda"
cfg.is_half = True
cfg.version = "v2Pro"
tts = TTS(cfg)

count = 0
for gpt_label, gpt_path in gpt_paths:
    tts.init_t2s_weights(gpt_path)
    for sovits_label, sovits_path in SOVITS:
        if sovits_filter and sovits_label not in sovits_filter:
            continue
        if not os.path.exists(sovits_path):
            print(f"skip missing {sovits_path}", flush=True)
            continue
        tts.init_vits_weights(sovits_path)
        group = f"{gpt_label}__{sovits_label}__{PROFILE}"
        for row in rows:
            req = {
                "text": row["text"],
                "text_lang": "ja",
                "ref_audio_path": REF,
                "prompt_text": PROMPT,
                "prompt_lang": "ja",
                "batch_size": 1,
                "speed_factor": 1.0,
                "text_split_method": "cut5",
                "seed": 1234,
                "parallel_infer": True,
                "return_fragment": False,
            }
            req.update(PARAMS[PROFILE])
            try:
                sr, audio = next(tts.run(req))
                out = OUT / f"{group}__{row['sample_id']}.wav"
                sf.write(out, audio, sr)
                count += 1
                if count % 25 == 0:
                    print(f"generated={count} last={out.name}", flush=True)
            except Exception as exc:
                print(f"FAIL {group} {row['sample_id']}: {exc}", flush=True)

print(f"DONE generated={count} out={OUT}")
