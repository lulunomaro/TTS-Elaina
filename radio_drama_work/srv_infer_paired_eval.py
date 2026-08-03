#!/usr/bin/env python3
"""Generate same-text paired samples for naturalness_score_v1.py."""
import json
import os
import sys

import soundfile as sf

REPO = "/root/GPT-SoVITS"
MANIFEST = "/root/autodl-tmp/paired_emo_eval.jsonl"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/infer_paired_emo_b16_v1"
DATASET = "/root/autodl-tmp/dataset_v1"
REF = os.path.join(DATASET, "wavs", "old_my_voice.mp3_0014764160_0014986560.wav.refmt_vocals.wav")
PROMPT = "/u738b/u7acb/u30bb/u30ec/u30b9/u30c6/u30ea/u30a2/u3068/u3044/u3046/u56fd/u304c/u3069/u3053/u304b/u79c1/u306e/u65c5/u305d/u306e/u3082/u306e/u3068/u91cd/u306a/u3063/u3066/u3044/u308b/u304b/u3089/u306a/u306e/u304b/u3082/u3057/u308c/u307e/u305b/u3093/u3002"
GPT = "/root/autodl-tmp/weights/GPT_ireina_s2_old_v2/ireina_s2_old_v2-e6.ckpt"
SOVITS = [
    ("b16_e2", "/root/autodl-tmp/weights/SoVITS_ireina_s2_balanced48_emo20_dlr2e5_b16_v1/ireina_s2_balanced48_emo20_dlr2e5_b16_v1_e2_s40.pth"),
    ("b16_e4", "/root/autodl-tmp/weights/SoVITS_ireina_s2_balanced48_emo20_dlr2e5_b16_v1/ireina_s2_balanced48_emo20_dlr2e5_b16_v1_e4_s80.pth"),
    ("b16_e8", "/root/autodl-tmp/weights/SoVITS_ireina_s2_balanced48_emo20_dlr2e5_b16_v1/ireina_s2_balanced48_emo20_dlr2e5_b16_v1_e8_s160.pth"),
    ("focus_e3", "/root/autodl-tmp/weights/SoVITS_ireina_s2_emo_focus_v2_b16_lr75e5_dlr1e5/ireina_s2_emo_focus_v2_b16_lr75e5_dlr1e5_e3_s60.pth"),
    ("focus_e6", "/root/autodl-tmp/weights/SoVITS_ireina_s2_emo_focus_v2_b16_lr75e5_dlr1e5/ireina_s2_emo_focus_v2_b16_lr75e5_dlr1e5_e6_s120.pth"),
]
PARAMS = dict(top_k=20, top_p=0.9, temperature=0.85, repetition_penalty=1.20)

rows = [json.loads(line) for line in open(MANIFEST, encoding="utf-8") if line.strip()]
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))
os.chdir(REPO)
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

os.makedirs(OUT, exist_ok=True)
cfg = TTS_Config(os.path.join(REPO, "GPT_SoVITS/configs/tts_infer.yaml"))
cfg.device = "cuda"
cfg.is_half = True
cfg.version = "v2Pro"
tts = TTS(cfg)
tts.init_t2s_weights(GPT)

for name, sovits in SOVITS:
    print(f"=== {name} ===", flush=True)
    tts.init_vits_weights(sovits)
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
        req.update(PARAMS)
        sr, audio = next(tts.run(req))
        path = os.path.join(OUT, f"{name}__{row['sample_id']}.wav")
        sf.write(path, audio, sr)
        print(f"[ok] {row['sample_id']} scene={row['scene']} {len(audio) / sr:.2f}s", flush=True)

print(f"DONE {OUT}")
