#!/usr/bin/env python3
"""Generate fixed and paired samples for the frozen-D candidate."""
import json
import os
import sys
import soundfile as sf

REPO = "/root/GPT-SoVITS"
DATASET = "/root/autodl-tmp/dataset_v1"
REF = os.path.join(DATASET, "wavs", "old_my_voice.mp3_0014764160_0014986560.wav.refmt_vocals.wav")
PROMPT = "/u738b/u7acb/u30bb/u30ec/u30b9/u30c6/u30ea/u30a2/u3068/u3044/u3046/u56fd/u304c/u3069/u3053/u304b/u79c1/u306e/u65c5/u305d/u306e/u3082/u306e/u3068/u91cd/u306a/u3063/u3066/u3044/u308b/u304b/u3089/u306a/u306e/u304b/u3082/u3057/u308c/u307e/u305b/u3093/u3002"
GPT = "/root/autodl-tmp/weights/GPT_ireina_s2_old_v2/ireina_s2_old_v2-e6.ckpt"
SOVITS = [("gonly_e1", "/root/autodl-tmp/weights/SoVITS_ireina_s2_b16e2_gonly_v1/ireina_s2_b16e2_gonly_v1_e1_s17.pth")]
FIXED = [
    ("normal", "/u3053/u3093/u306b/u3061/u306f/u3001/u79c1/u306f/u9b54/u5973/u30a4/u30ec/u30a4/u30ca/u3067/u3059/u3002/u4eca/u65e5/u306f/u3068/u3066/u3082/u3044/u3044/u5929/u6c17/u3067/u3059/u306d/u3002"),
    ("emo", "/u3048/u3063/u3001/u672c/u5f53/u3067/u3059/u304b/uff1f/u4fe1/u3058/u3089/u308c/u307e/u305b/u3093/u2026/u2026/u3067/u3082/u3001/u5b09/u3057/u3044/u3067/u3059/uff01"),
    ("sigh", "/u306f/u3041/u2026/u2026/u3069/u3046/u3057/u3066/u3053/u3093/u306a/u3053/u3068/u306b/u306a/u3063/u3066/u3057/u307e/u3063/u305f/u306e/u3067/u3057/u3087/u3046/u3002"),
    ("ending", "/u305d/u308c/u3067/u306f/u3001/u307e/u305f/u3069/u3053/u304b/u3067/u304a/u4f1a/u3044/u3057/u307e/u3057/u3087/u3046/u3002"),
]
rows = [json.loads(x) for x in open("/root/autodl-tmp/paired_emo_eval.jsonl", encoding="utf-8") if x.strip()]
OUT_FIXED = "/root/autodl-tmp/infer_gonly_correct_v1"
OUT_PAIRED = "/root/autodl-tmp/infer_paired_gonly_v1"
os.makedirs(OUT_FIXED, exist_ok=True)
os.makedirs(OUT_PAIRED, exist_ok=True)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))
os.chdir(REPO)
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
cfg = TTS_Config(os.path.join(REPO, "GPT_SoVITS/configs/tts_infer.yaml"))
cfg.device = "cuda"
cfg.is_half = True
cfg.version = "v2Pro"
tts = TTS(cfg)
tts.init_t2s_weights(GPT)
PARAMS = dict(top_k=20, top_p=0.9, temperature=0.85, repetition_penalty=1.20)

def gen(text):
    req = {"text": text, "text_lang": "ja", "ref_audio_path": REF,
           "prompt_text": PROMPT, "prompt_lang": "ja", "batch_size": 1,
           "speed_factor": 1.0, "text_split_method": "cut5", "seed": 1234,
           "parallel_infer": True, "return_fragment": False}
    req.update(PARAMS)
    return next(tts.run(req))

for name, path in SOVITS:
    tts.init_vits_weights(path)
    for sid, text in FIXED:
        sr, audio = gen(text)
        sf.write(os.path.join(OUT_FIXED, f"oldgpt_e6__{name}__{sid}.wav"), audio, sr)
    for row in rows:
        sr, audio = gen(row["text"])
        sf.write(os.path.join(OUT_PAIRED, f"{name}__{row['sample_id']}.wav"), audio, sr)
print("DONE")
