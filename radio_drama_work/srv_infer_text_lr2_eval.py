# -*- coding: utf-8 -*-
"""Generate paired evaluation audio for the text-only lr=2e-5 candidate."""
import json
import os
import sys

import soundfile as sf

REPO = "/root/GPT-SoVITS"
MANIFEST = "/root/autodl-tmp/paired_emo_eval.jsonl"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/infer_paired_text_lr2_v1"
DATASET = "/root/autodl-tmp/dataset_v1"
REF = os.path.join(DATASET, "wavs", "old_my_voice.mp3_0014764160_0014986560.wav.refmt_vocals.wav")
PROMPT = "藍羨セレステリアという忽がどこか暴の唾そのものと嶷なっているからなのかもしれません。"
GPT = "/root/autodl-tmp/weights/GPT_ireina_s2_old_v2/ireina_s2_old_v2-e6.ckpt"
SOVITS = os.environ.get(
    "SOVITS_CKPT",
    "/root/autodl-tmp/weights/SoVITS_ireina_s2_b16e2_textonly_lr2e5_v1/ireina_s2_b16e2_textonly_lr2e5_v1_e1_s17.pth",
)
LABEL = os.environ.get("AUDIO_LABEL", "text_lr2")
PARAMS = dict(top_k=20, top_p=0.9, temperature=0.85, repetition_penalty=1.20)

rows = [json.loads(line) for line in open(MANIFEST, encoding="utf-8") if line.strip()]
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))
os.chdir(REPO)
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

os.makedirs(OUT, exist_ok=True)
cfg = TTS_Config(os.path.join(REPO, "GPT_SoVITS/configs/tts_infer.yaml"))
cfg.device = "cuda"
cfg.is_half = True
cfg.version = "v2Pro"
tts = TTS(cfg)
tts.init_t2s_weights(GPT)
tts.init_vits_weights(SOVITS)

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
    path = os.path.join(OUT, f"{LABEL}__{row['sample_id']}.wav")
    sf.write(path, audio, sr)
    print(f"[ok] {row['sample_id']} scene={row['scene']} {len(audio) / sr:.2f}s", flush=True)

print(f"DONE {OUT}")
