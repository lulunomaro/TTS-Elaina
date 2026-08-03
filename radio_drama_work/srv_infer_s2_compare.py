# -*- coding: utf-8 -*-
"""固定官方 GPT，比较两套新 SoVITS 的中间轮次。"""

import glob
import os
import sys

import soundfile as sf

REPO = "/root/GPT-SoVITS"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/infer_s2_v2"
DATASET = "/root/autodl-tmp/dataset_v1"
BASE_GPT = f"{REPO}/GPT_SoVITS/pretrained_models/s1v3.ckpt"
BASE_SOVITS = f"{REPO}/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth"
REF = os.path.join(DATASET, "wavs", "old_my_voice.mp3_0014764160_0014986560.wav.refmt_vocals.wav")
PROMPT = "王立セレステリアという国がどこか私の旅そのものと重なっているからなのかもしれません"

sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))
os.chdir(REPO)
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402


def weights(directory: str):
    return [
        (f"e{epoch}", os.path.join(directory, f))
        for epoch in (2, 4, 6, 8)
        for f in sorted(glob.glob(os.path.join(directory, f"*_e{epoch}_s*.pth")))
    ][:4]


os.makedirs(OUT, exist_ok=True)
cfg = TTS_Config(os.path.join(REPO, "GPT_SoVITS/configs/tts_infer.yaml"))
cfg.device = "cuda"
cfg.is_half = True
cfg.version = "v2Pro"
tts = TTS(cfg)

cases = [("base_base", BASE_SOVITS)]
cases += [(f"old_{ep}", path) for ep, path in weights("/root/autodl-tmp/weights/SoVITS_ireina_s2_old_v2")]
cases += [(f"curated_{ep}", path) for ep, path in weights("/root/autodl-tmp/weights/SoVITS_ireina_s2_curated_v2")]
texts = [
    ("normal", "こんにちは、私は魔女のイレイナです。今日はとてもいい天気ですね。"),
    ("emo", "えっ、本当ですか？信じられません……でも、とても嬉しいです！"),
]
params = dict(top_k=20, top_p=0.9, temperature=0.85, repetition_penalty=1.20)

for name, sovits in cases:
    print(f"=== {name} ===", flush=True)
    tts.init_t2s_weights(BASE_GPT)
    tts.init_vits_weights(sovits)
    for sid, text in texts:
        req = {
            "text": text,
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
        req.update(params)
        sr, audio = next(tts.run(req))
        path = os.path.join(OUT, f"{name}__{sid}.wav")
        sf.write(path, audio, sr)
        print(f"[ok] {path} {sr} {len(audio) / sr:.2f}s", flush=True)

print(f"DONE {OUT}")
