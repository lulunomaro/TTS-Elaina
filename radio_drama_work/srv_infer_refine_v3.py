# -*- coding: utf-8 -*-
"""固定 D_old GPT e6，比较旧候选与平衡精修 SoVITS 的四类关键句。"""

import os
import sys

import soundfile as sf


REPO = "/root/GPT-SoVITS"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/infer_refine_v3"
DATASET = "/root/autodl-tmp/dataset_v1"
REF = os.path.join(DATASET, "wavs", "old_my_voice.mp3_0014764160_0014986560.wav.refmt_vocals.wav")
PROMPT = "王立セレステリアという国がどこか私の旅そのものと重なっているからなのかもしれません。"
GPT = "/root/autodl-tmp/weights/GPT_ireina_s2_old_v2/ireina_s2_old_v2-e6.ckpt"

SOVITS = [
    ("old_e2", "/root/autodl-tmp/weights/SoVITS_ireina_s2_old_v2/ireina_s2_old_v2_e2_s50.pth"),
    ("old_e4", "/root/autodl-tmp/weights/SoVITS_ireina_s2_old_v2/ireina_s2_old_v2_e4_s100.pth"),
    ("old_e6", "/root/autodl-tmp/weights/SoVITS_ireina_s2_old_v2/ireina_s2_old_v2_e6_s150.pth"),
    ("curated_e2", "/root/autodl-tmp/weights/SoVITS_ireina_s2_curated_v2/ireina_s2_curated_v2_e2_s76.pth"),
    ("curated_e4", "/root/autodl-tmp/weights/SoVITS_ireina_s2_curated_v2/ireina_s2_curated_v2_e4_s152.pth"),
    ("curated_e6", "/root/autodl-tmp/weights/SoVITS_ireina_s2_curated_v2/ireina_s2_curated_v2_e6_s228.pth"),
    ("curated_e8", "/root/autodl-tmp/weights/SoVITS_ireina_s2_curated_v2/ireina_s2_curated_v2_e8_s304.pth"),
    ("balanced48_e2", "/root/autodl-tmp/weights/SoVITS_ireina_s2_balanced48_v3/ireina_s2_balanced48_v3_e2_s62.pth"),
    ("balanced48_e4", "/root/autodl-tmp/weights/SoVITS_ireina_s2_balanced48_v3/ireina_s2_balanced48_v3_e4_s124.pth"),
    ("balanced48_e6", "/root/autodl-tmp/weights/SoVITS_ireina_s2_balanced48_v3/ireina_s2_balanced48_v3_e6_s186.pth"),
    ("plus18_e2", "/root/autodl-tmp/weights/SoVITS_ireina_s2_balanced48_plus18_v1/ireina_s2_balanced48_plus18_v1_e2_s66.pth"),
    ("plus18_e4", "/root/autodl-tmp/weights/SoVITS_ireina_s2_balanced48_plus18_v1/ireina_s2_balanced48_plus18_v1_e4_s132.pth"),
    ("plus18_e6", "/root/autodl-tmp/weights/SoVITS_ireina_s2_balanced48_plus18_v1/ireina_s2_balanced48_plus18_v1_e6_s198.pth"),
    ("human_focus14_e2", "/root/autodl-tmp/weights/SoVITS_ireina_s2_human_focus14_lr5e5_v1/ireina_s2_human_focus14_lr5e5_v1_e2_s54.pth"),
    ("human_focus14_e4", "/root/autodl-tmp/weights/SoVITS_ireina_s2_human_focus14_lr5e5_v1/ireina_s2_human_focus14_lr5e5_v1_e4_s108.pth"),
]
if os.environ.get("ONLY_NEW_BALANCED48_LR5E5") == "1":
    SOVITS = [
        ("balanced48_lr5e5_e2", "/root/autodl-tmp/weights/SoVITS_ireina_s2_balanced48_lr5e5_v1/ireina_s2_balanced48_lr5e5_v1_e2_s62.pth"),
        ("balanced48_lr5e5_e4", "/root/autodl-tmp/weights/SoVITS_ireina_s2_balanced48_lr5e5_v1/ireina_s2_balanced48_lr5e5_v1_e4_s124.pth"),
    ]
if os.environ.get("ONLY_NEW_EMO20_DLR2E5") == "1":
    SOVITS = [
        ("balanced48_emo20_dlr2e5_e2", "/root/autodl-tmp/weights/SoVITS_ireina_s2_balanced48_emo20_dlr2e5_v1/ireina_s2_balanced48_emo20_dlr2e5_v1_e2_s62.pth"),
        ("balanced48_emo20_dlr2e5_e4", "/root/autodl-tmp/weights/SoVITS_ireina_s2_balanced48_emo20_dlr2e5_v1/ireina_s2_balanced48_emo20_dlr2e5_v1_e4_s124.pth"),
    ]
if os.environ.get("ONLY_NEW_EMO20_DLR2E5_B16") == "1":
    SOVITS = [
        ("balanced48_emo20_dlr2e5_b16_e2", "/root/autodl-tmp/weights/SoVITS_ireina_s2_balanced48_emo20_dlr2e5_b16_v1/ireina_s2_balanced48_emo20_dlr2e5_b16_v1_e2_s40.pth"),
        ("balanced48_emo20_dlr2e5_b16_e4", "/root/autodl-tmp/weights/SoVITS_ireina_s2_balanced48_emo20_dlr2e5_b16_v1/ireina_s2_balanced48_emo20_dlr2e5_b16_v1_e4_s80.pth"),
        ("balanced48_emo20_dlr2e5_b16_e6", "/root/autodl-tmp/weights/SoVITS_ireina_s2_balanced48_emo20_dlr2e5_b16_v1/ireina_s2_balanced48_emo20_dlr2e5_b16_v1_e6_s120.pth"),
        ("balanced48_emo20_dlr2e5_b16_e8", "/root/autodl-tmp/weights/SoVITS_ireina_s2_balanced48_emo20_dlr2e5_b16_v1/ireina_s2_balanced48_emo20_dlr2e5_b16_v1_e8_s160.pth"),
    ]
TEXTS = [
    ("normal", "こんにちは、私は魔女のイレイナです。今日はとてもいい天気ですね。"),
    ("emo", "えっ、本当ですか？信じられません…でも、とても嬉しいです！"),
    ("sigh", "はぁ…そうですか。仕方ありませんね。"),
    ("ending", "それでは、またどこかでお会いしましょう。"),
]
PARAMS = dict(top_k=20, top_p=0.9, temperature=0.85, repetition_penalty=1.20)

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

for name, sovits in SOVITS:
    print(f"=== {name} ===", flush=True)
    tts.init_vits_weights(sovits)
    for sid, text in TEXTS:
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
        req.update(PARAMS)
        sr, audio = next(tts.run(req))
        path = os.path.join(OUT, f"oldgpt_e6__{name}__{sid}.wav")
        sf.write(path, audio, sr)
        print(f"[ok] {path} {len(audio) / sr:.2f}s", flush=True)

print(f"DONE {OUT}")
