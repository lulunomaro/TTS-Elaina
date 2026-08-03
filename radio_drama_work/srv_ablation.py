# -*- coding: utf-8 -*-
"""v2Pro 四路消融：区分底模、GPT 微调、SoVITS 微调各自影响。"""
import os
import sys
import soundfile as sf

REPO = "/root/GPT-SoVITS"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))
os.chdir(REPO)
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/ablation_v1"
GPT_DIR = "/root/autodl-tmp/weights/GPT_weights_v2Pro"
SOVITS_DIR = "/root/autodl-tmp/weights/SoVITS_weights_v2Pro"
REF = "/root/autodl-tmp/dataset_v1/wavs/old_my_voice.mp3_0014764160_0014986560.wav.refmt_vocals.wav"
PROMPT = "王立セレステリアという国がどこか私の旅そのものと重なっているからなのかもしれません"
TEXTS = [
    ("normal", "こんにちは、私は魔女のイレイナです。今日はとてもいい天気ですね。"),
    ("emo", "えっ、本当ですか？信じられません……でも、とても嬉しいです！"),
]
BASE_GPT = "/root/GPT-SoVITS/GPT_SoVITS/pretrained_models/s1v3.ckpt"
BASE_SOVITS = "/root/GPT-SoVITS/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth"
CASES = [
    ("base_base", BASE_GPT, BASE_SOVITS),
    ("finetuneGPT_baseS", os.path.join(GPT_DIR, "ireina_v1-e15.ckpt"), BASE_SOVITS),
    ("baseGPT_finetuneS", BASE_GPT, os.path.join(SOVITS_DIR, "ireina_v1_e8_s312.pth")),
    ("finetuneGPT_finetuneS", os.path.join(GPT_DIR, "ireina_v1-e15.ckpt"), os.path.join(SOVITS_DIR, "ireina_v1_e8_s312.pth")),
]

os.makedirs(OUT, exist_ok=True)
cfg = TTS_Config(os.path.join(REPO, "GPT_SoVITS/configs/tts_infer.yaml"))
cfg.device = "cuda"
cfg.is_half = True
cfg.version = "v2Pro"
tts = TTS(cfg)
params = dict(top_k=20, top_p=0.9, temperature=0.85, repetition_penalty=1.20)
for name, gpt, sovits in CASES:
    print("===", name, "===", flush=True)
    tts.init_t2s_weights(gpt)
    tts.init_vits_weights(sovits)
    for sid, text in TEXTS:
        req = {
            "text": text, "text_lang": "ja", "ref_audio_path": REF,
            "prompt_text": PROMPT, "prompt_lang": "ja", "batch_size": 1,
            "speed_factor": 1.0, "text_split_method": "cut5", "seed": 1234,
            "parallel_infer": True, "return_fragment": False,
        }
        req.update(params)
        try:
            sr, audio = next(tts.run(req))
            path = os.path.join(OUT, "%s__%s.wav" % (name, sid))
            sf.write(path, audio, sr)
            print("[ok]", path, sr, len(audio) / sr, flush=True)
        except Exception as e:
            print("[FAIL]", name, sid, str(e)[:300], flush=True)
print("DONE", OUT, flush=True)
