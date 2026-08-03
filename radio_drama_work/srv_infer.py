# -*- coding: utf-8 -*-
"""AutoDL 服务器端 v2Pro 推理：支持多权重 × 多采样参数扫描。

用法（在 /root/GPT-SoVITS 下用 gptsovits 环境跑）：
    python srv_infer.py <out_dir>
"""
import os
import sys
import glob

import soundfile as sf

REPO = "/root/GPT-SoVITS"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))
os.chdir(REPO)

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/out"
os.makedirs(OUT, exist_ok=True)

GPT_DIR = "/root/autodl-tmp/weights/GPT_weights_v2Pro"
SOVITS_DIR = "/root/autodl-tmp/weights/SoVITS_weights_v2Pro"
DATASET = "/root/autodl-tmp/dataset_v1"

# 用户听感判定：老数据的这条旁白参考效果最好（grid 测试中 D 组胜出）
REF_WAV = os.path.join(
    DATASET, "wavs",
    "old_my_voice.mp3_0014764160_0014986560.wav.refmt_vocals.wav")
REF_TEXT = ("王立セレステリアという国がどこか私の旅そのものと"
            "重なっているからなのかもしれません")

TEST_SENTENCES = [
    ("01_ja", "ja",
     "こんにちは、私は魔女のイレイナです。今日はとてもいい天気ですね。"),
    ("02_ja_emo", "ja",
     "えっ、本当ですか？信じられません……でも、とても嬉しいです！"),
]


def pick(pattern_dir, prefer_epochs):
    """按偏好的 epoch 顺序挑权重文件"""
    files = glob.glob(os.path.join(pattern_dir, "*"))
    files = [f for f in files if f.endswith((".ckpt", ".pth"))]
    out = []
    for ep in prefer_epochs:
        for f in sorted(files):
            base = os.path.basename(f)
            if ("-e%d." % ep) in base or ("_e%d_" % ep) in base or ("-e%d-" % ep) in base:
                out.append((ep, f))
                break
    return out


def main():
    gpts = pick(GPT_DIR, [15, 12, 9])
    sovits = pick(SOVITS_DIR, [8, 6, 4])
    print("GPT   candidates:", [(e, os.path.basename(f)) for e, f in gpts])
    print("SoVITS candidates:", [(e, os.path.basename(f)) for e, f in sovits])
    if not gpts or not sovits:
        print("!! 权重不足，退出")
        return

    cfg = TTS_Config(os.path.join(REPO, "GPT_SoVITS/configs/tts_infer.yaml"))
    cfg.device = "cuda"
    cfg.is_half = True
    cfg.version = "v2Pro"
    tts = TTS(cfg)

    # 逐一切换 GPT / SoVITS 权重，避免只测试到终轮组合。
    # 每个组合都输出两句测试文本 × 三组采样参数。
    combos = [(g_ep, g_path, s_ep, s_path) for g_ep, g_path in gpts for s_ep, s_path in sovits]
    print("Total weight combos:", len(combos))

    # 采样参数组：此前所有实验都只用过 P1(默认)，从未调过
    PARAM_SETS = [
        ("P1_default", dict(top_k=15, top_p=1.0, temperature=1.0, repetition_penalty=1.35)),
        ("P2_soft",    dict(top_k=20, top_p=0.9, temperature=0.85, repetition_penalty=1.20)),
        ("P3_lively",  dict(top_k=30, top_p=0.95, temperature=1.15, repetition_penalty=1.35)),
    ]

    for g_ep, g_path, s_ep, s_path in combos:
        tts.init_t2s_weights(g_path)
        tts.init_vits_weights(s_path)
        print("loaded GPT e%d + SoVITS e%d" % (g_ep, s_ep), flush=True)
        for pname, pset in PARAM_SETS:
            for sid, lang, text in TEST_SENTENCES:
                req = {
                    "text": text,
                    "text_lang": lang,
                    "ref_audio_path": REF_WAV,
                    "prompt_text": REF_TEXT,
                    "prompt_lang": "ja",
                    "batch_size": 1,
                    "speed_factor": 1.0,
                    "text_split_method": "cut5",
                    "seed": 1234,
                    "parallel_infer": True,
                    "return_fragment": False,
                }
                req.update(pset)
                try:
                    gen = tts.run(req)
                    sr, audio = next(gen)
                    name = "%s__g%d_s%d__%s.wav" % (sid, g_ep, s_ep, pname)
                    sf.write(os.path.join(OUT, name), audio, sr)
                    print("[ok] %s  (%d Hz, %.1fs)" % (name, sr, len(audio) / sr), flush=True)
                except Exception as e:
                    print("[FAIL] %s %s : %s" % (sid, pname, str(e)[:200]), flush=True)

    print("DONE ->", OUT)
    for f in sorted(os.listdir(OUT)):
        print("  ", f)


if __name__ == "__main__":
    main()
