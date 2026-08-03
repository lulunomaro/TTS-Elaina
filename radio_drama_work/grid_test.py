# -*- coding: utf-8 -*-
"""诊断"机械/AI感"：在同一进程内切换 GPT/SoVITS 轮次 + 参考音频，做网格对比。
需在 v2pro 包根运行。
"""
import io
import os
import sys
import traceback

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import soundfile as sf  # noqa: E402

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

ROOT = r"radio_drama_work/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50"
GPT_DIR = os.path.join(ROOT, "GPT_weights_v2Pro_merged")
SV_DIR = os.path.join(ROOT, "SoVITS_weights_v2Pro_merged")
OUT = r"radio_drama_work/grid"
os.makedirs(OUT, exist_ok=True)

# ---- 参考音频：新(广播剧,响度高) vs 老(mp3分离,偏轻) ----
REF_NEW = (
    r"radio_drama_work/radio_drama_pure"
    r"/113814583-1-160_vocals.wav_0022915200_0023071360.wav"
)
REF_NEW_TEXT = "この調子でいけば華麗に薬を混ぜたこともバレずにいけるかもしれません"
REF_OLD = os.path.join(
    ROOT, "logs", "my_voice", "vocals",
    "my_voice.mp3_0014764160_0014986560.wav.refmt_vocals.wav",
)
REF_OLD_TEXT = "王立セレステリアという国がどこか私の旅そのものと重なっているからなのかもしれません"

# 若新参考不在 radio_drama_pure，从 merged.list 里找
if not os.path.exists(REF_NEW):
    lst = os.path.join(ROOT, "logs", "my_voice_merged", "merged.list")
    for ln in open(lst, encoding="utf-8"):
        p = ln.strip().split("|")
        if len(p) >= 4 and "0022915200_0023071360" in p[0]:
            REF_NEW, REF_NEW_TEXT = p[0], p[3]
            break
print("REF_NEW exists:", os.path.exists(REF_NEW), REF_NEW)
print("REF_OLD exists:", os.path.exists(REF_OLD))

TEXT_JA = "わたしはイレイナ。旅を続ける魔女です。今日はどんな出会いが待っているのでしょうか。"

# ---- 网格：(标签, GPT轮, SoVITS轮, 参考) ----
GRID = [
    # 官方推荐轮数 + 新参考  —— 主推
    ("A_g12_s8_refnew", "my_voice_merged-e12.ckpt", "my_voice_merged_e8_s736.pth", "new"),
    # 稍多一点
    ("B_g16_s12_refnew", "my_voice_merged-e16.ckpt", "my_voice_merged_e12_s1104.pth", "new"),
    # 当前过拟合基线 + 新参考（隔离参考音频变量）
    ("C_g40_s40_refnew", "my_voice_merged-e40.ckpt", "my_voice_merged_e40_s3680.pth", "new"),
    # 官方轮数 + 老参考（隔离轮次变量）
    ("D_g12_s8_refold", "my_voice_merged-e12.ckpt", "my_voice_merged_e8_s736.pth", "old"),
]

cfg = TTS_Config({
    "custom": {
        "device": "cuda",
        "is_half": True,
        "version": "v2Pro",
        "t2s_weights_path": os.path.join(GPT_DIR, GRID[0][1]),
        "vits_weights_path": os.path.join(SV_DIR, GRID[0][2]),
        "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
    }
})
tts = TTS(cfg)

cur_g = cur_s = None
for tag, g, s, ref in GRID:
    try:
        if g != cur_g:
            tts.init_t2s_weights(os.path.join(GPT_DIR, g))
            cur_g = g
        if s != cur_s:
            tts.init_vits_weights(os.path.join(SV_DIR, s))
            cur_s = s
        rw, rt = (REF_NEW, REF_NEW_TEXT) if ref == "new" else (REF_OLD, REF_OLD_TEXT)
        res = tts.run({
            "text": TEXT_JA,
            "text_lang": "ja",
            "ref_audio_path": rw,
            "prompt_text": rt,
            "prompt_lang": "ja",
            "top_k": 20,
            "top_p": 1.0,
            "temperature": 1.0,
            "text_split_method": "cut5",
            "speed_factor": 1.0,
            "batch_size": 1,
            "seed": 1234,
            "parallel_infer": True,
        })
        sr = audio = None
        for sr, audio in res:
            pass
        out = os.path.join(OUT, tag + ".wav")
        sf.write(out, audio, sr)
        print("[ok] %-22s %.1fs  gpt=%s sovits=%s ref=%s" % (
            tag, len(audio) / sr, g, s, ref), flush=True)
    except Exception:
        print("[FAIL] %s" % tag)
        traceback.print_exc()

print("DONE ->", OUT)
