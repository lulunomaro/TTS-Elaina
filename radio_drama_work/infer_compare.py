# -*- coding: utf-8 -*-
"""用指定版本的权重合成对比音频。需在对应包根目录下运行（cwd=包根）。

用法:
  python infer_compare.py <version> <gpt_ckpt> <sovits_pth> <out_prefix>
例:
  python infer_compare.py v2Pro GPT_weights_v2Pro_merged/xxx.ckpt SoVITS_weights_v2Pro_merged/yyy.pth F:/out/v2pro
"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import soundfile as sf  # noqa: E402
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

VERSION = sys.argv[1]
GPT_CKPT = sys.argv[2]
SOVITS_PTH = sys.argv[3]
OUT_PREFIX = sys.argv[4]

# 参考音频（老数据中语气最典型的 イレイナ 旁白，7.0s）
REF_WAV = (
    r"radio_drama_work/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50"
    r"/logs/my_voice/vocals/my_voice.mp3_0014764160_0014986560.wav.refmt_vocals.wav"
)
REF_TEXT = "王立セレステリアという国がどこか私の旅そのものと重なっているからなのかもしれません"

# 对比用的统一测试句
CASES = [
    ("01_ja", "ja", "わたしはイレイナ。旅を続ける魔女です。今日はどんな出会いが待っているのでしょうか。"),
    ("02_ja_emo", "ja", "えっ、そんなことがあるんですか?まさか、信じられません。でも、面白そうですね。"),
    ("03_zh", "zh", "今天天气不错，我们一起出去散散步吧。"),
]

cfg = TTS_Config(
    {
        "custom": {
            "device": "cuda",
            "is_half": True,
            "version": VERSION,
            "t2s_weights_path": GPT_CKPT,
            "vits_weights_path": SOVITS_PTH,
            "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
            "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        }
    }
)
print("[cfg] version=%s/n  t2s=%s/n  vits=%s" % (cfg.version, cfg.t2s_weights_path, cfg.vits_weights_path))

tts = TTS(cfg)
os.makedirs(os.path.dirname(OUT_PREFIX), exist_ok=True)

for tag, lang, text in CASES:
    res = tts.run(
        {
            "text": text,
            "text_lang": lang,
            "ref_audio_path": REF_WAV,
            "prompt_text": REF_TEXT,
            "prompt_lang": "ja",
            "top_k": 15,
            "top_p": 1.0,
            "temperature": 1.0,
            "text_split_method": "cut5",
            "speed_factor": 1.0,
            "batch_size": 1,
            "seed": 1234,
            "parallel_infer": True,
        }
    )
    sr, audio = None, None
    for sr, audio in res:
        pass
    out = "%s_%s.wav" % (OUT_PREFIX, tag)
    sf.write(out, audio, sr)
    print("[ok] %s  (%.1fs)" % (out, len(audio) / sr))

print("DONE", VERSION)
