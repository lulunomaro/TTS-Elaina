# -*- coding: utf-8 -*-
"""伊蕾娜 v2Pro 最终推理脚本（定稿版）
组合：官方 GPT (s1v3) + 微调 S2 v2Pro (e2)
参考：QW1106（韵律正常）为主，QW0001 备用
参数：top_k=20 / top_p=0.60 / temperature=0.60
机制：重试直至完整句（2-15s），限长防超长
用法：
  python final_infer.py "こんにちは、私はイレイナです。よろしくお願いします。" out.wav [--ref QW1106]
"""
import argparse
import io
import os
import subprocess
import time

import numpy as np
import soundfile as sf

PKG = r"F:\GPTSOVITS\GPT-SoVITS-v2pro-20250604-nvidia50\GPT-SoVITS-v2pro-20250604-nvidia50"
PY = os.path.join(PKG, "runtime", "python.exe")
EXP = os.path.join(PKG, "logs", "ireina_qwen_v1")

GPT_MODEL = os.path.join(PKG, "GPT_SoVITS", "pretrained_models", "s1v3.ckpt")  # 官方 GPT（不动）
SOVITS_MODEL = os.path.join(EXP, "s2G_finetune_v2pro_e2.pth")  # 微调 S2 v2Pro e2

REFS = {
    "QW1106": (r"F:\GPTSOVITS\radio_drama_work\ireina_qwen_v1\audio\QW1106_majo_ep12.wav", "て激しい議論の末にようやく本のタイトルが決まったのでした"),
    "QW0001": (r"F:\GPTSOVITS\radio_drama_work\ireina_qwen_v1\audio\QW0001_113814583-1-160.wav", "魔女さんはこの島のことを知っていますか？私はまだよく知らないので"),
    "QW0005": (r"F:\GPTSOVITS\radio_drama_work\ireina_qwen_v1\audio\QW0005_113814583-1-160.wav", "私の一存で尽きるようなお話です。それがフレイアさんと前世の出会いのきっかけになったことになるのです"),
}
LIST = r"F:\GPTSOVITS\radio_drama_work\ireina_qwen_v1\ireina_qwen_v1_clean.list"


def real_text(name):
    for l in io.open(LIST, encoding="utf-8").read().splitlines():
        p = l.split("|")
        if os.path.basename(p[0]) == name:
            return p[3]
    return REFS[name.split("_")[0]][1] if "_" not in name else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", help="目标文本（日语）")
    ap.add_argument("output", help="输出 wav 路径")
    ap.add_argument("--ref", default="QW1106", choices=list(REFS))
    ap.add_argument("--max_attempts", type=int, default=15)
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = PKG + os.pathsep + os.path.join(PKG, "GPT_SoVITS") + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = "0"

    wav, fallback_txt = REFS[args.ref]
    ref_txt = real_text(os.path.basename(wav)) or fallback_txt
    ref_f = os.path.join(out_dir, "_ref.txt")
    tgt_f = os.path.join(out_dir, "_target.txt")
    io.open(ref_f, "w", encoding="utf-8").write(ref_txt)
    io.open(tgt_f, "w", encoding="utf-8").write(args.text)

    def check(path):
        a, sr = sf.read(path, dtype="float32")
        dur = len(a) / sr
        n = int(sr * 0.2)
        mid = float(np.sqrt(np.mean(a[len(a) // 2: len(a) // 2 + n] ** 2))) if len(a) > 2 * n else 0.0
        return 1.5 <= dur <= 15.0 and mid > 0.005

    for attempt in range(1, args.max_attempts + 1):
        t0 = time.time()
        r = subprocess.run([
            PY, "-s", os.path.join(PKG, "GPT_SoVITS", "inference_cli.py"),
            "--gpt_model", GPT_MODEL,
            "--sovits_model", SOVITS_MODEL,
            "--ref_audio", wav,
            "--ref_text", ref_f,
            "--ref_language", "日文",
            "--target_text", tgt_f,
            "--target_language", "日文",
            "--output_path", out_dir,
            "--top_k", "20", "--top_p", "0.60", "--temperature", "0.60",
        ], env=env, cwd=PKG)
        if r.returncode == 0:
            src = os.path.join(out_dir, "output.wav")
            if os.path.exists(args.output):
                os.remove(args.output)
            os.rename(src, args.output)
            if check(args.output):
                print(f"OK attempt {attempt} ({time.time()-t0:.0f}s) -> {args.output}")
                return 0
            print(f"attempt {attempt} reject (%.0fs)" % (time.time() - t0))
        else:
            print(f"attempt {attempt} FAIL")
    print("NO GOOD SAMPLE")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
