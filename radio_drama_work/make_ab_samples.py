# -*- coding: utf-8 -*-
"""
生成「分离前 vs 分离后」A/B 对照样本，供人耳判断广播剧数据质量。
对每个选中片段输出:
  *_1raw.wav   原始混音(分离前, 含BGM/音效/其他角色)
  *_2vocal.wav 分离后人声(实际训练用的那份)
  *_3inst.wav  分离出的伴奏(用于检查有没有把人声漏到伴奏里)
并打印频谱损伤指标。
"""
import os, sys, io
import numpy as np
import soundfile as sf

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

WORK = r"radio_drama_work/radio_drama_work"
OUT = os.path.join(WORK, "ab_check")
os.makedirs(OUT, exist_ok=True)

# (标签, 切片文件名, 说明)
PICKS = [
    ("best",   "113814583-1-160_vocals.wav_0008851200_0009012480.wav", "SNR 77.2dB 全库最干净档"),
    ("median", "280479210-1-192_vocals.wav_0002197760_0002446080.wav", "SNR 72.8dB 中位水平"),
    ("worst",  "113814583-1-160_vocals.wav_0016768640_0016999680.wav", "SNR 27.6dB 全库最脏档"),
    ("ref",    "113814583-1-160_vocals.wav_0022915200_0023071360.wav", "被选作新参考音频的那条"),
]

SR_SLICE = 32000  # 切片采样率, 文件名中的数字以此为单位


def spec_report(x, sr, tag):
    """打印频谱能量分布, 检查高频截止与频谱空洞"""
    if x.ndim > 1:
        x = x.mean(1)
    if len(x) < 2048:
        return
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    fr = np.fft.rfftfreq(len(x), 1 / sr)
    tot = (X ** 2).sum() + 1e-20
    bands = [(0, 1000), (1000, 4000), (4000, 8000), (8000, 12000),
             (12000, 16000), (16000, 20000), (20000, sr // 2)]
    parts = []
    for lo, hi in bands:
        m = (fr >= lo) & (fr < hi)
        if m.any():
            parts.append("%dk:%.1f%%" % (hi // 1000, 100 * (X[m] ** 2).sum() / tot))
    c = np.cumsum(X ** 2) / tot
    roll = fr[np.searchsorted(c, 0.99)]
    print("      %-7s rolloff99=%5.0fHz  %s" % (tag, roll, " ".join(parts)))


def main():
    for label, name, desc in PICKS:
        src = os.path.join(WORK, "radio_drama_pure", name)
        if not os.path.exists(src):
            src = os.path.join(WORK, "radio_drama", name)
        if not os.path.exists(src):
            print("[skip] not found:", name)
            continue

        # 解析文件名: <base>_vocals.wav_<start>_<end>.wav
        stem = name[:-4]
        parts = stem.split("_")
        start32, end32 = int(parts[-2]), int(parts[-1])
        base = stem.split("_vocals.wav_")[0]

        raw_p = os.path.join(WORK, "raw", base + ".wav")
        voc_p = os.path.join(WORK, "vocals", base + "_vocals.wav")
        ins_p = os.path.join(WORK, "ins", base + "_other.wav")

        info = sf.info(raw_p)
        sr_full = info.samplerate
        ratio = sr_full / SR_SLICE
        s = int(start32 * ratio)
        e = int(end32 * ratio)
        n = e - s

        print("/n=== [%s] %s" % (label, desc))
        print("      %s  %.1fs" % (name, (end32 - start32) / SR_SLICE))

        # 分离前原始混音
        raw, _ = sf.read(raw_p, start=s, frames=n)
        sf.write(os.path.join(OUT, "%s_1raw.wav" % label), raw, sr_full)
        spec_report(raw, sr_full, "raw")

        # 分离后人声(全采样率版, 便于公平比较)
        voc, _ = sf.read(voc_p, start=s, frames=n)
        sf.write(os.path.join(OUT, "%s_2vocal.wav" % label), voc, sr_full)
        spec_report(voc, sr_full, "vocal")

        # 伴奏
        if os.path.exists(ins_p):
            ins, _ = sf.read(ins_p, start=s, frames=n)
            sf.write(os.path.join(OUT, "%s_3inst.wav" % label), ins, sr_full)
            spec_report(ins, sr_full, "inst")

        # 实际训练用的那份(32k 单声道)
        tr, tsr = sf.read(src)
        sf.write(os.path.join(OUT, "%s_4train32k.wav" % label), tr, tsr)
        spec_report(tr, tsr, "train")

    print("/nDONE ->", OUT)


if __name__ == "__main__":
    main()
