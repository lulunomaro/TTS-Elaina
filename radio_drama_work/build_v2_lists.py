# -*- coding: utf-8 -*-
"""构建本轮训练使用的独立数据清单，不修改 dataset_v1。"""

from pathlib import Path


ROOT = Path(r"radio_drama_work/radio_drama_work")
SRC = ROOT / "dataset_v1" / "dataset.list"
OUT = ROOT / "training_v2"

# 这 12 段低于 old 中心相似度 0.70；先从 curated 训练集排除，保留在边界清单中，
# 后续试听后再决定是否作为气声/短音数据加入。
BAD_RADIO = {
    "113814583-1-160_vocals.wav_0000747520_0000830720.wav",
    "280479210-1-192_vocals.wav_0005022720_0005115520.wav",
    "280479210-1-192_vocals.wav_0002991360_0003101440.wav",
    "113814583-1-160_vocals.wav_0000629760_0000747520.wav",
    "280479210-1-192_vocals.wav_0020792320_0020936320.wav",
    "280479210-1-192_vocals.wav_0001879040_0002090880.wav",
    "280479210-1-192_vocals.wav_0005405440_0005538560.wav",
    "113814583-1-160_vocals.wav_0004064640_0004161280.wav",
    "113814583-1-160_vocals.wav_0001294720_0001427200.wav",
    "113814583-1-160_vocals.wav_0000853760_0000960640.wav",
    "280479210-1-192_vocals.wav_0012567040_0012675840.wav",
    "113814583-1-160_vocals.wav_0020286720_0020398080.wav",
}


def basename_without_prefix(path: str) -> str:
    name = Path(path.replace("//", "/")).name
    return name[6:] if name.startswith("radio_") else name


def main() -> None:
    lines = [line for line in SRC.read_text(encoding="utf-8").splitlines() if line.strip()]
    old = []
    radio_ok = []
    held_out = []
    for line in lines:
        fields = line.split("|")
        if len(fields) < 4:
            continue
        speaker = fields[1]
        if speaker == "old":
            old.append(line)
        elif speaker == "rd":
            if basename_without_prefix(fields[0]) in BAD_RADIO:
                held_out.append(line)
            else:
                radio_ok.append(line)

    curated = old + radio_ok
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "D_old.list").write_text("/n".join(old) + "/n", encoding="utf-8")
    (OUT / "D_radio_ok.list").write_text("/n".join(radio_ok) + "/n", encoding="utf-8")
    (OUT / "D_curated.list").write_text("/n".join(curated) + "/n", encoding="utf-8")
    (OUT / "held_out_low_sv.list").write_text("/n".join(held_out) + "/n", encoding="utf-8")

    print(f"source={len(lines)} old={len(old)} radio_ok={len(radio_ok)} held_out={len(held_out)} curated={len(curated)}")
    print(f"out={OUT}")


if __name__ == "__main__":
    main()
