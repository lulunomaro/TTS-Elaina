# -*- coding: utf-8 -*-
"""在服务器已有 D_old / D_radio_ok 清单上生成平衡精修清单，不上传音频或转写文本。"""

from pathlib import Path


DATA = Path("/root/autodl-tmp/ireina_v2/data")
NAMES = DATA / "radio_top48_names.txt"
OLD = DATA / "D_old.list"
RADIO = DATA / "D_radio_ok.list"
OUT = DATA / "D_balanced48.list"
RADIO_OUT = DATA / "D_radio_top48.list"


def key(line: str) -> str:
    name = Path(line.split("|", 1)[0].replace("//", "/")).name
    return name[6:] if name.startswith("radio_") else name


names = {line.strip() for line in NAMES.read_text(encoding="utf-8").splitlines() if line.strip()}
old = [line for line in OLD.read_text(encoding="utf-8").splitlines() if line.strip()]
radio = [line for line in RADIO.read_text(encoding="utf-8").splitlines() if line.strip()]
radio_top = [line for line in radio if key(line) in names]
missing = sorted(names - {key(line) for line in radio_top})
if missing:
    raise SystemExit(f"selection names missing from D_radio_ok.list: {missing}")
if len(radio_top) != len(names):
    raise SystemExit(f"selection mismatch: names={len(names)} selected={len(radio_top)}")

RADIO_OUT.write_text("/n".join(radio_top) + "/n", encoding="utf-8")
OUT.write_text("/n".join(old + radio_top) + "/n", encoding="utf-8")
print(f"old={len(old)} radio_top48={len(radio_top)} balanced={len(old) + len(radio_top)}")
print(f"wrote={OUT}")
