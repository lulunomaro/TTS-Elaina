#!/usr/bin/env python3
"""Build a broad emotion/prosody-focused list from accepted D_old data."""
from pathlib import Path

DATA = Path("/root/autodl-tmp/ireina_v2/data")
OLD = DATA / "D_old.list"
BASE = DATA / "D_balanced48.list"
OUT = DATA / "D_balanced48_emo_focus_v2.list"

# Expressive, hesitant, surprised, angry, sad, laughing, sighing and ending rows.
ROWS = [14, 15, 18, 19, 22, 26, 27, 34, 38, 42, 46, 47, 48, 49, 53, 54,
        55, 58, 63, 64, 65, 68, 69, 71, 73, 74, 76, 80, 81, 83, 84, 86,
        87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 107, 111, 113,
        114, 119, 120, 121, 122, 123, 124, 126, 127, 134, 136, 137, 138,
        139, 140, 141, 142, 145, 146, 147, 148, 151, 152, 153, 154]
# Rows most likely to expose synthetic tails get two extra repeats.
CRITICAL = [83, 86, 91, 96, 107, 124, 146, 152, 154]

old = [line for line in OLD.read_text(encoding="utf-8").splitlines() if line.strip()]
base = [line for line in BASE.read_text(encoding="utf-8").splitlines() if line.strip()]
selected = [old[index - 1] for index in ROWS]
extra = [old[index - 1] for index in CRITICAL for _ in range(2)]
OUT.write_text("/n".join(base + selected + extra) + "/n", encoding="utf-8")
print(f"base={len(base)} focus={len(selected)} critical_extra={len(extra)} total={len(base) + len(selected) + len(extra)}")
print(f"wrote={OUT}")
print("line83=", old[82].split("|", 3)[-1])
