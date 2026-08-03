#!/usr/bin/env python3
"""Make a conservative emotion-weighted list from already accepted D_old data."""
from pathlib import Path

DATA = Path("/root/autodl-tmp/ireina_v2/data")
OLD = DATA / "D_old.list"
BASE = DATA / "D_balanced48.list"
OUT = DATA / "D_balanced48_emo20.list"

# 1-based rows from D_old. They contain sighs, hesitation, surprise, anger,
# sadness or expressive sentence endings. The rows are deliberately explicit
# and reviewable; no new broadcast-drama material is introduced here.
ROWS = [5, 9, 14, 15, 18, 19, 22, 42, 46, 49, 54, 68, 71, 74, 81, 88, 91, 107, 124, 148]

old = [line for line in OLD.read_text(encoding="utf-8").splitlines() if line.strip()]
base = [line for line in BASE.read_text(encoding="utf-8").splitlines() if line.strip()]
selected = [old[index - 1] for index in ROWS]
OUT.write_text("/n".join(base + selected) + "/n", encoding="utf-8")
print(f"base={len(base)} emotion_repeat={len(selected)} total={len(base) + len(selected)}")
print(f"wrote={OUT}")
