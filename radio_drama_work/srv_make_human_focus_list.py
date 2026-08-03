#!/usr/bin/env python3
"""Build a conservative D_old + top-ranked radio list on the server."""
from pathlib import Path

DATA = Path("/root/autodl-tmp/ireina_v2/data")
OLD = DATA / "D_old.list"
RADIO = DATA / "D_radio_ok.list"
RANKED = DATA / "radio_top48_names.txt"
# These are the highest-ranked rows after removing obvious overlap,
# incomplete ASR, narrator-like fragments, and the row containing an ASCII
# interjection from another speaker.  Keep the selection explicit and
# reviewable instead of relying on a similarity threshold alone.
OUT = DATA / "D_human_focus14.list"
RANKS = [2, 3, 4, 7, 9, 12, 16, 17, 18, 19, 20, 21, 22, 23]


def key(line: str) -> str:
    name = Path(line.split("|", 1)[0].replace("//", "/")).name
    return name[6:] if name.startswith("radio_") else name


old = [line for line in OLD.read_text(encoding="utf-8").splitlines() if line.strip()]
radio = [line for line in RADIO.read_text(encoding="utf-8").splitlines() if line.strip()]
ranked = [line.strip() for line in RANKED.read_text(encoding="utf-8").splitlines() if line.strip()]
by_key = {key(line): line for line in radio}
selected = [by_key[ranked[index - 1]] for index in RANKS if index <= len(ranked) and ranked[index - 1] in by_key]
if len(selected) != len(RANKS):
    raise SystemExit(f"expected {len(RANKS)} ranked radio rows, got {len(selected)}")
OUT.write_text("/n".join(old + selected) + "/n", encoding="utf-8")
print(f"old={len(old)} radio={len(selected)} total={len(old) + len(selected)}")
print(f"wrote={OUT}")
