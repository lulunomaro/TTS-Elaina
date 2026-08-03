#!/bin/bash
set -u
PY=/root/miniconda3/envs/gptsovits/bin/python
"$PY" - <<'PY'
import importlib.util as u
names = ["librosa", "pyworld", "parselmouth", "speechmos", "transformers", "soundfile", "scipy", "sklearn", "torchaudio"]
print({n: bool(u.find_spec(n)) for n in names})
PY
find /root/autodl-tmp/pkg -maxdepth 5 -type f /( -iname '*mos*' -o -iname '*wavlm*' -o -iname '*hubert*' /) | head -40
