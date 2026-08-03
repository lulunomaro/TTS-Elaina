#!/bin/bash
set -euo pipefail
DATA=/root/autodl-tmp/ireina_v2/data
BASE=$DATA/D_balanced48.list
ADD=/root/autodl-tmp/new_fine_curated.list
OUT=$DATA/D_balanced48_plus18.list
cat "$BASE" "$ADD" > "$OUT"
echo "base=$(wc -l < "$BASE") add=$(wc -l < "$ADD") total=$(wc -l < "$OUT")"
echo "out=$OUT"
