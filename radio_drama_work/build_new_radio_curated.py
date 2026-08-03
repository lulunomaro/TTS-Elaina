import json, math, os
from pathlib import Path
import numpy as np
import soundfile as sf

BASE = Path('/root/autodl-tmp/ireina_v2/data/D_balanced48_emo_focus_v2.list')
ASR = Path('/root/autodl-tmp/ireina_new_materials_v1/asr/asr.json')
SV = {r['name']: r for r in json.loads(Path('/root/autodl-tmp/ireina_new_materials_v1/sv_scores.json').read_text())}
OUT_DIR = Path('/root/autodl-tmp/ireina_new_materials_v1')
OUT = OUT_DIR / 'new_radio_curated.list'

def tail_stats(path):
    y, sr = sf.read(path, always_2d=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1: y = y.mean(axis=1)
    n = len(y); tail = y[max(0, n-int(.20*sr)):]; prev = y[max(0, n-int(.70*sr)):max(0, n-int(.20*sr))]
    rms = lambda z: float(np.sqrt(np.mean(np.square(z)) + 1e-12))
    db = lambda x: 20*math.log10(max(x, 1e-6))
    z = tail*np.hanning(max(1,len(tail)))
    spec = np.abs(np.fft.rfft(z, n=max(2048, 1 << max(1, (len(z)-1).bit_length()))))
    freqs = np.fft.rfftfreq(len(spec)*2-2, 1/sr)
    hi = float(np.mean(spec[(freqs>=5000)&(freqs<=min(12000,sr/2))]))
    lo = float(np.mean(spec[(freqs>=80)&(freqs<=1000)]))
    return dict(sec=n/sr, tail_minus_prev_db=db(rms(tail))-db(rms(prev)), tail_high_low=hi/max(lo,1e-7))

reject_phrases = ('ブレーコー', 'なんか言ってる', 'バカになっちゃった', 'うん何', 'しかしいいんですか')
rows = []
for row in json.loads(ASR.read_text(encoding='utf-8')):
    s = SV[row['name']]
    text = (row.get('text') or '').strip()
    reasons=[]
    if s['median'] < .75: reasons.append('sv_median')
    if s['fraction_ge_075'] < .5: reasons.append('sv_fraction')
    if row.get('avg_logprob', -99) < -.45: reasons.append('asr_lp')
    if not (3 <= len(text) <= 30): reasons.append('text_len')
    if any(x in text for x in reject_phrases): reasons.append('mixed_or_uncertain')
    if reasons: continue
    # Correct the two recurring ASR name variants only; all other text remains ASR-verbatim.
    text = text.replace('イレーナ', 'イレイナ').replace('アビリア', 'アベリア')
    q = tail_stats(row['file'])
    if q['tail_minus_prev_db'] > 5.0: reasons.append('tail_jump')
    if reasons: continue
    rows.append({**row, **{k:s[k] for k in ('best_score','median','fraction_ge_075','fraction_ge_080')}, 'text': text, **q})

rows.sort(key=lambda r: (-r['median'], -r['best_score'], r['name']))
lines = [x for x in BASE.read_text(encoding='utf-8').splitlines() if x.strip()]
for r in rows:
    lines.append(f"{r['file']}|new_radio|JA|{r['text']}")
OUT.write_text('/n'.join(lines)+'/n', encoding='utf-8')
(OUT_DIR/'new_radio_curated.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT_DIR/'new_radio_curated_only.list').write_text('/n'.join(f"{r['file']}|new_radio|JA|{r['text']}" for r in rows)+'/n',encoding='utf-8')
print('baseline', len(lines)-len(rows), 'new_selected', len(rows), 'merged', len(lines))
for r in rows:
    print(f"{r['median']:.3f}/t{r['best_score']:.3f}/t{r['tail_minus_prev_db']:.1f}dB/t{r['name']}/t{r['text']}")
print('wrote', OUT)
