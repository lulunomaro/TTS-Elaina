import json, os, shutil
MAN='/root/autodl-tmp/paired_emo_eval.jsonl'; SRC='/root/autodl-tmp/infer_param_grid_v1'; EXP='/root/autodl-tmp/infer_expressive_full_v1'; OUT='/root/autodl-tmp/infer_router_v1'
special={'r018':'expressive','r054':'conservative','r083':'stable','r086':'expressive','r091':'stable','r107':'conservative','r124':'conservative','r146':'balanced','r152':'stable','r154':'conservative'}
os.makedirs(OUT,exist_ok=True)
for line in open(MAN,encoding='utf-8'):
 if not line.strip(): continue
 sid=json.loads(line)['sample_id']; preset=special.get(sid,'expressive')
 src=os.path.join(SRC if sid in special else EXP, f'{preset}__{sid}.wav' if sid in special else f'expressive__{sid}.wav')
 dst=os.path.join(OUT,f'router__{sid}.wav'); shutil.copy2(src,dst); print(sid,preset)
