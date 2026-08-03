import json, os, shutil
OUT='/root/autodl-tmp/infer_best_router_v1'; os.makedirs(OUT,exist_ok=True)
sources={
 '/root/autodl-tmp/naturalness_param_grid_v1/naturalness_scores_v1.json':'/root/autodl-tmp/infer_param_grid_v1',
 '/root/autodl-tmp/naturalness_param_remaining_v1/naturalness_scores_v1.json':'/root/autodl-tmp/infer_param_remaining_v1',
 '/root/autodl-tmp/naturalness_expressive_full_v1/naturalness_scores_v1.json':'/root/autodl-tmp/infer_expressive_full_v1',
}
best={}
for score_path, directory in sources.items():
 payload=json.load(open(score_path))
 for row in payload['files']:
  sid=row['sample_id']; score=row['naturalness_score_v1']
  if sid not in best or score>best[sid][0]: best[sid]=(score,row['name'],directory,row)
for sid,(score,name,directory,row) in sorted(best.items()):
 src=os.path.join(directory,name); dst=os.path.join(OUT,f'best_router__{sid}.wav'); shutil.copy2(src,dst); print(sid,f'{score:.2f}',name)
print('selected',len(best),'mean',sum(x[0] for x in best.values())/len(best))
