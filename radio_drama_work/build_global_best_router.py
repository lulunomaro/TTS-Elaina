import json, os, shutil

OUT='/root/autodl-tmp/infer_global_best_v1'
os.makedirs(OUT, exist_ok=True)
sources=[
 ('/root/autodl-tmp/naturalness_param_grid_v1/naturalness_scores_v1.json','/root/autodl-tmp/infer_param_grid_v1'),
 ('/root/autodl-tmp/naturalness_param_remaining_v1/naturalness_scores_v1.json','/root/autodl-tmp/infer_param_remaining_v1'),
 ('/root/autodl-tmp/naturalness_expressive_full_v1/naturalness_scores_v1.json','/root/autodl-tmp/infer_expressive_full_v1'),
 ('/root/autodl-tmp/naturalness_newradio_v1/naturalness_scores_v1.json','/root/autodl-tmp/infer_newradio_compare_v1'),
 ('/root/autodl-tmp/naturalness_dold_text_v1/naturalness_scores_v1.json','/root/autodl-tmp/infer_dold_text_v1'),
 ('/root/autodl-tmp/naturalness_dold_text14_v1/naturalness_scores_v1.json','/root/autodl-tmp/infer_dold_text14_v1'),
]
best={}
for score_path, directory in sources:
 for row in json.load(open(score_path))['files']:
  sid=row['sample_id']; val=row['naturalness_score_v1']
  if sid not in best or val>best[sid][0]: best[sid]=(val,row['name'],directory,row['group'])
for sid,(val,name,directory,group) in sorted(best.items()):
 shutil.copy2(os.path.join(directory,name),os.path.join(OUT,f'global_best__{sid}.wav'))
 print(sid,f'{val:.2f}',group,name)
print('selected',len(best),'mean',sum(x[0] for x in best.values())/len(best))
