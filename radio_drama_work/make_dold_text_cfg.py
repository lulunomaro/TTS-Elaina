import json, os, shutil
BASE='/root/autodl-tmp/exp/ireina_s2_old_v2'
NAME='ireina_s2_dold_text_lr1e5_b32_v1'
EXP=f'/root/autodl-tmp/exp/{NAME}'; OUT=f'/root/autodl-tmp/weights/SoVITS_{NAME}'
if os.path.exists(EXP): raise SystemExit(f'exists {EXP}')
os.makedirs(os.path.join(EXP,'cfg'))
for fn in ('2-name2text.txt','6-name2semantic.tsv'): shutil.copy2(os.path.join(BASE,fn),os.path.join(EXP,fn))
for dn in ('3-bert','4-cnhubert','5-wav32k','7-sv_cn'): os.symlink(os.path.join(BASE,dn),os.path.join(EXP,dn),target_is_directory=True)
with open('/root/autodl-tmp/exp/ireina_s2_emo_focus_v2_b16_lr75e5_dlr1e5/cfg/s2.json',encoding='utf-8') as f: cfg=json.load(f)
cfg['train'].update({'epochs':2,'batch_size':32,'save_every_epoch':1,'if_save_latest':False,'if_save_every_weights':True,'learning_rate':1e-5,'d_learning_rate':0.0,'text_low_lr_rate':1.0,'pretrained_s2G':'/root/autodl-tmp/weights/ireina_champion_text_lr2e5_e1_trainable.pth','pretrained_s2D':'/root/GPT-SoVITS/GPT_SoVITS/pretrained_models/v2Pro/s2Dv2Pro.pth','gpu_numbers':'0','grad_ckpt':False,'fp16_run':True,'max_steps_per_epoch':8,'text_only':True})
cfg['data']['exp_dir']=EXP; cfg['s2_ckpt_dir']=EXP; cfg['save_weight_dir']=OUT; cfg['name']=NAME; cfg['version']='v2Pro'; cfg['model']['version']='v2Pro'
os.makedirs(OUT)
with open(os.path.join(EXP,'cfg','s2.json'),'w',encoding='utf-8') as f: json.dump(cfg,f,indent=2,ensure_ascii=False)
print('created',EXP)
