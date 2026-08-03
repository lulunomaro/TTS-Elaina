import sys,os,torch
os.chdir('/root/GPT-SoVITS');sys.path.insert(0,'/root/GPT-SoVITS');sys.path.insert(0,'/root/GPT-SoVITS/GPT_SoVITS')
from GPT_SoVITS.process_ckpt import load_sovits_new
paths={
'base':'/root/autodl-tmp/weights/SoVITS_ireina_s2_b16e2_textonly_lr2e5_v1/ireina_s2_b16e2_textonly_lr2e5_v1_e1_s17.pth',
'new':'/root/autodl-tmp/weights/SoVITS_ireina_s2_newradio_text_lr15_b32_v2/ireina_s2_newradio_text_lr15_b32_v2_e1_s12.pth',
'blend':'/root/autodl-tmp/blends_ireina_v1/text25.pth'}
for tag,p in paths.items():
 x=load_sovits_new(p);m=x['config']['model'];w=x['weight'];print(tag,'gin',m['gin_channels'],'hidden',m['hidden_channels'],'sv',tuple(w['sv_emb.weight'].shape),'ssl',tuple(w['ssl_proj.weight'].shape),'ge',tuple(w['ge_to512.weight'].shape),'config_type',type(x['config']))
