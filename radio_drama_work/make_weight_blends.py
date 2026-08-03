import os, torch
os.chdir('/root/GPT-SoVITS'); import sys; sys.path.insert(0,'/root/GPT-SoVITS'); sys.path.insert(0,'/root/GPT-SoVITS/GPT_SoVITS')
from GPT_SoVITS.process_ckpt import load_sovits_new

base='/root/autodl-tmp/weights/SoVITS_ireina_s2_b16e2_textonly_lr2e5_v1/ireina_s2_b16e2_textonly_lr2e5_v1_e1_s17.pth'
models={
 'text': '/root/autodl-tmp/weights/SoVITS_ireina_s2_newradio_text_lr15_b32_v2/ireina_s2_newradio_text_lr15_b32_v2_e1_s12.pth',
 'dold': '/root/autodl-tmp/weights/SoVITS_ireina_s2_dold_text_lr1e5_b32_v1/ireina_s2_dold_text_lr1e5_b32_v1_e1_s8.pth',
 'gonly': '/root/autodl-tmp/weights/SoVITS_ireina_s2_newradio_gonly_lr5e6_b24_v2/ireina_s2_newradio_gonly_lr5e6_b24_v2_e1_s14.pth',
}
def load(p): return load_sovits_new(p)
bobj=load(base); b=bobj['weight']; os.makedirs('/root/autodl-tmp/blends_ireina_v1',exist_ok=True)
for tag, path, alpha in [('text25',models['text'],.25),('text50',models['text'],.50),('dold25',models['dold'],.25),('gonly20',models['gonly'],.20)]:
    x=load(path)['weight']; out={}
    for k,v in b.items():
        if k not in x: out[k]=v; continue
        if torch.is_floating_point(v): out[k]=(1-alpha)*v.float()+alpha*x[k].float()
        else: out[k]=v
    dst=f'/root/autodl-tmp/blends_ireina_v1/{tag}.pth'; torch.save({'weight':out,'config':bobj['config'],'info':f'blend {tag}'},dst); print(tag,os.path.getsize(dst),len(out),flush=True)
