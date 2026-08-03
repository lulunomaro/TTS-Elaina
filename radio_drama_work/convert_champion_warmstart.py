import os, sys, torch
os.chdir('/root/GPT-SoVITS')
sys.path.insert(0, '/root/GPT-SoVITS')
sys.path.insert(0, '/root/GPT-SoVITS/GPT_SoVITS')
from GPT_SoVITS.process_ckpt import load_sovits_new

src = '/root/autodl-tmp/weights/SoVITS_ireina_s2_b16e2_textonly_lr2e5_v1/ireina_s2_b16e2_textonly_lr2e5_v1_e1_s17.pth'
dst = '/root/autodl-tmp/weights/ireina_champion_text_lr2e5_e1_trainable.pth'
x = load_sovits_new(src)
print('loaded', type(x), list(x.keys()))
torch.save({'weight': x['weight'], 'info': x.get('info', 'champion_e1')}, dst)
print('saved', dst, os.path.getsize(dst))
y = torch.load(dst, map_location='cpu', weights_only=False)
print('verify', list(y.keys()), len(y['weight']))
