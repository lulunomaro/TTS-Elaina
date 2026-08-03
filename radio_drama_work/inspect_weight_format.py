import torch
for p in [
 '/root/autodl-tmp/weights/SoVITS_ireina_s2_b16e2_textonly_lr2e5_v1/ireina_s2_b16e2_textonly_lr2e5_v1_e1_s17.pth',
 '/root/autodl-tmp/weights/ireina_s2_b16e2_warmstart.pth',
 '/root/autodl-tmp/weights/SoVITS_ireina_s2_b16e2_textonly_v1/ireina_s2_b16e2_textonly_v1_e1_s17.pth',
]:
 try:
  x=torch.load(p,map_location='cpu',weights_only=False)
  print(p, type(x), list(x.keys())[:10] if isinstance(x,dict) else None, x.get('info') if isinstance(x,dict) else None)
 except Exception as e: print(p, type(e).__name__, e)
