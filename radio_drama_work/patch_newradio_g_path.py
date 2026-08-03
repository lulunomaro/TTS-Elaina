import json
for name in ('ireina_s2_newradio_text_lr15_b32_v2','ireina_s2_newradio_gonly_lr5e6_b24_v2'):
    p=f'/root/autodl-tmp/exp/{name}/cfg/s2.json'
    with open(p,encoding='utf-8') as f: cfg=json.load(f)
    cfg['train']['pretrained_s2G']='/root/autodl-tmp/weights/ireina_champion_text_lr2e5_e1_trainable.pth'
    with open(p,'w',encoding='utf-8') as f: json.dump(cfg,f,indent=2,ensure_ascii=False)
    print('patched',p)
