import json, os, shutil

BASE = '/root/autodl-tmp/exp/ireina_s2_newradio_curated_v1'
G = '/root/autodl-tmp/weights/SoVITS_ireina_s2_b16e2_textonly_lr2e5_v1/ireina_s2_b16e2_textonly_lr2e5_v1_e1_s17.pth'
D = '/root/GPT-SoVITS/GPT_SoVITS/pretrained_models/v2Pro/s2Dv2Pro.pth'

def make(name, *, text_only, lr, batch, epochs, max_steps, text_lr_rate):
    exp = f'/root/autodl-tmp/exp/{name}'
    out = f'/root/autodl-tmp/weights/SoVITS_{name}'
    if os.path.exists(exp): raise SystemExit(f'exists {exp}')
    os.makedirs(os.path.join(exp, 'cfg'))
    for fn in ('2-name2text.txt','6-name2semantic.tsv'):
        shutil.copy2(os.path.join(BASE,fn), os.path.join(exp,fn))
    for dn in ('3-bert','4-cnhubert','5-wav32k','7-sv_cn'):
        os.symlink(os.path.join(BASE,dn), os.path.join(exp,dn), target_is_directory=True)
    with open('/root/autodl-tmp/exp/ireina_s2_emo_focus_v2_b16_lr75e5_dlr1e5/cfg/s2.json', encoding='utf-8') as f: cfg=json.load(f)
    tr=cfg['train']; tr.update({
        'epochs':epochs, 'batch_size':batch, 'save_every_epoch':1,
        'if_save_latest':False, 'if_save_every_weights':True,
        'learning_rate':lr, 'd_learning_rate':0.0,
        'text_low_lr_rate':text_lr_rate, 'pretrained_s2G':G,
        'pretrained_s2D':D, 'gpu_numbers':'0', 'grad_ckpt':False,
        'fp16_run':True, 'max_steps_per_epoch':max_steps,
        'text_only':text_only,
    })
    cfg['data']['exp_dir']=exp; cfg['s2_ckpt_dir']=exp; cfg['save_weight_dir']=out
    cfg['name']=name; cfg['version']='v2Pro'; cfg['model']['version']='v2Pro'
    os.makedirs(out)
    with open(os.path.join(exp,'cfg','s2.json'),'w',encoding='utf-8') as f: json.dump(cfg,f,indent=2,ensure_ascii=False)
    print(f'created {name} text_only={text_only} lr={lr} batch={batch} seg={tr["segment_size"]} epochs={epochs} max_steps={max_steps}')

make('ireina_s2_newradio_text_lr15_b32_v2', text_only=True, lr=1.5e-5, batch=32, epochs=2, max_steps=12, text_lr_rate=1.0)
make('ireina_s2_newradio_gonly_lr5e6_b24_v2', text_only=False, lr=5e-6, batch=24, epochs=1, max_steps=14, text_lr_rate=0.03)
