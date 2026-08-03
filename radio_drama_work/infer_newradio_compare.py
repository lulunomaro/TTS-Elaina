import json, os, sys
import soundfile as sf

REPO='/root/GPT-SoVITS'
MANIFEST='/root/autodl-tmp/paired_emo_eval.jsonl'
OUT='/root/autodl-tmp/infer_newradio_compare_v1'
REF='/root/autodl-tmp/dataset_v1/wavs/old_my_voice.mp3_0014764160_0014986560.wav.refmt_vocals.wav'
PROMPT='王立セレステリアという国がどこか私の旅そのものと重なっているからなのかもしれません。'
GPT='/root/autodl-tmp/weights/GPT_ireina_s2_old_v2/ireina_s2_old_v2-e6.ckpt'
SOVITS=[
 ('newradio_text_e1', '/root/autodl-tmp/weights/SoVITS_ireina_s2_newradio_text_lr15_b32_v2/ireina_s2_newradio_text_lr15_b32_v2_e1_s12.pth'),
 ('newradio_gonly_e1', '/root/autodl-tmp/weights/SoVITS_ireina_s2_newradio_gonly_lr5e6_b24_v2/ireina_s2_newradio_gonly_lr5e6_b24_v2_e1_s14.pth'),
]
PARAMS=dict(top_k=20, top_p=0.9, temperature=0.85, repetition_penalty=1.20)
rows=[json.loads(line) for line in open(MANIFEST,encoding='utf-8') if line.strip()]
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,'GPT_SoVITS')); os.chdir(REPO)
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
os.makedirs(OUT,exist_ok=True)
cfg=TTS_Config(os.path.join(REPO,'GPT_SoVITS/configs/tts_infer.yaml')); cfg.device='cuda'; cfg.is_half=True; cfg.version='v2Pro'
tts=TTS(cfg); tts.init_t2s_weights(GPT)
for name, sovits in SOVITS:
    print('===',name,'===',flush=True); tts.init_vits_weights(sovits)
    for row in rows:
        req={'text':row['text'],'text_lang':'ja','ref_audio_path':REF,'prompt_text':PROMPT,'prompt_lang':'ja','batch_size':1,'speed_factor':1.0,'text_split_method':'cut5','seed':1234,'parallel_infer':True,'return_fragment':False}
        req.update(PARAMS); sr,audio=next(tts.run(req)); path=os.path.join(OUT,f'{name}__{row["sample_id"]}.wav'); sf.write(path,audio,sr)
        print('[ok]',row['sample_id'],len(audio)/sr,flush=True)
print('DONE',OUT)
