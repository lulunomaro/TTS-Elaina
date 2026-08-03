import json,os,sys,soundfile as sf
REPO='/root/GPT-SoVITS'; MAN='/root/autodl-tmp/paired_emo_eval.jsonl'; OUT='/root/autodl-tmp/infer_param_remaining_v1'; REF='/root/autodl-tmp/dataset_v1/wavs/old_my_voice.mp3_0014764160_0014986560.wav.refmt_vocals.wav'; PROMPT='王立セレステリアという国がどこか私の旅そのものと重なっているからなのかもしれません。'; GPT='/root/autodl-tmp/weights/GPT_ireina_s2_old_v2/ireina_s2_old_v2-e6.ckpt'; SOVITS='/root/autodl-tmp/weights/SoVITS_ireina_s2_b16e2_textonly_lr2e5_v1/ireina_s2_b16e2_textonly_lr2e5_v1_e1_s17.pth'
PARAMS={'stable':dict(top_k=20,top_p=.90,temperature=.85,repetition_penalty=1.20),'conservative':dict(top_k=15,top_p=.88,temperature=.70,repetition_penalty=1.28),'balanced':dict(top_k=10,top_p=.85,temperature=.75,repetition_penalty=1.25)}
special={'r018','r054','r083','r086','r091','r107','r124','r146','r152','r154'}
rows=[]
for line in open(MAN,encoding='utf-8'):
 if line.strip():
  row=json.loads(line)
  if row['sample_id'] not in special: rows.append(row)
sys.path.insert(0,REPO);sys.path.insert(0,os.path.join(REPO,'GPT_SoVITS'));os.chdir(REPO)
from GPT_SoVITS.TTS_infer_pack.TTS import TTS,TTS_Config
os.makedirs(OUT,exist_ok=True);cfg=TTS_Config(os.path.join(REPO,'GPT_SoVITS/configs/tts_infer.yaml'));cfg.device='cuda';cfg.is_half=True;cfg.version='v2Pro';tts=TTS(cfg);tts.init_t2s_weights(GPT);tts.init_vits_weights(SOVITS)
for name,params in PARAMS.items():
 print('===',name,flush=True)
 for row in rows:
  req={'text':row['text'],'text_lang':'ja','ref_audio_path':REF,'prompt_text':PROMPT,'prompt_lang':'ja','batch_size':1,'speed_factor':1.0,'text_split_method':'cut5','seed':1234,'parallel_infer':True,'return_fragment':False};req.update(params);sr,a=next(tts.run(req));sf.write(os.path.join(OUT,f'{name}__{row["sample_id"]}.wav'),a,sr)
print('DONE')
