import json, os, sys, soundfile as sf
REPO='/root/GPT-SoVITS'; MAN='/root/autodl-tmp/paired_emo_eval.jsonl'; OUT='/root/autodl-tmp/infer_dold_text14_v1'; REF='/root/autodl-tmp/dataset_v1/wavs/old_my_voice.mp3_0014764160_0014986560.wav.refmt_vocals.wav'; PROMPT='王立セレステリアという国がどこか私の旅そのものと重なっているからなのかもしれません。'; GPT='/root/autodl-tmp/weights/GPT_ireina_s2_old_v2/ireina_s2_old_v2-e6.ckpt'; SOVITS='/root/autodl-tmp/weights/SoVITS_ireina_s2_dold_text_lr1e5_b32_s14_v1/ireina_s2_dold_text_lr1e5_b32_s14_v1_e1_s14.pth'
rows=[json.loads(x) for x in open(MAN,encoding='utf-8') if x.strip()]; sys.path.insert(0,REPO);sys.path.insert(0,os.path.join(REPO,'GPT_SoVITS'));os.chdir(REPO)
from GPT_SoVITS.TTS_infer_pack.TTS import TTS,TTS_Config
os.makedirs(OUT,exist_ok=True);cfg=TTS_Config(os.path.join(REPO,'GPT_SoVITS/configs/tts_infer.yaml'));cfg.device='cuda';cfg.is_half=True;cfg.version='v2Pro';tts=TTS(cfg);tts.init_t2s_weights(GPT);tts.init_vits_weights(SOVITS)
for row in rows:
 req={'text':row['text'],'text_lang':'ja','ref_audio_path':REF,'prompt_text':PROMPT,'prompt_lang':'ja','batch_size':1,'speed_factor':1.0,'text_split_method':'cut5','seed':1234,'parallel_infer':True,'return_fragment':False,'top_k':30,'top_p':.95,'temperature':.95,'repetition_penalty':1.15};sr,a=next(tts.run(req));sf.write(os.path.join(OUT,f'dold_text14__{row["sample_id"]}.wav'),a,sr)
print('DONE')
