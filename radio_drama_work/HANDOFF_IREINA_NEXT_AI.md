# 伊蕾娜 GPT-SoVITS v2Pro 精修交接文档（交给下一位 AI）

更新时间：2026-08-01 13:09（Asia/Shanghai；本次追加“广播剧混入其他说话人”专项交接）
项目目录：radio_drama_work/radio_drama_work
服务器仓库：/root/GPT-SoVITS
服务器 Python：/root/miniconda3/envs/gptsovits/bin/python
服务器连接脚本：radio_drama_work/remote.py

## 0. 交接时必须先知道的结论

1. 本轮已经停止继续训练。最后一个 batch12/max_sec34 的 S1 GPT 实验在停止命令前已经自然跑完 e12；screen 已不存在，GPU 应为空闲。
2. 当前最高的“单一组合”是：

~~~
GPT high-capacity D_old e12
× SoVITS balanced48_v3 e2_s62
× expressive 采样参数
= naturalness_score_v1 76.96
~~~

3. 旧实验的逐句路由仍然更高：77.38。它是逐句从不同模型/参数选择的上限，不是一个单一模型，不能宣称为单一模型 90 分。
4. “90 分”目前没有达到。这里的分数是代理指标，不是人工 MOS；当前数据和模型条件下，不能把 76～77 说成 90。
5. 不要立即继续堆 SoVITS epoch。历史结果反复显示：SoVITS e2 往往比 e4/e6/e8 更稳，后续 epoch 容易出现伪声、机械感和句尾异常。
6. 服务器数据盘最后一次检查只剩约 456M，使用率 100%。在清理空间前禁止启动任何新训练或大规模推理。

## 1. 用户目标与听感重点

用户要的是尽量贴近原声的伊蕾娜模型，优先级如下：

~~~
真人感 / 自然感
> 机械感、AI 感
> 电音、金属音、伪声
> 叹气和情绪句的自然呼吸
> 普通句语气、停顿、句尾
> 单纯音色相似度
~~~

已确认的用户反馈：

- base_base__emo 机械感最低，但语气不对。
- base GPT 比 fine-tuned GPT 机械感少；fine-tuned GPT 在叹气处机械感重且有电音。
- D_old GPT e6 × D_old e4/e6、D_old GPT e6 × D_curated e4 听起来不错。
- D_old GPT e6 × balanced48 e6 和 D_old GPT e6 × human_focus14 e2 的普通句曾经最好，但情绪句不够好。
- 新精修一度降低真人感，出现伪声；广播剧素材末尾可能有很短的 BGM，因此已经做过句尾污染检测。
- 用户希望把 5090 的 32GB 显存利用起来；本轮已做过 b8 和 b12 的 S1 试验。

## 2. 数据现状

服务器主要数据：

~~~
/root/autodl-tmp/dataset_v1/
/root/autodl-tmp/dataset_v1/dataset.list
/root/autodl-tmp/ireina_v2/data/D_old.list
/root/autodl-tmp/ireina_v2/data/D_radio_ok.list
/root/autodl-tmp/ireina_v2/data/D_curated.list
/root/autodl-tmp/ireina_v2/data/D_balanced48.list
/root/autodl-tmp/ireina_v2/data/D_balanced48_plus18.list
/root/autodl-tmp/ireina_v2/data/D_human_focus14.list
/root/autodl-tmp/ireina_v2/data/D_balanced48_emo20.list
/root/autodl-tmp/ireina_v2/data/D_balanced48_emo_focus_v2.list
~~~

当前清单规模：

| 清单 | 条数 | 说明 |
|---|---:|---|
| D_old.list | 154 | 原始目标人声，当前最可靠的核心数据 |
| D_radio_ok.list | 98 | 广播剧分离 vocal，仍可能混入其他角色/声态 |
| D_curated.list | 252 | old + radio_ok |
| D_radio_top48.list | 48 | speaker 相似度/人工筛选后的广播剧子集 |
| D_balanced48.list | 202 | old + top48，历史上比较稳 |
| D_balanced48_plus18.list | 220 | balanced48 再加 18 条，效果不稳定 |
| D_human_focus14.list | 168 | old + 14 条重点广播剧片段 |
| D_balanced48_emo20.list | 222 | 情绪重点片段，情绪覆盖提升但伪声风险增大 |
| D_balanced48_emo_focus_v2.list | 292 | 更激进，不建议直接作为单说话人全量训练 |
| 当前合并训练集 | 307 左右 | 三段新广播剧最终只增加 15 条高置信度片段 |

三段新广播剧已经经过：BS-Roformer 人声分离、ERes2Net 目标说话人相似度、日语 ASR、句尾 BGM/残留检测和人工/代理筛选。455 个候选最终只留下 15 条高置信度片段加入训练。不要因为广播剧干净就把全部片段加入单说话人训练。

本地数据与新素材处理脚本：

~~~
radio_drama_work/dataset_v1/
radio_drama_work/dataset_v1.zip
radio_drama_work/new_radio_materials.tar
radio_drama_work/build_v2_lists.py
radio_drama_work/build_new_radio_curated.py
radio_drama_work/score_new_radio_sv.sh
radio_drama_work/report_new_radio_candidates.py
~~~

## 3. 服务器空间和训练状态（优先处理）

最后一次只读检查：

~~~
/root/autodl-tmp: 50G total, 50G used, 456M available
~~~

已完成的清理：

- 已删除旧的本轮 ireina_v1 实验中间 S2 G_*.pth、D_*.pth。
- 已删除旧的本轮 ireina_v1 S1 Lightning logs_s1_v2Pro/ckpt。
- 没有删除正式导出权重和数据。

现在占空间的主要新增内容是两轮 high-capacity S1 的 Lightning checkpoint：

~~~
/root/autodl-tmp/exp/ireina_s1_dold_hicap_b8s28_v1/logs_s1_v2Pro/ckpt/
/root/autodl-tmp/exp/ireina_s1_dold_hicap_b12s34_v1/logs_s1_v2Pro/ckpt/
~~~

这两个目录里的 checkpoint 不是推理必需品；对应导出 GPT 权重在独立的 /root/autodl-tmp/weights/GPT_* 目录中。但本交接时没有擅自删除它们，因为用户明确授权过的是本轮新实验的中间 G/D checkpoint，S1 Lightning checkpoint 需要下一位 AI 删除前再次核对/取得确认。

清理前必须先执行：

~~~
df -h /root/autodl-tmp
du -sh /root/autodl-tmp/exp/ireina_s1_dold_hicap_b8s28_v1/*
du -sh /root/autodl-tmp/exp/ireina_s1_dold_hicap_b12s34_v1/*
find /root/autodl-tmp/weights/GPT_ireina_s1_dold_hicap_b8s28_v1 -maxdepth 1 -type f -printf '%f %s/n'
find /root/autodl-tmp/weights/GPT_ireina_s1_dold_hicap_b12s34_v1 -maxdepth 1 -type f -printf '%f %s/n'
~~~

若决定删除，先确认导出权重齐全，再只针对明确的 logs_s1_v2Pro/ckpt 目录操作；不要删除整个实验目录，因为里面有配置、特征链接和日志。

## 4. 已保留的正式权重

### 4.1 旧 D_old GPT

~~~
/root/autodl-tmp/weights/GPT_ireina_s2_old_v2/ireina_s2_old_v2-e3.ckpt
/root/autodl-tmp/weights/GPT_ireina_s2_old_v2/ireina_s2_old_v2-e6.ckpt
/root/autodl-tmp/weights/GPT_ireina_s2_old_v2/ireina_s2_old_v2-e9.ckpt
~~~

历史上 e6 是最值得保留的对照 GPT：

~~~
/root/autodl-tmp/weights/GPT_ireina_s2_old_v2/ireina_s2_old_v2-e6.ckpt
~~~

### 4.2 high-capacity D_old GPT：b8/max_sec28

训练设置：

~~~
实验：ireina_s1_dold_hicap_b8s28_v1
数据：D_old，原始 154 条，经 S1 loader 过滤后有效 129 条
batch_size=8
max_sec=28
先试跑 2 epoch，再从 e2 恢复到 e12
S1 显存实测约 12.5GB / 32GB
~~~

导出目录：

~~~
/root/autodl-tmp/weights/GPT_ireina_s1_dold_hicap_b8s28_v1/
~~~

已存在候选：e1、e2、e4、e6、e8、e10、e12。

### 4.3 high-capacity D_old GPT：b12/max_sec34

训练设置：

~~~
实验：ireina_s1_dold_hicap_b12s34_v1
数据：D_old，S1 loader 有效 130 条
batch_size=12
max_sec=34
先试跑 2 epoch，再从 e2 恢复到 e12
曾观察到显存约 18.97GB / 32GB
~~~

日志已显示：

~~~
Restored all states from .../epoch=1-step=4.ckpt
Trainer.fit stopped: max_epochs=12 reached
~~~

因此 b12 实验实际上已经完成 e12。导出目录：

~~~
/root/autodl-tmp/weights/GPT_ireina_s1_dold_hicap_b12s34_v1/
~~~

已存在候选：e1、e2、e4、e6、e8、e10、e12。注意：b12 只完成了训练，尚未生成配对试听、尚未评分；不能因为 batch 更大就直接宣布更好。

### 4.4 已验证 SoVITS 权重

最重要的候选：

~~~
/root/autodl-tmp/weights/SoVITS_ireina_s2_old_v2/ireina_s2_old_v2_e4_s100.pth
/root/autodl-tmp/weights/SoVITS_ireina_s2_old_v2/ireina_s2_old_v2_e6_s150.pth
/root/autodl-tmp/weights/SoVITS_ireina_s2_balanced48_v3/ireina_s2_balanced48_v3_e2_s62.pth
/root/autodl-tmp/weights/SoVITS_ireina_s2_curated_v2/ireina_s2_curated_v2_e4_s152.pth
/root/autodl-tmp/weights/SoVITS_ireina_s2_balanced48_emo20_dlr2e5_b16_v1/ireina_s2_balanced48_emo20_dlr2e5_b16_v1_e2_s40.pth
~~~

当前最值得优先试听的声学权重：

~~~
/root/autodl-tmp/weights/SoVITS_ireina_s2_balanced48_v3/ireina_s2_balanced48_v3_e2_s62.pth
~~~

不要把 curated_e8、balanced48_e6 等高 epoch 默认当作更好；后期可能过拟合。

## 5. 本轮 high-capacity 评测结果

评测固定参考音频、prompt_text、日语文本、seed=1234，并使用配对 manifest 的自然感代理评分。相关文件：

~~~
/root/autodl-tmp/srv_infer_highcap_compare.py
/root/autodl-tmp/naturalness_score_v1.py
/root/autodl-tmp/paired_emo_eval.jsonl
/root/autodl-tmp/naturalness_highcap_expr_v1/naturalness_groups_v1.tsv
/root/autodl-tmp/naturalness_highcap_stable_v1/naturalness_groups_v1.tsv
/root/autodl-tmp/infer_highcap_paired_v1/
/root/autodl-tmp/infer_highcap_stable_v1/
~~~

### 5.1 expressive 参数

~~~
top_k=30
top_p=0.95
temperature=0.95
repetition_penalty=1.15
seed=1234
~~~

最高组：

| GPT | SoVITS | 分数 |
|---|---|---:|
| b8 e12 | balanced48 e2 | 76.96 |
| b8 e4 | balanced48 e2 | 76.66 |
| b8 e10 | balanced48 e2 | 76.47 |
| b8 e12 | emo_balanced e2 | 76.41 |
| b8 e6 | balanced48 e2 | 76.07 |
| b8 e2 | balanced48 e2 | 75.97 |
| b8 e12 | old e4 | 74.81 |
| b8 e4 | old e6 | 74.80 |

旧对照：

~~~
旧单模型 expressive：75.62
旧逐句 global_best 路由：77.38
~~~

### 5.2 stable 参数

~~~
top_k=20
top_p=0.90
temperature=0.85
repetition_penalty=1.20
seed=1234
~~~

补测结果：

| GPT | SoVITS | 分数 |
|---|---|---:|
| b8 e10 | balanced48 e2 | 76.15 |
| b8 e4 | balanced48 e2 | 75.84 |
| b8 e12 | balanced48 e2 | 75.43 |

结论：本轮 expressive 参数优于 stable；当前新增单模型的首选是 b8 e12 × balanced48 e2 × expressive。

## 6. 当前生产建议

如果需要一个单一模型，优先使用：

~~~
GPT:
/root/autodl-tmp/weights/GPT_ireina_s1_dold_hicap_b8s28_v1/ireina_s1_dold_hicap_b8s28_v1-e12.ckpt

SoVITS:
/root/autodl-tmp/weights/SoVITS_ireina_s2_balanced48_v3/ireina_s2_balanced48_v3_e2_s62.pth

采样：
top_k=30
top_p=0.95
temperature=0.95
repetition_penalty=1.15
seed=1234
~~~

正式替换旧模型前，下一位 AI 必须补听：普通叙述、情绪/惊讶句、叹气句、句尾收音句，以及用户之前指出的 to te mo ... 类情绪句。

如果允许逐句路由，现有本地 global_best 试听包仍是更高上限：

~~~
radio_drama_work/final_router_listen/ireina_global_best.html
radio_drama_work/ireina_global_best_listen.tar.gz
~~~

它不是一个单模型权重，只适合试听和证明不同句子需要不同 GPT/SoVITS/采样参数。

## 7. 下一位 AI 的安全执行顺序

### 第一步：不要训练，先清点磁盘

~~~
df -h /root/autodl-tmp
screen -ls
ps -ef | grep -E 's1_train|s2_train|srv_infer|naturalness' | grep -v grep
~~~

确认没有残留进程。服务器目前不应再启动训练。

### 第二步：清理前核对正式导出权重

先确认 b8/b12 的 e1/e2/e4/e6/e8/e10/e12 都存在，再决定是否删除对应 S1 Lightning checkpoint。不要删除 weights/GPT_* 导出权重。当前只剩约 456M，任何新任务前必须先处理空间。

### 第三步：只评测 b12，不重训

当前 srv_infer_highcap_compare.py 的 GPT_DIR 默认写的是 b8 目录。若要评测 b12，应先把脚本中的 GPT 目录改为：

~~~
/root/autodl-tmp/weights/GPT_ireina_s1_dold_hicap_b12s34_v1
~~~

最好把脚本改成读取环境变量 GPT_DIR，再只测试：

~~~
b12 e2/e4/e6/e8/e10/e12
× balanced48_e2
× old_e4/old_e6
× expressive
~~~

先做 6×3×22 的配对评测，不要再次生成 770 条全矩阵。若 b12 e12 没超过 76.96，停止训练扩张，保留 b8 e12。

### 第四步：若要追求真正 90，方向必须转到数据

目前瓶颈不是显存，也不是简单的 SoVITS epoch：

- D_old 只有 154 条，情绪/叹气/自然句尾覆盖不够；
- 广播剧中可加入的目标人声只有 15 条高置信度片段；
- 情绪句质量和说话状态一致性仍不足；
- 代理评分最高也只到 77 左右。

真正有效的下一步应是收集更多同一目标角色的干净、单人、无 BGM、无混响/电音、包含叹气和情绪变化的录音，并重新做严谨筛选。继续用当前小数据集堆 epoch 不应被当成冲 90 的主方案。

## 8. 已知坑，禁止重复

1. 不要再用旧的 srv_train_bg.sh 直接覆盖公共 GPT_weights_v2Pro / SoVITS_weights_v2Pro；它会清空对应目录并可能误 resume。
2. 不要使用 GPTSoVits conda 环境；5090 上曾出现 no kernel image is available for execution on the device。必须使用 /root/miniconda3/envs/gptsovits/bin/python。
3. 不要默认 S1 e15、S2 e8 最好；必须用固定句试听和代理评分共同决定。
4. 不要把 top_3_acc 当作自然感指标；它高可能只是过拟合。
5. 不要把所有 radio vocal 片段直接并入单说话人数据。
6. 不要只靠 top-k/top-p/temperature 修复训练阶段产生的电音；采样只能改善一部分表现。
7. 不要继续尝试此前失败的 v2Pro speaker embedding 权重插值；维度/config 不兼容，相关无效产物已删除。
8. 不要覆盖目前的旧 GPT e6、balanced48 e2、curated e4 等对照权重。
9. 每次比较必须固定参考音频、prompt_text、prompt_lang、测试文本、speed_factor、seed，并记录 GPT/S2/参数三者。
10. 任何“90 分”结论都必须注明是代理分数还是人工试听结论；当前没有人工 MOS 90 的证据。

## 9. 本地关键文件

~~~
radio_drama_work/HANDOFF_IREINA_V2PRO.md
radio_drama_work/HANDOFF_IREINA_NEXT_AI.md
radio_drama_work/FINAL_IREINA_OPTIMIZATION_REPORT.md
radio_drama_work/NATURALNESS_SCORING_STANDARD.md
radio_drama_work/human_likeness_eval.py
radio_drama_work/human_likeness_eval_v2.py
radio_drama_work/naturalness_score_v1.py
radio_drama_work/srv_infer_highcap_compare.py
radio_drama_work/srv_make_highcap_s1.py
radio_drama_work/srv_extend_highcap_s1.py
radio_drama_work/launch_s1_highcap.sh
radio_drama_work/final_router_listen/ireina_global_best.html
radio_drama_work/ireina_global_best_listen.tar.gz
~~~

## 10. 最后一条交接指令

接手后先读完整本文件，再做只读服务器检查；先处理磁盘空间，再评测 b12。不要重新训练，不要删除正式权重，不要把当前 76～77 的代理分数包装成 90。当前最稳妥的可用版本是 b8 e12 × balanced48 e2 × expressive，旧 global_best HTML/路由用于试听对照。

---

# 11. 2026-08-01 追加交接：广播剧切片混入其他说话人

## 11.1 本次发现的核心问题

用户发现了一个比“静音阈值不合适”更严重的问题：广播剧中存在两个人几乎无停顿、接着说话的桥段。原来的处理链是：

~~~text
视频/音频
  -> BS-Roformer 人声分离
  -> 按静音切片
  -> ERes2Net 目标说话人相似度
  -> ASR、时长、句尾和人工/代理筛选
  -> 候选训练片段
~~~

这个链条只能保证“片段中有声音”，不能保证“整段始终是伊蕾娜”。如果两名角色之间没有足够长的静音，静音切片会把两个人合成一个候选；ERes2Net 对混合片段也可能给出较高的整体相似度。于是单个高分、甚至 ASR 文本正常，都不能证明整个片段是目标人声。

用户实际试听并确认的反例是：

~~~text
28055832533-1-192_vocals_0017516160_0017573120.wav
ASR：誰か説明して!
原 single-center speaker score：约 0.7646
用户结论：不是伊蕾娜
~~~

这条反例非常重要：它证明旧的“speaker score >= 0.76”规则会把非目标人声放进候选，不能再把旧清单中的 0.76～0.80 分片段视为安全数据。

## 11.2 本次已经实现的新流程

新脚本：

~~~text
radio_drama_work/speaker_aware_reslice.py
~~~

它不是训练脚本，也不会自动修改训练清单。它是一个“目标说话人召回/复核器”，现在的逻辑是：

1. 读取静音切出来的原片段。
2. 不再把整段一次性送入 speaker encoder，而是使用重叠滑窗。
3. 默认窗口长度 1.5 秒，步长 0.5 秒。
4. 每个窗口都用 ERes2NetV2 与 164 条已知目标参考音频比较。
5. 同时记录：
   - `center`：与目标参考中心向量的相似度；
   - `p25`：与 164 条参考的相似度 25 分位数，用于避免只像极少数参考；
   - `median`、`p75`：参考集合中的中位数和较高分位数；
   - `rms`：窗口能量；
   - `known_negative_sim`：与用户确认的非目标反例的相似度；
   - `target_negative_margin = center - known_negative_sim`：与目标中心相比，是否更像目标而不是已知反例。
6. 只有连续重叠窗口都满足目标条件，才会标记为 `stable=true`。
7. 单窗口命中只输出到 `review` 目录，绝不自动写入训练清单。
8. 如果一个静音片段前半是伊蕾娜、后半是其他角色，滑窗分数会在说话人切换处发生变化；后续可以依据 active window 的连续区间重新裁边，而不是继续使用整条静音片段。

默认严格门槛为：

~~~text
window_sec = 1.5
hop_sec = 0.5
center >= 0.79
median >= 0.64
p25 >= 0.56
rms >= 0.004
target_negative_margin >= 0.08
至少 2 个连续重叠窗口，才是 stable
~~~

脚本已支持环境变量覆盖参数，供“高召回但仅复核”的第二轮扫描使用：

~~~text
SAW_WIN_SEC
SAW_HOP_SEC
SAW_CENTER_MIN
SAW_MEDIAN_MIN
SAW_P25_MIN
SAW_MARGIN_MIN
SAW_MIN_STABLE_WINDOWS
SAW_MIN_SEG_SEC
~~~

第二轮细扫实际使用过：

~~~text
SAW_WIN_SEC=1.0
SAW_HOP_SEC=0.25
SAW_CENTER_MIN=0.78
SAW_MEDIAN_MIN=0.61
SAW_P25_MIN=0.52
SAW_MARGIN_MIN=0.05
SAW_MIN_STABLE_WINDOWS=2
SAW_MIN_SEG_SEC=0.75
~~~

这套细扫仍然只用于试听复核，不能因为它召回了某条音频，就直接把该音频加入训练。

## 11.3 本次实际处理过的六部广播剧

用户提供的六个原始视频为：

~~~text
C:/Users/lulu/Downloads/202607311250.mp4
C:/Users/lulu/Downloads/28055832533-1-192.mp4
C:/Users/lulu/Downloads/1311215796_u1-1-192.mp4
C:/Users/lulu/Downloads/1405440540-1-192.mp4
C:/Users/lulu/Downloads/280479210-1-192.mp4
C:/Users/lulu/Downloads/113814583-1-160.mp4
~~~

本地状态：

- `1311215796_u1-1-192.mp4`、`1405440540-1-192.mp4`、`28055832533-1-192.mp4` 之前已经完成抽音频、BS-Roformer 分离和静音切片。
- `280479210-1-192.mp4`、`113814583-1-160.mp4` 属于旧广播剧素材，本地已有 `vocals`、`radio_drama_fine` 等处理结果，也做过旧全轨/精细片段复扫。
- `202607311250.mp4` 本轮补做了本地抽音频和分离。它时长约 13.434 秒，抽出的原始音频为：

~~~text
radio_drama_work/new_raw/202607311250.wav
~~~

- 第六部的 BS-Roformer 输出文件为：

~~~text
radio_drama_work/new_vocals/202607311250_vocals.wav
~~~

但该分离输出极低，测得约为：

~~~text
peak 约 0.002136
RMS 约 0.000137
duration 约 13.419 秒
~~~

因此固定 `-40 dB` 静音切片得到 0 条，不能把“得到 0 条切片”解释为“其中没有伊蕾娜”。对这部短素材还做了原始音频的 speaker-aware 扫描，结果是 0 个命中；这只能记为“当前自动筛选未找到可靠候选”，不能当成人工排除结论。

## 11.4 本轮扫描结果

### A. 三部已有新广播剧的 455 个静音片段

目录：

~~~text
radio_drama_work/new_slices
~~~

重新统一分离后，三部原来能正常切出的素材仍合计 455 条；`202607311250_vocals.wav` 因分离输出过低，未贡献静音片段。

使用严格的 speaker-aware 扫描后：

~~~text
输入：455 条
review-only 提示：8 条
stable training candidate：0 条
~~~

报告：

~~~text
radio_drama_work/speaker_aware_candidates_contrastive_v2/speaker_aware_report.json
~~~

试听文件：

~~~text
radio_drama_work/speaker_aware_candidates_contrastive_v2/review/
~~~

8 条提示的来源和区间如下。这里的 `p25` 和 `center` 只是机器复核信息，不是“已确认伊蕾娜”的标签：

~~~text
1. 1311215796_u1-1-192_vocals_0016493440_0016910720.wav  3.50-5.00s  p25≈0.561 center≈0.803
2. 1405440540-1-192_vocals_0012969600_0013081600.wav      1.50-3.00s  p25≈0.595 center≈0.829
3. 28055832533-1-192_vocals_0013268480_0013539200.wav     5.50-7.00s  p25≈0.579 center≈0.817
4. 28055832533-1-192_vocals_0014717440_0014883200.wav     0.00-1.50s  p25≈0.568 center≈0.806
5. 28055832533-1-192_vocals_0015066240_0015190400.wav     1.00-2.50s  p25≈0.568 center≈0.817
6. 28055832533-1-192_vocals_0015066240_0015190400.wav     2.38-3.88s  p25≈0.566 center≈0.813
7. 28055832533-1-192_vocals_0018346880_0018502400.wav     2.00-3.50s  p25≈0.590 center≈0.824
8. 28055832533-1-192_vocals_0024772480_0024890880.wav     2.20-3.70s  p25≈0.561 center≈0.794
~~~

### B. 1 秒细扫

报告：

~~~text
radio_drama_work/speaker_aware_fine_review/speaker_aware_report.json
~~~

结果：

~~~text
review-only 提示：5 条
stable：0 条
~~~

这 5 条是为捕捉短目标说话段而做的高召回复核窗口，不可与上面的 8 条简单相加，因为有重叠/同源片段：

~~~text
1. 1311215796_u1-1-192_vocals_0018844160_0018899200.wav  0.50-1.50s
2. 28055832533-1-192_vocals_0005351680_0005528320.wav     2.50-3.50s
3. 28055832533-1-192_vocals_0012453760_0012586240.wav     1.75-2.75s
4. 28055832533-1-192_vocals_0015066240_0015190400.wav     2.75-3.75s
5. 28055832533-1-192_vocals_0018346880_0018502400.wav     2.50-3.50s
~~~

### C. 旧两部广播剧全轨复扫

目录：

~~~text
radio_drama_work/speaker_aware_old_full
~~~

结果为 4 个 review-only 单窗口提示、0 个 stable：

~~~text
113814583-1-160_vocals.wav 约 79.5-81.0s
113814583-1-160_vocals.wav 约 99.5-101.0s
113814583-1-160_vocals.wav 约 176.0-177.5s
113814583-1-160_vocals.wav 约 691.5-693.0s
~~~

另外，旧 `radio_drama_fine` 503 条精细窗口的多参考排序中，机器认为相对靠前的 3 条是：

~~~text
113814583-1-160_vocals.wav_0027455360_0027570880.wav
ASR：思っただけですそれ私のセリフですね

113814583-1-160_vocals.wav_0008081600_0008195520.wav
ASR：最近はあまり言ってませんから

113814583-1-160_vocals.wav_0003184960_0003277440.wav
ASR：再び味見をしてみると
~~~

它们仍然只是机器排序靠前，未得到用户人工确认，不得直接加入单说话人训练。

## 11.5 已确认反例的处理状态

本轮已把用户确认的反例从本地精筛清单中移除，并在生成清单的脚本中加入硬排除：

~~~text
radio_drama_work/make_final_window_list.py
~~~

硬排除名：

~~~text
28055832533-1-192_vocals_0017516160_0017573120.wav
~~~

修正后的本地清单：

~~~text
radio_drama_work/new_fine_windows_v2/new_fine_curated.list
~~~

当前数量：

~~~text
keep：17 条
drop：9 条
confirmed_non_target：1 条
confirmed negative 已不在 new_fine_curated.list 中
~~~

原 18 条清单在修改前已备份为：

~~~text
radio_drama_work/review_recovery_candidates/new_fine_curated_before_confirmed_negative.list
~~~

注意：以下历史文件仍可能包含这条反例，它们只是历史候选/审计材料，不能再当作已验证训练清单：

~~~text
radio_drama_work/new_curated/new_curated.json
radio_drama_work/new_candidates_fine_sv.json
radio_drama_work/new_fine_windows_v2/new_fine_windows_asr.json
radio_drama_work/review_recovery_candidates/new_fine_curated_before_confirmed_negative.list
~~~

服务器目前没有连接，本轮没有上传修正后的清单，也没有改动服务器训练进程。因此不能假设“已经完成的 b12 训练一定没有吃到这条反例”。服务器开机后必须只读核对实际训练使用的 list、配置和日志；如果服务器实际使用的是旧的 `new_fine_curated.list` 或 `plus18` list，就要把该模型标记为“可能含有已确认非目标片段”，不能把它当作干净模型结论。

## 11.6 当前“15 条”与本地“18/17 条”的解释

旧交接里写过“三部新广播剧最终只增加 15 条高置信度片段”。但本地生成流程中存在：

~~~text
radio_drama_work/new_fine_windows_v2/new_fine_curated.list
~~~

该文件原来是 18 条；`make_plus18_list.sh` 也明确把它作为 `ADD` 输入。它与交接里写的“最终增加 15 条”不是同一个可直接核对的计数口径。可能存在服务器侧再次删减、训练配置使用另一份 list、或交接记录按手工最终版统计的情况。

因此下一位 AI 不得简单写成“15 条全部安全”，也不得简单写成“17 条全部安全”。正确做法是：

1. 先在服务器只读核对实际运行配置和 list。
2. 搜索反例文件名是否出现在训练 list、打包数据或运行日志中。
3. 对训练前实际 list 与本地修正版做逐行对比。
4. 在用户没有人工确认剩余片段前，统一标为“候选/待审”，不要标为“纯伊蕾娜”。

## 11.7 试听/复核包

当前最值得先让用户试听的是：

~~~text
radio_drama_work/speaker_aware_candidates_contrastive_v2/review/review_0002_p250.595_1405440540-1-192_vocals_0012969600_0013081600_0000048000_0000096000.wav
radio_drama_work/speaker_aware_candidates_contrastive_v2/review/review_0007_p250.590_28055832533-1-192_vocals_0018346880_0018502400_0000064000_0000112000.wav
radio_drama_work/speaker_aware_candidates_contrastive_v2/review/review_0003_p250.579_28055832533-1-192_vocals_0013268480_0013539200_0000176000_0000224000.wav
~~~

选择这些只是因为机器的多参考 `p25` 较高，不代表它们已被人工认定为伊蕾娜。试听时必须重点听：

- 开头是否已经是另一名角色；
- 结尾是否在同一个静音片段内切入另一名角色；
- 是否有两个人重叠说话；
- 是否有明显 BGM、混响、电话音或音效盖住人声；
- 音色是否与已知伊蕾娜参考音频一致，而不是只因为日语台词清楚就误判。

如果用户说其中某条是伊蕾娜，下一位 AI 仍应根据滑窗边界重新导出“最小纯目标区间”，不要直接把整条 review 文件加入训练。

## 11.8 下一位 AI 的正确执行顺序

### 第一步：先读本文件和脚本，不启动服务器训练

优先阅读：

~~~text
radio_drama_work/HANDOFF_IREINA_NEXT_AI.md
radio_drama_work/speaker_aware_reslice.py
radio_drama_work/make_final_window_list.py
radio_drama_work/new_fine_windows_v2/new_fine_curated.json
radio_drama_work/speaker_aware_candidates_contrastive_v2/speaker_aware_report.json
~~~

当前没有用户授权删除服务器数据，也没有必要删除本地历史候选。先保留证据和审计备份。

### 第二步：服务器开机后只读检查

服务器开机、SSH 恢复后，先执行旧交接里的磁盘和进程检查：

~~~bash
df -h /root/autodl-tmp
screen -ls
ps -ef | grep -E 's1_train|s2_train|srv_infer|naturalness' | grep -v grep
~~~

然后核对本轮实际训练数据，重点搜索：

~~~bash
find /root/autodl-tmp -type f /( -name '*.list' -o -name '*.json' -o -name '*.log' /) -print0 | xargs -0 grep -n '0017516160_0017573120' 2>/dev/null
find /root/autodl-tmp -type f /( -name '*.list' -o -name '*.json' /) -print0 | xargs -0 grep -n 'new_fine/|plus18/|plus15' 2>/dev/null
~~~

如果空间仍约 456M，禁止做任何大规模推理、解压或训练。先按旧交接核对正式导出权重，再考虑清理中间 checkpoint。

### 第三步：先人工确认 review 音频，再生成 verified list

当前自动流程没有得到 `stable` 新片段。下一步不是调低阈值后直接训练，而是：

1. 让用户试听 `speaker_aware_candidates_contrastive_v2/review` 中的候选。
2. 记录用户明确确认的文件名和纯目标时间区间。
3. 对确认区间重新导出 32k 单声道/合适响度 WAV。
4. 运行 ASR，人工核对文本。
5. 生成一个新的、独立的 `verified_target.list`。
6. 训练前对该 list 做重复、时长、峰值、BGM 和 speaker-aware 复核。
7. 只有用户确认后，才考虑上传/训练。

建议新清单不要覆盖历史文件，名称应类似：

~~~text
new_fine_verified_after_human_20260801.list
~~~

### 第四步：如果用户确认候选不是目标声

把它加入硬负例/排除记录，但不能只靠一个负例把所有相似角色自动判掉。应保留：

~~~text
negative filename
用户判断
原始分数
多参考分数
与负例的相似度
对应试听文件路径
~~~

之后再用多个已确认负例计算 contrastive margin，重新跑候选排序。

### 第五步：只有在数据明确变干净后，才评测 b12

当前任务重点是数据纯度，不是继续训练。服务器恢复后，如需使用 b12，只做旧交接指定的只读评测：

~~~text
b12 e2/e4/e6/e8/e10/e12
× balanced48_e2
× old_e4/old_e6
× expressive
~~~

不要在没有重新核对数据 list 的情况下，用可能含有非目标角色的新增数据继续训练。不要把“把混合角色喂进去后分数上升”解释为模型变好；这很可能只是评测泄漏或音色污染。

## 11.9 本次修改和产物清单

本轮新增/修改的关键文件：

~~~text
修改：radio_drama_work/speaker_aware_reslice.py
修改：radio_drama_work/make_final_window_list.py
重新生成：radio_drama_work/new_fine_windows_v2/new_fine_curated.list
重新生成：radio_drama_work/new_fine_windows_v2/new_fine_curated.json
备份：radio_drama_work/review_recovery_candidates/new_fine_curated_before_confirmed_negative.list
新增报告：radio_drama_work/speaker_aware_candidates_contrastive_v2/speaker_aware_report.json
新增试听目录：radio_drama_work/speaker_aware_candidates_contrastive_v2/review/
新增报告：radio_drama_work/speaker_aware_fine_review/speaker_aware_report.json
新增试听目录：radio_drama_work/speaker_aware_fine_review/review/
新增报告：radio_drama_work/speaker_aware_202607_raw/speaker_aware_report.json
新增原始音频：radio_drama_work/new_raw/202607311250.wav
新增分离音频：radio_drama_work/new_vocals/202607311250_vocals.wav
~~~

本轮没有：

- 启动服务器训练；
- 修改或删除服务器权重；
- 把 speaker-aware review 结果自动写入训练 list；
- 宣称发现了新的、已经确认的伊蕾娜训练片段；
- 宣称达到了 90 分。

## 11.10 最重要的结论

这次工作的结论不是“广播剧只能找到 15 条”，而是：

~~~text
静音切片的候选数量 ≠ 目标说话人训练数据数量。
speaker score 高 ≠ 整段都是目标人声。
ASR 文本正常 ≠ 说话人正确。
单窗口命中 ≠ 稳定目标片段。
~~~

当前最安全的状态是：保留旧模型和旧实验证据；把确认的非目标片段从本地训练候选移除；继续用滑窗/连续性/人工试听建立 verified list；服务器开机后先核对实际训练 list 和磁盘，再决定是否评测或训练。没有人工确认之前，宁可少加数据，也不要再次把其他角色混入伊蕾娜单说话人训练。

---

# 12. 2026-08-01 再追加：用户复核前三条后扩大召回

## 12.1 用户新增的三条人工反例

用户继续试听了上一节列出的前三个优先候选，并确认它们也混有其他非目标说话人的声音。新增反例为：

~~~text
review_0002 ... 1405440540-1-192_vocals_0012969600_0013081600 ... 1.50-3.00s
review_0007 ... 28055832533-1-192_vocals_0018346880_0018502400 ... 2.00-3.50s
review_0003 ... 28055832533-1-192_vocals_0013268480_0013539200 ... 5.50-7.00s
recall_004 ... 1311215796_u1-1-192_vocals_0006943360_0007132160 ... 4.40-5.90s
~~~

这三条已经写入：

~~~text
radio_drama_work/confirmed_non_target_or_mixed.txt
~~~

当前已知人工负例/混合片段总数为 5 条：原先用户确认的 1 条、上一轮新增的 3 条，以及本轮用户确认的 `recall_004`。它们都不得进入训练，也不能在后续“高分候选”试听包中重复出现。

## 12.2 对比筛选器已升级为五个负例

`speaker_aware_reslice.py` 原先只拿一个确认反例做 contrastive margin；现在会读取 `confirmed_non_target_or_mixed.txt`，对所有存在的负例计算 embedding，并使用最相似负例：

~~~text
known_negative_sim = max(与五个确认负例的相似度)
target_negative_margin = target_center - known_negative_sim
~~~

这比只对比一个负例更保守，但仍不是完整的说话人分类器。五个负例只代表目前人工确认过的错误类型，不能覆盖所有广播剧角色。

重新扫描 `new_slices` 的结果：

~~~text
输入：455 条静音片段
四负例严格筛选：2 条 review-only
stable：0 条
~~~

报告：

~~~text
radio_drama_work/speaker_aware_candidates_multinegative/speaker_aware_report.json
~~~

这里的“2 条 review-only”不是新增训练数据，只是仍有少量窗口在机器指标上靠前；用户没有人工确认前，不得使用。

## 12.3 为避免严格门槛造成低估，已建立全量召回包

之前的 8 条是严格门槛下的 review 提示，不能用来回答“广播剧是不是只有这么一点”。因此新增了一个只做排序和试听的脚本：

~~~text
radio_drama_work/build_recall_review_pack.py
~~~

它把每个滑窗按照多参考目标相似度、参考中位数和四负例 margin 排序，做非重叠抽样，输出 WAV 和 JSON manifest。它仍然不会生成训练 list。

### 新三部广播剧召回结果

对 455 条静音片段的全部滑窗进行排序，排除确认反例区间后，得到最新结果：

~~~text
有效排序窗口：652 条
导出试听窗口：30 条
~~~

目录和清单：

~~~text
radio_drama_work/recall_review_pack_v3/
radio_drama_work/recall_review_pack_v3/recall_review_manifest.json
radio_drama_work/recall_review_pack_v2/（上一轮版本；v3 是排除新反例后的当前版）
~~~

注意：30 条是“值得听的窗口”，不是“30 条伊蕾娜”。其中有些可能是其他角色、混合片段、短到无法可靠识别的窗口，必须人工复核。

### 旧两部广播剧召回结果

对本地已有的 503 条 `radio_drama_fine` 精细片段重新用五负例扫描并排序：

~~~text
输入：503 条精细片段
有效排序窗口：947 条
导出试听窗口：24 条
~~~

报告、试听包和 ASR：

~~~text
radio_drama_work/speaker_aware_old_fine_multinegative/speaker_aware_report.json
radio_drama_work/recall_review_pack_old_v1/
radio_drama_work/recall_review_pack_old_v1/recall_review_manifest.json
radio_drama_work/recall_review_pack_old_v1/recall_review_asr.json
~~~

因此目前不能说“广播剧只有 15 条”或“只有 8 条”。更准确的状态是：

~~~text
机器严格确认可训练：0 条新增
新三部待人工复核：30 条
旧两部待人工复核：24 条
总待人工复核窗口：54 条
已确认纯目标并可直接训练：仍然没有新增条目
~~~

这 54 条是扩大召回后的检查范围，不是最终训练数量。用户逐条试听后，可能留下若干条，也可能大部分都被排除；必须以人工说话人判断为准。

## 12.4 第六部 202607311250 的当前判断

第六部只有约 13.4 秒。BS-Roformer 输出音量过低，静音切片为 0；原始音频直接滑窗扫描的最高机器结果约为：

~~~text
center 约 0.701
p25 约 0.449
median 约 0.568
~~~

它目前没有进入 54 条召回包。不能说这部一定没有目标声，只能说当前素材质量/分离结果/目标相似度都不足以把它列为优先候选。若后续要继续处理，应该先人工听原声或换分离策略，再决定是否值得细切；不要仅仅因为它是新文件就强行加入训练。

## 12.5 现在继续工作的正确方向

下一位 AI 或本线程继续工作时，应按下面顺序推进：

1. 先让用户试听 `recall_review_pack_v2` 的前若干条，再试听 `recall_review_pack_old_v1` 的前若干条；每次只确认少量，避免用户疲劳。
2. 用户确认“纯伊蕾娜”的窗口后，只截取最小纯目标时间区间；如果窗口边缘仍有另一角色，继续向内裁切。
3. 用户确认“混合/其他角色”的窗口后，加入 `confirmed_non_target_or_mixed.txt`，并在召回脚本中排除对应的精确区间，而不是粗暴删除整部广播剧。
4. 不要根据 `rank`、`p25`、`center` 或 ASR 置信度单独决定训练。
5. 先建立人工确认的 `verified_target` 清单，至少记录文件、绝对时间区间、ASR、人工结论和备注。
6. 只有 verified 清单形成后，才重新计算训练数据数量，才讨论是否上传服务器和训练。

本次新增的召回脚本和负例记录不改变任何服务器模型；服务器仍需开机后只读核对，不能据此假设 b12 训练已经使用修正版数据。
# 13. 2026-08-01 ClearerVoice-Studio 实测

## 13.1 用户确认的四条纯净正样本

用户已试听并确认 `recall_review_pack_v3` 的前四条都是纯净伊蕾娜声音。已记录在：

~~~text
radio_drama_work/confirmed_target_pure.txt
~~~

这四条是人工确认的正样本/声纹参考，不要自动加入训练清单；加入训练前仍需按绝对时间区间核对是否与现有样本重复。

## 13.2 ClearVoice 安装和模型

本地隔离安装位置：

~~~text
radio_drama_work/third_party/clearvoice_pkg/
radio_drama_work/third_party/clearvoice_download/clearvoice-0.1.2-py3-none-any.whl
~~~

已下载官方 `MossFormer2_SS_16K` 权重到：

~~~text
radio_drama_work/clearvoice_test/checkpoints/MossFormer2_SS_16K/last_best_checkpoint.pt
~~~

权重约 670 MB。测试脚本和结果目录：

~~~text
radio_drama_work/score_clearvoice_outputs.py
radio_drama_work/clearvoice_test/
radio_drama_work/clearvoice_test/clearvoice_scores.json
~~~

## 13.3 实测结论

ClearVoice 的 `MossFormer2_SS_16K` 可以把混合音频拆成两路匿名说话人，但它本身不知道哪一路是伊蕾娜；必须再用 ERes2Net 声纹评分挑选输出路。对一个较长混合源片段 `140544...0012969600_0013081600.wav`，整段测试的结果为：

~~~text
s1: center 0.6781 / median 0.5393 / p25 0.4769
s2: center 0.2819 / median 0.2188 / p25 0.1787
~~~

这个结果说明 `s1` 更像目标声，但分离后仍明显低于纯净目标参考，存在分离损伤，不能直接视为无损纯净音频。

对三个已知混合区间截短后再测试：

~~~text
140544_window_s1: center 0.6736 / median 0.5259 / p25 0.4730
140544_window_s2: center 0.2422 / median 0.1915 / p25 0.1435
280558_1326_window_s1: center 0.3109 / median 0.2791 / p25 0.1603
280558_1326_window_s2: center 0.2133 / median 0.1660 / p25 0.1249
280558_1834_window_s1: center 0.2876 / median 0.2407 / p25 0.1475
280558_1834_window_s2: center 0.1672 / median 0.1292 / p25 0.0845
~~~

因此目前只能确认：ClearVoice 对 `140544` 这组有可利用迹象；对另外两组没有可靠分离证据，不能自动选一路进入训练。输出文件位于：

~~~text
radio_drama_work/clearvoice_test/outputs_windows/MossFormer2_SS_16K/
~~~

## 13.4 为什么没有继续使用 AV 目标说话人提取

ClearVoice 的 AV 模型依赖画面中的说话人脸/嘴部运动，不是“用伊蕾娜声音参考做音频身份提取”。抽查 `202607311250.mp4` 的画面是 Minecraft 游戏场景，没有可用于目标说话人的真人脸部线索，因此 AV 模式不适合当前素材。其余广播剧视频若也是游戏画面或静态画面，也不能依赖 AV 模式。

## 13.5 后续建议

1. 先试听 `outputs_windows` 中的六条 stem，重点确认 `140544_window_s1` 是否确实只保留伊蕾娜，以及是否出现机械化/断字/音色变薄。
2. 不要对已经纯净的四条正样本使用 `MossFormer2_SS_16K`；它是两人分离模型，单人输入可能制造伪影。
3. 如果要继续扩大数据，ClearVoice 只作为“混合候选的辅助分离器”，流程应是：短时间窗分离 → ERes2Net 比较两路 → 人工试听 → 仅保存人工确认的伊蕾娜区间。
4. 当前不能宣称 ClearVoice 已经解决混入问题，也不能因 `s1` 分数较高就自动改写训练 list。
# 14. 2026-08-01 按 ClearVoice 官方 Numpy2Numpy 示例重跑

用户指出上一版分离结果有问题。复核官方文档后确认，官方 `demo_Numpy2Numpy.py` 的 `MossFormer2_SS_16K` 流程是：先将音频重采样为 16 kHz，整理为 `[batch, length]` 的 `float32` numpy，调用 `ClearVoice(audio, False)`，再将返回的 `[speaker, batch, length]` 按 `output_wav[spk, 0, :]` 写出。官方文档也明确该模型是两路匿名 speech separation，不是“输入伊蕾娜参考音频后直接输出伊蕾娜”。

已按该官方示例重跑三个已知混合区间，输出目录：

~~~text
radio_drama_work/clearvoice_test/outputs_official_numpy/
~~~

对应评分：

~~~text
140544_window_spk1: center 0.5672 / median 0.4663 / p25 0.3742
140544_window_spk2: center 0.4207 / median 0.3290 / p25 0.2427
280558_1326_window_spk1: center 0.6756 / median 0.5816 / p25 0.4606
280558_1326_window_spk2: center 0.6311 / median 0.5062 / p25 0.4206
280558_1834_window_spk1: center 0.6929 / median 0.5745 / p25 0.4663
280558_1834_window_spk2: center 0.5461 / median 0.4175 / p25 0.3212
~~~

评分报告：

~~~text
radio_drama_work/clearvoice_test/clearvoice_scores_official_numpy.json
~~~

这版与之前文件 I/O 输出不同，试听时以 `outputs_official_numpy` 为准。尤其 `280558_1326` 两路都保持较高目标相似度，说明它仍有目标声泄漏/分离不干净；不能只按固定的 `spk1` 或 `spk2` 作为伊蕾娜。后续若继续做，应按更短时间窗逐窗比较两路，并人工确认后再拼接；不可直接把整条 stem 放进训练。
# 15. 2026-08-01 放弃 ClearVoice 重建，改为原始波形词级硬切

用户试听确认 ClearVoice 分离输出发糊、不可直接训练。原因是 `MossFormer2_SS_16K` 会重建两路匿名人声；它不是目标说话人识别器，且对当前日语广播剧连续换人场景产生明显伪影。因此 ClearVoice 输出不得进入训练。

新方案保留 BS-Roformer 的原始人声轨，只做：

~~~text
Faster-Whisper 日语 word_timestamps
-> 每个词周围 1.20 秒原始波形窗口
-> ERes2NetV2 与 168 条目标参考比较
-> 与 2 条纯非目标参考做 margin
-> 同一候选内部 0.80 秒/0.20 秒滑窗做两类球面聚类
-> 仅保留目标类且词级门槛通过的时间区间
-> 对原始波形硬切/淡入淡出，不经过任何声源重建
~~~

相关脚本：

~~~text
radio_drama_work/word_target_hardcut.py
radio_drama_work/word_target_cluster_hardcut.py
radio_drama_work/make_safe_hardcut_eval.py
~~~

重要修正：三条“混合片段”不能当作纯非目标负例，否则其中包含的伊蕾娜会把目标词全部压掉。词级 margin 只使用用户确认的两条纯非目标；混合片段只作为验证阶段的已知污染区间。

对当前三条问题片段的自动结果：

~~~text
140544：自动保留 0 词；
280558_1326：自动保留 4 词，其中 2 词位于用户确认的混合区间；排除后保留 2 词；
280558_1834：自动保留 3 词，全部位于用户确认的混合区间；排除后保留 0 词。
~~~

已知混合区间排除后的试听输出：

~~~text
radio_drama_work/word_target_hardcut_safe_eval_v1/
radio_drama_work/word_target_hardcut_safe_eval_v1/safe_eval_report.json
~~~

目前只有 `280558_1326` 的 `みたい`、`ってる` 两个短词通过验证，合计不到 1 秒，暂时不够作为训练样本。它们是原始波形截取，不会出现 ClearVoice 那种重建发糊；但仍需用户试听确认，不能自动加入训练 list。

当前正确结论不是“调一个阈值就能把三条全部救回来”，而是：人物识别已经能做到更细的词级排除，但这三条在已知污染区间之外没有足够长、经过人工确认的纯目标段。下一步可把同一脚本批量跑 `new_slices`，但输出仍应先进入 review，不得自动训练。

# 16. 2026-08-02 最新状态：12 部全轨同源聚类、逐词硬切与边界救援已完成

本文件前面的若干“只有不到 1 秒”“下一步跑 new_slices”等结论是历史阶段记录，已经被后续全轨方案取代。最新、最完整且可复跑的交接文档是：

```text
radio_drama_work/FULL_RADIO_MINING_EXPERIENCE_V1.md
```

最新数字：

- 12 部 vocals 共 199.760 分钟，37,432 个 ASR 词全部重新评分；
- 已人工确认广播剧素材 57 条 / 166.785 秒；
- 逐词与边界严格机器池 189 条 / 549.605 秒；
- 源内 NISQA-TTS 与声学门后主试听池 143 条 / 419.605 秒；
- 另有边界复核 49 条 / 142.290 秒；
- 优先盲验网页：`full_semantic_acceptance_v1/index.html`（20 条机器候选 + 4 条 gold）；
- 没有启动训练、没有修改旧训练清单、没有宣称达到 90 分。

后续不得再把 `boundary_cut` 当作人物负例。应先读取新文档，再基于盲听反馈决定 143 条机器主池能晋级多少。

# 17. 2026-08-02 首轮 24 条全量盲验失败与边界规则修正

用户已完整打标：24 条中 7 条通过、13 条切头/切尾、2 条混入、2 条音质问题。20 条机器候选实际只有 5 条直接通过；4 条旧 gold 里也有 1 条边界失败、1 条音质失败。因此 `full_semantic_quality_strict_review_v1.list` 的 143 条不能直接晋级。

反馈归档与解析：

```text
radio_drama_work/human_full_semantic_acceptance_v1/
```

新确认的两段混入硬掩码：

```text
37239655783-1-192  1094.970–1098.420
1405440540-1-192     42.720–44.500
```

13 条边界失败仍是人物正证据，不得当负例。真正根因是首轮分段器把任何 `strict_speaker_pass=False` 的日语短词都当换人，并固定再腐蚀 0.18 秒；这会系统性产出「完成さ」「許してく」「描か」等半句。该规则已废止，后续必须区分短词 `soft_fail` 与持续强非目标 `hard_speaker_break`，并新增语义/音素完整性硬门。

已按完整词句从原始 BS-Roformer vocals 重切 13 个问题、15 条版本、71.931 秒：

```text
radio_drama_work/full_semantic_acceptance_boundary_rescue_v1/index.html
```

全部避开两段新混入掩码，不做 ClearVoice/VASR 重建，不改训练清单。3 条超过 6 秒只用于确认完整句；等待用户这轮反馈后，才能据此实现并重跑全量语义边界 v2。完整经验以 `FULL_RADIO_MINING_EXPERIENCE_V1.md` 第 10 节为准。

# 18. 2026-08-02 边界救援 12/15 通过，等待最终六条

`full_semantic_acceptance_boundary_rescue_v1` 反馈已收回：12 条通过、1 条仍切开头、1 条完整扩边带入非目标、1 条有环境音。去重后冻结 11 个原问题 / 54.158 秒，其中 9 条 / 37.725 秒满足 1.5–6 秒：

```text
radio_drama_work/human_verified_semantic_rescues_v1.list
radio_drama_work/human_verified_semantic_rescues_v1_train_1p5_6s.list
```

A001 只有保守版通过，新增精确非目标硬掩码：

```text
1405372904-1-192  496.375–497.950
```

最后 6 条复核页已经生成：

```text
radio_drama_work/full_semantic_boundary_rescue_round2_v1/index.html
```

它包含 A021 补回「もう」、A011 缩到 6 秒、A004 长句拆成两段、A017 环境音拆段检查。必须等这 6 条反馈后再冻结全量 v2 边界规则。当前没有训练、没有修改旧训练清单。最新细节以 `FULL_RADIO_MINING_EXPERIENCE_V1.md` 第 11 节为准。

# 19. 2026-08-02 最终六条 4/6 通过，只剩 A021 三个毫秒版本

最终六条反馈：T001、T004、T005、T006 通过；T003 因翻书/合书环境音剔除；T002 正文纯净但开头仍残留极短「诶」。冻结集已更新为：

```text
radio_drama_work/human_verified_semantic_rescues_v2.list
```

当前 13 条 / 55.823 秒，全部 1.5–6 秒且为用户实际听过的具体波形。

A021 最后三个毫秒级起点盲测页：

```text
radio_drama_work/full_semantic_edge_final_review_v1/index.html
```

三个版本分别从 475.590、475.610、475.630 秒开始。任一版无残留「诶」且「もう」未切头即可；多个通过时保留最早起点。收到该 3 条反馈后，边界救援阶段即可收口并开始全量 v2 重建。当前仍未训练、未改旧训练清单。最新细节见总经验文档第 12 节。

# 20. 2026-08-02 最终边界冻结、全量 v2 重建和 GPU 质量门已完成

最终三版本反馈已解析：475.610 和 475.630 秒通过，475.590 秒仍带前一角色极短“诶”，因此冻结最早安全起点 475.610 秒。

最终人工边界救援集：

```text
radio_drama_work/human_verified_semantic_rescues_v3.list
```

14 条 / 60.567875 秒，全部 1.5–6 秒、32 kHz mono PCM16、用户实际听过并通过。新增边界排除：

```text
280479210-1-192  475.565–475.610  # 脏过渡，不是纯负例
```

翻书/合书质量排除仍为：

```text
306672160-1-192   82.425–87.207637  # 质量问题，不是人物负例
```

全量 v2 新脚本：

```text
radio_drama_work/calibrate_full_semantic_word_states_v2.py
radio_drama_work/build_full_semantic_candidates_v2.py
radio_drama_work/validate_full_semantic_v2.py
radio_drama_work/score_full_semantic_quality_v2.py
radio_drama_work/build_full_semantic_acceptance_v2.py
```

最新数字：

- 12/12 来源、37,432 个词完成 v2 重建；
- 人工确认总集 75 条 / 238.563 秒，其中训练长度安全集 70 条 / 203.137906 秒；
- 机器严格复核池 168 条 / 561.826375 秒；
- 语义/边界复核池 448 条 / 927.311063 秒；
- GPU NISQA + 声学 + 严重边界门后保留 115 条 / 380.155969 秒；
- 质量复核 53 条 / 181.670406 秒；
- 机器池已对人工区间做 80% 重叠去重；
- 70 条人工安全集 + 115 条机器待验集的潜在总量为 185 条 / 583.293875 秒（约 9 分 43 秒）。

硬验收已通过：路径、格式、时长、文本、重复、硬掩码和两个 `safe_word` 均无问题。GPU 质量失败原因：来源内后 20% 共 37、异常点击 14、严重边界能量 8；原因可重叠。没有削波、低 SNR 或相对原区间退化超过 0.20 的失败。

下一步唯一优先入口：

```text
radio_drama_work/full_semantic_acceptance_v2/index.html
```

它是可直接双击打开的本地盲听页，24 条 = 20 条 v2 机器候选 + 4 条 gold；11 个有候选来源全部覆盖。反馈文件名为 `full_semantic_acceptance_feedback_v2_*.json`。

反馈解释必须严格区分：

- “开头被切 / 结尾被切”仍是人物正证据，进入边界救援；
- “非伊蕾娜 / 混入无缝接话 / 重叠说话”进入人物硬掩码；
- “杂音 / 环境音 / 道具声”进入质量排除，不能做人物负例。

当前没有训练、没有改旧训练清单。115 条仍是机器待验，不能称为可训练数据。总潜在时长低于 15 分钟，不能宣称具备 90 分训练基础。完整算法、校准率和分来源统计见 `FULL_RADIO_MINING_EXPERIENCE_V1.md` 第 13–16 节。

# 21. 2026-08-02 v2 首轮反馈已收回，下一步听 14 条救援版本

`full_semantic_acceptance_v2` 的 24 条已全部打标：4/4 gold 通过；20 条机器候选中 13 条通过、3 条边界失败、3 条混入、1 条有脚步声。因为机器通过率只有 65% 且有明确非目标，115 条质量池不能整批晋级。

13 条机器直接通过波形已冻结：

```text
radio_drama_work/human_verified_full_semantic_v2_direct.list
```

13 条 / 42.750125 秒。与旧 70 条合并后的当前严格人工确认清单：

```text
radio_drama_work/human_confirmed_broadcast_v3_train_1p5_6s.list
```

83 条 / 245.888031 秒（约 4 分 05.9 秒），全部为用户听过的准确波形，32 kHz mono PCM16、1.5–6 秒、路径与波形哈希唯一。

七条失败已定位：B005 缺「嬉」和「よ」，B007 缺「匿」，B022 缺问句「か」；B009 是前一角色「ダメです」无缝接目标「でもダメです」；B013 只有最开头极短前人尾音；B024 前一角色「そっくりだ」后才是目标「そうですね」；B018 有脚步声并已质量排除。

三个 mixed 失败没有被整段写成硬负例，等待安全起点确定后再建立精确边缘掩码。当前优先页面：

```text
radio_drama_work/full_semantic_acceptance_v2_rescue_v1/index.html
```

它可直接双击打开，共 14 条盲版本：B005 三版、B007 一版、B022 一版、B009/B013/B024 各三个毫秒版本。反馈文件名为 `full_semantic_acceptance_v2_rescue_feedback_v1_*.json`。

下一 AI 收到反馈后应：

1. 按 origin 只冻结人工通过的最早安全版本；
2. B005 的 6.335 秒完整句只作对照，最终优先保留两个 ≤6 秒且各自通过的分句；
3. 对 B009/B013/B024，用原失败起点到最早安全通过起点生成精确污染掩码；
4. 边界失败继续作为目标身份正证据，脚步声继续只作质量排除；
5. 不批量晋级剩余 95 条未听机器候选，先用这轮错误更新边界/局部切换规则。

没有训练、没有修改已有训练清单、没有宣称达到 90 分。完整细节见总经验文档第 17–18 节。

# 22. 2026-08-02 救援已结算，新增 12 集动画全轨

14 条 v2 救援反馈已经结算：5 条人工通过，最终冻结 4 条训练长度片段 / 11.016219 秒：

```text
radio_drama_work/human_verified_full_semantic_v2_rescues_v1.list
```

与旧 83 条人工确认集去重合并后的最新独立清单：

```text
radio_drama_work/human_confirmed_broadcast_v4_train_1p5_6s.list
```

共 87 条 / 256.904250 秒，路径、波形哈希、文本、格式和时长已全部验收通过；没有覆盖 v3 或其他旧清单。

B009 原片及三个重切版本均被确认不是伊蕾娜，已建立 `306672160-1-192` 209.765–215.540 秒精确硬负例。B013/B024 的毫秒裁边都失败，最后必须完整丢掉上一角色短语，从 B013 词 1483「先生」和 B024 词 3293「クリソツですね」重新开始。最后 3 条复核入口：

```text
radio_drama_work/full_semantic_acceptance_v2_edge_final2_v1/index.html
```

新增动画资产为《魔女之旅》12 集正片，共 286.419083 分钟；NCOP/NCED 是歌曲，已排除。12/12 原始 AAC 音轨已提取为 44.1 kHz stereo PCM16，GPU BS-Roformer 正在写入独立目录：

```text
radio_drama_work/anime_batch_v1/raw
radio_drama_work/anime_batch_v1/vocals
radio_drama_work/anime_batch_v1/other
radio_drama_work/anime_batch_v1/manifest.json
```

动画域不能使用全局声纹阈值直接晋级。已实现的后续顺序是：GPU 日语逐词 ASR → GPU ERes2Net 全轨独立扫描 → FunASR VAD + CAM++ 每集缓存 → 每集多 K 聚类 → 36 条身份校准 HTML → 用人工标签建立每集独立说话人模型 → 全词评分和完整语义切分 → 质量门 → 48 条分层盲听验收。

第一轮动画页面只判断“伊蕾娜 / 非伊蕾娜 / 混合 / 不确定”，暂时忽略切头切尾；第二轮才同时判断边界与音质。这是为了先获得每集可靠的正负说话人锚点，避免再次出现大量别人的声音被全局高分误收。

关键入口脚本：

```text
radio_drama_work/prepare_anime_batch_v1.py
radio_drama_work/run_anime_asr_v1.py
radio_drama_work/run_anime_eres_scan_v1.py
radio_drama_work/prepare_anime_source_local_campp_v1.py
radio_drama_work/build_anime_cluster_calibration_v1.py
radio_drama_work/make_speaker_identity_review_page_v1.py
```

所有重任务继续使用本机 RTX 5070 Ti，CPU 线程限制为 2。当前没有训练、没有修改旧训练清单、没有 90 分声明。详细方法见 `FULL_RADIO_MINING_EXPERIENCE_V1.md` 第 19 节。

预校准现已完成，不再是“正在分离”状态：

```text
12/12 raw
12/12 vocals
12/12 other
286.419083 分钟
34,358 个日语 ASR 词
14,865 个 CAM++ 说话人窗口
68,712 个 ERes2Net 窗口
```

第一轮动画身份页已经生成并通过 36/36 路径、格式、时长、唯一波形哈希自检：

```text
radio_drama_work/anime_speaker_calibration_v1/index.html
```

36 条 / 121.720 秒，每集 3 条，构成为机器预测 24 条 target-like + 12 条 competitor，但这些预测绝对不能冒充真标签。用户在页面上只选“伊蕾娜 / 非伊蕾娜 / 混合 / 不确定”，本轮忽略切头切尾和音质。反馈下载前缀为：

```text
anime_speaker_calibration_feedback_v1_*.json
```

收到反馈后运行：

```powershell
runtime/python.exe ingest_anime_speaker_calibration_v1.py --feedback <反馈.json>
```

然后先检查 `human_anime_speaker_calibration_v1/report.json`：每集必须至少有一个 target 和一个 pure_non_target；缺任一锚点都要做定向第二轮，不能跳过。只有锚点覆盖完整后，才依次运行 `build_anime_source_local_clusters_v1.py`、`score_anime_source_local_words_v1.py`、`build_anime_semantic_candidates_v1.py`、`score_anime_semantic_quality_v1.py`、`build_anime_semantic_acceptance_v1.py`。当前仍未训练、未改旧清单、未声明 90 分。

## 23. 2026-08-02 12:27 动画首轮反馈已导入，等待 15 条定向补标

已导入：

```text
C:/Users/lulu/Downloads/anime_speaker_calibration_feedback_v1_2026-08-02T12-19-11-475Z.json
```

结果：36/36 完成，27 target、9 pure_non_target。所有 12 集都有 target，但 `majo_ep04`、`majo_ep09`、`majo_ep12` 缺 pure_non_target；原因是这三集首轮的 competitor 也被用户确认成伊蕾娜。不能在这三集上拟合无负锚点模型。

已生成定向补标页：

```text
radio_drama_work/anime_speaker_calibration_round2_v1/index.html
```

共 15 条 / 34.420 秒，每个缺失集 5 条。候选只来自未试听 competitor pool，已经排除该集所有人工 target cluster；路径与 SHA256 波形均唯一。用户只需判断“伊蕾娜 / 非伊蕾娜 / 混合 / 不确定”，暂时忽略切头切尾和音质。

收到 `anime_speaker_calibration_round2_feedback_v1_*.json` 后立即运行：

```powershell
runtime/python.exe ingest_anime_speaker_calibration_round2_v1.py --feedback <反馈.json>
```

随后检查：

```text
human_anime_speaker_calibration_v2/report.json
```

只有 `calibration_complete=true` 才进入每集聚类和 34,358 词全量重评分；否则只继续补缺失集。

同一批收到的广播剧最终边界反馈 3/3 通过。去掉同一句的一个近重复毫秒版本后，冻结 2 条 / 5.540 秒，并生成：

```text
radio_drama_work/human_confirmed_broadcast_v5_train_1p5_6s.list
```

v5 为 89 条 / 262.444250 秒，全部通过路径、唯一波形、32 kHz mono PCM16、1.5–6 秒和文本检查。未启动训练、未修改旧训练清单、未宣称 90 分。

## 24. 动画第二轮已导入；第 12 集等待稳定窗口第三轮

第二轮 15/15 反馈已导入，合并后为 29 target、19 pure_non_target、3 mixed。第 4、9 集锚点完整，但第 12 集第二轮只有 2 target + 3 mixed，仍缺纯负例。禁止把 mixed 当纯负锚点。

已生成第 12 集稳定 CAM++ 窗口页：

```text
radio_drama_work/anime_speaker_calibration_round3_v1/index.html
```

12 条 / 18 秒，来自 K=16 clusters `9,14,8,7`，每簇 3 条；人工 target clusters `0,5,10,11` 和 mixed clusters `1,13` 已全部排除。文件与 SHA256 波形唯一。

收到 `anime_speaker_calibration_round3_feedback_v1_*.json` 后运行：

```powershell
runtime/python.exe ingest_anime_speaker_calibration_round3_v1.py --feedback <反馈.json>
```

确认 `human_anime_speaker_calibration_v3/report.json` 的 `calibration_complete=true` 后，再把动画下游脚本的校准输入统一切到 v3，随后运行 12 集源内聚类、34,358 词重评分、完整句候选和音质门。当前未训练、未改旧训练清单、未声明 90 分。

## 25. 动画全量挖掘和人工验收已完成

第 12 集第三轮反馈已导入，12 集校准完整。最终身份统计：63 条，29 target、30 pure_non_target、3 mixed、1 uncertain。源内模型自检通过：29/29 target 保留、30/30 pure negative 拒绝、3/3 mixed 拒绝。

全量 GPU 结果：34,358 个词全部评分，严格状态通过 4,131 词；25 条完整句进入质量门，11 条 / 37.685031 秒通过，并由用户 11/11 验收无误。S007 原先漏勾，用户明确确认没有问题，已写入 `anime_semantic_acceptance_v1/feedback_completed_user_confirmed.json` 后导入。

动画严格人工清单：

```text
radio_drama_work/human_verified_anime_semantic_v1_direct.list
```

广播剧 v5 与动画清单合并为：

```text
radio_drama_work/human_confirmed_radio_anime_v1_train_1p5_6s.list
```

合并结果：100 条 / 300.129281 秒 / 约 5.002 分钟。路径、唯一波形、32 kHz mono PCM16、1.5–6 秒和文本检查全部通过。没有修改旧训练清单、没有启动训练、没有 90 分声明。

## 26. 动画第一批语义复核与边界救援（2026-08-02）

第一批 48 条语义候选已由用户试听：19 条直接通过，26 条属于目标声但边界被切，1 条与 OP 重叠，1 条杂音/音质问题，1 条不确定。边界候选没有用 ClearVoice/VASR 重建，而是从原始 BS-Roformer `vocals` 按 ASR 词边界生成 `word_expand` 和 `pad_220ms` 两种版本。

边界救援共 52 个版本，用户反馈后选择最佳版本并按 origin 去重，冻结 10 条 / 26.345094 秒：

```text
radio_drama_work/human_verified_anime_boundary_rescue_v1.list
radio_drama_work/human_anime_boundary_rescue_batch_v1/report.json
```

直接通过清单：

```text
radio_drama_work/human_verified_anime_review_batch_v1.list
```

动画目前人工确认 40 条 / 115.505125 秒。与广播剧 v5 合并后，新清单为 129 条 / 377.949375 秒 / 约 6.30 分钟：

```text
radio_drama_work/human_verified_anime_v2_train_1p5_6s.list
radio_drama_work/human_confirmed_radio_anime_v2_train_1p5_6s.list
radio_drama_work/human_confirmed_radio_anime_v2_report.json
```

旧 v1 清单仍保留，未覆盖；训练仍未启动。

## 27. 第二批动画试听

第一批的 48 个 origin 已从 431 条语义复核池排除，第二批重新选 48 条，覆盖 12 集。第 10 集剩余候选只有 1 条，所以第二批采用每集至少 1 条、总计 48 条，不重复优先于固定每集数量。

试听页：

```text
radio_drama_work/anime_semantic_review_batch_v2/index.html
```

用户反馈下载后导入：

```powershell
runtime/python.exe ingest_anime_semantic_review_batch_v2.py --feedback <反馈.json>
```

反馈分类：目标声音通过；非目标；混合/无缝接话；开头/结尾切断；杂音/音质；重叠；不确定。确认是伊蕾娜、但切头/切尾的片段应标 `boundary_cut`，后续从原始 vocals 救援；不要为了增加数量把混合或非目标标成通过。

## 28. 第二批反馈与救援状态

第二批 48 条已人工反馈：20 target、22 boundary_cut、5 uncertain、1 pure_non_target。20 条直接通过已冻结并合并。

当前人工确认清单为 149 条 / 442.164406 秒：

```text
radio_drama_work/human_confirmed_radio_anime_v3_train_1p5_6s.list
```

第二批的 22 个边界 origin 已从原始 BS-Roformer `vocals` 生成 44 条版本，输出页：

```text
radio_drama_work/anime_boundary_rescue_batch_v2/index.html
```

`word_expand` 优先恢复相邻 ASR 词，`pad_220ms` 只做原始波形边缘扩展；两者都不做声音重建。请只判断救援后是否目标声音完整自然。确认后运行：

```powershell
runtime/python.exe ingest_anime_boundary_rescue_batch_v2.py --feedback <反馈.json>
```

5 条不确定候选暂未加入训练清单，也没有训练或覆盖旧清单。
