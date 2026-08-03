# 伊蕾娜 12 部广播剧全量挖掘：处理经验与续作说明（v1）

更新时间：2026-08-02  
工作目录：`radio_drama_work/radio_drama_work`  
当前状态：全量本地挖掘、逐词重评分、语义切分、相对音质门控和盲听验收包均已生成；没有启动训练、没有连接服务器、没有修改旧训练清单，也没有宣称达到 90 分。

## 1. 这轮最终处理了什么

输入固定为 `full_track_manifest_v1.json` 中的 12 条 BS-Roformer `vocals.wav`，排除 `202607311250`。

- 总音轨时长：11,985.579 秒，即 199.760 分钟。
- 完整日语 ASR 词数：37,432。
- CAM++ 全轨说话人窗口：13,209。
- 12/12 来源均有 ASR、CAM++ 缓存和同剧内说话人模型。
- 所有新导出音频都直接来自原始 BS-Roformer vocals；没有使用 ClearVoice 或 VASR 重建人声。
- 重计算使用本地 RTX 5070 Ti；CPU 线程限制为 2。

## 2. 当前产量应该怎样解读

### 2.1 已经人工确认的广播剧素材

- 57 条，166.785 秒，即 2 分 46.8 秒。
- 组成：此前三轮标注中的 45 条纯目标 + 本次最终救回的 12 条边界片段。
- 清单：`human_confirmed_broadcast_v1.list`。
- 这些文件已验证为 32 kHz、单声道、PCM16，文本非空且无 U+FFFD。

### 2.2 机器说话人和边界严格池

- 189 条，549.605 秒，即 9 分 09.6 秒。
- 清单：`full_semantic_machine_strict_review_v1.list`。
- 这是逐词硬切、边界内缩和已知污染硬排除后的机器候选，仍然需要人工验收，不能直接称为纯净训练集。

### 2.3 再经过相对音质门控后的主试听池

- 143 条，419.605 秒，即 6 分 59.6 秒。
- 清单：`full_semantic_quality_strict_review_v1.list`。
- 质量条件：源内 NISQA-TTS 前 80% + 无削波 + 点击组不超过 3 次/秒 + SNR 代理不低于 5 dB。
- NISQA 只用于同一来源内相对排序，不是人工 MOS，更不是“90 分”。
- 试听网页：`full_semantic_quality_v1/index.html`，可以直接双击，不需要启动服务器。

### 2.4 尚可继续救的池

- 边界复核池：49 条，142.290 秒，即 2 分 22.3 秒。
- 音质复核池：46 条，130.000 秒，即 2 分 10.0 秒。它们是 189 条机器严格池中被相对音质门降级的部分，不是额外重复计入的素材。
- 边界试听页：`full_semantic_boundary_review_v1/index.html`。
- 音质试听页：`full_semantic_quality_review_v1/index.html`。

### 2.5 当前合理的数量结论

- 已确定可用：57 条 / 2 分 46.8 秒。
- “已确认 + 主机器候选”的高质量潜力：200 条 / 9 分 46.4 秒。
- 如果 189 条机器严格池和 49 条边界池最终全部人工通过，去重后的理论上限约为 295 条 / 14 分 18.7 秒。
- 上限不是保证。任何人工发现的非目标词、重叠说话、截断或明显伪影都必须继续剔除。
- 当前还不足以诚实宣称已经有 15–40 分钟纯净数据，也不能据此保证训练达到 90 分。

## 3. 12 条边界救援真正教会了什么

### 3.1 “边界坏”不等于“说话人错”

旧流程把切头、切尾或无缝接话片段整体当作不可用，甚至可能把它们当作声纹负例。这会同时造成两种损失：

1. 片段中本来很长的纯伊蕾娜部分被整段丢掉；
2. 如果混合片段含有伊蕾娜，拿它训练“非目标”会反向压制真正的目标声。

现在的规则是：

- `pure_non_target` 才能作为纯负锚点；
- `mixed/overlap` 只做验证和硬排除，不作为纯负锚点；
- 原始 `boundary_cut` 不进入声纹模型；
- 只有用户听过并通过的最终内缩修正版，才作为新的目标锚点。

### 3.2 同剧内聚类比全局固定相似度可靠

伊蕾娜在不同剧、不同情绪、不同录音条件下的全局声纹分布变化很大。全局阈值曾经同时出现：

- 目标声音因情绪变化被大量拒绝；
- 同一声优或相近女声被错误放行。

新流程按每一部广播剧独立聚类，用该剧内的人工目标、纯非目标和混合验证样本决定目标簇。v4 模型结果：

- 目标锚点保留 53/57；
- 纯非目标拒绝 53/53；
- 混合样本拒绝 21/23。

剩余两个混合样本说明“整段聚类”仍不能处理无缝换人，因此它只能确定候选区域，不能直接输出训练音频。

### 3.3 无缝换人必须主动向内损失

用户发现过典型错误：前半句是伊蕾娜，后面其他角色无停顿接上，旧窗口会把对方的一个短词（例如「はい」或一个「飲」）一起归入目标。

新规则：

- 每个 ASR 词单独做 1.2 秒 CAM++ 判断；
- 任一词失败，前后片段立即硬切断；
- 不允许用“前后都是目标”去填回中间的失败词；
- 目标 run 与失败词无缝相邻时，先从目标侧额外删除至少 0.18 秒，再寻找切点；
- 这种主动损失召回率的做法，是防止一字污染最有效的手段。

### 3.4 先保证人物，再追求完整句子

旧流程曾为了句子完整向外扩展，结果把相邻角色重新扩进来。现在顺序固定为：

1. 逐词说话人硬门；
2. 已知污染硬掩码；
3. 无缝切换边缘内缩；
4. 只在剩余连续目标词内部寻找语义结尾、停顿和能量谷；
5. 超过 6 秒时在内部安全边界拆分；
6. 不能找到完整安全边界的片段只进 review。

“完整”只能在已经安全的目标 run 内优化，绝不能跨过失败词。

### 3.5 切点不能只靠静音

广播剧里经常没有静音就换人，单纯静音切片无法解决人物边界。最终切点同时使用：

- ASR 词起止；
- 句尾词形（例如 `でした/ました/です/ます/ですか`）；
- 词间间隔；
- 20 ms RMS、10 ms hop 的局部能量谷；
- 说话人切换时的向内夹紧；
- 5 ms 淡入淡出。

能量谷只负责在已经确定的目标词附近找自然切点，不能替代说话人判断。

### 3.6 不再用 ClearVoice/VASR 修复训练波形

ClearVoice 的匿名双路分离会重建声音，之前试听已确认明显发糊；VASR 也可能产生高频伪影。它们可以用于诊断或重叠说话辅助判断，但当前训练候选必须保留原始 BS-Roformer vocals，只允许截取、重采样和极短淡入淡出。

## 4. 本轮完整流水线

```text
12 条 BS-Roformer vocals
  -> FunASR VAD 生成 1.5s / 0.75s CAM++ 窗口
  -> 12 部各自建立 source-local spherical clusters
  -> 45 个旧纯目标 + 12 个最终边界救援作为目标锚点
  -> 53 个纯非目标作为负锚点；混合只验证
  -> 对 37,432 个 ASR 词重新做 1.2s CAM++ GPU embedding
  -> 同源目标簇、上下文窗口、正负锚点四项逐词门控
  -> 已知非目标/混合/质量/dirty-boundary 区间硬掩码
  -> 失败词硬切 + 无缝边缘向内删除至少 0.18s
  -> 语义句尾/停顿/能量谷内切，输出 1.5–6.0s
  -> 32kHz mono PCM16，5ms fade
  -> 点击/削波/SNR + 源内 NISQA-TTS p20
  -> 20 个机器候选 + 4 个已知 gold 的盲听验收包
```

## 5. 关键脚本与复跑顺序

Python 运行时：

```powershell
$py = 'radio_drama_work/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50/runtime/python.exe'
```

### 5.1 补齐 12 部 CAM++ 缓存

```powershell
& $py radio_drama_work/prepare_full_source_local_campp_v1.py --resume
```

产物：`full_source_local_campp_v1/`。七部复用旧缓存，五部由 GPU 新生成；缓存总覆盖必须为 12/12。

### 5.2 重建同剧内目标簇

```powershell
& $py radio_drama_work/source_local_cluster_anchor_v4.py
```

产物：`source_local_cluster_anchor_v4/models/` 和 `report.json`。

### 5.3 全轨逐词 GPU 重评分

```powershell
& $py radio_drama_work/score_full_source_local_words_v1.py --resume
```

产物：`full_source_local_word_scores_v1/`。每个来源都有独立 embedding cache 和词级 JSON；完整词数必须为 37,432。

### 5.4 语义合并与边界内缩

```powershell
& $py radio_drama_work/build_full_semantic_candidates_v1.py
```

产物：`full_semantic_mining_v1/`、189 条机器严格池和 49 条边界池。

### 5.5 GPU NISQA-TTS 与声学门控

```powershell
& $py radio_drama_work/score_full_semantic_quality_v1.py --resume
```

产物：`full_semantic_quality_v1/` 和 143 条主试听池。

### 5.6 生成盲听验收包

```powershell
& $py radio_drama_work/build_full_semantic_acceptance_v1.py
& $py radio_drama_work/make_human_review_page_v2.py `
  --root radio_drama_work/full_semantic_acceptance_v1 `
  --title '伊蕾娜全量挖掘盲听验收' `
  --round-label 'Blind acceptance · 20 machine + 4 gold · 24 clips' `
  --store-key ireina_full_semantic_acceptance_v1 `
  --feedback-version full_semantic_acceptance_feedback_v1 `
  --download-prefix full_semantic_acceptance_feedback_v1
```

试听页：`full_semantic_acceptance_v1/index.html`。直接双击打开，无需服务器。

## 6. 下一步人工验收规则

先听 24 条盲验包，不要先看 `private_manifest.json`：

- 20 条是机器候选；
- 4 条是已知纯净 gold 对照；
- 页面只显示 `A001–A024`。

反馈要求仍然是：

- 不勾错误：整条只有伊蕾娜、边界完整、清楚自然、可训练；
- 勾错误并选择：非伊蕾娜、混入/无缝接话、重叠、切头切尾、音质问题或无法确定；
- 如果只是极短边缘问题，备注“开头/结尾多少秒”，不要把整条简单归为非目标。

晋级规则：

- 20 条机器样本里出现任何明确非目标声：整批不自动晋级，把该精确区间加入硬掩码，只重跑受影响来源的候选构建和质量阶段；
- 没有非目标声，且至少 18/20 被认为自然、完整、可训练：可以冻结当前阈值；
- 切头切尾但人物正确的样本进入新一轮内缩救援，不作为声纹负例；
- 通过盲验后，再决定是否需要把 143 条全部听完或做分层抽验。

## 7. 已知风险和不能忘的例外

- `28243657968-1-192` 本轮没有新增 1.5 秒以上机器严格片段；它仍有人工确认目标，但自动词 run 太碎。不能为了让每个来源都有产量而桥接失败词。
- `113814583-1-160` 没有纯同源负锚点；采用更严格词门和硬掩码，仍必须重点抽检。
- `28055832533-1-192` 有一个已知无缝混合片段曾被整段目标簇接纳，因此同样是 guarded 来源。
- `1405372904-1-192` 与 `306672160-1-192` 历史上出现过一字混入；虽然 v4 模型表现改善，验收时仍要重点听边界。
- 两个 `safe_word` 文件始终强制排除。
- 三个 `243358894` 用户明确否决区间始终强制排除。
- 自动筛选只能叫“机器高置信度候选”，不能写成“逐条人工听审绝对纯净”。
- 任何数量目标都不能凌驾于纯净度；不足 15 分钟时必须如实写“数据仍不足”。

## 8. 重要产物索引

```text
full_source_local_campp_v1/report.json
source_local_cluster_anchor_v4/report.json
full_source_local_word_scores_v1/report.json
full_semantic_mining_v1/report.json
full_semantic_mining_v1/index.html                  # 189 条说话人/边界严格池
full_semantic_quality_v1/report.json
full_semantic_quality_v1/index.html                 # 143 条主质量池
full_semantic_boundary_review_v1/index.html         # 49 条边界救援池
full_semantic_quality_review_v1/index.html          # 46 条音质复核池
full_semantic_acceptance_v1/index.html              # 优先听：24 条盲验
human_confirmed_broadcast_v1.list                    # 57 条已人工确认
full_semantic_quality_strict_review_v1.list          # 143 条待验主池
```

## 9. 一句话交接

> 本节原来的“任何失败词硬切、边缘固定向内损失”是首轮方案，已被第 10 节人工盲验结果正式废止。最新交接是：复用 12 部同源 CAM++ 模型和 37,432 个逐词缓存，但必须区分短词 `soft_fail` 与持续强非目标 `hard_speaker_break`；先补全完整词句，再做词外边界；人工混入掩码永远不可跨越，边界错误永远不可改成人物负例。

## 10. 2026-08-02 首轮全量盲验结果：旧“逐词硬切”规则必须废止

### 10.1 人工结果

用户已完整听完 `full_semantic_acceptance_v1` 的 24 条盲测，反馈文件为：

```text
C:/Users/lulu/Downloads/full_semantic_acceptance_feedback_v1_2026-08-02T09-13-25-238Z.json
```

解析结果：

- 7 条通过；
- 13 条切头/切尾；
- 2 条混入/无缝接话；
- 2 条杂音/音质问题。

角色拆分后：

- 20 条机器候选中仅 5 条通过，12 条边界失败，2 条混入，1 条音质失败；
- 4 条旧 gold 对照中 2 条通过，1 条边界失败，1 条音质失败。

因此旧 gold 只能证明“历史上曾被认为是目标人物”，不能自动证明当前切点完整或音质可训练。当前 143 条主池绝对不能因 NISQA/声纹通过就整体晋级。

反馈已安全落盘到：

```text
human_full_semantic_acceptance_v1/labels_resolved.json
human_full_semantic_acceptance_v1/hard_exclusion_intervals.json
human_full_semantic_acceptance_v1/training_exclusion_intervals.json
human_full_semantic_acceptance_v1/report.json
```

### 10.2 真正根因

旧流程的两个假设被人工试听证伪：

1. `strict_speaker_pass=False` 不等于该短词一定是别人。日语单字、无声辅音、句尾助词和 0.1–0.2 秒短词的 1.2 秒声纹窗很容易受前后文影响。同一个伊蕾娜句子里会出现一两个低 margin 词；把它们全部当换人点，会得到「完成さ」「許してく」「丁寧に描か」「記憶力はゼロと言っても過」等明显半句。
2. 对无缝邻接失败词一律删除 0.18 秒，会把本来通过的完整词再次削掉。例如被删除的「よ」「れる」「から」「よね」恰好都是形成自然句尾所需的词。

原来的 1.5–6 秒拆分器随后又从碎 run 中挑一个合法时长，造成“人物大多正确，但开头和结尾像被刀切掉”的系统性错误。NISQA、RMS、点击检测无法识别这种语义/音素完整性失败。

### 10.3 立即废止的旧规则

本文件 3.3、3.4、4、9 节所写的下列规则只代表首轮实现，现已过时：

- 任一失败词无条件硬切；
- 不允许填回任何失败短词；
- 无缝邻接失败词时固定向目标侧删除至少 0.18 秒；
- 只要切后时长落在 1.5–6 秒且边界能量低，就可称为“机器严格边界”。

保留不变的原则只有：不得跨越**人工确认的非目标/混合区间**，不得跨越持续、强烈且多判据一致的非目标段；边界失败不能回灌成负声纹。

### 10.4 新边界规则

后续全量重建应采用“词级迟滞 + 完整短句”而不是“单词一票否决”：

1. 人工确认目标的核心片段可作为该处的局部正锚；
2. 一个低分短词若前后均为同一目标簇、source-local context 仍偏目标、未碰人工硬掩码，只标成 `soft_fail`，允许为补全同一句而桥接；
3. 只有连续多个强非目标词、target-negative 明显为负、source-local context 转为非目标，或命中人工混入掩码时，才是 `hard_speaker_break`；
4. 先恢复完整 ASR 词和句尾，再在词外 20–25ms 或真实停顿的局部能量谷切；
5. 不再固定腐蚀 0.18 秒；无缝边界只做约 25ms 的词级保护，并交人工边界验收；
6. 超过 6 秒的完整句先进入 review。确认人物与边界后，再在同一说话人内部的真实停顿拆分，不能先任意截成半句；
7. 语义完整性必须成为独立硬门：明显以助词、半个词或未完成谓语开头/结尾的候选不得自动晋级。

### 10.5 本轮新增硬排除

两条人工确认混入已写入说话人硬掩码，扩边不得覆盖：

```text
37239655783-1-192  1094.970–1098.420  # A006，开头非伊蕾娜
1405440540-1-192     42.720–44.500   # A013，混入/无缝接话
```

它们是验证/硬排除区间，不应把整段混合音频直接当纯负声纹参考。

质量失败同样不作为说话人负例。当前明确质量复核点包括 A005、A018，以及带“音质不清晰”备注的边界样本 A004。

### 10.6 已生成的边界救援包

新脚本：

```text
ingest_full_semantic_acceptance_feedback_v1.py
build_full_semantic_acceptance_boundary_rescue_v1.py
```

新试听目录：

```text
full_semantic_acceptance_boundary_rescue_v1/index.html
```

内容为 13 个边界问题对应的 15 条盲测版本，共 71.931 秒：

- 音频全部从原始 BS-Roformer vocals 重切；
- 补全被拆掉的完整日语词句；
- 使用真实停顿内的局部能量谷；无缝处仅留 25ms 词级保护；
- 只加 2ms 防点击淡入淡出；
- 不使用 ClearVoice/VASR 重建；
- 15/15 文件存在、HTML 引用有效、无重复、无人工混入掩码重叠；
- 3 条完整句超过 6 秒，仅供边界验收，暂不进入训练。

试听时仍按最严格标准：整条只有伊蕾娜、首尾字完整、音质可训练才通过。同一问题有两个版本时，只保留人工通过的版本；若都失败则继续精确裁边。

### 10.7 当前状态

- 尚未启动训练；
- 尚未修改任何已有训练清单；
- 旧 143 条主池仍是“待重建/待验候选”，不是可训练数据；
- 尚未宣称可达到 90 分；
- 下一步必须先收回 `full_semantic_acceptance_boundary_rescue_v1` 的人工反馈，再据通过/失败边界更新全量 v2 分段器。

## 11. 2026-08-02 完整词句救援首轮反馈：12/15 通过

用户已完整试听 `full_semantic_acceptance_boundary_rescue_v1` 的 15 条：

- 12 条通过；
- 1 条仍切开头（A021 的救援版漏掉了前置词「もう」）；
- 1 条扩边后左侧带入非目标声音（A001 的完整句版）；
- 1 条存在环境音（A017）。

两个存在备选版本的来源均已正确去重：

- A001 的完整扩边版失败，但保守版通过，只保留保守版；
- A012 的完整句版和保守版都通过，优先保留语义更完整的版本。

最终已冻结 11 个不同原问题、11 条人工听过的通过音频，共 54.158 秒：

```text
human_verified_semantic_rescues_v1/
human_verified_semantic_rescues_v1.list
```

其中直接满足 1.5–6.0 秒训练长度的是 9 条 / 37.725 秒：

```text
human_verified_semantic_rescues_v1_train_1p5_6s.list
```

这只是本轮边界救援冻结清单，尚未与旧 57 条人工素材、首轮直接通过机器样本或未来全量 v2 候选做来源时间去重，也没有加入任何正式训练清单。

### 11.1 新增精确硬掩码

A001 的失败完整句版明确说明向左扩得过远。通过的保守兄弟版本给出了安全起点，因此只屏蔽多出来的左边缘，不屏蔽通过版：

```text
1405372904-1-192  496.375–497.950  # R006 开头非伊蕾娜；R013 从 497.955 开始并已通过
```

这一例验证了新规则的两面：短词 `soft_fail` 确实经常需要补回，但语义扩边不能无条件跨越整串低分词。全量 v2 应同时生成保守/完整候选；只要完整版跨入持续低上下文区，就必须进入人工 review 或被硬掩码截断。

### 11.2 最终六条复核包

已生成最后 6 条快速复核：

```text
full_semantic_boundary_rescue_round2_v1/index.html
```

内容：

- A021：补回遗漏的「もう」；
- A011：将已通过的 6.075 秒句仅缩短前置静音，得到 6.000 秒版本；
- A004：将已通过的 10.358 秒长句按完整词边界拆成 5.985 秒和 4.353 秒两段；
- A017：将带环境音的 6.543 秒句拆成 1.760 秒和 4.833 秒两段，判断环境音是否只污染局部。

6/6 文件存在、页面引用有效、无重复、无人工硬掩码重叠，全部仍来自原始 BS-Roformer vocals，未降噪、未重建。每个新切点仍需听审；上一条长句通过不能自动替代新拆分边界的验收。

### 11.3 当前停止点

- 等待 `full_semantic_boundary_rescue_round2_v1` 的 6 条反馈；
- 反馈回来后冻结最终边界规则，随后才能重建全量 v2 候选；
- 不启动训练，不修改旧训练清单，不宣称具备 90 分数据基础。

## 12. 2026-08-02 最终六条反馈：4/6 通过，只剩一个毫秒级边界

`full_semantic_boundary_rescue_round2_v1` 已完整听审：

- T001（6.000 秒训练长度版）通过；
- T004（A017 环境音句的前半段，1.760 秒）通过；
- T005、T006（A004 长句拆分后的两段）均通过；
- T002 补回「もう」后正文纯净，但开头仍残留极短「诶」；
- T003 仍有明显翻书和合书声，剔除。

现在冻结版为：

```text
human_verified_semantic_rescues_v2/
human_verified_semantic_rescues_v2.list
```

统计：13 条 / 55.823 秒，全部为 32kHz mono PCM16、1.5–6.0 秒，全部是用户实际听过并通过的具体波形。它们仍只是边界救援阶段的冻结集，尚未并入正式训练清单。

环境音案例给出的新经验：质量问题必须定位到源时间，不能因一句中存在翻书声就丢弃整句。可以按完整词边界拆成子段分别听审；本例成功保留前半段，后半段继续作为质量排除而不是人物负例。

### 12.1 A021 最后三个起点版本

T002 的失败版从 475.565 秒开始，目标词「もう」的 ASR 起点为 475.590 秒。已从原始 vocals 生成三个版本：

```text
full_semantic_edge_final_review_v1/index.html
```

- 475.590：目标 ASR 起点；
- 475.610：向内 20ms；
- 475.630：向内 40ms。

三个版本正文和结尾完全相同，只用于在“残留前一人元音”和“削掉目标词 /m/”之间确定最早安全起点。页面为盲测；任一版本无残留且「もう」完整即可通过，若多个通过则保留起点最早者。

3/3 文件存在、HTML 引用有效、无硬掩码重叠。等待这 3 条反馈后即可冻结边界救援规则并开始全量 v2 候选重建。

## 13. 2026-08-02 最终毫秒边界已冻结：475.610 秒

`full_semantic_edge_final_feedback_v1_2026-08-02T09-47-17-672Z.json` 已完整解析：

- U001（475.610 秒）通过；
- U002（475.630 秒）通过；
- U003（475.590 秒）失败，人工备注为“这个前头有诶的声音”。

按“多个版本通过时保留最早安全起点”的规则，最终选择 475.610 秒。由此得到最终边界救援冻结集：

```text
human_verified_semantic_rescues_v3/
human_verified_semantic_rescues_v3.list
```

统计为 14 条 / 60.567875 秒；全部为用户实际听过并通过的具体波形，全部是 32 kHz、mono、PCM16、1.5–6.0 秒。

新增的精确边界排除是：

```text
280479210-1-192  475.565–475.610  # 前一角色残留“诶”到目标「もう」之间的脏过渡
```

它是混合边界，不是纯非目标负声纹参考。另保留质量排除：

```text
306672160-1-192   82.425–87.207637  # 翻书/合书环境声
```

环境声也不得用作人物负例。

## 14. 2026-08-02 全量语义边界 v2 重建完成

新增可复跑脚本：

```text
calibrate_full_semantic_word_states_v2.py
build_full_semantic_candidates_v2.py
validate_full_semantic_v2.py
```

校准集由人工反馈映射回全轨逐词分数：

- 78 个目标区间、988 个目标词；
- 53 个纯非目标区间、510 个纯非目标词；
- 26 个混合/不安全区间、217 个混合词。

关键校准结果：

- 旧 strict：目标召回 79.96%，纯负误收 0.98%；
- `soft_balanced`：目标召回 89.37%，纯负误收 0.20%；
- `edge_soft_anchor`：目标召回 89.98%，纯负误收 0%；
- `hard_negative_balanced`：目标误杀 0.30%，纯负识别 92.35%。

因此 v2 允许经校准的短词 soft/edge-soft 回归，但仍禁止跨越任意 unknown 词拼句；每个候选至少需要一个 strong 目标锚点。已知混合掩码继续是硬否决，因为模型对混合词的自动通过率仍约 42%–47%，不能只靠阈值解决无缝接话。

重建结果：

- 12/12 部来源全部处理；
- 37,432 个缓存 ASR 词全部重新归类；
- 168 条机器严格复核候选 / 561.826375 秒；
- 448 条语义/边界复核候选 / 927.311063 秒；
- 64 个与人工确认素材高度重叠的机器候选已优先去重；
- `28243657968-1-192` 完成扫描，但没有片段同时满足 strict 说话人、语义完整和边界门槛。

人工素材同步聚合并按源区间去重：

- 全部人工确认：75 条 / 238.563 秒；
- 满足 1.5–6.0 秒训练长度：70 条 / 203.137906 秒。

对应清单：

```text
human_confirmed_broadcast_v2.list
human_confirmed_broadcast_v2_train_1p5_6s.list
full_semantic_machine_strict_review_v2.list
```

`validate_full_semantic_v2.py` 已通过：168 条机器候选和 70 条人工训练安全片段均路径真实、无重复、无空/乱码文本、不覆盖硬排除区间、不含两个 `safe_word`，并统一为 32 kHz mono PCM16。

## 15. 2026-08-02 GPU NISQA 与声学质量门完成

新增脚本：

```text
score_full_semantic_quality_v2.py
```

该脚本强制 CUDA，CPU 线程限制为 2，并同时评分：

1. 导出的 32 kHz PCM16 候选；
2. 同一 BS-Roformer vocals 原始时间区间的内存重采样版本。

这样可以区分“来源本身音质一般”和“导出过程造成退化”。最终 168 条中：

- 115 条通过质量门 / 380.155969 秒（约 6 分 20 秒）；
- 53 条移入质量复核 / 181.670406 秒；
- 37 条触发来源内 NISQA 后 20%；
- 14 条触发异常点击门；
- 8 条触发源边界能量比 ≥0.85 的严重切头/切尾风险；
- 0 条因削波失败；
- 0 条因低 SNR proxy 失败；
- 0 条因相对同一原始区间下降超过 0.20 失败。

原因计数可以重叠，因此 37+14+8 不等于 53。NISQA 只作来源内相对排序，不是人工 MOS，更不是 90 分评测。

质量通过池按来源统计：

| 来源 | 条数 | 秒 |
|---|---:|---:|
| 113814583-1-160 | 19 | 66.615 |
| 1311215796_u1-1-192 | 2 | 4.135 |
| 1405372904-1-192 | 1 | 1.985 |
| 1405440540-1-192 | 12 | 46.055 |
| 243358894-1-192 | 9 | 24.605 |
| 253810599-1-192 | 4 | 14.545 |
| 280479210-1-192 | 33 | 101.925 |
| 28055832533-1-192 | 7 | 22.920 |
| 306672160-1-192 | 6 | 22.225 |
| 37141415568-1-192 | 2 | 6.950 |
| 37239655783-1-192 | 20 | 68.195 |

输出：

```text
full_semantic_quality_v2/report.json
full_semantic_quality_v2/audit_manifest.json
full_semantic_quality_strict_review_v2.list
full_semantic_quality_review_v2.list
```

把 70 条人工训练安全素材与 115 条机器待验素材相加，在机器候选已对人工区间去重的前提下，当前潜在总量为 185 条 / 583.293875 秒（约 9 分 43 秒）。其中只有前 70 条已人工确认；115 条仍不能写入训练清单。总量尚未达到原计划 15 分钟下限，因此仍应标记为“数据不足以宣称具备 90 分训练基础”。

## 16. 2026-08-02 全量 v2 盲听验收页

新增：

```text
build_full_semantic_acceptance_v2.py
full_semantic_acceptance_v2/index.html
```

页面可直接双击打开，不需要 HTTP 服务；已自检 24/24 音频存在且页面无任何网络依赖。内容为：

- 20 条机器候选；
- 4 条已确认 gold 对照；
- 11 个有质量通过候选的来源全部覆盖；
- `1405372904` 的唯一通过片段强制抽入；
- `306672160` 高风险源抽入 3 条；
- 其余按最低声纹余量、较高边缘能量、接近来源 NISQA p20 等风险优先抽样。

新版反馈类型把“开头被切”“结尾被切”“开头和结尾都被切”分开。只要声音本人但边界不完整，应标对应边界类型；后续从原始 vocals 自动救援，不能把它当人物负例。明确非目标、无缝接话或重叠说话才进入人物硬排除。

反馈文件前缀：

```text
full_semantic_acceptance_feedback_v2_*.json
```

当前停止点是等待这 24 条盲听反馈。没有启动训练，没有修改任何已有训练清单，没有宣称达到 90 分。

## 17. 2026-08-02 v2 首轮 24 条反馈：13/20 机器样本直接通过

已解析并归档：

```text
full_semantic_acceptance_feedback_v2_2026-08-02T10-38-49-359Z.json
human_full_semantic_acceptance_v2/
```

24 条全部听完，结果为：

- 4/4 条 gold 对照通过；
- 13/20 条机器候选直接通过；
- 3 条机器候选为切头/切尾；
- 3 条机器候选混入非目标声音；
- 1 条机器候选有脚步走动声。

因此机器直接通过率为 65%，未达到 18/20，且出现了 3 条明确非目标声音。`full_semantic_quality_strict_review_v2.list` 的其余候选不得整批晋级。四个 gold 已存在于旧人工池，不重复计数。

13 条新通过机器波形已逐文件复制、SHA256 校验并冻结：

```text
human_verified_full_semantic_v2_direct/
human_verified_full_semantic_v2_direct.list
```

统计为 13 条 / 42.750125 秒，全部是用户实际听过并通过的准确波形，全部 32 kHz mono PCM16、1.5–6.0 秒。

与此前 70 条人工训练安全素材合并后的新独立版本：

```text
human_confirmed_broadcast_v3_train_1p5_6s.list
human_confirmed_broadcast_v3_report.json
```

当前严格人工确认总量为 83 条 / 245.888031 秒（约 4 分 05.9 秒）；路径、文本、格式、时长、重复路径和重复波形哈希均已通过检查。没有覆盖或修改 v2 清单。

### 17.1 七条失败的精确含义

- B005：文本从「しいこと…」开始且漏掉句尾「よ」；ASR 显示应补回词 615「嬉」和词 648「よ」。
- B007：文本为「名希望さん…」，缺词 770「匿」。
- B022：问句「先生こそ本当に作ったんです」缺词 1240「か」。
- B009：前一角色的残句「ダメです」与目标回复「でもダメです」无缝相接；目标可从词 644 / 210.380 秒附近重切。
- B013：目标正文正确，但前一角色在 413.750 秒结束时留下极短尾音；原 +20ms 滞后仍不够，应继续做 20ms 步进。
- B024：前一角色说「そっくりだ」，目标从词 3292「そうですね」/ 956.610 秒开始回复；原局部声纹受上下文平滑影响，把前半句误判为 strong。
- B018：1.635 秒句中有脚步走动声；片段太短且污染位置未确认，不做重建/降噪，整条暂时质量排除。

三个 mixed 候选目前只进入：

```text
human_full_semantic_acceptance_v2/contamination_localization_queue.json
```

没有把整条 mixed 片段写成说话人硬负例，也没有屏蔽其中已确认“其他部分很好”的目标语音。必须等重切版本通过后，才能用“原失败起点到最早安全起点”形成精确边缘掩码。

## 18. 2026-08-02 v2 边界与无缝接话救援页

新增脚本和页面：

```text
build_full_semantic_acceptance_v2_rescue_v1.py
full_semantic_acceptance_v2_rescue_v1/index.html
```

页面可直接双击打开，无网络依赖。共 6 个原问题、14 个盲版本、46.451374 秒：

- B005：完整 6.335 秒句作身份/连续性对照，并生成 2.198 秒和 4.030 秒两个训练长度完整词分句；
- B007：补回「匿」后的 2.458 秒版本；
- B022：补回问句末尾「か」后的 2.330 秒版本；
- B009：从目标第二句起点开始的 0/20/40ms 三版本；
- B013：在原失败起点基础上再向内 20/40/60ms 三版本；
- B024：从目标回复「そうですね」开始的 0/20/40ms 三版本。

14/14 音频存在、编号和路径唯一、不覆盖既有硬掩码；最短 1.630 秒，最长 6.335 秒。所有音频仅从原始 BS-Roformer vocals 重切并加 2ms 防点击淡入淡出，没有 ClearVoice/VASR 重建或 AI 降噪。

反馈文件前缀：

```text
full_semantic_acceptance_v2_rescue_feedback_v1_*.json
```

当前停止点是等待这 14 条反馈。通过后按每个 origin 选择最早安全、语义最完整且训练长度合规的版本；B005 的 6.335 秒完整句本身只作对照，不直接进入 1.5–6 秒清单。没有训练、没有修改已有训练清单、没有 90 分声明。

## 19. 2026-08-02 v2 救援反馈结论与动画新域处理方案

`full_semantic_acceptance_v2_rescue_v1` 的 14 条反馈已经解析并归档。人工结果为 5 条通过、9 条失败，其中训练长度内最终冻结 4 条 / 11.016219 秒：

```text
human_verified_full_semantic_v2_rescues_v1.list
human_verified_full_semantic_v2_rescues_v1/report.json
```

具体结论：

- B005 的 6.335 秒完整句通过，但只保留为连续性参考；实际冻结两个完整分句，约 2.198 秒和 4.030 秒；
- B007 补回「匿」的完整版本通过；
- B022 补回问句末尾「か」的完整版本通过；
- B009 原候选和三个第二回复重切版本全部被确认不是伊蕾娜，因此只对 `306672160-1-192` 的 209.765–215.540 秒建立精确人物硬负例；
- B013 的 20/40/60ms 版本都仍带前一说话者，必须整句丢掉「ご安心ください」，从词 1483「先生」重新开始；
- B024 的三个版本都仍带前一说话者，必须整句丢掉「そうですね」，从词 3293「クリソツですね」重新开始；
- B018 的脚步声仍是质量排除，不作为人物负例。

上述 4 条训练长度救援已与此前 83 条人工确认集做路径与波形哈希去重，生成新的独立合并版本：

```text
human_confirmed_broadcast_v4_train_1p5_6s.list
human_confirmed_broadcast_v4_report.json
```

当前严格人工确认总量为 87 条 / 256.904250 秒（约 4 分 16.9 秒）。全部路径真实、波形哈希唯一、文本有效、格式为 32 kHz mono PCM16、时长 1.5–6.0 秒；没有覆盖任何旧清单。

B013/B024 最后 3 条极小复核页：

```text
full_semantic_acceptance_v2_edge_final2_v1/index.html
```

这次不再以 20ms 继续蚕食同一前置短语，而是完整放弃已证实属于上一角色的短语，再从下一完整目标词开始。该经验必须用于动画：无缝接话时不能依赖长窗声纹平滑后的单词标签，也不能只做毫秒裁边；应把明确非目标词两侧切断，从下一完整目标词或短语重新起句。

### 19.1 新增 12 集动画资产

新输入为《魔女之旅》12 集主剧集，容器位于 `F:/BaiduNetdiskDownload`。每集只有一条 48 kHz 双声道 AAC 音轨，字幕为硬字幕、没有可读取字幕流。12 集实测总时长 286.419083 分钟（约 4 小时 46 分）；NCOP/NCED 均为歌曲资产，明确排除，不参与对白挖掘。

隔离目录：

```text
anime_batch_v1/raw/
anime_batch_v1/vocals/
anime_batch_v1/other/
anime_batch_v1/manifest.json
anime_mining_v1/
```

12/12 原始音轨已经提取为 44.1 kHz stereo PCM16。GPU BS-Roformer 分离使用 `model_bs_roformer_ep_317_sdr_12.9755`、CUDA FP32；CPU 线程固定为 2。所有训练候选最终仍从 BS-Roformer `vocals.wav` 一次导出，不使用 ClearVoice 或 VASR 重建目标声音。

### 19.2 动画不能直接沿用广播剧全局阈值

广播剧阶段已经证明：168 条老目标参考的全局 CAM++/ERes2Net 绝对阈值会漏掉大量目标，也会因声线、情绪和上下文平滑误收其他年轻女性角色。动画又引入了新的配乐、混响、声优表演域和大量同龄女性角色，因此不能把“全局高分”直接晋级训练。

动画采用以下两级校准：

1. BS-Roformer 分离后，分别运行 Faster-Whisper 日语逐词 ASR、ERes2Net 全轨扫描、FunASR VAD + CAM++ 窗口缓存；两种 embedding 始终分开计算和保存；
2. CAM++ 在每一集内部做 K=8/12/16/20 多尺度球面聚类，168 条老纯净伊蕾娜参考只用于给簇排序；ERes2Net 作为独立支持/硬负例警告；
3. 每集先抽 2 条目标候选和 1 条竞争说话人，共 36 条身份校准样本；本轮只判断“伊蕾娜 / 非伊蕾娜 / 混合 / 不确定”，暂不把切头切尾当身份错误；
4. 人工标签回来后，为每集建立独立目标/非目标簇，再对所有 ASR 词做 1.2 秒 CAM++ 局部评分；
5. 完整短语构建继续使用语义终点、能量谷和完整词边界。无缝硬切边界永不自动进入 strict，只进入人工边界复核；
6. strict 候选还必须获得独立 ERes2Net 支持并通过点击、削波、噪声、边缘能量和源内相对质量门；
7. 生成分层盲听页后，由用户确认具体波形，只有人工通过的 1.5–6 秒片段才能进入新的动画训练候选清单。

新增脚本：

```text
prepare_anime_batch_v1.py
run_anime_asr_v1.py
prepare_anime_source_local_campp_v1.py
run_anime_eres_scan_v1.py
build_anime_cluster_calibration_v1.py
make_speaker_identity_review_page_v1.py
ingest_anime_speaker_calibration_v1.py
build_anime_source_local_clusters_v1.py
score_anime_source_local_words_v1.py
build_anime_semantic_candidates_v1.py
build_anime_semantic_acceptance_v1.py
```

当前仍未启动训练、未修改任何旧训练清单、未把机器候选称为可训练数据，也没有 90 分声明。

### 19.3 动画预校准已完整跑通

12 集 GPU BS-Roformer 分离已全部完成：raw/vocals/other 均为 12/12，总时长 286.419083 分钟，分离耗时 2945.304 秒。随后可恢复流水线 `run_anime_precalibration_v1.py --phase all --resume` 已完成：

- Faster-Whisper large-v3-turbo：12/12，共 34,358 个日语词；无空文本、U+FFFD 乱码或非单调时间戳；
- FunASR VAD + CAM++：12/12，共 14,865 个 1.5 秒说话人窗口；
- ERes2Net：12/12，共 68,712 个 1.0 秒 / 0.25 秒 hop 窗口；
- 老广播剧绝对 active 门仅通过 10 个窗口，证明该绝对阈值在动画域失效；窗口原始 center/p25/margin 全部保留，后续改用每集源内分位和人工锚点；
- 每集多 K CAM++ 聚类都成功选出 2 条 target-like 和 1 条 competitor；
- 最终身份校准包 36 条 / 121.720 秒，每集恰好 3 条，全部 32 kHz mono PCM16、1.5–5 秒、路径和波形哈希唯一。

可直接双击打开：

```text
anime_speaker_calibration_v1/index.html
```

该页只判断说话人身份，不判断训练边界：

- 全程只有伊蕾娜：选“伊蕾娜”；
- 全程不是伊蕾娜：选“非伊蕾娜”；
- 两人无缝接话、前后混入或重叠：选“混合/无缝接话”；
- 无法确认：选“不确定”。

本轮故意暂时忽略切头切尾和音质，因为这些样本只用于给每一集建立目标/非目标说话人锚点，不会直接成为训练音频。反馈文件前缀：

```text
anime_speaker_calibration_feedback_v1_*.json
```

收到完整反馈后先运行 `ingest_anime_speaker_calibration_v1.py`。每集必须至少有一个人工 target 和一个人工 pure_non_target；不足时生成定向第二轮，禁止拟合无负锚点模型。满足后才运行每集独立聚类、34,358 词重评分、完整语义边界、ERes2Net 共识、GPU 相对 NISQA/点击/削波/截断质量门及 48 条训练可用性盲听页。

### 19.4 首轮动画身份反馈与定向负锚点补标

首轮 36/36 人工反馈已经导入。人工结果为 27 条伊蕾娜、9 条纯非目标；12 集都有正锚点，但第 4、9、12 集的机器 competitor 也被人工确认为伊蕾娜，因此这三集缺少纯负锚点。这个现象说明动画中的旧广播剧绝对声纹距离和单一竞争簇都不能直接作为角色真值。

处理规则如下：

- 第 1、2、3、5、6、7、8、10、11 集已有逐集正负锚点；
- 第 4、9、12 集禁止在无负锚点情况下拟合或跑全量词级模型；
- 定向第二轮从每集尚未试听的 competitor pool 中各选 5 条；
- 排除所有已被人工判为伊蕾娜的 CAM++ cluster；
- 尽量覆盖不同竞争 cluster，并避免与首轮区间重叠；
- 本轮仍只判断身份，不把临时 ASR 边界误当训练边界。

第二轮共 15 条 / 34.420 秒，路径、音频哈希均唯一，可直接双击：

```text
anime_speaker_calibration_round2_v1/index.html
```

反馈前缀为 `anime_speaker_calibration_round2_feedback_v1_*.json`。导入命令：

```powershell
runtime/python.exe ingest_anime_speaker_calibration_round2_v1.py --feedback <反馈.json>
```

导入后必须检查 `human_anime_speaker_calibration_v2/report.json` 中 `calibration_complete=true`。若仍有缺失负锚点，只补缺失集，不允许靠机器标签填空。

同轮广播剧最终边界反馈 3/3 通过。B013 的 exact 与 +20 ms 属于同一句近重复，只冻结完整起音的 exact 版本；连同 B024 共新增 2 条 / 5.540 秒。新的严格人工广播剧清单为：

```text
human_confirmed_broadcast_v5_train_1p5_6s.list
```

其结果为 89 条 / 262.444250 秒；路径、波形哈希、格式、时长和文本检查全部通过。没有启动训练、没有覆盖旧清单、没有 90 分声明。

### 19.5 第 12 集必须用稳定 CAM++ 窗口寻找纯负锚点

动画身份第二轮 15/15 已导入。合并结果为 29 target、19 pure_non_target、3 mixed。第 4、9 集各得到 5 条纯负例；第 12 集的 5 条却为 2 target + 3 mixed，仍没有 pure_non_target。混合片段不能充当纯负锚点。

第 12 集因此改用源内 K=16 CAM++ 的 1.5 秒稳定窗口，而不再使用容易跨无缝接话边界的整句 competitor pool。已排除人工 target clusters `0,5,10,11`、人工 mixed clusters `1,13`，并从尚未试听的低目标稳定 clusters `9,14,8,7` 各取 3 条，共 12 条 / 18 秒。所有路径和波形哈希唯一，同簇局部稳定窗口数为 3–5。

直接打开：

```text
anime_speaker_calibration_round3_v1/index.html
```

反馈前缀为 `anime_speaker_calibration_round3_feedback_v1_*.json`。只有导入后 `human_anime_speaker_calibration_v3/report.json` 的 `calibration_complete=true`，才允许继续 12 集源内模型与 34,358 词全量重评分。

### 19.6 动画全量挖掘与人工验收完成

第 12 集第三轮反馈已导入，12 集人工身份校准完整：63 条身份样本中 29 target、30 pure_non_target、3 mixed、1 uncertain。源内聚类校准自检为 29/29 target 保留、30/30 pure negative 拒绝、3/3 mixed 拒绝。

全量结果：34,358/34,358 个 ASR 词完成 GPU 评分；4,131 个词通过严格目标状态；25 条机器严格完整句进入相对质量门；11 条 / 37.685031 秒通过质量门并由用户 11/11 验收无误。最后漏标的 S007 根据用户明确确认补记为通过。

动画严格人工清单为 `human_verified_anime_semantic_v1_direct.list`。与广播剧 v5 合并后的独立总候选清单为：

```text
human_confirmed_radio_anime_v1_train_1p5_6s.list
```

总计 100 条、300.129281 秒（约 5.002 分钟）。全部通过路径存在、路径唯一、波形 SHA256 唯一、32 kHz mono PCM16、1.5–6 秒和文本非空检查；旧清单未修改，训练未启动，没有 90 分声明。

### 19.7 动画语义复核第一批与边界救援

第一批 48 条动画语义候选已由用户完整试听。结果为：19 条直接确认目标声音，26 条主要是开头/结尾切断，1 条与 OP 重叠，1 条有杂音/音质问题，1 条不确定。直接通过清单为：

```text
human_verified_anime_review_batch_v1.list
```

针对 26 条边界问题，从原始 BS-Roformer `vocals` 重新按 ASR 词边界扩展，并另生成 220ms padding 版本；没有使用 ClearVoice 或 VASR 重建。52 个救援版本经用户试听后，15 个版本中去重冻结出 10 条完整目标片段，共 26.345094 秒：

```text
human_verified_anime_boundary_rescue_v1.list
```

当前动画人工确认累计为 40 条、115.505125 秒；与 89 条广播剧 v5 合并后，新版本为 129 条、377.949375 秒（约 6.30 分钟）：

```text
human_verified_anime_v2_train_1p5_6s.list
human_confirmed_radio_anime_v2_train_1p5_6s.list
human_confirmed_radio_anime_v2_report.json
```

旧 v1 合并清单没有覆盖，训练没有启动。

### 19.8 第二批试听规则

第一批已试听的 48 个原始候选已经从剩余池排除，第二批重新抽取 48 条，覆盖全部 12 集；由于第 10 集剩余候选只有 1 条，因此第二批采用“每集至少 1 条、总量 48 条”，不重复优先于每集固定数量。试听页为：

```text
anime_semantic_review_batch_v2/index.html
```

每条只需要做以下判断：目标声音通过；非目标；两人混合/无缝接话；开头或结尾切断；杂音/音质；重叠；不确定。开头/结尾切断但主体声线确实是伊蕾娜时，标记 `boundary_cut`，不要标成非目标；这类片段会进入下一轮原始 vocals 边界救援。只有直接确认目标或救援后确认完整的片段才进入冻结清单。

反馈 JSON 下载后使用：

```powershell
runtime//python.exe ingest_anime_semantic_review_batch_v2.py --feedback <反馈文件.json>
```

自动判定仍然不等于逐条人工听审；现阶段清单是人工确认候选，不宣称达到 90 分。

### 19.9 第二批反馈与边界救援

第二批 48 条反馈已导入：20 条直接确认目标，22 条目标声音但边界被切，5 条不确定，1 条非目标。20 条直接通过已并入新版本清单；当前人工确认累计为 149 条 / 442.164406 秒（约 7.37 分钟）：

```text
human_verified_anime_v3_train_1p5_6s.list
human_confirmed_radio_anime_v3_train_1p5_6s.list
human_confirmed_radio_anime_v3_report.json
```

22 条边界片段从原始 BS-Roformer `vocals` 生成了 44 条救援版本（每条 `word_expand` 和 `pad_220ms` 各一版），没有加载旧的 ASR/说话人重型模块，不使用 ClearVoice/VASR 重建：

```text
anime_boundary_rescue_batch_v2/index.html
```

这 44 条目前仍是试听池，不计入 v3 训练候选。用户确认后，使用：

```powershell
runtime//python.exe ingest_anime_boundary_rescue_batch_v2.py --feedback <反馈文件.json>
```
