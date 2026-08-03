# 交接给 Luna：判别器辅助挖掘（v1）

更新时间：2026-08-02 22:50
作者：监控/验证方（不干扰现有挖掘流程）
目录：`radio_drama_work/discriminator_probe/`

## 0. 一句话交接

> 我在独立目录训练了一个**说话人判别器**（不是替换你的规则系统，是给 review 池打分排序，**把你需要用户逐条听的 431+448 条候选自动分成 高置信/待核/低分 三档**）。四轮验证全部通过，可以直接用来减少用户听音负担、提升挖掘产量。**你现有的管线（词状态/语义切分/质量门/边界救援）全部保留不动。**

## 1. 判别器是什么

- 模型：Logistic Regression，输入 **CAM++ 192 维词级嵌入**（完全复用你已生成的 embedding 缓存，零新计算）
- 训练集：1554 词 = 752 正（人工确认 target 区间内词）+ 802 负（纯非目标区间内词）
  - 正/负来自 `human_calibration_combined_v3` + `human_anime_speaker_calibration_v3`
- 输出：每段音频的 `prob`（0~1，伊蕾娜概率，段内词概率平均）

## 2. 验证结果（四轮，全部通过）

| 实验 | 结果 | 结论 |
|---|---|---|
| LOO 可分性（1554 词，逐标签留出） | **AUC = 0.970** | 远超 0.90 门槛 |
| 操作点（exp1） | prob≥0.7 → 精确率 95.6%/召回 84.3%；prob≥0.9 → 精确率 99.0% | 高置信档很干净 |
| 负例排除（exp4b，人工确认段） | confirmed non-target=0.091，mixed=0.067/0.078，全 <0.1 | ✅ 不混入 |
| 人工通过段召回（exp4，60 条冻结动画段） | mean 0.861，**91.7% ≥0.7** | ✅ 不漏杀 |
| 高分段质量（exp4，p≥0.9 的 199 条） | 75.4% 直接过声学门（点击≤3/无削波），SNR 均值 48.8dB | ✅ 与你的质量门同量级 |

## 3. 打分清单（已生成，直接用）

```
radio_drama_work/discriminator_probe/anime_review_431_scored.list   （431 条全量排序）
radio_drama_work/discriminator_probe/radio_review_448_scored.list   （448 条全量排序）
格式：<audio路径>|prob=<0.xxxx>|<source>|<start>-<end>|<duration>|<text>
按 prob 降序排列
```

分档统计：

| 档位 | 动画 431 池 | 广播剧 448 池 | 建议 |
|---|---|---|---|
| prob ≥ 0.9 | 199 条 | 102 条 | **自动候选**（仍走你的质量门） |
| 0.7-0.9 | 206 条 | 191 条 | **优先听音核验**（高概率通过） |
| 0.5-0.7 | 23 条 | 86 条 | 抽验 |
| 0.3-0.5 | 2 条 | 45 条 | 低优先级 |
| < 0.3 | 1 条 | 24 条 | **自动排除** |

## 4. 建议 Luna 的用法（不破坏你的规则）

1. **下一轮 review 批量包**：不再从 431 池随机/均衡抽 48 条，而是**优先抽 prob≥0.9 的段**（预计通过率远高于 19/48）→ 用户听音量骤降
2. **prob<0.3 的段**（动画 1 + 广播剧 24）直接不进听音包，可先标记低分候选
3. **你的质量门/语义门照跑**：判别器只管说话人，音质仍由 NISQA/点击/边界能量把关——两者互补
4. **边界救援逻辑不变**：boundary_cut 段判别器给分高（0.885），不影响你的救援流程
5. **训练集可迭代**：每轮新的人工确认段都可以追加进训练集重训（脚本已支持，30 秒重训），判别器会越来越准

## 5. 脚本清单（都在 discriminator_probe/）

```
exp1_feasibility.py           # LOO AUC（证据）
exp2_rank_review.py           # 打分排序（旧版，top200）
exp3_human_crosscheck.py      # 与人工判定对比
exp4_purity_quality.py        # 质量验证
exp4b_negatives.py            # 负例排除验证
rank_all_review_pools.py      # ★ 全量打分清单生成（431+448）
```

复跑全量打分（如果需要更新）：
```powershell
& radio_drama_work/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50/runtime/python.exe radio_drama_work/discriminator_probe/rank_all_review_pools.py
```

## 6. 铁律（与你的红线一致）

- 判别器输出**只用于排序和分档**，不自动写入任何训练清单
- 不进训练清单的候选仍需你现有的验收流程
- 不修改 `radio_drama_work` 下任何现有脚本/清单/掩码
- 不启动训练，不连服务器

## 7. 当前用户待办（不受本交接影响）

用户正在听：`anime_boundary_rescue_batch_v2/index.html`（44 条，22 个边界 origin）
导入：`ingest_anime_boundary_rescue_batch_v2.py --feedback <反馈.json>`
完成后合并到 `human_confirmed_radio_anime_v3`（当前 149 条 / 442.2s）。
