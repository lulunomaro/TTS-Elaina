# 伊蕾娜真人感精修结果

更新时间：2026-07-31

## 当前推荐

固定 GPT：

```text
/root/autodl-tmp/weights/GPT_ireina_s2_old_v2/ireina_s2_old_v2-e6.ckpt
```

当前优先试听/使用的 SoVITS：

```text
/root/autodl-tmp/weights/SoVITS_ireina_s2_balanced48_v3/ireina_s2_balanced48_v3_e2_s62.pth
```

在固定参考音、参考文本、测试句和采样参数下，它仍然是当前真人感排序第一名。

## 本轮新实验

| 实验 | 数据 | 关键设置 | 结果 |
|---|---:|---|---|
| `human_focus14_lr5e5_v1` | 168 段 | D_old + 14 段完整广播剧；S2 lr=5e-5；text lr=0.2 | e2/e4 均未超过 balanced48 e2 |
| `balanced48_lr5e5_v1` | 202 段 | balanced48；S2 lr=5e-5；text lr=0.2 | e2/e4 均未超过 balanced48 标准 e2 |

新权重仍保留在服务器：

```text
/root/autodl-tmp/weights/SoVITS_ireina_s2_human_focus14_lr5e5_v1/
/root/autodl-tmp/weights/SoVITS_ireina_s2_balanced48_lr5e5_v1/
```

## 评分器

脚本：`human_likeness_eval_v2.py`

它结合频谱/F0/RMS 伪影特征和“按候选模型留出”的真实/合成 Logistic 分类器，输出：

- `classifier_realness`
- `human_like_score_v2`
- 每句、每组模型明细

这是用于排序和缩小试听范围的代理分数，不是绝对 MOS；最终仍以固定句人工试听为准。

汇总：`human_likeness_groups_v2.tsv`

