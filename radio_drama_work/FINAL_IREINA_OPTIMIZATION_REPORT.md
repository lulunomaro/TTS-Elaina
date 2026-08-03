# 伊蕾娜本轮精修结果

## 当前结论

固定 22 条同文本评测的最高“单一模型”结果仍是冠军 SoVITS 权重配合 expressive 推理参数，评分 75.6；当前冠军默认参数为 74.9。

把所有已生成模型和采样设置按句型选择最优试听，得到 77.4，但这是“逐句路由上限”，不是一个单独权重的绝对分数，不能当作单模型 90 分宣传。

## 保留的单一模型

```text
/root/autodl-tmp/weights/SoVITS_ireina_s2_b16e2_textonly_lr2e5_v1/ireina_s2_b16e2_textonly_lr2e5_v1_e1_s17.pth
```

推荐推理参数：

```text
top_k=30
top_p=0.95
temperature=0.95
repetition_penalty=1.15
seed=1234
```

## 本轮数据与训练

- 三段新广播剧经 BS-Roformer 分离后得到 455 个切片。
- 经过说话人相似度、日语 ASR、句尾残留检测，只加入 15 条高置信度片段。
- 新素材文本精修：74.55；低学习率声学精修：73.83。
- 纯 D_old 文本精修：75.5，仍未超过冠军 expressive 75.6，因此不替换冠军。
- 本轮新实验的 G/D 中间 checkpoint 已清理，保留导出权重和评测结果。

## 评测文件（服务器）

```text
/root/autodl-tmp/naturalness_expressive_full_v1/naturalness_groups_v1.tsv
/root/autodl-tmp/naturalness_global_best_v1/naturalness_groups_v1.tsv
/root/autodl-tmp/infer_global_best_v1/
```

评分是用于模型迭代的代理指标，不是人工 MOS；“90 分”目前尚未达到。下一轮真正冲击 90 需要继续改善 GPT/韵律建模和更大规模、严格分离的人声数据，而不是继续堆 SoVITS epoch。
