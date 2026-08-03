# TTS-Elaina

GPT-SoVITS v2Pro 日语语音克隆：角色「イレイナ」（Elaina）音色训练素材挖掘与建模。

## 项目目标

从多说话人广播剧 + 动画中，用纯机器管线挖掘出高纯度、可直接训练的单说话人（伊蕾娜）语音素材，训练 GPT-SoVITS v2Pro 模型。

## 核心管线

```
原始视频/音频
  -> BS-Roformer 人声分离（去 BGM，保留原始波形）
  -> FunASR VAD + CAM++ 说话人窗口
  -> 同剧内球形聚类 + 人工锚点（source-local modeling）
  -> 逐词 1.2s CAM++ 打分 + 词状态（strong/soft/edge/hard-negative）
  -> 语义合并切段（soft-fail 迟滞 + 语义完整性硬门）
  -> 质量门（NISQA 源内相对 + 点击/削波/SNR）
  -> 人工盲听验收闭环 + 边界救援
  -> 判别器辅助分档（Logistic Regression on CAM++ embeddings）
```

## 目录结构

```
radio_drama_work/        主挖掘管线：脚本、清单、报告、交接文档
discriminator_probe/     说话人判别器实验：训练/打分/验证（AUC 0.97）
```

## 安全声明

- **音频素材、模型权重、嵌入缓存一律不入库**（版权内容 + 可再生产物）
- **凭据文件（askpass.sh / remote.py / models.json 等）一律不入库**
- 本仓库只包含：代码、清单（文本）、处理报告、方法论文档

## 关键文档

- `radio_drama_work/HANDOFF_IREINA_NEXT_AI.md` — 完整交接文档（数据/训练/评分/坑）
- `radio_drama_work/FULL_RADIO_MINING_EXPERIENCE_V1.md` — 挖掘经验与管线说明
- `radio_drama_work/NATURALNESS_SCORING_STANDARD.md` — 自然度评分标准
- `discriminator_probe/HANDOFF_TO_LUNA_DISCRIMINATOR.md` — 判别器交接说明
