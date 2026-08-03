# GPT-SoVITS イレイナ 音色克隆项目交接文档

> 交给下一位 AI 继续执行。请先完整阅读本文件，再进行任何训练或文件操作。  
> 更新时间：2026-07-31

## 1. 最终目标

用同一句日语测试音频，训练出最接近角色「イレイナ」的 GPT-SoVITS v2Pro 模型：

- 音色接近目标；
- 语气、停顿、叹气和情绪自然；
- 不要机械感、AI 感、电音或金属音；
- 重点关注叹气、气声、句尾和情绪变化。

用户最新试听结论：

1. `base_base__emo.wav` 的机械感最低，但语气不太对；
2. `baseGPT_finetuneS__emo.wav`（官方 GPT + 微调 SoVITS）比使用微调 GPT 的版本机械感明显少；
3. `finetuneGPT_finetuneS__emo.wav` 的叹气部分机械感非常重，并且有明显电音；
4. 因此当前最强证据是：**微调 GPT（S1）是机械感，尤其是叹气机械化/电音的主要来源；SoVITS 微调主要负责音色变化，但也可能引入伪影。**

## 2. 已完成的工作

### 2.1 数据

当前训练集共 264 段、约 31 分钟：

- `old`：154 段原目标音色素材；
- `radio`：110 段广播剧分离 vocal；
- 分离 vocal 已经过用户试听确认，用户认为“非常纯净优质”，不要再把分离伪影当作主要嫌疑；
- 但“干净”不等于“全部是同一个说话人”。目前仍没有证明 110 段 radio 全部属于同一目标角色。

本地数据：

```text
radio_drama_work/dataset_v1/
radio_drama_work/dataset_v1.zip
```

服务器数据：

```text
/root/autodl-tmp/dataset_v1/
/root/autodl-tmp/dataset_v1/dataset.list
```

数据清单格式：

```text
绝对音频路径|说话人|语言|文本
```

当前 speaker 标签大致为 `old` 和 `rd`，但训练时本质上被当作同一个目标说话人使用。

### 2.2 v2Pro 训练

训练平台和模型已经固定：

- 所有训练都在 AutoDL 服务器上执行；
- 只使用 GPT-SoVITS v2Pro；
- 不要切换回 V4；
- 本地只做轻量数据处理、打包和上传。

服务器仓库：

```text
/root/GPT-SoVITS
```

训练环境：

```text
/root/miniconda3/envs/gptsovits/bin/python
```

绝对不要使用 `GPTSoVits` conda 环境。该环境的 torch/cu121 不支持 RTX 5090 的 sm_120，实际计算会报：

```text
no kernel image is available for execution on the device
```

本轮已完成训练：

- S1 GPT：15 轮，实际使用 batch=6、max_sec=22、每 3 轮保存；
- S2 SoVITS：8 轮，实际使用 batch=8、每 2 轮保存；
- 训练期间曾经发生过一次 S1 OOM，原因是 batch=8 + max_sec=54；已经清理坏状态并用 batch=6 + max_sec=22 重训成功；
- 日志显示 `===== ALL TRAINING DONE =====`，本轮训练已完成。

训练产物：

```text
/root/autodl-tmp/exp/ireina_v1/
/root/autodl-tmp/weights/GPT_weights_v2Pro/
/root/autodl-tmp/weights/SoVITS_weights_v2Pro/
```

GPT 权重：

```text
ireina_v1-e3.ckpt
ireina_v1-e6.ckpt
ireina_v1-e9.ckpt
ireina_v1-e12.ckpt
ireina_v1-e15.ckpt
```

SoVITS 权重：

```text
ireina_v1_e2_s*.pth
ireina_v1_e4_s*.pth
ireina_v1_e6_s*.pth
ireina_v1_e8_s*.pth
```

训练配置：

```text
/root/autodl-tmp/exp/ireina_v1/cfg/s1.yaml
/root/autodl-tmp/exp/ireina_v1/cfg/s2.json
```

本地配置生成脚本：

```text
radio_drama_work/srv_make_cfg.py
```

### 2.3 推理扫描和消融实验

完整推理扫描已完成：

- 9 组 GPT×SoVITS 权重组合；
- 3 组采样参数；
- 2 句日语测试句；
- 共 54 个 WAV；
- 服务器输出：`/root/autodl-tmp/infer_v1/`；
- 本地代表样本：`radio_drama_work/infer_v1/`。

消融实验已完成，样本在：

```text
radio_drama_work/ablation_v1/
```

四路定义：

```text
base_base              = 官方 GPT + 官方 SoVITS
finetuneGPT_baseS      = 微调 GPT + 官方 SoVITS
baseGPT_finetuneS      = 官方 GPT + 微调 SoVITS
finetuneGPT_finetuneS  = 微调 GPT + 微调 SoVITS
```

每路都有 `normal` 和 `emo` 样本。

消融结论：

- 官方 v2Pro 底模本身不是当前机械感的主要来源；
- 加入微调 GPT 后，韵律、时长和叹气明显变僵；
- 加入微调 SoVITS 后，音色发生变化，也可能有一定质感损失；
- 两个微调权重同时使用时，机械感最明显；
- 当前不应继续盲目调 top-k、top-p、temperature，因为同一权重换采样参数后机械感仍然明显。

## 3. 说话人向量分析结果

服务器预处理生成了每条样本的 speaker embedding：

```text
/root/autodl-tmp/exp/ireina_v1/7-sv_cn/
```

已得到的统计：

- old 内部平均相似度：0.672；
- radio 内部平均相似度：0.693；
- old/radio 中心余弦相似度：0.940；
- old 与 radio 跨组平均相似度：0.641；
- 存在离群片段，跨组一致性不足。

解读：

- radio vocal 本身是干净的；
- 但 radio 可能混有其他角色或不同发声状态；
- 264 段全部混成一个单说话人训练集，可能让 GPT 学到互相矛盾的韵律、叹气方式和音色；
- 这比“分离音频有杂声”更值得优先处理。

已有相关脚本：

```text
radio_drama_work/cluster_sv.py
radio_drama_work/speaker_filter.py
radio_drama_work/speaker_scores.json
radio_drama_work/speaker_scores_fine.json
radio_drama_work/cluster_result.json
```

## 4. 服务器和磁盘硬性规则

SSH 连接信息已写入本地：

```text
radio_drama_work/remote.py
```

不要在交接文本、日志或新的脚本中重复输出密码。使用本地 `remote.py` 执行远程命令，例如：

```bash
C:/Users/lulu/.workbuddy/binaries/python/envs/default/Scripts/python.exe remote.py run "命令"
C:/Users/lulu/.workbuddy/binaries/python/envs/default/Scripts/python.exe remote.py put 本地文件 远程路径
C:/Users/lulu/.workbuddy/binaries/python/envs/default/Scripts/python.exe remote.py get 远程路径 本地路径
```

服务器规则：

- 训练、预处理、推理全部使用 `/root/miniconda3/envs/gptsovits/bin/python`；
- 运行需要外网时先执行 `source /etc/network_turbo`；
- 所有数据、日志、exp、权重放 `/root/autodl-tmp`；
- 系统盘只剩约 4.5G，不要把训练产物放 `/root` 或系统盘；
- 不要删除现有训练权重，除非先确认有备份和明确的删除清单；
- 训练前检查是否残留 Lightning ckpt，避免自动 resume 到坏状态；
- 服务器训练完成后，使用 `remote.py get` 下载音频回本地试听。

## 5. 下一步：必须按这个顺序执行

### 第一步：不要重复训练当前 264 段混合数据

不要直接再次运行原来的 `srv_train_bg.sh`。当前混合数据已经证明会让微调 GPT 的叹气变机械并产生电音。

保留当前权重作为对照，不覆盖：

```text
/root/autodl-tmp/weights/GPT_weights_v2Pro/
/root/autodl-tmp/weights/SoVITS_weights_v2Pro/
```

### 第二步：筛选 radio 数据

使用 `7-sv_cn` speaker embedding，对 110 段 radio 做聚类和离群点筛选：

1. 以 old 目标素材中心和 radio 主簇中心作为参考；
2. 列出 radio 样本与目标中心的相似度；
3. 先剔除明显离群的 radio 片段；
4. 最好人工试听边界样本，确认是否为同一角色；
5. 不要仅因为音频干净就保留全部 radio。

建议构造至少三个数据版本：

```text
D_old       = 仅 old 154 段
D_radio_ok  = 仅筛选后的 radio 主簇
D_curated   = old + 筛选后的 radio
```

优先从 `D_old` 和 `D_curated` 开始，不要直接使用全部 264 段。

### 第三步：先做 SoVITS-only 验证

因为用户已经明确听出官方 GPT 比微调 GPT 更自然，下一轮先固定官方 GPT，只训练/测试 SoVITS：

```text
官方 GPT + 官方 SoVITS       基线
官方 GPT + D_old 微调SoVITS
官方 GPT + D_curated 微调SoVITS
```

目的：验证在不改变 GPT 韵律的情况下，SoVITS 能否把音色拉近目标，同时不制造明显电音。

每个版本至少保存 e2/e4/e6/e8 中间权重，并用同一参考音频、同一 prompt_text、同一句测试文本试听。

### 第四步：再单独验证 GPT 微调

只有 SoVITS-only 结果可接受后，才训练 GPT：

1. 先用 `D_old` 训练 GPT；
2. 试听 e3/e6/e9/e12/e15；
3. 重点听叹气、气声、句尾、停顿；
4. 如果 D_old 的 GPT 仍有电音，降低 GPT 训练强度或减少轮数；
5. 不要默认 e15 最好，必须按试听选择；
6. 再与 `D_curated` 的 GPT 对比，确认 radio 数据是否导致问题。

训练建议：

- S1 先尝试 6~12 轮并保存中间轮次；
- 保持 `if_save_latest=False`；
- RTX 5090 上使用 batch=6、max_sec=22 作为安全起点；
- 不要再使用 40 轮；
- 不要因为 `top_3_acc` 很高就判断模型更好，高准确率可能意味着过拟合。

### 第五步：统一推理条件

每次对比必须固定：

- 同一参考音频；
- 与参考音频严格匹配的 `prompt_text`；
- 同一 `prompt_lang`；
- 同一句测试文本；
- 同一 speed factor；
- 先使用默认采样参数，再单独调采样参数。

推荐先做四路最小矩阵：

```text
官方GPT + 官方SoVITS
官方GPT + 新SoVITS(old)
官方GPT + 新SoVITS(curated)
新GPT(old/curated) + 新SoVITS
```

不要一次同时改变数据、GPT、SoVITS 和采样参数，否则无法判断问题来自哪里。

## 6. 试听验收标准

每个候选模型使用同一组测试句，至少检查：

- 普通叙述是否自然；
- 情绪句是否有正确语气；
- 叹气是否像真实呼吸，而不是周期性噪声；
- 是否出现电音、金属音、颤音或机器人式音高跳变；
- 句尾是否僵硬；
- 音色是否稳定；
- 是否只是“音色像”，但语气完全不对。

用户当前偏好不是单纯追求音色相似度，而是优先消除：

```text
机械感 > AI感 > 叹气电音 > 语气不自然
```

`base_base__emo` 可作为“最低机械感基线”，但不能作为最终模型，因为用户认为语气不对。

## 7. 不要重复踩的坑

- 不要把 40 轮训练当作更充分；此前已严重过拟合；
- 不要把 `top_3_acc=0.994` 当作好听的证据；
- 不要把全部 radio 素材直接和 old 混合成单说话人；
- 不要只调采样参数来解决训练阶段产生的电音；
- 不要使用 `GPTSoVits` 环境；
- 不要把文件写到系统盘；
- 不要让 Lightning 自动 resume 旧的中断 ckpt；
- 不要覆盖当前 v1 权重；
- 不要改变参考音频却把结果归因于训练配置；
- 不要在没有下载音频并让用户试听的情况下宣布“已经解决”。

## 8. 关键本地文件索引

```text
radio_drama_work/remote.py             # AutoDL SSH/SFTP
radio_drama_work/pack_dataset.py      # 数据打包
radio_drama_work/srv_prepare.sh       # 服务器预处理
radio_drama_work/srv_merge_parts.py   # 合并预处理分片
radio_drama_work/srv_make_cfg.py      # 生成 v2Pro 配置
radio_drama_work/srv_train_bg.sh      # 后台训练脚本
radio_drama_work/srv_infer.py         # 权重/采样扫描
radio_drama_work/srv_ablation.py      # 四路消融
radio_drama_work/cluster_sv.py        # speaker embedding 分析
radio_drama_work/infer_v1/           # v1 推理样本
radio_drama_work/ablation_v1/        # 四路消融样本
```

## 9. 给下一位 AI 的第一条执行指令

先不要训练。先完成以下只读检查并报告：

1. 读取本交接文档；
2. 检查当前本地 `ablation_v1` 文件是否齐全；
3. 通过 `remote.py` 检查服务器 v1 权重、`7-sv_cn` 和磁盘空间；
4. 读取/整理 110 段 radio 的 speaker embedding 排名；
5. 设计 `D_old`、`D_radio_ok`、`D_curated` 的具体清单；
6. 在得到清单前不要启动任何新训练。

当前最合理的实验方向一句话概括：

> **保留官方 GPT 的自然韵律，先用筛选后的单说话人数据重新训练 SoVITS；确认音色可接受后，再用更干净、更一致的数据低强度微调 GPT。**

