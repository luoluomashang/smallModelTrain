# Qwen3 QLoRA 整章正文执行器设计

> 版本：v2.0
> 日期：2026-06-17
> 目标模型：Qwen/Qwen3-4B-Instruct-2507
> 训练路线：QLoRA SFT + 评分闭环 + 偏好优化候选
> 目标输出：整章 2000-2500 中文汉字正文

## 1. 背景与目标

本项目训练的不是独立小说创作模型，而是“正文执行器”。外部大模型负责编剧、章节规划、人物动机、冲突推进、伏笔和前后文一致性；本地小模型只负责根据章节执行卡生成符合作者个人文风的整章正文。

最终目标是让模型在 16GB 显存、32GB 内存、Codex 同机运行的本地环境中，稳定完成以下任务：

```text
风格契约 + 章节执行卡
-> 2000-2500 中文汉字正文
-> 低 AI 味
-> 低擅自加戏
-> 可人工轻修
```

第一阶段的成功标准不是一次训练出生产模型，而是跑通可复现闭环：数据清洗、章节卡构造、QLoRA SFT、固定评测、失败归因、偏好数据准备。

## 2. 关键决策

### 2.1 基座模型

采用：

```text
Qwen/Qwen3-4B-Instruct-2507
```

理由：

- Qwen3-4B-Instruct-2507 是非 thinking 指令模型，适合直接生成正文，不需要处理 `<think>` 输出。
- 官方模型卡标注其原生上下文为 262K，远高于本项目训练需要。
- 与 Qwen2.5-3B 相比，4B 容量对中文长文、指令跟随、风格执行更有余量。
- 4B 仍处于 16GB 显存可通过 QLoRA 微调的范围内。

本项目不会利用 262K 超长上下文做训练。第一版训练长度控制在 8192 tokens，必要时降到 6144 tokens。

### 2.2 训练方式

采用 QLoRA 作为主线，而不是 bf16 LoRA。

理由：

- 整章任务需要同时容纳章节卡、风格契约和 2000-2500 汉字输出，序列长度比普通指令微调更吃显存。
- 16GB 显存还要给 Codex、系统、可能的浏览器和监控进程留余量。
- QLoRA 冻结 4-bit 量化基座，只训练 LoRA adapter，更适合保住 8192 cutoff_len。
- 对本项目而言，稳定跑完整闭环比理论上的微小质量上限更重要。

### 2.3 训练路线

推荐主路线：

```text
V0: baseline 评测
V1: QLoRA SFT v1
V2: QLoRA SFT v2
V3: 偏好优化实验
V4: 固定评测与坏例回流
```

第一版不做：

- 全参微调。
- 大规模 CPT。
- 从零训练。
- 直接 GRPO/RLVR 训练正文质量。
- 多 adapter 题材拆分。

## 3. 最新训练方法取舍

### 3.1 SFT 仍是主干

本项目的第一问题是“章节卡到正文”的格式执行和风格拟合，因此 SFT 是必要主干。没有足够好的 SFT，后续偏好优化会变成修补随机错误，收益不稳定。

SFT 数据应以高质量 A 类作者正文为主，并确保章节卡不会泄漏原文句子。

### 3.2 QLoRA 是资源约束下的默认微调方式

QLoRA 适合本项目的原因不是“文风更强”，而是能让训练保留更长上下文。若在 16GB 显存上使用 bf16 LoRA，最容易被迫降低 cutoff_len，这会伤害整章结尾、章末钩子和后半章节奏。

### 3.3 偏好优化方法候选

偏好阶段不锁死 DPO，采用方法矩阵选择。

| 方法 | 输入数据 | 资源压力 | 本项目定位 |
|---|---|---:|---|
| DPO | 成对 chosen/rejected | 高 | 高质量偏好对足够后使用 |
| ORPO | SFT 与偏好目标合并 | 中 | 小显存第一优先候选 |
| KTO | 好/坏单样本标签 | 中 | 当成对数据不足时使用 |
| SimPO | 成对偏好，无 reference model | 中 | 已被 LLaMA-Factory 支持，可作为 DPO 的轻量替代实验 |
| GRPO/RLVR | 可验证奖励 | 高 | 暂不用于小说质量，只能用于字数、禁词等规则实验 |

推荐顺序：

```text
QLoRA SFT -> ORPO/KTO 小规模偏好 -> DPO/SimPO 对照 -> 规则型 RL 实验暂缓
```

选择原因：

- DPO 对 chosen/rejected 质量要求高，且偏好训练通常比 SFT 更吃显存。
- ORPO/KTO 更适合先做小显存、低风险偏好压制。
- SimPO 由于不需要 reference model，理论上更适合资源受限环境；LLaMA-Factory 已支持该方法，适合在 DPO 前后做小样本对照。
- GRPO/RLVR 更适合数学、代码、工具调用或明确 verifier 的任务；小说正文质量不应直接交给规则奖励。

## 4. 数据设计

### 4.1 数据分层

原始文本分为：

```text
A 类：当前最认可、最接近目标风格的正文。
B 类：可用但风格不稳定、较旧或题材不完全匹配。
C 类：废稿、设定、笔记、实验文本、非正文。
```

第一版只用 A 类进入 SFT。B 类可进入风格分析或后续扩展。C 类不参与训练。

### 4.2 数据规模

建议阶段规模：

| 阶段 | 数据量 | 目标 |
|---|---:|---|
| baseline eval | 50 张固定章节卡 | 建立原始模型基线 |
| SFT v1 | 500-1000 条整章样本 | 验证方向和资源配置 |
| SFT v2 | 3000-5000 条整章样本 | 提升稳定性 |
| 偏好 v1 | 300-1000 条偏好样本 | 压制明显坏例 |
| 固定 eval | 50-200 张章节卡 | 永不进训练 |

### 4.3 章节卡防泄漏

章节卡从原文反推，但不能贴近原文句子。每张卡需要做泄漏检查：

- 章节卡不得包含原文连续 12 个以上中文字符。
- 章节卡不得保留原文标志性对白。
- 章节卡不得把动作写到句子级细节。
- `source_text` 只能作为离线溯源字段，不进入训练 prompt。

章节卡应保留结构，而不是复述正文。

### 4.4 SFT 样本格式

SFT 输入包含：

```text
【角色】
你是作者的正文执行器，只负责根据章节执行卡写正文。

【风格契约】
作者风格规则、对白规则、禁止表达。

【前情摘要】
上一章关键状态。

【本章目标】
本章剧情目标。

【章节结构】
5-8 个结构步骤，每步有目标和字数预算。

【人物状态】
核心人物当前动机、情绪、说话方式。

【必须出现】
必须出现的信息。

【禁止事项】
不得出现的信息、不得提前揭露的伏笔。

【目标字数】
2000-2500 中文汉字。

【输出要求】
只输出正文，不输出提纲、小标题、解释或分析。
```

SFT 输出只包含作者原正文或人工修订后的目标正文。

## 5. 训练配置

### 5.1 QLoRA SFT 默认配置

```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
template: qwen3
stage: sft
do_train: true
finetuning_type: lora
quantization_bit: 4
quantization_method: bitsandbytes

lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: all

cutoff_len: 8192
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 3.0e-5
num_train_epochs: 2

bf16: true
gradient_checkpointing: true
logging_steps: 10
save_steps: 200
save_total_limit: 3
```

### 5.2 OOM 降级顺序

如果训练 OOM，按以下顺序处理：

```text
1. cutoff_len: 8192 -> 6144
2. lora_rank: 16 -> 8
3. gradient_accumulation_steps: 16 -> 32
4. max_samples 降低，先跑通 100-200 条冒烟训练
```

不优先把 `cutoff_len` 降到 4096。整章任务如果频繁截断后半章，模型会学不到结尾、反转和钩子。

### 5.3 推理配置

固定评测推理从以下参数开始：

```yaml
max_new_tokens: 5120
temperature: 0.7
top_p: 0.8
top_k: 20
repetition_penalty: 1.05
```

调参规则：

- 如果输出少于 2000 汉字，先提高 `max_new_tokens` 到 6144。
- 如果正文灌水，先把 `temperature` 降到 0.65。
- 如果重复严重，把 `repetition_penalty` 提到 1.08。
- 固定评测必须记录每次推理参数，不能混用。

## 6. 评分与验收

### 6.1 硬门槛

以下任一项失败，则该样本判定为未通过硬门槛：

```text
中文汉字数不在 2000-2500。
输出包含提纲、小标题、解释、分析或提示语。
出现明显复读。
违反 must_not_include。
提前泄露伏笔。
大段照抄训练原文。
```

### 6.2 100 分制

| 维度 | 权重 | 评分内容 |
|---|---:|---|
| 章节卡执行 | 25 | must_include 覆盖、结构顺序、禁止项、无擅自加戏 |
| 整章完成度 | 15 | 字数、起承转合、后半章不塌、章末钩子 |
| 作者文风相似度 | 20 | 句长、段落、对白、动作承接、低 AI 味 |
| 小说可读性 | 20 | 冲突推进、人物一致、场景连贯、节奏自然 |
| 人工轻修可用性 | 20 | 修改量、硬伤数量、可发表潜力 |

### 6.3 阶段验收线

| 阶段 | 均分 | 硬门槛通过率 | 轻修可用率 |
|---|---:|---:|---:|
| baseline | 记录即可 | 记录即可 | 记录即可 |
| SFT v1 | >= 72 | >= 65% | >= 30% |
| SFT v2 | >= 78 | >= 75% | >= 40% |
| 偏好 v1 | >= 82 | >= 85% | >= 50% |
| 生产候选 | >= 86 | >= 90% | >= 60% |

### 6.4 失败类型枚举

评测报告必须按失败类型归因：

```text
length_short
length_long
outline_leak
must_include_missing
forbidden_violation
plot_drift
early_wrap
front_heavy
wordy_filler
ai_trace
over_explaining
dialogue_stiff
style_mismatch
repetition
hook_missing
source_memorization
```

这些失败类型直接用于构造偏好数据。

## 7. Codex 智能体设计

Codex 作为总控，不在第一版引入复杂多进程智能体框架。每个智能体是一个可复跑的工作单元，有明确输入、输出和报告。

| 智能体 | 输入 | 输出 | 职责 |
|---|---|---|---|
| DataCuratorAgent | `data_raw/` | `data_clean/chapters.jsonl` | 清洗、切章、A/B/C 分层、去重、统计中文汉字 |
| CardBuilderAgent | `chapters.jsonl` | `data_cards/chapter_cards.jsonl` | 反推章节卡，检查泄漏，生成固定 eval cards |
| StyleProfilerAgent | A 类正文 | `style_contract.md`, `style_profile.json` | 统计句长、段落、对白比例、禁用表达 |
| DatasetBuilderAgent | 章节卡、风格契约、正文 | `data_sft/`, `data_pref/` | 构造 SFT 和偏好数据 |
| TrainRunnerAgent | 训练数据、配置 | `outputs/sft_v*` | 启动 QLoRA、监控 OOM、记录 loss 和资源 |
| InferenceAgent | eval cards、adapter | `outputs/eval/*` | 固定参数批量生成整章 |
| RuleJudgeAgent | 生成正文、章节卡 | `metrics.jsonl` | 字数、重复、禁词、must_include、must_not_include |
| LLMJudgeAgent | 生成正文、rubric | `judge_scores.jsonl` | 按 100 分制盲评，输出证据 |
| PreferenceMinerAgent | 失败样本、人工轻修 | `data_pref/*` | 构造 ORPO/KTO/DPO 候选数据 |
| AcceptanceAgent | metrics、judge、人工标注 | `reports/*.md` | 汇总验收结论，决定是否进入下一阶段 |

### 7.1 调度原则

训练、推理、LLM 评分不要并行抢显存。

```text
数据处理阶段：CPU 为主，可以由 Codex 自动化执行。
训练阶段：GPU 独占，暂停本地推理和浏览器自动化。
推理阶段：GPU 独占，生成完释放模型。
评分阶段：规则评分走 CPU；LLM 评分优先走外部 API 或训练结束后本地单独运行。
```

### 7.2 报告要求

每个阶段输出一份 Markdown 报告：

```text
reports/baseline_report.md
reports/sft_v1_report.md
reports/sft_v2_report.md
reports/pref_v1_report.md
```

报告必须包含：

- 训练配置快照。
- 数据集版本和样本数量。
- 推理参数。
- 硬门槛通过率。
- 100 分制均分和分项分。
- 失败类型分布。
- 最差 10 条样本索引。
- 最值得构造偏好数据的 20 条失败样本。
- 是否进入下一阶段的结论。

## 8. 偏好优化设计

### 8.1 数据来源

偏好数据来自：

```text
baseline 失败输出
SFT v1 失败输出
SFT v2 失败输出
人工轻修版本
规则生成的 hard negative
外部大模型生成的 hard negative
```

### 8.2 hard negative 设计

rejected 不应只是明显低质文本，而要尽量与 chosen 同信息、同长度、同章节卡，只在某个质量维度上失败。

优先构造：

```text
同剧情但 AI 味重。
同字数但后半章灌水。
同信息但漏掉章末钩子。
同结构但对白说明化。
同风格但违反禁止事项。
同章节卡但擅自新增设定。
同开头但提前收尾。
```

### 8.3 方法选择

第一轮偏好优化优先尝试 ORPO 或 KTO。只有当成对 chosen/rejected 质量足够高，且显存测试通过，再进行 DPO 或 SimPO 对照。

偏好优化验收不是 loss，而是固定 eval 的轻修可用率和硬门槛通过率。

## 9. 项目目录

```text
configs/
  sft_qlora_qwen3_4b.yaml
  orpo_qlora_qwen3_4b.yaml
  dpo_qlora_qwen3_4b.yaml
  infer_eval_qwen3_4b.yaml

data_raw/
  novels/
  drafts/
  notes/

data_clean/
  chapters.jsonl

data_cards/
  chapter_cards.jsonl
  eval_cards_50.jsonl
  eval_cards_200.jsonl

data_sft/
  sft_chapter_v1.jsonl
  sft_chapter_v2.jsonl

data_pref/
  kto_v1.jsonl
  orpo_v1.jsonl
  dpo_v1.jsonl
  preference_candidates.jsonl

scripts/
  ingest_raw_text.py
  clean_chapters.py
  split_train_eval.py
  build_style_contract.py
  build_sft_dataset.py
  build_preference_dataset.py
  run_inference.py
  evaluate_outputs.py
  detect_ai_trace.py
  score_outputs.py

outputs/
  baseline/
  sft_v1/
  sft_v2/
  pref_v1/

reports/
  baseline_report.md
  sft_v1_report.md
  sft_v2_report.md
  pref_v1_report.md
```

## 10. 第一阶段交付范围

第一阶段只实现训练前与评测闭环，不要求开始正式训练。

交付内容：

```text
1. 项目目录结构。
2. 数据清洗与章节切分脚本。
3. 中文汉字统计与章节过滤。
4. train/eval split。
5. 风格契约模板和风格统计报告。
6. SFT 数据构造脚本。
7. 偏好数据候选构造脚本。
8. 规则评分脚本。
9. 固定 eval 输出评测报告。
10. QLoRA SFT 配置文件。
11. 推理配置文件。
12. README 使用说明。
```

第一阶段完成标准：

```text
输入一批 txt/md 正文
-> 输出 chapters.jsonl
-> 人工或外部模型补 chapter_cards.jsonl
-> 输出 sft_chapter_v1.jsonl
-> 对任意 outputs/eval 目录生成评分报告
-> 具备启动 QLoRA SFT 的配置
```

## 11. 风险与控制

### 11.1 OOM 风险

控制方式：

- 默认 QLoRA 4-bit。
- 默认 `per_device_train_batch_size: 1`。
- 默认开启 gradient checkpointing。
- 先 100 条样本冒烟训练。
- 训练、推理、评分分阶段运行。

### 11.2 章节卡分布偏差

训练章节卡来自原文反推，生产章节卡来自外部大模型规划，两者可能不一致。

控制方式：

- 固定 eval cards 中加入生产式章节卡。
- 章节卡 schema 固定。
- 外部大模型生成章节卡时使用同一模板。

### 11.3 风格过拟合

控制方式：

- 不做重度 CPT。
- 检测训练原文相似度。
- eval 按作品或章节隔离。
- rejected 中加入 source_memorization 类型。

### 11.4 评分漂移

控制方式：

- 规则评分固定。
- LLM judge 使用固定 rubric。
- LLM judge 采用盲评，不暴露模型版本。
- 最终仍以人工轻修可用率作为核心指标。

## 12. 参考资料

- Qwen3-4B-Instruct-2507 model card: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- QLoRA: https://arxiv.org/abs/2305.14314
- LoRA: https://arxiv.org/abs/2106.09685
- DPO: https://arxiv.org/abs/2305.18290
- ORPO: https://arxiv.org/abs/2403.07691
- KTO: https://arxiv.org/abs/2402.01306
- SimPO: https://arxiv.org/abs/2405.14734
- LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory
