# 第三阶段真实数据准备指南

## 第三阶段目标

第三阶段的目标，是把项目从“代码测试通过”推进到“真实训练数据可用”。本阶段只准备和校验数据，不启动真实 GPU 训练；真实 smoke 训练属于第四阶段。

进入第四阶段前，需要得到明确的 readiness 决策。只有 `ready_for_stage4_smoke_training` 允许进入第四阶段真实训练。

## 1. 准备原始文本

把第一批原始文本放入 `data_raw/novels/`。当前导入脚本支持 `.txt` 和 `.md` 文件。

第一批数据应优先选择 A 类目标风格正文：已经定稿、符合目标叙事节奏和语言风格、可作为训练目标的章节文本。不要混入草稿、创作笔记、人物设定、世界观备忘、片段灵感或未整理材料。

## 2. 生成清洗后章节

```powershell
python scripts/ingest_raw_text.py --input-dir data_raw/novels --output data_clean/chapters_raw.jsonl
python scripts/clean_chapters.py --input data_clean/chapters_raw.jsonl --output data_clean/chapters.jsonl --min-chars 500 --max-chars 5000
```

`chapters_raw.jsonl` 用于保留导入后的原始章节记录，`chapters.jsonl` 是经过长度和文本清洗后的章节集合。

## 3. 固定训练集和评测集

先用 20 条评测卡进行小批量准备：

```powershell
python scripts/split_train_eval.py --input data_clean/chapters.jsonl --output data_clean/chapters_split.jsonl --eval-output data_cards/eval_cards_20.jsonl --eval-count 20
```

数据量足够时，再固定 50 条评测卡：

```powershell
python scripts/split_train_eval.py --input data_clean/chapters.jsonl --output data_clean/chapters_split.jsonl --eval-output data_cards/eval_cards_50.jsonl --eval-count 50
```

评测集一旦选定，应保持稳定，不要把评测样本混入 SFT 训练样本。

## 4. 生成风格契约

```powershell
python scripts/build_style_contract.py --chapters data_clean/chapters_split.jsonl --contract-output style_contract.md --profile-output style_profile.json
```

`style_contract.md` 用于章节卡和训练提示中的风格约束；`style_profile.json` 用于复查篇幅、对白比例、段落长度等统计分布。

## 5. 准备章节卡

人工或使用受控辅助流程准备 `data_cards/chapter_cards.jsonl`。每行是一张章节卡，字段应描述写作目标、结构、人物状态和约束，不要把目标正文直接写进提示字段。

示例：

```json
{"id":"train_001","style_contract":"只输出正文；保持短句推进，动作和感官细节清楚。","previous_summary":"上一章林默发现交易地点临时改到旧仓库。","chapter_goal":"林默进入旧仓库，与对方完成试探式谈判，并发现箱子异常。","chapter_structure":[{"step":1,"name":"入场","goal":"交代地点、天气和警惕感","estimated_chars":"300-500"},{"step":2,"name":"谈判","goal":"通过短对白推进利益冲突","estimated_chars":"900-1200"},{"step":3,"name":"异常","goal":"让箱子响动形成收束悬念","estimated_chars":"300-500"}],"character_states":[{"name":"林默","state":"冷静但警惕，先观察再行动","speech_style":"短句，少解释"},{"name":"周衡","state":"试图压价，又不愿失去交易","speech_style":"客气里带威胁"}],"must_include":["旧仓库","加钱","箱子响动"],"must_not_include":["真相大白","旁白解释全部背景"],"ending_hook":"箱子在无人触碰时响了一下。","target_word_count":"2000-2500中文汉字","source_text":"离线溯源文本，可记录对应原文或来源说明；构建训练提示时不应进入 prompt。"}
```

## 6. 构建 SFT 数据集

```powershell
python scripts/build_sft_dataset.py --cards data_cards/chapter_cards.jsonl --chapters data_clean/chapters_split.jsonl --output data_sft/sft_chapter_v1.jsonl
```

构建结果应来自章节卡和 `chapters_split.jsonl` 的配对；章节正文是训练目标，章节卡是模型输入。

## 7. 生成 readiness 报告

对 20 条评测卡生成报告：

```powershell
python scripts/check_stage3_data_readiness.py --eval-cards data_cards/eval_cards_20.jsonl --run-smoke-dry-run
```

对 50 条评测卡生成报告：

```powershell
python scripts/check_stage3_data_readiness.py --eval-cards data_cards/eval_cards_50.jsonl --run-smoke-dry-run
```

`--run-smoke-dry-run` 只验证第四阶段 smoke 启动命令是否能被正确构造，不会启动真实 GPU 训练。

## 8. 阶段出口

阅读 `reports/stage3_data_readiness_report.md` 中的 decision：

- `ready_for_stage4_smoke_training`：允许进入第四阶段真实 smoke 训练。
- 其他 decision：继续修正数据、章节卡、评测集或 SFT 数据集；不要进入真实训练。
