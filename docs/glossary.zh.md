# 术语表

这页把项目里反复出现的词翻译成人话。你不需要先背完，遇到不懂的词再回来查。

## 训练相关

### 基座模型

基座模型就是还没有用本项目数据继续训练的原始大模型。本项目默认把本地模型放在 `E:\models\Qwen3-4B-Instruct-2507`。脚本里的 `--model-dir` 通常指向这个目录。

### SFT

SFT 是监督微调。你可以把它理解成“给模型看很多输入和标准答案，让它学会按这种格式回答”。本项目里的 SFT 数据主要来自章节卡和真实章节正文。

### QLoRA

QLoRA 是一种省显存的微调方式。它不直接改完整大模型，而是在量化后的模型旁边训练一小组可保存的参数。

### Adapter

Adapter 是训练后得到的小型适配器。它不是完整大模型，通常放在 `outputs/sft_smoke/` 或 `outputs/sft_v1/`。推理时需要基座模型加 adapter 一起使用。

### Smoke training

Smoke training 是冒烟训练。它的目的不是追求最终质量，而是确认训练链路能跑通、adapter 能生成、日志和报告能落盘。

### Full training

Full training 是更正式的训练。它更耗时、更占显存，也更需要先通过数据 readiness、模型检查、环境检查和 smoke training。

### Dry-run

Dry-run 是试运行。它通常只检查参数、路径和即将执行的命令，不真正启动耗时训练。

## 数据相关

### JSONL

JSONL 是一行一个 JSON 对象的文本文件。它适合保存很多条样本。本项目的章节、卡片、生成结果和 metrics 大多用 JSONL。

### 章节

章节是从原始小说文本里切出来的一段正文。清洗后的章节会进入 `data_clean/chapters.jsonl`。

### 章节卡

章节卡告诉模型“这一章应该写什么”。它包含标题、简介、人物、场景、结构和禁忌等信息，但不应该泄漏完整原文。

### Eval card

Eval card 是评测卡。它用于固定评测题目，让不同模型或不同训练版本在同一批要求上生成文本。

### Execution card

Execution card 是更严格的执行卡。它会被推理和质量评测使用，要求字段完整、可验证、可绑定到生成结果。

### SFT 数据集

SFT 数据集是训练时喂给模型的输入和输出。默认文件是 `data_sft/sft_chapter_v1.jsonl`。

## 评测相关

### Inference

Inference 是推理，也就是让模型根据评测卡生成正文。生成结果通常写入 `outputs/.../generated.jsonl`。

### Metrics

Metrics 是评分结果。它通常记录覆盖率、长度、重复、AI 痕迹、结构问题等指标。

### Report

Report 是给人读的 Markdown 报告。它通常在 `reports/` 下面。

### Quality gate

Quality gate 是质量门槛。它不是“感觉不错就过”，而是用固定规则和报告判断是否允许扩大训练规模。

### Agent review

Agent review 是智能体审阅。它读取评测卡、生成结果和 metrics，再按规则产出审阅记录、投票和总结。

## 故障相关

### OOM

OOM 是显存或内存不够。训练或推理时如果显卡显存不足，常见表现是 CUDA out of memory。

### CUDA

CUDA 是 NVIDIA 显卡用于深度学习计算的环境。真实 GPU 训练需要它，普通文档阅读和部分数据处理不需要。

### Token budget

Token budget 是生成预算。预算太小会导致输出太短；预算太大可能更慢、更占显存。
