# Small Model Train

Small Model Train 是一个用于训练和评测中文整章小说执行器的 Python 项目。它围绕 Qwen3-4B-Instruct-2507 做 QLoRA/SFT 训练准备、冒烟训练、推理评测、质量报告和智能体审阅。

如果你是第一次打开这个项目，先读这两篇：

- [零基础使用手册](docs/zero-start.zh.md)
- [项目文档索引](docs/index.zh.md)

如果你已经知道要做什么，可以直接跳到：

- [项目目录地图](docs/project-map.zh.md)
- [完整数据流说明](docs/pipeline-flow.zh.md)
- [代码设计说明](docs/code-design.zh.md)
- [常见问题排查](docs/troubleshooting.zh.md)
- [术语表](docs/glossary.zh.md)

## 这个项目做什么

这个项目把原始小说文本整理成训练数据，用章节卡约束模型要写什么，再用 SFT/QLoRA 训练一个适合写中文整章正文的适配器。训练后，它会用固定评测卡生成文本、打分、生成报告，并通过质量门槛决定是否能扩大训练规模。

一句话版流程：

```text
原始小说 -> 清洗章节 -> 章节卡 -> SFT 数据 -> 冒烟训练 -> 评测生成 -> 打分报告 -> 质量/审阅决策
```

## 最安全的第一次检查

先确认你在项目根目录：

```powershell
Get-Location
```

安装轻量开发依赖：

```powershell
python -m pip install -e ".[dev]"
```

运行测试：

```powershell
python -m pytest
```

这些检查不会启动真实 GPU 训练。

## 常用入口

数据准备从这里开始：

```powershell
python scripts/ingest_raw_text.py --input-dir data_raw/novels --output data_clean/chapters_raw.jsonl
python scripts/clean_chapters.py --input data_clean/chapters_raw.jsonl --output data_clean/chapters.jsonl --min-chars 500 --max-chars 5000
python scripts/split_train_eval.py --input data_clean/chapters.jsonl --output data_clean/chapters_split.jsonl --eval-output data_cards/eval_cards_50.jsonl --eval-count 50
```

训练前检查从这里开始：

```powershell
python scripts/check_local_model.py --model-dir E:\models\Qwen3-4B-Instruct-2507 --report reports/model_check_report.md --json-output reports/model_check_report.json
python scripts/check_training_env.py --report reports/training_env_report.md --json-output reports/training_env_report.json
python scripts/check_stage3_data_readiness.py --eval-cards data_cards/eval_cards_50.jsonl --config configs/sft_qlora_qwen3_4b_smoke_6144.yaml --run-smoke-dry-run
```

真实训练会占用显卡资源。第一次操作前，请先读 [零基础使用手册](docs/zero-start.zh.md) 和 [常见问题排查](docs/troubleshooting.zh.md)。

## 现有阶段指南

- [第一阶段数据管线中文说明](docs/stage1-pipeline-guide.zh.md)
- [第三阶段真实数据准备指南](docs/stage3-data-bring-up-guide.zh.md)
- [第四阶段 Smoke Eval 指南](docs/stage4-smoke-eval-guide.zh.md)
- [Stage 4.1 Quality Eval Hardening 指南](docs/stage4-1-quality-eval-guide.zh.md)
- [Stage 5A 证据链修正操作指南](docs/stage5a-evidence-chain-hardening.zh.md)
- [Stage 5B StyleContract 闭环指南](docs/stage5b-style-contract-closure.zh.md)
- [Stage 5C 正式章节执行卡与数据完整性指南](docs/stage5c-formal-execution-cards-data-integrity.zh.md)
- [Stage 5D 作者反馈与 AI 味降低指南](docs/stage5d-author-feedback-ai-taste-reduction.zh.md)
- [Stage 5E 受控实验与效率指南](docs/stage5e-controlled-experimentation-efficiency.zh.md)
- [Stage 4 决策日志](docs/stage4-decision-log.zh.md)
- [第四阶段总结与下一阶段前瞻](docs/stage4-summary-next-outlook.zh.md)

## 重要提醒

- `data_raw/`、`data_clean/`、`data_cards/`、`data_sft/` 是数据层级。
- `outputs/` 是模型生成和训练产物。
- `reports/` 是人读的报告。
- `logs/` 是排错时看的运行日志。
- `src/small_model_train/` 放核心逻辑。
- `scripts/` 放命令入口。
- `tests/` 放自动化测试。

真实 GPU 训练依赖本地模型路径、CUDA、PyTorch、transformers、peft、bitsandbytes 和 LLaMA-Factory 环境。轻量数据处理和测试不等于真实训练环境已经可用。
