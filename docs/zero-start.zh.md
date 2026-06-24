# 零基础使用手册

这篇给第一次接触这个项目的人。目标不是让你马上懂所有机器学习概念，而是让你知道下一步该点哪里、跑什么命令、结果去哪看。

## 先用一句话理解项目

这个项目把中文小说章节整理成训练样本，用 Qwen3-4B-Instruct-2507 做 QLoRA/SFT 训练，然后用固定评测题检查模型能不能按章节卡写出合格正文。

## 你需要准备什么

- Windows + PowerShell。
- Python 3.10 或更高版本。
- 能打开这个仓库目录。
- 原始小说文本，放到 `data_raw/novels/`。
- 如果要跑真实训练，还需要 NVIDIA 显卡、CUDA、PyTorch、transformers、peft、bitsandbytes、LLaMA-Factory 和本地模型 `E:\models\Qwen3-4B-Instruct-2507`。

只读文档、跑轻量测试、理解代码，不需要先准备完整 GPU 训练环境。

## 打开项目目录

在 PowerShell 里进入项目：

```powershell
cd E:\codex\smallModelTrain
Get-Location
```

如果 `Get-Location` 显示的路径不是 `E:\codex\smallModelTrain`，先不要继续跑命令。

## 安装轻量开发依赖

```powershell
python -m pip install -e ".[dev]"
```

成功标志：命令结束时没有红色异常，之后可以运行 `python -m pytest`。

## 第一次安全检查

```powershell
python -m pytest
```

这个命令只跑自动化测试，不会启动真实训练。

## 准备原始小说

把 `.txt` 小说文本放进：

```text
data_raw/novels/
```

建议文件名能看出作品名。不要把训练输出、压缩包、图片或无关文件放进去。

## 跑第一段数据准备

```powershell
python scripts/ingest_raw_text.py --input-dir data_raw/novels --output data_clean/chapters_raw.jsonl
python scripts/clean_chapters.py --input data_clean/chapters_raw.jsonl --output data_clean/chapters.jsonl --min-chars 500 --max-chars 5000
python scripts/split_train_eval.py --input data_clean/chapters.jsonl --output data_clean/chapters_split.jsonl --eval-output data_cards/eval_cards_50.jsonl --eval-count 50
```

成功后你会看到：

- `data_clean/chapters_raw.jsonl`
- `data_clean/chapters.jsonl`
- `data_clean/chapters_split.jsonl`
- `data_cards/eval_cards_50.jsonl`

## 生成风格信息

```powershell
python scripts/build_style_contract.py --chapters data_clean/chapters_split.jsonl --contract-json-output data_style/style_contract_author_main_v1.json --contract-output style_contract.md --metrics-output data_style/style_metrics_author_main_v1.json --style-contract-id author_main_v1
```

成功后你会看到：

- `data_style/style_contract_author_main_v1.json`
- `style_contract.md`
- `data_style/style_metrics_author_main_v1.json`

Stage 5B 起，formal SFT 使用 `data_style/style_contract_author_main_v1.json` 作为机器门禁源；`style_contract.md` 只用于人工审阅。旧的 `style_profile.json` 仍可通过 `--profile-output` 作为兼容统计输出，但默认说明以 `data_style/style_metrics_author_main_v1.json` 为准。

## 构建训练数据

先生成章节卡：

```powershell
python scripts/build_chapter_cards.py --chapters data_clean/chapters_split.jsonl --output data_cards/chapter_cards.jsonl --count 50 --min-chars 2000 --max-chars 3000
```

再构建 SFT 数据：

```powershell
python scripts/build_sft_dataset.py --cards data_cards/chapter_cards.jsonl --chapters data_clean/chapters_split.jsonl --output data_sft/sft_chapter_v1.jsonl --dataset-info-output data_sft/dataset_info.json
```

成功后你会看到：

- `data_cards/chapter_cards.jsonl`
- `data_sft/sft_chapter_v1.jsonl`
- `data_sft/dataset_info.json`

## 训练前先做检查

数据 readiness：

```powershell
python scripts/check_stage3_data_readiness.py --eval-cards data_cards/eval_cards_50.jsonl --config configs/sft_qlora_qwen3_4b_smoke_6144.yaml --run-smoke-dry-run
```

模型检查：

```powershell
python scripts/check_local_model.py --model-dir E:\models\Qwen3-4B-Instruct-2507 --report reports/model_check_report.md
```

环境检查：

```powershell
python scripts/check_training_env.py --report reports/training_env_report.md
```

先读报告，再决定是否训练：

- `reports/stage3_data_readiness_report.md`
- `reports/model_check_report.md`
- `reports/training_env_report.md`

## 准备 Stage 4 执行卡

`data_cards/eval_execution_cards_50.jsonl` 不是前面入门数据准备命令自动生成的文件。它需要基于固定评测样本手动或用 Agent 准备，并包含执行字段：`id`、`target_platform`、`genre_tags`、`style_contract`、`chapter_goal`、`chapter_structure`、`conflict_beat`、`payoff_beat`、`must_include`、`must_not_include`、`ending_hook`、`target_word_count`。

训练前先验证执行卡：

```powershell
python scripts/run_eval_inference.py --cards data_cards/eval_execution_cards_50.jsonl --output outputs/sft_smoke/generated_dry_run.jsonl --model-name sft_smoke --dry-run
```

如果命令失败，先按 [第四阶段 Smoke Eval 指南](stage4-smoke-eval-guide.zh.md) 或当前 Stage 4 数据准备路径修正执行卡，不要继续训练。

## 什么是 dry-run

Dry-run 是试运行。它用来确认路径、参数、配置和即将执行的命令是否合理。它通常不会真正训练模型。

下面的命令会使用 `data_cards/eval_execution_cards_50.jsonl`。这是 Stage 4 脚本使用的更严格执行卡版本；如果这个文件不存在，先停下来，按 [第四阶段 Smoke Eval 指南](stage4-smoke-eval-guide.zh.md) 或当前 Stage 4 数据准备路径补齐后再训练。

示例：

```powershell
python scripts/run_sft_smoke.py --config configs/sft_qlora_qwen3_4b_smoke_6144.yaml --eval-cards data_cards/eval_execution_cards_50.jsonl --dry-run
```

## 什么是真实训练

真实训练会占用显卡和较长时间。不要在 readiness、模型检查、环境检查失败时硬跑。

下面的真实训练命令同样使用 `data_cards/eval_execution_cards_50.jsonl`。它不是前面入门数据准备命令自动生成的文件；如果缺失，先按 [第四阶段 Smoke Eval 指南](stage4-smoke-eval-guide.zh.md) 或当前 Stage 4 数据准备路径处理，不要直接开训。

真实 GPU 训练前检查清单：

- `data_cards/eval_execution_cards_50.jsonl` 已存在，且 `run_eval_inference.py --dry-run` 验证通过。
- 已读 `reports/stage3_data_readiness_report.md`。
- `reports/model_check_report.md` 通过。
- `reports/training_env_report.md` 没有阻断项。
- `configs/sft_qlora_qwen3_4b_smoke_6144.yaml` 对应的 smoke dry-run 已通过。

Smoke training 示例：

```powershell
python scripts/run_sft_smoke.py --config configs/sft_qlora_qwen3_4b_smoke_6144.yaml --eval-cards data_cards/eval_execution_cards_50.jsonl
```

## 训练后看哪里

训练和 adapter 检查会产生：

- Adapter：`outputs/sft_smoke/`
- 训练日志：`logs/training/`
- Adapter 检查报告：`reports/sft_smoke_report.md`

后续 eval 命令会产生：

- 生成结果：`outputs/sft_smoke/generated.jsonl`
- 评分结果：`outputs/sft_smoke/metrics.jsonl`
- 评测报告：`reports/sft_smoke_eval_report.md`

## 常见安全原则

- 不懂某个词，先查 [术语表](glossary.zh.md)。
- 不知道文件在哪，先查 [项目目录地图](project-map.zh.md)。
- 不知道下一步输入输出，先查 [完整数据流说明](pipeline-flow.zh.md)。
- 命令失败时不要连续盲试，先查 [常见问题排查](troubleshooting.zh.md)。
- 真实训练前必须看 readiness、模型检查和环境检查报告。
