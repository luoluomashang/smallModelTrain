# Zero-Base Documentation System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a layered Chinese documentation system that lets a complete beginner understand, run, inspect, and safely extend the Small Model Train project.

**Architecture:** Keep `README.md` as the short front door, add focused beginner docs under `docs/`, and use `docs/index.zh.md` as the hub that connects new beginner material to the existing stage guides. The documentation explains the same project from four angles: how to start, where files live, how data flows, and how the code is designed.

**Tech Stack:** Markdown, PowerShell command examples, Python CLI scripts in `scripts/`, Python package modules in `src/small_model_train/`, pytest, and ripgrep.

---

## File Structure

- Modify: `README.md`
  - Responsibility: short project front door, safest first links, quick-start commands, and warnings about real GPU training.
- Create: `docs/index.zh.md`
  - Responsibility: documentation hub with beginner, operator, and developer reading paths.
- Create: `docs/zero-start.zh.md`
  - Responsibility: zero-base user manual that explains the project from first principles and shows the safe first workflows.
- Create: `docs/project-map.zh.md`
  - Responsibility: directory and artifact map for source files, generated data, logs, reports, configs, and outputs.
- Create: `docs/pipeline-flow.zh.md`
  - Responsibility: end-to-end data flow from raw novels to training, eval, scoring, reports, and review gates.
- Create: `docs/code-design.zh.md`
  - Responsibility: code organization guide for maintainers and future contributors.
- Create: `docs/troubleshooting.zh.md`
  - Responsibility: symptom-based failure diagnosis guide.
- Create: `docs/glossary.zh.md`
  - Responsibility: plain-language glossary for project terms.

The implementation must not change training, inference, scoring, or review logic.

## Task 1: README Front Door And Document Index

**Files:**
- Modify: `README.md`
- Create: `docs/index.zh.md`

- [ ] **Step 1: Run navigation pre-check**

Run:

```powershell
$required = @("docs/index.zh.md")
$missing = $required | Where-Object { -not (Test-Path $_) }
if ($missing) {
    Write-Output ("missing docs: " + ($missing -join ", "))
    exit 1
}
Write-Output "all navigation docs exist"
```

Expected: command exits `1` and prints `missing docs: docs/index.zh.md`.

- [ ] **Step 2: Replace `README.md` with a short front door**

Use this structure and content in `README.md`:

````markdown
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
````

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
python scripts/check_local_model.py --model-dir E:\models\Qwen3-4B-Instruct-2507 --report reports/model_check_report.md
python scripts/check_training_env.py --report reports/training_env_report.md
python scripts/check_stage3_data_readiness.py --eval-cards data_cards/eval_cards_50.jsonl --run-smoke-dry-run
```

真实训练会占用显卡资源。第一次操作前，请先读 [零基础使用手册](docs/zero-start.zh.md) 和 [常见问题排查](docs/troubleshooting.zh.md)。

## 现有阶段指南

- [第一阶段数据管线中文说明](docs/stage1-pipeline-guide.zh.md)
- [第三阶段真实数据准备指南](docs/stage3-data-bring-up-guide.zh.md)
- [第四阶段 Smoke Eval 指南](docs/stage4-smoke-eval-guide.zh.md)
- [Stage 4.1 Quality Eval Hardening 指南](docs/stage4-1-quality-eval-guide.zh.md)
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
```

- [ ] **Step 3: Create `docs/index.zh.md`**

Use this structure and content:

```markdown
# 项目文档索引

这页是 Small Model Train 的文档导航。你不需要一次读完所有文档，按自己的角色选择路线即可。

## 我完全没基础

按这个顺序读：

1. [零基础使用手册](zero-start.zh.md)
2. [术语表](glossary.zh.md)
3. [项目目录地图](project-map.zh.md)
4. [完整数据流说明](pipeline-flow.zh.md)
5. [常见问题排查](troubleshooting.zh.md)

读完这条路线，你应该能知道项目在做什么、文件放哪里、先跑哪些安全命令、输出结果去哪里看。

## 我只想跑流程

按这个顺序读：

1. [零基础使用手册](zero-start.zh.md) 的“最安全的第一次运行”和“常用流程”
2. [第三阶段真实数据准备指南](stage3-data-bring-up-guide.zh.md)
3. [第四阶段 Smoke Eval 指南](stage4-smoke-eval-guide.zh.md)
4. [Stage 4.1 Quality Eval Hardening 指南](stage4-1-quality-eval-guide.zh.md)
5. [Stage 4 决策日志](stage4-decision-log.zh.md)

这条路线适合已经会用 PowerShell、Python，并且只想按步骤复跑数据准备、训练检查、评测和报告的人。

## 我要改代码

按这个顺序读：

1. [代码设计说明](code-design.zh.md)
2. [项目目录地图](project-map.zh.md)
3. [完整数据流说明](pipeline-flow.zh.md)
4. [常见问题排查](troubleshooting.zh.md)
5. `tests/` 里的相关测试文件

改代码前先确认要改的是命令入口还是核心逻辑。一般来说，`scripts/` 负责接收参数和读写文件，`src/small_model_train/` 负责可测试的业务逻辑。

## 阶段文档

- [第一阶段数据管线中文说明](stage1-pipeline-guide.zh.md)：解释从原稿到基础数据、评分和报告的第一阶段流程。
- [第三阶段真实数据准备指南](stage3-data-bring-up-guide.zh.md)：解释真实数据接入、章节卡、SFT 数据和 readiness 报告。
- [第四阶段 Smoke Eval 指南](stage4-smoke-eval-guide.zh.md)：解释固定 50 卡 smoke training、adapter 检查、eval 推理和评分。
- [Stage 4.1 Quality Eval Hardening 指南](stage4-1-quality-eval-guide.zh.md)：解释长生成质量评测、质量子集、预算报告和审阅门槛。
- [Stage 4 决策日志](stage4-decision-log.zh.md)：记录当前是否允许扩大训练规模，以及原因。
- [Stage 4.1 Full50 归档](stage4-1-full50-archive.zh.md)：归档 full50 运行证据。
- [Stage 4.1 Full50 Manual Review Findings](stage4-1-full50-manual-review.zh.md)：记录人工复核发现。
- [第四阶段总结与下一阶段前瞻](stage4-summary-next-outlook.zh.md)：总结当前状态和下一阶段建议。
- [两阶段实现审计报告](two-stage-implementation-audit.zh.md)：审计早期实现是否符合计划。

## 产物去哪里看

- `data_raw/novels/`：你放进去的原始小说文本。
- `data_clean/`：清洗、切章、划分后的章节数据。
- `data_cards/`：章节卡、评测卡、质量子集卡。
- `data_sft/`：给训练用的 SFT 数据和数据集元信息。
- `outputs/`：训练产物、adapter、生成结果、metrics。
- `reports/`：模型检查、环境检查、readiness、评测和质量报告。
- `logs/`：训练、推理、OOM 探测等运行日志。
- `configs/`：训练和推理配置。
```

- [ ] **Step 4: Verify navigation docs**

Run:

```powershell
$required = @("README.md", "docs/index.zh.md")
$missing = $required | Where-Object { -not (Test-Path $_) }
if ($missing) { throw ("missing docs: " + ($missing -join ", ")) }
rg -n "zero-start.zh.md|project-map.zh.md|pipeline-flow.zh.md|code-design.zh.md|troubleshooting.zh.md|glossary.zh.md" README.md docs/index.zh.md
```

Expected: command exits `0`; the `rg` output shows each new document linked from `README.md` or `docs/index.zh.md`.

- [ ] **Step 5: Commit navigation work**

Run:

```powershell
git add README.md docs/index.zh.md
git commit -m "docs: add beginner documentation entry points"
```

Expected: commit succeeds with `README.md` modified and `docs/index.zh.md` created.

## Task 2: Glossary And Project Map

**Files:**
- Create: `docs/glossary.zh.md`
- Create: `docs/project-map.zh.md`

- [ ] **Step 1: Run glossary/map pre-check**

Run:

```powershell
$required = @("docs/glossary.zh.md", "docs/project-map.zh.md")
$missing = $required | Where-Object { -not (Test-Path $_) }
if ($missing) {
    Write-Output ("missing docs: " + ($missing -join ", "))
    exit 1
}
Write-Output "glossary and project map exist"
```

Expected: command exits `1` and prints both missing files.

- [ ] **Step 2: Create `docs/glossary.zh.md`**

Write a plain-language glossary with these exact sections and terms:

```markdown
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
```

- [ ] **Step 3: Create `docs/project-map.zh.md`**

Write the project map with these sections:

```markdown
# 项目目录地图

这页解释每个目录和关键文件是干什么的。你可以先把项目想成一条流水线：原始数据进来，脚本处理它，核心代码负责规则，结果写到数据目录、输出目录、报告目录和日志目录。

## 根目录

- `README.md`：项目入口，负责告诉你从哪里开始。
- `pyproject.toml`：Python 项目配置，包含测试路径和开发依赖。
- `style_contract.md`：由脚本生成的风格契约，用来概括训练数据的风格特征。
- `style_profile.json`：由脚本生成的风格统计数据。
- `mlflow.db`：训练或实验记录数据库，属于运行产物。

## 源码目录

- `scripts/`：命令入口。你在 PowerShell 里运行的大多数命令都来自这里。
- `src/small_model_train/`：核心逻辑。脚本会调用这里的函数完成清洗、校验、评分、报告、训练封装等工作。
- `tests/`：自动化测试。改代码后先跑相关测试，再跑全量 `python -m pytest`。

## 数据目录

- `data_raw/novels/`：放原始小说文本。
- `data_clean/chapters_raw.jsonl`：原始文本切章后的中间结果。
- `data_clean/chapters.jsonl`：清洗和长度过滤后的章节。
- `data_clean/chapters_split.jsonl`：标记 train/eval 划分后的章节。
- `data_cards/`：章节卡、评测卡、执行卡、质量子集。
- `data_sft/`：训练数据和 LLaMA-Factory 数据集元信息。

## 配置、输出、报告和日志

- `configs/`：训练和推理配置。
- `outputs/`：adapter、生成结果、metrics 等模型相关产物。
- `reports/`：人读报告，例如模型检查报告、环境检查报告、readiness 报告、质量报告。
- `logs/`：训练、推理、OOM 探测产生的 stdout、stderr 和事件日志。

## 文档目录

- `docs/index.zh.md`：文档索引。
- `docs/zero-start.zh.md`：零基础使用手册。
- `docs/project-map.zh.md`：当前目录地图。
- `docs/pipeline-flow.zh.md`：完整数据流。
- `docs/code-design.zh.md`：代码设计说明。
- `docs/troubleshooting.zh.md`：常见问题排查。
- `docs/glossary.zh.md`：术语表。
- `docs/stage*.zh.md`：已有阶段指南、决策日志和归档。
- `docs/superpowers/`：设计 spec 和实施计划，主要给协作代理和开发者追溯使用。

## 哪些文件可以手动改

通常可以手动改：

- `README.md`
- `docs/`
- `configs/`
- `data_raw/novels/` 里的原始文本
- 小规模手写的测试样例

通常不要手动改：

- `outputs/` 里的模型和生成结果
- `logs/` 里的运行日志
- 自动脚本生成的 `metrics.jsonl`
- 自动脚本生成的报告，除非你明确是在补人工说明

## 判断一个文件是不是生成物

如果文件来自某条脚本命令的 `--output`、`--report`、`--stdout-log`、`--stderr-log` 或 `--event-log` 参数，它通常就是生成物。生成物可以删除后重建，但删除前要确认没有正在使用它的报告或决策记录。
```

- [ ] **Step 4: Verify glossary/map docs**

Run:

```powershell
$required = @("docs/glossary.zh.md", "docs/project-map.zh.md")
$missing = $required | Where-Object { -not (Test-Path $_) }
if ($missing) { throw ("missing docs: " + ($missing -join ", ")) }
rg -n "SFT|QLoRA|Adapter|JSONL|OOM" docs/glossary.zh.md
rg -n "scripts/|src/small_model_train/|data_raw/|outputs/|reports/|logs/" docs/project-map.zh.md
```

Expected: command exits `0` and shows matches for all required concepts and directories.

- [ ] **Step 5: Commit glossary/map work**

Run:

```powershell
git add docs/glossary.zh.md docs/project-map.zh.md
git commit -m "docs: explain terms and project layout"
```

Expected: commit succeeds with the two new docs created.

## Task 3: Pipeline Flow And Zero-Start Manual

**Files:**
- Create: `docs/pipeline-flow.zh.md`
- Create: `docs/zero-start.zh.md`

- [ ] **Step 1: Run beginner-flow pre-check**

Run:

```powershell
$required = @("docs/pipeline-flow.zh.md", "docs/zero-start.zh.md")
$missing = $required | Where-Object { -not (Test-Path $_) }
if ($missing) {
    Write-Output ("missing docs: " + ($missing -join ", "))
    exit 1
}
Write-Output "pipeline and zero-start docs exist"
```

Expected: command exits `1` and prints both missing files.

- [ ] **Step 2: Create `docs/pipeline-flow.zh.md`**

Write a flow guide with this exact high-level structure:

````markdown
# 完整数据流说明

这页按“文件怎么变成另一个文件”的顺序解释项目。你不需要先懂训练原理，只要记住：每一步都有输入、命令、输出和检查方式。

## 总览

```text
data_raw/novels/
  -> data_clean/chapters_raw.jsonl
  -> data_clean/chapters.jsonl
  -> data_clean/chapters_split.jsonl + data_cards/eval_cards_50.jsonl
  -> style_contract.md + style_profile.json
  -> data_cards/chapter_cards.jsonl
  -> data_sft/sft_chapter_v1.jsonl
  -> reports/stage3_data_readiness_report.md
  -> outputs/sft_smoke/
  -> outputs/sft_smoke/generated.jsonl
  -> outputs/sft_smoke/metrics.jsonl
  -> reports/sft_smoke_eval_report.md
  -> reports/stage4_1_quality_eval_budget_report.md
````

## 1. 原始小说到原始章节

输入：`data_raw/novels/`

命令：

```powershell
python scripts/ingest_raw_text.py --input-dir data_raw/novels --output data_clean/chapters_raw.jsonl
```

输出：`data_clean/chapters_raw.jsonl`

成功标志：文件存在，每一行是一章或一段可处理文本。

## 2. 原始章节到清洗章节

输入：`data_clean/chapters_raw.jsonl`

命令：

```powershell
python scripts/clean_chapters.py --input data_clean/chapters_raw.jsonl --output data_clean/chapters.jsonl --min-chars 500 --max-chars 5000
```

输出：`data_clean/chapters.jsonl`

成功标志：过短、过长或格式不合适的章节被过滤。

## 3. 划分训练集和评测集

输入：`data_clean/chapters.jsonl`

命令：

```powershell
python scripts/split_train_eval.py --input data_clean/chapters.jsonl --output data_clean/chapters_split.jsonl --eval-output data_cards/eval_cards_50.jsonl --eval-count 50
```

输出：

- `data_clean/chapters_split.jsonl`
- `data_cards/eval_cards_50.jsonl`

成功标志：评测集数量符合 `--eval-count`，训练数据仍然保留在 split 文件里。

## 4. 生成风格契约

输入：`data_clean/chapters_split.jsonl`

命令：

```powershell
python scripts/build_style_contract.py --chapters data_clean/chapters_split.jsonl --contract-output style_contract.md --profile-output style_profile.json
```

输出：

- `style_contract.md`
- `style_profile.json`

成功标志：Markdown 里有风格摘要，JSON 里有统计字段。

## 5. 生成章节卡

输入：`data_clean/chapters_split.jsonl`

命令：

```powershell
python scripts/build_chapter_cards.py --chapters data_clean/chapters_split.jsonl --output data_cards/chapter_cards.jsonl --count 50 --min-chars 2000 --max-chars 3000
```

输出：`data_cards/chapter_cards.jsonl`

成功标志：章节卡字段完整，没有直接塞入完整原文。

## 6. 构建 SFT 数据

输入：

- `data_cards/chapter_cards.jsonl`
- `data_clean/chapters_split.jsonl`

命令：

```powershell
python scripts/build_sft_dataset.py --cards data_cards/chapter_cards.jsonl --chapters data_clean/chapters_split.jsonl --output data_sft/sft_chapter_v1.jsonl --dataset-info-output data_sft/dataset_info.json
```

输出：

- `data_sft/sft_chapter_v1.jsonl`
- `data_sft/dataset_info.json`

成功标志：SFT 数据可以被训练脚本引用。

## 7. 训练前 readiness

命令：

```powershell
python scripts/check_stage3_data_readiness.py --eval-cards data_cards/eval_cards_50.jsonl --run-smoke-dry-run
```

输出：`reports/stage3_data_readiness_report.md`

成功标志：报告给出可以进入 smoke training 的状态。

## 8. 模型和环境检查

命令：

```powershell
python scripts/check_local_model.py --model-dir E:\models\Qwen3-4B-Instruct-2507 --report reports/model_check_report.md
python scripts/check_training_env.py --report reports/training_env_report.md
```

输出：

- `reports/model_check_report.md`
- `reports/training_env_report.md`

成功标志：模型文件存在，训练环境报告没有阻断项。

## 9. Smoke training

命令：

```powershell
python scripts/run_sft_smoke.py --config configs/sft_qlora_qwen3_4b_smoke_6144.yaml --eval-cards data_cards/eval_execution_cards_50.jsonl
```

输出：`outputs/sft_smoke/`

成功标志：adapter 文件生成，训练日志写入 `logs/training/`。

## 10. Adapter 检查

命令：

```powershell
python scripts/check_adapter.py --adapter-dir outputs/sft_smoke --report reports/sft_smoke_report.md --title "SFT Smoke Adapter Check"
```

输出：`reports/sft_smoke_report.md`

成功标志：报告确认 adapter 结构可读。

## 11. Eval 推理和评分

命令：

```powershell
python scripts/run_eval_inference.py --cards data_cards/eval_execution_cards_50.jsonl --adapter-dir outputs/sft_smoke --output outputs/sft_smoke/generated.jsonl --model-name sft_smoke --max-new-tokens 256
python scripts/score_outputs.py --cards data_cards/eval_execution_cards_50.jsonl --outputs outputs/sft_smoke/generated.jsonl --output outputs/sft_smoke/metrics.jsonl
python scripts/evaluate_outputs.py --scores outputs/sft_smoke/metrics.jsonl --report reports/sft_smoke_eval_report.md --title "SFT Smoke Eval Report"
```

输出：

- `outputs/sft_smoke/generated.jsonl`
- `outputs/sft_smoke/metrics.jsonl`
- `reports/sft_smoke_eval_report.md`

成功标志：生成结果、metrics 和 Markdown 报告三者数量能对上。

## 12. Stage 4.1 质量评测

命令：

```powershell
python scripts/build_eval_quality_subset.py --cards data_cards/eval_execution_cards_50.jsonl --metrics outputs/sft_smoke/metrics.jsonl --output data_cards/eval_cards_quality_subset.jsonl --count 8
python scripts/run_eval_inference.py --cards data_cards/eval_cards_quality_subset.jsonl --adapter-dir outputs/sft_smoke --output outputs/sft_smoke/generated_subset_1024.jsonl --model-name sft_smoke_subset_1024 --max-new-tokens 1024
python scripts/score_outputs.py --cards data_cards/eval_cards_quality_subset.jsonl --outputs outputs/sft_smoke/generated_subset_1024.jsonl --output outputs/sft_smoke/metrics_subset_1024.jsonl
python scripts/build_stage4_quality_report.py --cards data_cards/eval_cards_quality_subset.jsonl --generated outputs/sft_smoke/generated_subset_1024.jsonl --metrics outputs/sft_smoke/metrics_subset_1024.jsonl --report reports/stage4_1_quality_eval_budget_report.md --title "Stage 4.1 Quality Eval Budget Report"
```

输出：`reports/stage4_1_quality_eval_budget_report.md`

成功标志：报告说明是否通过质量门槛，而不是只看命令有没有跑完。

## 阶段边界

- Stage 1：数据管线和基础评分报告。
- Stage 2：训练执行封装、模型检查、环境检查、adapter 检查、OOM 探测。
- Stage 3：真实数据接入和 readiness 证据，不直接追求最终训练质量。
- Stage 4：固定 50 卡 smoke training 和评测决策。
- Stage 4.1：长生成质量、预算和审阅门槛。
```

- [ ] **Step 3: Create `docs/zero-start.zh.md`**

Write a beginner manual with these sections:

````markdown
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
````

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
python scripts/build_style_contract.py --chapters data_clean/chapters_split.jsonl --contract-output style_contract.md --profile-output style_profile.json
```

成功后你会看到：

- `style_contract.md`
- `style_profile.json`

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
python scripts/check_stage3_data_readiness.py --eval-cards data_cards/eval_cards_50.jsonl --run-smoke-dry-run
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

## 什么是 dry-run

Dry-run 是试运行。它用来确认路径、参数、配置和即将执行的命令是否合理。它通常不会真正训练模型。

示例：

```powershell
python scripts/run_sft_smoke.py --eval-cards data_cards/eval_execution_cards_50.jsonl --dry-run
```

## 什么是真实训练

真实训练会占用显卡和较长时间。不要在 readiness、模型检查、环境检查失败时硬跑。

Smoke training 示例：

```powershell
python scripts/run_sft_smoke.py --config configs/sft_qlora_qwen3_4b_smoke_6144.yaml --eval-cards data_cards/eval_execution_cards_50.jsonl
```

## 训练后看哪里

- Adapter：`outputs/sft_smoke/`
- 训练日志：`logs/training/`
- Adapter 检查报告：`reports/sft_smoke_report.md`
- 生成结果：`outputs/sft_smoke/generated.jsonl`
- 评分结果：`outputs/sft_smoke/metrics.jsonl`
- 评测报告：`reports/sft_smoke_eval_report.md`

## 常见安全原则

- 不懂某个词，先查 [术语表](glossary.zh.md)。
- 不知道文件在哪，先查 [项目目录地图](project-map.zh.md)。
- 不知道下一步输入输出，先查 [完整数据流说明](pipeline-flow.zh.md)。
- 命令失败时不要连续盲试，先查 [常见问题排查](troubleshooting.zh.md)。
- 真实训练前必须看 readiness、模型检查和环境检查报告。
```

- [ ] **Step 4: Verify beginner flow docs**

Run:

```powershell
$required = @("docs/pipeline-flow.zh.md", "docs/zero-start.zh.md")
$missing = $required | Where-Object { -not (Test-Path $_) }
if ($missing) { throw ("missing docs: " + ($missing -join ", ")) }
rg -n "ingest_raw_text.py|clean_chapters.py|split_train_eval.py|build_sft_dataset.py|run_sft_smoke.py|run_eval_inference.py|build_stage4_quality_report.py" docs/pipeline-flow.zh.md docs/zero-start.zh.md
rg -n "data_raw/novels|data_clean/chapters|data_cards|data_sft|outputs/sft_smoke|reports/" docs/pipeline-flow.zh.md docs/zero-start.zh.md
```

Expected: command exits `0` and shows every major script and artifact path.

- [ ] **Step 5: Commit beginner flow docs**

Run:

```powershell
git add docs/pipeline-flow.zh.md docs/zero-start.zh.md
git commit -m "docs: add zero-start pipeline guide"
```

Expected: commit succeeds with the two new docs created.

## Task 4: Code Design And Troubleshooting

**Files:**
- Create: `docs/code-design.zh.md`
- Create: `docs/troubleshooting.zh.md`

- [ ] **Step 1: Run maintainer-doc pre-check**

Run:

```powershell
$required = @("docs/code-design.zh.md", "docs/troubleshooting.zh.md")
$missing = $required | Where-Object { -not (Test-Path $_) }
if ($missing) {
    Write-Output ("missing docs: " + ($missing -join ", "))
    exit 1
}
Write-Output "maintainer docs exist"
```

Expected: command exits `1` and prints both missing files.

- [ ] **Step 2: Create `docs/code-design.zh.md`**

Write the code design guide with this structure:

```markdown
# 代码设计说明

这页给以后要改代码的人。先记住一个原则：`scripts/` 是命令入口，`src/small_model_train/` 是核心逻辑，`tests/` 是保护网。

## 总体分层

- `scripts/`：解析命令行参数，调用核心函数，读写文件，返回进程退出码。
- `src/small_model_train/`：放可测试的业务逻辑。
- `tests/`：用 pytest 验证模块行为和命令入口边界。

脚本应该尽量薄。复杂逻辑应该放进 `src/small_model_train/`，这样测试可以直接调用函数。

## 数据准备模块

- `io_utils.py`：读写文本和 JSONL。
- `text_utils.py`：中文字符计数、段落、对话比例、重复 ngram。
- `chapter_splitter.py`：清洗原文和切章节。
- `dataset_split.py`：确定训练集和评测集划分。
- `style_profile.py`：生成风格统计和风格契约。

相关脚本：

- `scripts/ingest_raw_text.py`
- `scripts/clean_chapters.py`
- `scripts/split_train_eval.py`
- `scripts/build_style_contract.py`

## 卡片和训练数据模块

- `chapter_cards.py`：构建和校验章节卡。
- `execution_cards.py`：校验执行卡。
- `sft_builder.py`：把章节卡和章节正文组合成 SFT 样本，并阻止原文泄漏。
- `preference_builder.py`：构建偏好数据候选。

相关脚本：

- `scripts/build_chapter_cards.py`
- `scripts/build_sft_dataset.py`
- `scripts/build_preference_dataset.py`

## Stage 2 训练执行模块

- `stage2_config.py`：读写扁平 YAML、构建 LLaMA-Factory 命令。
- `stage2_model_check.py`：检查本地模型文件和 transformers 加载。
- `stage2_env_check.py`：检查 Python、CUDA、GPU 显存和训练依赖。
- `stage2_training.py`：封装训练命令、日志、失败报告和 GPU 采样。
- `stage2_monitoring.py`：事件日志、错误分类和失败摘要。
- `stage2_oom_probe.py`：分阶段 OOM 探测。
- `stage2_adapter.py`：检查 adapter 目录结构。
- `stage2_inference.py`：渲染评测 prompt、清理生成结果、构建生成行。

相关脚本：

- `scripts/check_local_model.py`
- `scripts/check_training_env.py`
- `scripts/run_sft_smoke.py`
- `scripts/run_sft_train.py`
- `scripts/check_adapter.py`
- `scripts/run_oom_probe.py`
- `scripts/run_eval_inference.py`

## Stage 3 和 Stage 4 质量模块

- `stage3_data_readiness.py`：检查真实数据、章节卡、评测卡和 SFT 数据是否达到训练前门槛。
- `quality_rules.py`：检测质量问题。
- `stage4_quality.py`：选择质量子集、识别大纲泄漏、汇总质量预算、渲染质量报告。
- `agent_review.py`：校验和聚合智能体审阅记录，渲染审阅报告。
- `scoring.py`：给生成结果打规则分。
- `reporting.py`：汇总 metrics 并生成 Markdown 报告。

相关脚本：

- `scripts/check_stage3_data_readiness.py`
- `scripts/build_eval_quality_subset.py`
- `scripts/build_stage4_quality_report.py`
- `scripts/run_agent_review.py`
- `scripts/score_outputs.py`
- `scripts/evaluate_outputs.py`

## 新增脚本时怎么做

1. 先把可测试逻辑写进 `src/small_model_train/`。
2. 在 `tests/` 里写对应测试。
3. 在 `scripts/` 里只做参数解析、调用函数、读写文件和返回退出码。
4. 给失败路径明确错误信息。
5. 在文档里说明输入、命令、输出和成功标志。

## 新增测试时怎么做

- 测核心逻辑时直接导入 `src/small_model_train/` 里的函数。
- 测脚本行为时用临时目录，不污染真实 `data_*`、`outputs/`、`reports/`。
- 测失败路径时断言错误类型或返回码。
- 改训练、推理、评分、报告逻辑时至少跑相关测试，再跑 `python -m pytest`。
```

- [ ] **Step 3: Create `docs/troubleshooting.zh.md`**

Write the troubleshooting guide with this structure:

````markdown
# 常见问题排查

这页按症状查问题。命令失败时先看报错和产物，不要连续盲试。

## 找不到文件或目录

症状：PowerShell 报 `No such file`、`FileNotFoundError`、路径不存在。

通常原因：

- 没在项目根目录运行命令。
- 上一步输出没有生成。
- 参数里的路径拼错。

先检查：

```powershell
Get-Location
Test-Path data_raw/novels
Test-Path data_clean/chapters.jsonl
````

安全处理：

- 回到 `E:\codex\smallModelTrain`。
- 按 [完整数据流说明](pipeline-flow.zh.md) 补跑缺失的上游步骤。
- 不要手写伪造中间 JSONL 来骗过后续脚本。

## JSONL 格式错误

症状：报 JSON 解析错误，或提示某一行无法读取。

通常原因：

- 文件不是一行一个 JSON。
- 手动编辑时删了引号、逗号或括号。
- 文件混入了普通文本。

先检查具体文件，例如：

```powershell
Get-Content data_cards/chapter_cards.jsonl -TotalCount 3
```

安全处理：

- 优先重新运行生成该文件的脚本。
- 如果必须手工修，只修报错行，并保持一行一个完整 JSON 对象。

## 章节卡或评测卡校验失败

症状：提示字段缺失、结构不合法、id 重复、样本数量不对。

通常原因：

- `data_cards/chapter_cards.jsonl` 或 eval cards 不是当前脚本期望的格式。
- 章节卡和章节 split 文件没有对齐。
- 生成结果、metrics、review 三类文件的 sample id 对不上。

先看：

- `reports/stage3_data_readiness_report.md`
- `data_cards/chapter_cards.jsonl`
- `data_cards/eval_execution_cards_50.jsonl`

安全处理：

- 重新生成章节卡。
- 重新构建 SFT 数据。
- 不要跳过 readiness 报告直接训练。

## 本地模型路径错误

症状：模型检查失败，提示 `config.json`、tokenizer、safetensors 或 shard 文件缺失。

先运行：

```powershell
python scripts/check_local_model.py --model-dir E:\models\Qwen3-4B-Instruct-2507 --report reports/model_check_report.md
```

然后读：

```text
reports/model_check_report.md
```

安全处理：

- 确认 `--model-dir` 指向真实模型目录。
- 不要把 adapter 目录当成基座模型目录。

## CUDA 不可用或显存不够

症状：`CUDA unavailable`、`CUDA out of memory`、进程被系统杀掉、训练中断。

先运行：

```powershell
python scripts/check_training_env.py --report reports/training_env_report.md
python scripts/run_oom_probe.py --dry-run
```

如果需要定位显存问题，再运行：

```powershell
python scripts/run_oom_probe.py
```

重点看：

- `reports/training_env_report.md`
- `reports/oom_probe_report.md`
- `logs/training/`

安全处理：

- 先关掉占用显卡的其他进程。
- 先跑 smoke 配置，不要直接跑 full training。
- OOM 后先读报告，不要反复重试同一个大配置。

## LLaMA-Factory 启动失败

症状：训练命令启动后很快退出，stderr 有 import、命令不存在或配置字段错误。

重点看：

- `logs/training/*stderr*`
- `logs/training/*stdout*`
- `logs/training/*events*`
- `reports/training_env_report.md`

安全处理：

- 先确认环境检查报告。
- 再确认 `configs/sft_qlora_qwen3_4b.yaml` 或 `configs/sft_qlora_qwen3_4b_smoke_6144.yaml`。
- 不要改完多个配置再试。一次只改一个变量，方便回溯。

## 生成结果为空或太短

症状：`generated.jsonl` 存在，但正文为空、很短或明显被截断。

通常原因：

- `--max-new-tokens` 太小。
- adapter 质量不足。
- prompt 或 eval card 异常。
- 推理过程被中断。

重点看：

- `outputs/sft_smoke/generated.jsonl`
- `outputs/sft_smoke/metrics.jsonl`
- `reports/sft_smoke_eval_report.md`
- `reports/stage4_1_quality_eval_budget_report.md`

安全处理：

- 先用固定质量子集做长生成评测。
- 不要只因为命令成功就扩大训练规模。

## Scoring 或报告数量对不上

症状：评分脚本提示 sample id 缺失、重复或生成结果和评测卡不匹配。

先确认三类文件使用同一批卡：

```powershell
python scripts/score_outputs.py --cards data_cards/eval_execution_cards_50.jsonl --outputs outputs/sft_smoke/generated.jsonl --output outputs/sft_smoke/metrics.jsonl
python scripts/evaluate_outputs.py --scores outputs/sft_smoke/metrics.jsonl --report reports/sft_smoke_eval_report.md --title "SFT Smoke Eval Report"
```

安全处理：

- 不要混用 20 卡、50 卡、quality subset 的生成结果和 metrics。
- 用文件名明确区分 full50、subset、1024 token 等运行。

## Agent review 产物对不上

症状：智能体审阅提示 reviews、votes、summary 或 metrics id 不一致。

重点看：

- `scripts/run_agent_review.py` 的参数。
- 输入的 cards、outputs、metrics 是否来自同一批样本。
- 输出的 review、votes、summary 是否被上一次运行覆盖。

安全处理：

- 每次审阅使用同一套 sample id。
- 出现重复或缺失 id 时，先修输入文件，不要改审阅聚合结果。
```

- [ ] **Step 4: Verify maintainer docs**

Run:

```powershell
$required = @("docs/code-design.zh.md", "docs/troubleshooting.zh.md")
$missing = $required | Where-Object { -not (Test-Path $_) }
if ($missing) { throw ("missing docs: " + ($missing -join ", ")) }
rg -n "stage2_training.py|stage3_data_readiness.py|stage4_quality.py|agent_review.py|sft_builder.py" docs/code-design.zh.md
rg -n "FileNotFoundError|JSONL|CUDA out of memory|LLaMA-Factory|generated.jsonl|metrics.jsonl|Agent review" docs/troubleshooting.zh.md
```

Expected: command exits `0` and shows module names and troubleshooting symptoms.

- [ ] **Step 5: Commit design/troubleshooting docs**

Run:

```powershell
git add docs/code-design.zh.md docs/troubleshooting.zh.md
git commit -m "docs: add code design and troubleshooting guides"
```

Expected: commit succeeds with the two new docs created.

## Task 5: Final Linkage And Verification

**Files:**
- Modify if needed: `README.md`
- Modify if needed: `docs/index.zh.md`
- Verify: all docs created in Tasks 1-4

- [ ] **Step 1: Check every expected doc exists**

Run:

```powershell
$required = @(
    "README.md",
    "docs/index.zh.md",
    "docs/zero-start.zh.md",
    "docs/project-map.zh.md",
    "docs/pipeline-flow.zh.md",
    "docs/code-design.zh.md",
    "docs/troubleshooting.zh.md",
    "docs/glossary.zh.md"
)
$missing = $required | Where-Object { -not (Test-Path $_) }
if ($missing) { throw ("missing docs: " + ($missing -join ", ")) }
Write-Output "all beginner docs exist"
```

Expected: command exits `0` and prints `all beginner docs exist`.

- [ ] **Step 2: Check README and index route to every new doc**

Run:

```powershell
$targets = @(
    "zero-start.zh.md",
    "project-map.zh.md",
    "pipeline-flow.zh.md",
    "code-design.zh.md",
    "troubleshooting.zh.md",
    "glossary.zh.md"
)
foreach ($target in $targets) {
    $matches = rg -n $target README.md docs/index.zh.md
    if (-not $matches) { throw "missing link to $target" }
    $matches
}
```

Expected: command exits `0` and prints at least one link match for each target.

- [ ] **Step 3: Check commands point to existing scripts**

Run:

```powershell
$scripts = @(
    "scripts/ingest_raw_text.py",
    "scripts/clean_chapters.py",
    "scripts/split_train_eval.py",
    "scripts/build_style_contract.py",
    "scripts/build_chapter_cards.py",
    "scripts/build_sft_dataset.py",
    "scripts/check_stage3_data_readiness.py",
    "scripts/check_local_model.py",
    "scripts/check_training_env.py",
    "scripts/run_sft_smoke.py",
    "scripts/check_adapter.py",
    "scripts/run_eval_inference.py",
    "scripts/score_outputs.py",
    "scripts/evaluate_outputs.py",
    "scripts/build_eval_quality_subset.py",
    "scripts/build_stage4_quality_report.py",
    "scripts/run_agent_review.py",
    "scripts/run_oom_probe.py"
)
foreach ($script in $scripts) {
    if (-not (Test-Path $script)) { throw "missing script $script" }
}
Write-Output "all documented scripts exist"
```

Expected: command exits `0` and prints `all documented scripts exist`.

- [ ] **Step 4: Search for unfinished markers in the beginner docs**

Run:

```powershell
$pattern = ("TO" + "DO|TB" + "D|FIX" + "ME|" + ([char]0x5F85) + ([char]0x8865))
$docs = @(
    "README.md",
    "docs/index.zh.md",
    "docs/zero-start.zh.md",
    "docs/project-map.zh.md",
    "docs/pipeline-flow.zh.md",
    "docs/code-design.zh.md",
    "docs/troubleshooting.zh.md",
    "docs/glossary.zh.md"
)
$matches = rg -n $pattern $docs
if ($matches) {
    $matches
    throw "unfinished marker found"
}
Write-Output "no unfinished markers found"
```

Expected: command exits `0` and prints `no unfinished markers found`.

- [ ] **Step 5: Run full tests**

Run:

```powershell
python -m pytest
```

Expected: command exits `0` with no failed tests.

- [ ] **Step 6: Inspect git diff**

Run:

```powershell
git status --short
git diff --stat
```

Expected: output shows only documentation changes that are not already committed, or an empty worktree if every prior task was committed.

- [ ] **Step 7: Commit final linkage fixes if any were needed**

If Task 5 changed `README.md` or `docs/index.zh.md`, run:

```powershell
git add README.md docs/index.zh.md
git commit -m "docs: finalize beginner documentation links"
```

Expected: commit succeeds if there were linkage edits. If no files changed, do not create an empty commit.

## Self-Review Checklist

- Spec coverage: the plan implements the approved docs system with README, index, zero-start guide, project map, pipeline flow, code design guide, troubleshooting guide, and glossary.
- Scope: the plan changes documentation only and does not alter training, inference, scoring, or review logic.
- Command consistency: every command in the plan references scripts that exist in the repository.
- Link consistency: Task 5 verifies README and index links to all new beginner docs.
- Verification: the plan includes existence checks, content checks, unfinished-marker checks, and `python -m pytest`.
