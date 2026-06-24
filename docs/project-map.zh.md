# 项目目录地图

这页解释每个目录和关键文件是干什么的。你可以先把项目想成一条流水线：原始数据进来，脚本处理它，核心代码负责规则，结果写到数据目录、输出目录、报告目录和日志目录。

## 根目录

- `README.md`：项目入口，负责告诉你从哪里开始。
- `pyproject.toml`：Python 项目配置，包含测试路径和开发依赖。
- `style_contract.md`：由脚本生成的风格契约 Markdown 摘要。Stage 5B 起，它用于人工审阅，formal SFT 的机器门禁源是 `data_style/` 中的 StyleContract JSON。
- `style_profile.json`：旧版/可选的风格统计输出，可通过 `--profile-output` 生成；Stage 5B 默认 metrics 产物在 `data_style/style_metrics_author_main_v1.json`。
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
- `data_style/`：Stage 5B 生成的 StyleContract JSON 和 style metrics。默认包括 `style_contract_author_main_v1.json` 和 `style_metrics_author_main_v1.json`。
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

如果文件来自某条脚本命令的输出、报告或日志类参数，它通常就是生成物。常见例子包括 `--output`、`--eval-output`、`--contract-output`、`--profile-output`、`--dataset-info-output`、`--output-dir`、`--report`、`--stdout-log`、`--stderr-log`、`--event-log`、`--log-dir`、`--votes-output` 和 `--summary-output`。

根目录里的生成文件也可能只在跑过命令后才出现，例如 `style_contract.md`、`style_profile.json` 和 `mlflow.db`。Stage 5B 起，风格资产的机器门禁文件在 `data_style/`，`style_contract.md` 只用于人工审阅。生成物可以删除后重建，但删除前要确认没有正在使用它的报告、决策记录或 formal SFT 绑定。
