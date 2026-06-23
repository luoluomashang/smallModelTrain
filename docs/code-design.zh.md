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
