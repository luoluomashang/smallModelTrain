# 两阶段实现审计报告

## 1. 审计结论摘要

本轮审计以两个阶段计划文件为准，对 `src`、`scripts`、`tests`、`README` 的实现逐项核对。当前实现总体符合 Stage 1 数据管线与 Stage 2 训练执行前置/监督/诊断设计；未发现把 dry-run 描述成真实训练完成的代码路径，也未发现用空函数或固定假结果冒充训练、推理、评分成功的实现。

需要明确的是：自动化测试覆盖的是数据处理、命令构造、报告生成、错误分类和诊断框架；真实 QLoRA 训练、真实 GPU 推理、真实 adapter 质量仍必须在本地训练环境中按 README 命令执行后确认。这属于计划边界，不属于已完成真实训练的证明。

本报告审阅的主要计划与说明文件包括 `docs/superpowers/plans/2026-06-17-qwen3-qlora-stage1-pipeline.md`、`docs/superpowers/plans/2026-06-18-qwen3-qlora-stage2-training-execution.md` 和 `docs/stage1-pipeline-guide.zh.md`。实现核对覆盖相关 `src/small_model_train/` 模块、`scripts/` 入口、配置、测试清单与 README 命令链路。

## 2. Stage 1 计划符合性矩阵

| 计划任务 | 对应实现文件 | 对应测试文件 | 结论 | 简要说明 |
| --- | --- | --- | --- | --- |
| 项目脚手架与共享文本工具 | `pyproject.toml`, `src/small_model_train/io_utils.py`, `text_utils.py` | `tests/test_text_utils.py`, `tests/test_pipeline_smoke.py` | 符合 | 已有包结构、pytest 配置、JSONL 和文本工具。 |
| 原稿清洗与切章 | `chapter_splitter.py`, `scripts/clean_chapters.py`, `scripts/ingest_raw_text.py` | `tests/test_chapter_splitter.py`, `tests/test_pipeline_smoke.py` | 符合 | 支持文本读取、清洗、章节抽取与最小/最大长度过滤。 |
| 确定性 train/eval 切分 | `dataset_split.py`, `scripts/split_train_eval.py` | `tests/test_dataset_split.py` | 符合 | 通过 seed 固定 eval，避免后续比较漂移。 |
| 风格画像与风格契约 | `style_profile.py`, `scripts/build_style_contract.py` | `tests/test_style_profile.py` | 符合 | 输出统计画像和 Markdown 风格约束。 |
| SFT 数据构造 | `sft_builder.py`, `scripts/build_sft_dataset.py` | `tests/test_sft_builder.py` | 符合 | 从章节卡和章节正文构造 instruction/input/output，并检查 source_text 泄漏。 |
| 规则评分与 AI 味检测 | `scoring.py`, `scripts/score_outputs.py`, `scripts/detect_ai_trace.py` | `tests/test_scoring.py` | 符合 | 规则评分覆盖长度、情节覆盖、重复和 AI 套话命中。 |
| 偏好候选构造 | `preference_builder.py`, `scripts/build_preference_dataset.py` | `tests/test_preference_builder.py` | 合理边界 | 只准备候选，不声明已训练 reward/DPO。 |
| Markdown 评测报告 | `reporting.py`, `scripts/evaluate_outputs.py` | `tests/test_reporting.py` | 符合 | 汇总规则分数和失败标签，生成可读报告。 |
| QLoRA 与推理配置 | `configs/sft_qlora_qwen3_4b.yaml`, `configs/infer_eval_qwen3_4b.yaml` | Stage 2 配置测试间接覆盖 | 符合 | 配置作为 Stage 2 输入存在，真实训练在 Stage 2 执行。 |
| 端到端 smoke test | `tests/test_pipeline_smoke.py` | `tests/test_pipeline_smoke.py` | 符合 | 覆盖 Stage 1 主要数据流。 |

## 3. Stage 2 计划符合性矩阵

| 计划任务 | 对应实现文件 | 对应测试文件 | 结论 | 简要说明 |
| --- | --- | --- | --- | --- |
| 本地模型文件与 transformers 加载检查 | `stage2_model_check.py`, `scripts/check_local_model.py` | `tests/test_stage2_model_check.py` | 符合 | 检查必需文件、safetensors 分片和可选 transformers 加载。 |
| 训练环境、CUDA、依赖与 VRAM 检查 | `stage2_env_check.py`, `scripts/check_training_env.py` | `tests/test_stage2_env_check.py` | 符合 | 收集包版本、CUDA 可用性、nvidia-smi 与显存建议。 |
| 配置快照与 LLaMA-Factory 命令构造 | `stage2_config.py`, `stage2_training.py` | `tests/test_stage2_config.py`, `tests/test_stage2_training.py` | 符合 | 训练前写出快照，再构造 `llamafactory-cli train` 命令。 |
| 训练事件日志、GPU 采样和错误分类 | `stage2_monitoring.py`, `stage2_training.py` | `tests/test_stage2_monitoring.py`, `tests/test_stage2_training.py` | 符合 | 保存事件、GPU 样本、stdout/stderr 和失败摘要。 |
| smoke/full 训练启动器 | `scripts/run_sft_smoke.py`, `scripts/run_sft_train.py` | `tests/test_stage2_training.py` | 符合 | dry-run 与真实 subprocess 路径分离，真实执行保留 exit code。 |
| adapter 静态校验 | `stage2_adapter.py`, `scripts/check_adapter.py` | `tests/test_stage2_adapter.py` | 符合 | 检查 adapter config、权重文件和 safetensors header。 |
| OOM probe 执行框架 | `stage2_oom_probe.py`, `scripts/run_oom_probe.py`, `scripts/stage2_oom_probe_worker.py` | `tests/test_stage2_oom_probe.py` | 符合 | 父进程逐个启动 worker，并为每个 probe 记录 stdout、stderr、event、GPU 日志。 |
| 固定 eval 推理和 Stage 1 评分衔接 | `stage2_inference.py`, `scripts/run_eval_inference.py`, `scripts/stage2_eval_worker.py`, `scripts/score_outputs.py` | `tests/test_stage2_inference.py`, `tests/test_scoring.py` | 合理边界 | 推理入口存在，真实生成依赖本地 GPU 与 adapter；评分复用 Stage 1 规则。 |
| README Stage 2 命令序列 | `README.md` | 手动命令链路审查 | 符合 | README 先检查模型/环境，再 dry-run、smoke、OOM probe、full train、eval。 |

## 4. 降级、掩饰、虚假实现专项审查

未发现的问题：

- 未发现真实训练函数直接返回成功而不启动 subprocess。
- 未发现 OOM probe 只写静态报告；默认入口会执行 probes，`--dry-run` 才只写计划。
- 未发现训练失败被吞掉后仍作为成功返回；非零 exit code 会进入 failure report。
- 未发现 Stage 1 评分用固定高分冒充评测；评分由输入文本和章节卡规则计算。
- 未发现 adapter 检查只看目录存在；它会检查配置、权重和 safetensors header。

需要如实说明的边界：

- dry-run 是命令和配置预演，不代表训练成功。
- 单元测试不会加载真实 4-bit Qwen3，也不会跑真实 GPU 训练。
- Stage 1 章节卡需要额外准备，系统不自动从正文生成完整章节卡。
- 固定 eval 推理入口存在，但真实输出质量要等本地 adapter 生成后评分。

补充核对结论：

- `stage2_training.py` 的错误分类由调用点把已捕获的 stdout/stderr 合并后传入 classifier；报告不应表述成 classifier 独立读取两个流。
- OOM probe 与训练监督会保留父进程捕获到的事件、GPU 样本和输出日志；如果子进程被硬杀且输出未 flush，仍只能保留已经捕获到的证据。

## 5. 已知合理边界

- Stage 1 的目标是数据资产准备和可复查评测闭环，不是模型训练完成证明。
- Stage 2 的自动化代码覆盖启动前检查、命令构造、日志保存、失败分类、adapter 静态检查和 eval 推理入口，不覆盖真实长时间训练的稳定性。
- `scripts/run_eval_inference.py --dry-run` 会写预览式生成行，便于检查 schema 和 prompt；这不是 adapter 真实输出。
- 规则评分适合发现长度、覆盖、重复、格式泄漏和常见套话问题，但不能证明小说质量、人设稳定性或章节节奏。
- `build_preference_dataset.py` 只把失败样本整理成 rejected 候选；没有实现 reward model、DPO 或偏好优化训练。

## 6. 风险清单与建议后续动作

| 级别 | 风险 | 影响 | 建议 |
| --- | --- | --- | --- |
| Important | `stage2_oom_probe_worker.py` 中 probes 5-7 将 `dataset` 指向 JSONL 路径，并将 `dataset_dir` 指向其父目录，是否完全匹配本地 LLaMA-Factory 数据集注册方式取决于本机配置。 | one-step training probe 可能因数据集解析方式失败，需要结合 stderr 判断。 | 首次执行时保留 probe 配置快照和 stderr；如确认为数据集注册问题，再增加 LLaMA-Factory dataset_info 适配任务。 |
| Minor | Stage 1 规则评分只能发现结构化问题，不能证明文学质量。 | 低分样本便于排查，高分仍需人工抽检。 | 在评测报告中保留人工复核样本。 |
| Minor | 自动测试使用小样本和模拟输出，不覆盖长章节真实显存压力。 | 测试通过不等于真实训练不会 OOM。 | 按 README 先跑 smoke 和 OOM probe，再 full train。 |

## 7. 验证命令与结果

| 命令 | 结果 |
| --- | --- |
| Task 1 baseline full tests：`python -m pytest -q` | 既有证据：implementer `128 passed in 1.07s`；spec reviewer `128 passed in 1.46s`；quality reviewer `128 passed in 1.37s`。 |
| Task 2 targeted Stage 1 tests | 既有证据：implementer `53 passed in 1.11s`；reviewers `53 passed in 1.08s`。 |
| Task 3 targeted Stage 2 tests | 既有证据：`46 passed in 0.24s`；fixup 后 `46 passed in 0.25s`。 |
| Task 1 red-flag scan over `docs src scripts` | 既有证据：Task 1 时间点无命中；后续如果扫描包含本审计报告或历史计划/规格，可能出现被审查类别或计划模板词命中，应按具体文件判断。 |
| Task checks：`git diff --check` | 既有证据：任务检查中无 whitespace errors。 |
| `python -m pytest -q` | Fresh check after report creation: `128 passed in 1.04s`。 |
| `$patterns = @("to" + "do", "tb" + "d", "place" + "holder", "fa" + "ke", "st" + "ub", "not" + "implemented", "pass$"); rg -n -i ($patterns -join "|") docs src scripts` | Fresh check exit code 1, no output；当前 `docs src scripts` 无命中。 |
| `git diff --check` | Fresh check exit code 0, no output；未发现 whitespace errors。 |
| `git status --short` | Fresh pre-commit check: only `?? docs/two-stage-implementation-audit.zh.md`。 |
