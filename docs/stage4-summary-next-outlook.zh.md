# 第四阶段总结与下一阶段前瞻

- 日期: 2026-06-19
- 当前代码位置: `main`
- 当前阶段结论: 第四阶段基础设施闭环已经跑通, 质量闭环尚未跑通。

## 结论摘要

第四阶段已经证明: 真实数据可以进入 SFT 数据集, QLoRA smoke training 可以在本机完成, LoRA adapter 可以落盘并通过检查, eval inference 可以用真实 adapter 生成 50 条结果, scoring/reporting 链路可以产出可复查报告。

第四阶段还没有证明: 当前模型输出已经满足 2000-2500 中文汉字的质量门槛, 或者可以直接扩展到 100/500 条训练。`--max-new-tokens 256` 的 eval 只证明推理与评分基础设施可用, 不代表文学质量通过。

因此当前决策是: 不扩展到 100/500。下一阶段应先做质量评估闭环, 再考虑扩样。

## 已完成内容

- 建立章节卡生成与校验逻辑, 统一 `chapter_structure` 为 `step`、`name`、`goal`、`estimated_chars`。
- SFT prompt 渲染改为严格结构校验, 非 canonical 结构会在写训练数据前失败。
- `build_sft_dataset.py` 支持同步生成 LLaMA-Factory `dataset_info.json`。
- Stage 3 readiness 会阻断畸形章节卡, 并区分 source leak 与普通 render error。
- eval worker 支持逐条写入 JSONL, launcher 支持实时进度输出。
- eval inference 增加 `--max-new-tokens` 显式预算参数, 生成结果会记录实际预算。
- 新增第四阶段决策日志与 smoke eval runbook, README 已标注 Stage 4 当前路径和旧 Stage 2 legacy 段落。

## 关键证据

数据 readiness:

- decision: `ready_for_stage4_smoke_training`
- 原始文本文件数: 2
- `chapters_raw`: 2041
- `chapters`: 1797
- `split`: 1797
- `train`: 1747
- `eval split`: 50
- chapter cards: 50
- eval cards: 50
- SFT samples: 50
- card schema/render/source-leak blockers: 无

Smoke training:

- local model: `E:\models\Qwen3-4B-Instruct-2507`
- training data: `data_sft/sft_chapter_v1.jsonl`
- adapter: `outputs/sft_smoke`
- default `cutoff_len: 8192` OOM
- 成功 retry 使用 `cutoff_len: 6144`
- total optimization steps: 4
- `train_loss`: `4.00523042678833`
- `train_runtime`: `261.1415`
- adapter check: `允许进入下一步`

Eval smoke:

- default quality eval with `max_new_tokens: 5120` 在约 7.5 分钟后仍为 0 completed rows, 因 smoke 预算过高而停止。
- budgetized smoke eval 使用 `max_new_tokens: 256` 完成 50 条真实 eval cards。
- generated rows: 50
- metrics rows: 50
- generated params 全部记录 `max_new_tokens: 256`
- average Chinese chars: `298.18`
- hard gate pass: `0/50`
- failures: `length_short: 50`, `outline_leak: 12`

## 残余风险

- 成功训练依赖 `outputs/sft_smoke_retry_6144.yaml`, 但 `outputs/` 被 git ignore。下一阶段需要把等价 6144 smoke config 固化到可追踪路径, 或明确复建流程。
- 256-token eval 的 `length_short: 50` 是预算化 smoke 的预期副作用, 不能用作模型质量结论。
- `outline_leak: 12` 仍需人工检查。需要区分是 prompt 诱导、输出格式问题、截断副作用, 还是 scoring 规则过敏。
- SFT/Readiness 已能阻断畸形结构, 但 invalid CLI path 仍可能以 Python traceback 形式失败。后续如果面向人工频繁运行, 可以补友好错误报告。
- source leak 分类目前基于错误前缀。若后续渲染校验继续增长, 建议改为自定义异常类型。

## 下一阶段定位

建议下一阶段先命名为 `Stage 4.1 Quality Eval Hardening`, 而不是直接进入 100-sample training。

阶段目标:

- 固化可复现的 smoke training 配置。
- 找到可在本机完成长生成的 eval 预算。
- 判断模型在长生成条件下是否能接近 2000-2500 中文汉字目标。
- 修复或解释 outline leak。
- 用 full 50 long eval 证明质量门槛, 再决定是否扩到 100。

## 建议任务顺序

1. 固化 6144 smoke config

   新增 checked-in config, 例如 `configs/sft_qlora_qwen3_4b_smoke_6144.yaml`, 内容等价于成功 retry 的 `cutoff_len: 6144` 配置。后续 smoke training 不再依赖 ignored `outputs/` 文件。

2. 做 long-generation subset 预算扫描

   从 `data_cards/eval_cards_50.jsonl` 固定抽取 5-10 条作为 quality subset, 分别尝试 `max_new_tokens` 1024、2048、4096 或更高。记录每档耗时、显存、完成行数、平均中文字符数、失败类型。

3. 分析 outline leak 样本

   对 12 条 `outline_leak` 样本抽样人工查看, 只记录问题类型和修复建议, 不把原文正文贴进文档。优先判断是否需要调整 prompt 的输出要求、后处理截断规则或 scoring marker。

4. 跑 full 50 long eval

   在 subset 预算稳定后, 对 50 条 eval cards 跑一次 long eval。只有完整生成、完整评分、报告可复查后, 才进入扩样讨论。

5. 决定是否进入 100-sample expansion

   如果 full 50 long eval 达到约定 gate, 才生成 100 条训练卡/SFT 样本并进入下一轮 smoke training。500 样本继续 out of scope, 直到 100-sample evidence 通过。

## 下一阶段验收门槛

进入 100 样本前至少需要:

- long-generation subset 完成, 无 OOM 或中途无产物卡死。
- generation budget 能支撑 2000-2500 中文汉字目标。
- full 50 long eval 完成, `generated=50`, `metrics=50`。
- `outline_leak` 降到可接受阈值, 优先目标为 0；如非 0, 必须有明确原因和处理决定。
- hard gate 是否通过要基于 long eval 判断, 不再使用 256-token smoke eval 的长度结果。
- 决策日志更新, 明确是否允许扩到 100。

## 建议产物

- `configs/sft_qlora_qwen3_4b_smoke_6144.yaml`
- `data_cards/eval_cards_quality_subset.jsonl`
- `outputs/sft_smoke/generated_subset_long.jsonl`
- `outputs/sft_smoke/metrics_subset_long.jsonl`
- `reports/sft_smoke_eval_subset_long_report.md`
- `reports/stage4_1_quality_eval_budget_report.md`
- `docs/stage4-1-quality-eval-decision.zh.md`

## 参考文档

- [Stage 4 决策日志](stage4-decision-log.zh.md)
- [Stage 4 Smoke Eval 指南](stage4-smoke-eval-guide.zh.md)
