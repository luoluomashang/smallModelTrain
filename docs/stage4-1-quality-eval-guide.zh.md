# Stage 4.1 Quality Eval Hardening 指南

## 目标

Stage 4.1 不扩展训练样本数。它先把 50-card control set 的质量评估闭环变得可复现：固定 6144 cutoff smoke 配置，抽取小规模 quality subset，扫描长生成预算，汇总长度、失败类型、outline leak 和缺失行证据，再决定是否进入 full 50 long eval。

## 前置条件

- Stage 3 readiness 报告为 `ready_for_stage4_smoke_training`。
- `data_cards/eval_cards_50.jsonl`、`data_sft/sft_chapter_v1.jsonl` 已存在。
- 本地模型目录为 `E:\models\Qwen3-4B-Instruct-2507`。
- `outputs/sft_smoke` 中已有可用 adapter，或准备重新运行 smoke training。
- 不把 generated 正文复制进文档；报告只记录计数、ID、marker 和决策。

## 1. 使用可追踪 6144 Smoke 配置

默认 `cutoff_len: 8192` 曾触发 OOM。Stage 4.1 使用可追踪配置复现成功 retry，不再依赖 ignored 的 `outputs/sft_smoke_retry_6144.yaml`。

```powershell
python scripts/run_sft_smoke.py --config configs/sft_qlora_qwen3_4b_smoke_6144.yaml --eval-cards data_cards/eval_cards_50.jsonl
```

训练完成后检查 adapter：

```powershell
python scripts/check_adapter.py --adapter-dir outputs/sft_smoke --report reports/sft_smoke_report.md --title "SFT Smoke Adapter Check"
```

## 2. 构建固定 Quality Subset

如果已有 `outputs/sft_smoke/metrics.jsonl`，subset 会优先包含旧 eval 中的 `outline_leak` 样本，再按原 eval 顺序补齐。没有 metrics 时，则按 eval cards 顺序取前 N 条。

```powershell
python scripts/build_eval_quality_subset.py --cards data_cards/eval_cards_50.jsonl --metrics outputs/sft_smoke/metrics.jsonl --output data_cards/eval_cards_quality_subset.jsonl --count 8
```

## 3. 跑 Long-Generation 预算扫描

先从 1024 token 开始。若无 OOM、无长时间 0 completed rows，再尝试 2048、4096 或更高预算。每一档都保留独立 generated 和 metrics 文件。

```powershell
python scripts/run_eval_inference.py --cards data_cards/eval_cards_quality_subset.jsonl --adapter-dir outputs/sft_smoke --output outputs/sft_smoke/generated_subset_1024.jsonl --model-name sft_smoke_subset_1024 --max-new-tokens 1024
python scripts/score_outputs.py --cards data_cards/eval_cards_quality_subset.jsonl --outputs outputs/sft_smoke/generated_subset_1024.jsonl --output outputs/sft_smoke/metrics_subset_1024.jsonl
python scripts/build_stage4_quality_report.py --cards data_cards/eval_cards_quality_subset.jsonl --generated outputs/sft_smoke/generated_subset_1024.jsonl --metrics outputs/sft_smoke/metrics_subset_1024.jsonl --report reports/stage4_1_quality_eval_budget_report.md --title "Stage 4.1 Quality Eval Budget Report"
```

更高预算只替换文件名和 `--max-new-tokens`：

```powershell
python scripts/run_eval_inference.py --cards data_cards/eval_cards_quality_subset.jsonl --adapter-dir outputs/sft_smoke --output outputs/sft_smoke/generated_subset_2048.jsonl --model-name sft_smoke_subset_2048 --max-new-tokens 2048
python scripts/score_outputs.py --cards data_cards/eval_cards_quality_subset.jsonl --outputs outputs/sft_smoke/generated_subset_2048.jsonl --output outputs/sft_smoke/metrics_subset_2048.jsonl
python scripts/build_stage4_quality_report.py --cards data_cards/eval_cards_quality_subset.jsonl --generated outputs/sft_smoke/generated_subset_2048.jsonl --metrics outputs/sft_smoke/metrics_subset_2048.jsonl --report reports/stage4_1_quality_eval_budget_report.md --title "Stage 4.1 Quality Eval Budget Report"
```

## Agent Review Gate

Stage 4.1 不再只依赖 rule metrics。`score_outputs.py` 完成后，必须继续运行 agent review coordinator，验证正文是否执行了外部执行卡中的结构、冲突、爽点和章末钩子。

```powershell
python scripts/run_agent_review.py `
  --cards data_cards/eval_cards_quality_subset.jsonl `
  --outputs outputs/sft_smoke/generated_subset_2048.jsonl `
  --metrics outputs/sft_smoke/metrics_subset_2048.jsonl `
  --target-platform hybrid_fanqie_qidian `
  --backend mock `
  --output outputs/sft_smoke/agent_reviews.jsonl `
  --votes-output outputs/sft_smoke/agent_votes.jsonl `
  --report reports/stage4_agent_review_report.md
```

真实三类 reviewer 审核完成后，用 `--reviews-import` 导入 JSONL 审核行。只有 rule gate 与 agent gate 都通过，且 blocker arbitration 已记录并处理，Stage 4.1 才能作为扩展训练证据。

## 4. 当前 256-token Baseline 报告

已有 Stage 4 的 256-token eval 只证明基础设施，不证明质量。可以用同一个报告脚本生成 baseline 证据，确认当前决策仍然是长度阻断。

```powershell
python scripts/build_stage4_quality_report.py --cards data_cards/eval_cards_50.jsonl --generated outputs/sft_smoke/generated.jsonl --metrics outputs/sft_smoke/metrics.jsonl --report reports/stage4_1_quality_eval_budget_report.md --title "Stage 4.1 Quality Eval Budget Report"
```

预期 baseline 决策为 `blocked_length_short`。

## 5. 晋级边界

只有满足以下条件，才进入 full 50 long eval：

- quality subset 完整生成，`generated_rows == expected_rows`。
- quality subset 完整评分，`metrics_rows == expected_rows`。
- 生成预算能接近 2000-2500 中文汉字目标。
- `outline_leak` 已降到可接受阈值；优先目标为 0，如非 0，必须记录明确原因。
- 没有 OOM、launcher failure、长时间 0 completed rows 或中途无产物卡死。

只有 full 50 long eval 通过约定 gate 后，才讨论 100-sample expansion。500-sample expansion 继续 out of scope，直到 100-sample evidence 通过。
