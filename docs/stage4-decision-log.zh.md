# Stage 4 决策日志

- 日期: 2026-06-19
- 范围: chapter card 修复后的 Stage 4 repaired 50-sample real-data smoke loop。

## 决策

不扩展到 100/500 训练。500-sample expansion 在 100-sample evidence 通过之前继续保持 out of scope。

Stage 4 基础设施闭环已经证明: data -> SFT -> smoke training -> adapter check -> eval inference -> scoring report。

质量闭环尚未证明: budgetized eval 的生成预算对 2000-2500 字硬门槛明显过短, 且 12 个样本仍存在 outline leak。

## Agent Review Acceptance Gate

- design spec: `docs/superpowers/specs/2026-06-21-agent-review-acceptance-system-design.md`
- implementation plan: `docs/superpowers/plans/2026-06-21-male-webnovel-agent-review-acceptance.md`
- scope: 评估正文执行质量，不改变当前训练方法；外部执行卡负责情节、结构、冲突、爽点和章末钩子。
- expansion 前置: execution-card schema guard、deterministic quality rules、agent majority gate，以及 blocker votes 的人工仲裁记录与处理。

## 数据就绪证据

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
- warning: SFT dataset has 50 rows, below preferred 100

## Smoke Training 证据

- local model: `E:\models\Qwen3-4B-Instruct-2507`
- adapter: `outputs/sft_smoke`
- training data: repaired SFT data, `data_sft/sft_chapter_v1.jsonl`
- actual retry: default `cutoff_len: 8192` OOM 后, 使用 `cutoff_len: 6144`
- total optimization steps: 4
- `train_loss`: `4.00523042678833`
- `train_runtime`: `261.1415`
- adapter check decision: `允许进入下一步`

## Eval 证据

- default eval with `max_new_tokens: 5120` 运行约 7.5 分钟后停止, 当时 GPU 活跃但完成行数为 0。
- 根因判断: smoke eval 预算对本机过高, 不是 adapter load failure。
- launcher/worker 已修复: stream progress, incremental row writes, explicit `--max-new-tokens`。
- 已完成 budgetized real eval: eval cards `data_cards/eval_cards_50.jsonl`, adapter `outputs/sft_smoke`, `max_new_tokens: 256`。
- generated rows: 50
- metrics rows: 50
- 所有 generated rows 均记录 `max_new_tokens: 256`
- average Chinese chars: `298.18`
- hard gate pass: `0/50`
- failures: `length_short: 50`, `outline_leak: 12`

## 下一步

- 保留 50-sample set 作为 control baseline。
- 修复并测量 quality eval 的 generation budget; 先用更小 subset 跑更大的 `max_new_tokens`, 再考虑 full 50 long generation。
- 检查 outline leak 样本以及 prompt/output formatting。
- promotion criteria: long-generation subset 成功完成; generation budget 足以支撑 2000-2500 个中文汉字; outline leak 降到可接受阈值, 优先目标为 0, 如非 0 需明确理由; full 50 long eval 通过约定 gate。
- 只有上述质量闭环通过后, 才考虑 100-sample expansion; 只有 100-sample evidence 也通过后, 才重新讨论 500-sample expansion。

## 证据产物

- `reports/stage3_data_readiness_report.md`
- `outputs/sft_smoke/train_results.json`
- `reports/sft_smoke_report.md`
- `outputs/sft_smoke/generated.jsonl`
- `outputs/sft_smoke/metrics.jsonl`
- `reports/sft_smoke_eval_report.md`
