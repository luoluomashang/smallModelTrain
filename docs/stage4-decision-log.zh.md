# Stage 4 决策日志

- 日期: 2026-06-19
- 范围: chapter card 修复后的 Stage 4 repaired 50-sample real-data smoke loop。

## 决策

不扩展到 100/500 训练。

Stage 4 基础设施闭环已经证明: data -> SFT -> smoke training -> adapter check -> eval inference -> scoring report。

质量闭环尚未证明: budgetized eval 的生成预算对 2000-2500 字硬门槛明显过短, 且 12 个样本仍存在 outline leak。

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
- training data: repaired SFT data
- actual retry: default `cutoff_len: 8192` OOM 后, 使用 `cutoff_len: 6144`
- total optimization steps: 4
- `train_loss`: `4.00523042678833`
- `train_runtime`: `261.1415`
- adapter check decision: `允许进入下一步`

## Eval 证据

- default 5120-token eval 运行约 7.5 分钟后停止, 当时 GPU 活跃但完成行数为 0。
- 根因判断: smoke eval 预算对本机过高, 不是 adapter load failure。
- launcher/worker 已修复: stream progress, incremental row writes, explicit `--max-new-tokens`。
- 已完成 budgetized real eval: 50 张 real eval cards, adapter `outputs/sft_smoke`, `max_new_tokens: 256`。
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
- 完成上述质量闭环后, 再考虑 100-sample expansion。

## 证据产物

- `reports/stage3_data_readiness_report.md`
- `outputs/sft_smoke/train_results.json`
- `reports/sft_smoke_report.md`
- `outputs/sft_smoke/generated.jsonl`
- `outputs/sft_smoke/metrics.jsonl`
- `reports/sft_smoke_eval_report.md`
