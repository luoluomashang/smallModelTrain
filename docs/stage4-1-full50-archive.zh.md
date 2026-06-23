# Stage 4.1 Full50 Quality Eval 归档

- 日期: 2026-06-21
- adapter: `outputs/sft_smoke`
- model: `E:\models\Qwen3-4B-Instruct-2507`
- control cards: `data_cards/eval_cards_50.jsonl`
- superseded by: `docs/stage4-1-full50-manual-review.zh.md`
- 当前归档结论: 旧机器 hard gate 以 targeted retry 策略通过，但人工复核后撤销晋级结论。

## Supersession Notice

本文件保留 full50 运行事实与旧评分结果，但不再作为 100-sample expansion 的前置通过证据。

人工复核发现：输出存在补字数式语义重复、非正文残留、疑似截断/不自然收尾，并且 full50 eval 输入使用了缺少章节执行字段的 `data_cards/eval_cards_50.jsonl`。详细问题见 `docs/stage4-1-full50-manual-review.zh.md`。

当前状态应以人工复核结论为准：

`blocked_by_eval_schema_mismatch_and_semantic_repetition`

## 最终结果

- final hard gate: `50/50`
- generated rows: `50`
- metrics rows: `50`
- character count: min `2171`, max `2499`, avg `2443.68`
- repeated n-gram ratio: avg `0.0145`, max `0.0419`
- must-include coverage: min `1.0`, avg `1.0`
- hard failures: none
- outline leak: none
- soft findings: `ai_trace: 1`

这不是单次 `2560` 全量通过。本次归档结果采用两段式 eval 策略：

1. full50 baseline 使用 `max_new_tokens=2560`, `repetition_penalty=1.12`, `no_repeat_ngram_size=4`。
2. 仅对 baseline 中的 `length_short` 样本使用 `max_new_tokens=3072` 定向补跑，重复控制参数保持不变。

## 运行摘要

### Baseline Full50

- event log: `logs/training/sft_smoke_eval_full50_2560_rp112_nr4_events.jsonl`
- start: `2026-06-21T15:45:34+08:00`
- end: `2026-06-21T20:02:22+08:00`
- status: `ok`
- output: `outputs/sft_smoke/generated_full50_2560_rp112_nr4.jsonl`
- metrics: `outputs/sft_smoke/metrics_full50_2560_rp112_nr4.jsonl`
- report: `reports/stage4_1_quality_eval_full50_2560_rp112_nr4.md`
- hard gate: `44/50`
- failures: `length_short: 6`, soft `ai_trace: 1`
- hard-risk notes: no `outline_leak`, no `repetition`

### Targeted Retry

- retry cards: `data_cards/eval_cards_full50_retry_length_short.jsonl`
- event log: `logs/training/sft_smoke_eval_full50_retry_length_short_3072_rp112_nr4_events.jsonl`
- start: `2026-06-21T20:05:22+08:00`
- end: `2026-06-21T20:45:53+08:00`
- status: `ok`
- output: `outputs/sft_smoke/generated_full50_retry_length_short_3072_rp112_nr4.jsonl`
- metrics: `outputs/sft_smoke/metrics_full50_retry_length_short_3072_rp112_nr4.jsonl`
- hard gate: `6/6`
- retry character count range: `2449-2494`

Retried IDs:

- `天启校对__26fb8c26_chapter_0043`
- `天启校对__26fb8c26_chapter_0584`
- `天启校对__26fb8c26_chapter_0696`
- `天启校对__26fb8c26_chapter_0702`
- `天启校对__26fb8c26_chapter_1316`
- `龙族校对__e82774d6_chapter_0204`

### Final Merged Result

- generated: `outputs/sft_smoke/generated_full50_merged_2560_rp112_nr4_retry3072.jsonl`
- metrics: `outputs/sft_smoke/metrics_full50_merged_2560_rp112_nr4_retry3072.jsonl`
- report: `reports/stage4_1_quality_eval_full50_merged_2560_rp112_nr4_retry3072.md`
- expected/generated/metrics: `50/50/50`
- final hard gate: `50/50`
- failure counts: soft `ai_trace: 1`
- outline leak triage: none

## 判断

Stage 4.1 早期暴露的核心风险已被压住：

- output sanitizer 消除了 outline/meta-text leakage。
- `no_repeat_ngram_size=4` 后 repetition 远低于 hard threshold。
- incremental row writes 和 event logs 证明 full50 eval 可以完整追踪。

仍需保留的工程策略是 targeted retry。单次 `2560` baseline 会有少量长度方差，`3072` retry 能补齐短样本且未重新引入 repetition 或 outline leak。

## 下一门槛

Stage 4.1 full50 control evidence 可以作为下一步 100-sample expansion 的前置证据。500-sample expansion 继续保持 out of scope，直到 100-sample evidence 也通过。
