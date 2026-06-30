# Stage 5E 受控实验与效率指南

Stage 5E 只能在 Stage 5D 入场检查通过，并且完整 `python -m pytest -q` 通过后开始。它的第一批产物是 experiment manifest、paired eval report 和 dry-run experiment matrix，用来记录受控实验边界、配对评测结果和候选训练命令。

Stage 5E 是控制面脚手架，不会自动运行 DPO、SimPO、ORPO、KTO 或 reward model training。Stage 5D 的 preference rows 仍只是候选数据，不表示已经做过 preference optimization。

## 入场检查

先运行 Stage 5E 入场检查：

```powershell
python scripts/check_stage5e_entry.py --summary reports/stage5d_review_summary.json --review-records data_review/stage5d_review_records.jsonl --revisions data_review/stage5d_revisions.jsonl --rejection-sampling-rows data_sft/stage5d_rejection_sampling_sft.jsonl --preference-rows data_pref/stage5d_same_plot_preference.jsonl --generation-records outputs/stage5d_generation_records.jsonl --output reports/stage5e_entry_check.json
```

然后运行完整测试：

```powershell
python -m pytest -q
```

只有 `reports/stage5e_entry_check.json` 中 `"passed": true`，并且完整 pytest 通过后，才允许构建 Stage 5E 实验产物。

## 实验 manifest

必备输入：

- Stage 5E gate JSON，例如 `reports/stage5e_entry_check.json`。
- 本轮候选训练 config，例如 `configs/sft_qlora_qwen3_4b_smoke_6144.yaml`。
- Stage 5D rejection-sampling SFT 数据，例如 `data_sft/stage5d_rejection_sampling_sft.jsonl`。
- 固定 eval cards，例如 `data_cards/eval_execution_cards_50.jsonl`。
- 配对 judgments/metrics 的计划占位或已产物路径，例如 `reports/stage5e_paired_eval_report.md`。

构建 experiment manifest：

```powershell
python scripts/build_stage5e_experiment_manifest.py --experiment-id stage5e_lr_probe_001 --baseline-run-id stage5d_baseline --candidate-run-id stage5e_candidate_lr --primary-variable-name learning_rate --primary-baseline-value 1e-4 --primary-candidate-value 8e-5 --controlled-variable seed=20260628=20260628 --stage5e-entry-check reports/stage5e_entry_check.json --artifact config=configs/sft_qlora_qwen3_4b_smoke_6144.yaml --artifact sft_dataset=data_sft/stage5d_rejection_sampling_sft.jsonl --artifact eval_cards=data_cards/eval_execution_cards_50.jsonl --paired-eval-json '{"planned_report":"reports/stage5e_paired_eval_report.md","planned_judgments":"data_review/stage5e_paired_judgments.jsonl","planned_metrics":"outputs/stage5e/candidate_metrics.jsonl"}' --output reports/stage5e_experiment_manifest.json
```

`--artifact name=path` 会把 config、SFT dataset、eval cards 等实验输入写入 manifest，并记录文件 hash。Manifest 只能记录一个 primary variable 的变化；controlled variables 必须在 baseline 和 candidate 中保持相同。比如本轮改 learning rate，就不能同时改 base model、LoRA rank、dataset、eval cards 或 generation params。

## Dry-Run 实验矩阵

先用 dry-run 写出候选命令：

```powershell
python scripts/run_experiment_matrix.py --manifest reports/stage5e_experiment_manifest.json --output reports/stage5e_experiment_commands.jsonl --dry-run
```

Dry-run 只写出 experiment commands，不会启动训练，也不会运行 preference optimization。使用 `--dry-run` 时，生成的 `run_sft_train.py` 命令也会带上 `--dry-run`，并使用实际支持的参数：`--config`、`--output-dir outputs/stage5e/<candidate_run_id>`，以及 manifest 中存在时的 `--sft-dataset` 和 `--eval-cards`。真正执行前要人工确认 manifest 中的 primary variable 和 controlled variables 仍然符合单变量实验边界。

## Paired Eval 报告

构建 paired eval summary 和 Markdown 报告：

```powershell
python scripts/build_paired_eval_report.py --baseline-metrics outputs/stage5e/baseline_metrics.jsonl --candidate-metrics outputs/stage5e/candidate_metrics.jsonl --judgments data_review/stage5e_paired_judgments.jsonl --summary-output reports/stage5e_paired_eval_summary.json --report-output reports/stage5e_paired_eval_report.md
```

Paired eval report 记录 candidate 相对 baseline 的 win/loss/tie 和 regression samples。它不是最终作者审阅，也不能替代 sealed evaluation、作者接受率或人工 prose 判断。

## 边界

- 每个实验只能改变一个 primary variable。
- 不能用 sanitized-only artifact 推进实验或报告。
- 不能把 rule projection 当作 human review。
- Stage 5D preference rows 不是已经训练过的 preference optimization。
- Efficiency win 不能忽略 plot execution、author acceptance 和 regressions。
