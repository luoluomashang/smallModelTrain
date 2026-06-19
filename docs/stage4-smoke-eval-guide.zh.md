# 第四阶段：真实 Smoke Training 与固定 Eval 指南

## 目标

第四阶段不是直接跳到 500 条样本。它要在修复后的 50 条真实样本上证明完整闭环：章节卡修复、SFT 重建、readiness 复查、真实 smoke training、adapter 检查、eval inference、scoring、阶段决策。

本阶段已经证明基础设施闭环可以跑通，但还没有证明质量闭环。256-token eval 只用于证明推理和评分链路，不代表输出质量通过。

## 前置条件

进入本阶段前，Stage 3 readiness 必须是 `ready_for_stage4_smoke_training`。

章节卡必须使用 canonical `chapter_structure` 项，每项包含 `step`、`name`、`goal`、`estimated_chars`。不要使用旧版节拍字段，也不要保留点号加中文冒号一类的畸形结构。

## 1. 重建章节卡与 SFT 数据

先用修复后的章节卡生成逻辑重建 50 条训练卡：

```powershell
python scripts/build_chapter_cards.py --chapters data_clean/chapters_split.jsonl --output data_cards/chapter_cards.jsonl --count 50 --min-chars 2000 --max-chars 3000
```

再重建 SFT 数据集，并同步写出 dataset info：

```powershell
python scripts/build_sft_dataset.py --cards data_cards/chapter_cards.jsonl --chapters data_clean/chapters_split.jsonl --output data_sft/sft_chapter_v1.jsonl --dataset-info-output data_sft/dataset_info.json
```

最后复查 Stage 3 readiness。只有保持 `ready_for_stage4_smoke_training` 才继续：

```powershell
python scripts/check_stage3_data_readiness.py --eval-cards data_cards/eval_cards_50.jsonl --run-smoke-dry-run
```

## 2. Smoke Training

训练前先确认本地模型和训练环境：

```powershell
python scripts/check_local_model.py --model-dir E:\models\Qwen3-4B-Instruct-2507 --report reports/model_check_report.md
python scripts/check_training_env.py --report reports/training_env_report.md
```

先做 dry run，确认命令和数据入口可读：

```powershell
python scripts/run_sft_smoke.py --eval-cards data_cards/eval_cards_50.jsonl --dry-run
```

默认真实 smoke training 命令：

```powershell
python scripts/run_sft_smoke.py --eval-cards data_cards/eval_cards_50.jsonl
```

实际证据中，默认 `cutoff_len: 8192` 发生 OOM；成功 run 使用 retry config，将 `cutoff_len` 降到 `6144`。本机当时使用的 ignored retry config 路径是 `outputs/sft_smoke_retry_6144.yaml`，可用以下命令重跑：

```powershell
python scripts/run_sft_smoke.py --config outputs/sft_smoke_retry_6144.yaml --eval-cards data_cards/eval_cards_50.jsonl
```

因为 `outputs/` 被 git ignore，未来复跑前不要假设这个文件一定存在。需要先重新生成 `outputs/sft_smoke_retry_6144.yaml`，或在明确改动范围允许时提交等价的 checked-in config，再依赖这条 retry 命令。记录报告时必须写明 retry config。不要把 resume 后出现的 `train_loss=0.0` 当作有效训练指标。

adapter 检查命令：

```powershell
python scripts/check_adapter.py --adapter-dir outputs/sft_smoke --report reports/sft_smoke_report.md --title "SFT Smoke Adapter Check"
```

## 3. Eval Inference 与 Scoring

默认质量 eval 使用 `max_new_tokens: 5120`。在本机 smoke 尝试中，5120-token eval 运行约 7.5 分钟后被停止，当时完成行数为 0。为证明 smoke 基础设施链路，显式使用 `--max-new-tokens 256`，并在报告中记录为 budgetized smoke，不要记录为质量通过。

真实 smoke eval 命令带完整日志：

```powershell
python scripts/run_eval_inference.py --cards data_cards/eval_cards_50.jsonl --adapter-dir outputs/sft_smoke --output outputs/sft_smoke/generated.jsonl --model-name sft_smoke --event-log logs/training/sft_smoke_eval_events.jsonl --stderr-log logs/training/sft_smoke_eval_stderr.log --stdout-log logs/training/sft_smoke_eval_stdout.log --max-new-tokens 256
```

scoring 命令：

```powershell
python scripts/score_outputs.py --cards data_cards/eval_cards_50.jsonl --outputs outputs/sft_smoke/generated.jsonl --output outputs/sft_smoke/metrics.jsonl
```

报告命令：

```powershell
python scripts/evaluate_outputs.py --scores outputs/sft_smoke/metrics.jsonl --report reports/sft_smoke_eval_report.md --title "SFT Smoke Eval Report"
```

## 4. 已完成 run 的证据

- readiness: `ready_for_stage4_smoke_training`
- training `train_loss`: `4.00523042678833`
- training `train_runtime`: `261.1415`
- total steps: `4`
- generated rows: `50`
- metrics rows: `50`
- generated rows 全部记录 `max_new_tokens: 256`
- eval report hard gate pass: `0/50`
- avg Chinese chars: `298.18`
- failures: `length_short: 50`, `outline_leak: 12`

## 5. 决策边界

不要扩展到 100/500。当前结论是：基础设施闭环已经证明，质量闭环尚未证明。

进入 100 样本前必须满足：

- long-generation subset 能成功完成。
- generation budget 能支撑 2000-2500 个中文汉字。
- `outline_leak` 降到可接受阈值，优先目标为 0。
- full 50 long eval 通过约定 gate。

500 样本继续保持 out of scope，直到 100 样本证据通过。

当前决策见 [Stage 4 决策日志](stage4-decision-log.zh.md)。
