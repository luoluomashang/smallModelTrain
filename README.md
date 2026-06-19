# Small Model Train

This project builds a Qwen3-4B-Instruct-2507 QLoRA pipeline for a Chinese whole-chapter novel executor.

## Stage 1 Pipeline

```powershell
python scripts/ingest_raw_text.py --input-dir data_raw/novels --output data_clean/chapters_raw.jsonl
python scripts/clean_chapters.py --input data_clean/chapters_raw.jsonl --output data_clean/chapters.jsonl --min-chars 500 --max-chars 5000
python scripts/split_train_eval.py --input data_clean/chapters.jsonl --output data_clean/chapters_split.jsonl --eval-output data_cards/eval_cards_50.jsonl --eval-count 50
python scripts/build_style_contract.py --chapters data_clean/chapters_split.jsonl --contract-output style_contract.md --profile-output style_profile.json
```

Prepare `data_cards/chapter_cards.jsonl` separately from the cleaned chapters using the agreed chapter-card schema before building the SFT dataset.

```powershell
python scripts/build_sft_dataset.py --cards data_cards/chapter_cards.jsonl --chapters data_clean/chapters_split.jsonl --output data_sft/sft_chapter_v1.jsonl
```

## Evaluation

```powershell
python scripts/score_outputs.py --cards data_cards/eval_cards_50.jsonl --outputs outputs/baseline/generated.jsonl --output outputs/baseline/metrics.jsonl
python scripts/evaluate_outputs.py --scores outputs/baseline/metrics.jsonl --report reports/baseline_report.md --title "Baseline Report"
```

## Training Config

Use `configs/sft_qlora_qwen3_4b.yaml` for the downstream LLaMA-Factory/Stage 2 QLoRA SFT training run. The repaired Stage 4 path now starts with a 50-sample smoke run. Expansion to 100 or 500 samples is blocked until the decision-log quality criteria pass.

## Stage 3 Data Bring-Up

Stage 3 prepares real chapter data and readiness evidence only. It does not start real GPU training.

```powershell
python scripts/ingest_raw_text.py --input-dir data_raw/novels --output data_clean/chapters_raw.jsonl
python scripts/clean_chapters.py --input data_clean/chapters_raw.jsonl --output data_clean/chapters.jsonl --min-chars 500 --max-chars 5000
python scripts/split_train_eval.py --input data_clean/chapters.jsonl --output data_clean/chapters_split.jsonl --eval-output data_cards/eval_cards_20.jsonl --eval-count 20
python scripts/build_style_contract.py --chapters data_clean/chapters_split.jsonl --contract-output style_contract.md --profile-output style_profile.json
python scripts/build_sft_dataset.py --cards data_cards/chapter_cards.jsonl --chapters data_clean/chapters_split.jsonl --output data_sft/sft_chapter_v1.jsonl
python scripts/check_stage3_data_readiness.py --eval-cards data_cards/eval_cards_20.jsonl --run-smoke-dry-run
```

See `docs/stage3-data-bring-up-guide.zh.md` for the full Chinese runbook.

## Stage 4 Smoke Eval

Stage 4 starts only after Stage 3 reports `ready_for_stage4_smoke_training`. It uses the repaired 50-card path to rebuild data, run real smoke training, check the adapter, run budgetized eval inference, score outputs, and make the expansion decision.

```powershell
python scripts/build_chapter_cards.py --chapters data_clean/chapters_split.jsonl --output data_cards/chapter_cards.jsonl --count 50 --min-chars 2000 --max-chars 3000
python scripts/build_sft_dataset.py --cards data_cards/chapter_cards.jsonl --chapters data_clean/chapters_split.jsonl --output data_sft/sft_chapter_v1.jsonl --dataset-info-output data_sft/dataset_info.json
python scripts/check_stage3_data_readiness.py --eval-cards data_cards/eval_cards_50.jsonl --run-smoke-dry-run
python scripts/check_local_model.py --model-dir E:\models\Qwen3-4B-Instruct-2507 --report reports/model_check_report.md
python scripts/check_training_env.py --report reports/training_env_report.md
python scripts/run_sft_smoke.py --eval-cards data_cards/eval_cards_50.jsonl --dry-run
python scripts/run_sft_smoke.py --eval-cards data_cards/eval_cards_50.jsonl
python scripts/run_sft_smoke.py --config outputs/sft_smoke_retry_6144.yaml --eval-cards data_cards/eval_cards_50.jsonl
python scripts/check_adapter.py --adapter-dir outputs/sft_smoke --report reports/sft_smoke_report.md --title "SFT Smoke Adapter Check"
python scripts/run_eval_inference.py --cards data_cards/eval_cards_50.jsonl --adapter-dir outputs/sft_smoke --output outputs/sft_smoke/generated.jsonl --model-name sft_smoke --event-log logs/training/sft_smoke_eval_events.jsonl --stderr-log logs/training/sft_smoke_eval_stderr.log --stdout-log logs/training/sft_smoke_eval_stdout.log --max-new-tokens 256
python scripts/score_outputs.py --cards data_cards/eval_cards_50.jsonl --outputs outputs/sft_smoke/generated.jsonl --output outputs/sft_smoke/metrics.jsonl
python scripts/evaluate_outputs.py --scores outputs/sft_smoke/metrics.jsonl --report reports/sft_smoke_eval_report.md --title "SFT Smoke Eval Report"
```

The `--max-new-tokens 256` eval proves infrastructure only. Quality expansion requires the criteria in `docs/stage4-decision-log.zh.md`: long-generation subset success, budget for 2000-2500 Chinese chars, leak reduction, and full 50 long eval passing the agreed gate.

See `docs/stage4-smoke-eval-guide.zh.md` for the full Chinese runbook and `docs/stage4-decision-log.zh.md` for the current decision.

For a handoff summary and next-stage outlook, see `docs/stage4-summary-next-outlook.zh.md`.

## Stage 2 Training Execution

Run Stage 2 from a shell with the training environment activated. The local base model path is `E:\models\Qwen3-4B-Instruct-2507`.

This section is legacy Stage 2 execution context. The current repaired Stage 4 smoke path is the preceding 50-card sequence and its decision-log gates.

```powershell
python scripts/check_local_model.py --model-dir E:\models\Qwen3-4B-Instruct-2507
python scripts/check_training_env.py
python scripts/run_sft_smoke.py --eval-cards data_cards/eval_cards_20.jsonl --dry-run
python scripts/run_sft_smoke.py --eval-cards data_cards/eval_cards_20.jsonl
python scripts/check_adapter.py --adapter-dir outputs/sft_smoke --report reports/sft_smoke_report.md --title "SFT Smoke Adapter Check"
python scripts/run_oom_probe.py --dry-run
python scripts/run_oom_probe.py
python scripts/run_sft_train.py
python scripts/check_adapter.py --adapter-dir outputs/sft_v1 --report reports/sft_v1_training_report.md --title "SFT v1 Adapter Check"
python scripts/run_eval_inference.py
python scripts/score_outputs.py --cards data_cards/eval_cards_50.jsonl --outputs outputs/sft_v1/generated.jsonl --output outputs/sft_v1/metrics.jsonl
python scripts/evaluate_outputs.py --scores outputs/sft_v1/metrics.jsonl --report reports/sft_v1_report.md --title "SFT v1 Report"
```

If a training or eval subprocess exits with CUDA OOM, launcher failure, or a crash, use the event log, stderr log, and `run_oom_probe.py` report to identify whether the failure is memory pressure, environment drift, or process termination before retrying.
