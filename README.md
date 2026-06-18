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

Use `configs/sft_qlora_qwen3_4b.yaml` for the downstream LLaMA-Factory/Stage 2 QLoRA SFT training run. Once the Stage 2 scripts and environment are ready, make the first real training attempt a 100-sample smoke run before a full run on 500-1000 samples.

## Stage 2 Training Execution

Run Stage 2 from a shell with the training environment activated. The local base model path is `E:\models\Qwen3-4B-Instruct-2507`.

```powershell
python scripts/check_local_model.py --model-dir E:\models\Qwen3-4B-Instruct-2507
python scripts/check_training_env.py
python scripts/run_sft_smoke.py --dry-run
python scripts/run_sft_smoke.py
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
