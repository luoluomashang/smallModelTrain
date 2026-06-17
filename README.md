# Small Model Train

This project builds a Qwen3-4B-Instruct-2507 QLoRA pipeline for a Chinese whole-chapter novel executor.

## Stage 1 Pipeline

```powershell
python scripts/ingest_raw_text.py --input-dir data_raw/novels --output data_clean/chapters_raw.jsonl
python scripts/clean_chapters.py --input data_clean/chapters_raw.jsonl --output data_clean/chapters.jsonl --min-chars 500 --max-chars 5000
python scripts/split_train_eval.py --input data_clean/chapters.jsonl --output data_clean/chapters_split.jsonl --eval-output data_cards/eval_cards_50.jsonl --eval-count 50
python scripts/build_style_contract.py --chapters data_clean/chapters_split.jsonl --contract-output style_contract.md --profile-output style_profile.json
python scripts/build_sft_dataset.py --cards data_cards/chapter_cards.jsonl --chapters data_clean/chapters_split.jsonl --output data_sft/sft_chapter_v1.jsonl
```

## Evaluation

```powershell
python scripts/score_outputs.py --cards data_cards/eval_cards_50.jsonl --outputs outputs/baseline/generated.jsonl --output outputs/baseline/metrics.jsonl
python scripts/evaluate_outputs.py --scores outputs/baseline/metrics.jsonl --report reports/baseline_report.md --title "Baseline Report"
```

## Training Config

Use `configs/sft_qlora_qwen3_4b.yaml` as the first QLoRA SFT configuration. Start with a 100-sample smoke run before a full run on 500-1000 samples.
