# Stage 5 Closure Report

## Decision

Stage 5 acceptance is pending final Task 7 verification evidence. Stage 5 is accepted only after all acceptance commands listed below pass and final evidence is recorded.

This closure report defines the Stage 5 acceptance boundary for engineering/control-plane/data-integrity closure only. It does not claim model-quality improvement, efficiency win, real preference optimization, DPO/SimPO/ORPO/KTO/reward-model training, sealed-eval success, author-acceptance improvement, production-scale formal training, or `eval_execution_cards_50.jsonl` validation.

## Required Artifacts

Stage 5A evidence reports:

- `reports/stage5a_review_model_check_report.json`
- `reports/stage5a_review_training_env_report.json`

Stage 5B full-corpus style artifacts:

- `data_style/style_contract_author_main_v1.json`
- `data_style/style_metrics_author_main_v1.json`
- `style_contract.md`
- Note: full-corpus StyleContract remains `pending_review` and is not used for formal SFT gates.

Stage 5C minimal formal closure-probe artifacts:

- `data_clean/stage5_closure_formal_corpus.jsonl`
- `data_style/stage5_closure_style_contract_author_main_v1.json`
- `data_style/stage5_closure_style_metrics_author_main_v1.json`
- `data_style/stage5_closure_style_contract.md`
- `data_cards/stage5_closure_chapter_execution_cards_approved.jsonl`
- `data_sft/sft_chapter_formal.jsonl`
- `data_sft/dataset_info_formal.json`
- `data_sft/sft_chapter_formal_manifest.json`

Stage 5D review/candidate artifacts:

- `data_review/stage5d_review_records.jsonl`
- `data_review/stage5d_revisions.jsonl`
- `data_sft/stage5d_rejection_sampling_sft.jsonl`
- `data_pref/stage5d_same_plot_preference.jsonl`
- `reports/stage5d_review_summary.json`
- `reports/stage5d_review_report.md`
- `outputs/stage5d_generation_records.jsonl`

Stage 5E control-plane artifacts:

- `configs/stage5e_candidate_lr_probe.yaml`
- `reports/stage5e_entry_check.json`
- `outputs/stage5e/baseline_metrics.jsonl`
- `outputs/stage5e/candidate_metrics.jsonl`
- `data_review/stage5e_paired_judgments.jsonl`
- `reports/stage5e_paired_eval_summary.json`
- `reports/stage5e_paired_eval_report.md`
- `reports/stage5e_experiment_manifest.json`
- `reports/stage5e_experiment_commands.jsonl`
- Note: `data_cards/eval_cards_50.jsonl` is recorded as fixed eval artifact in the control-plane probe. This does not claim stricter `data_cards/eval_execution_cards_50.jsonl` exists or validates.

## Acceptance Commands

Run these commands in Task 7 before recording final evidence:

```powershell
python scripts/check_stage5e_entry.py --summary reports/stage5d_review_summary.json --review-records data_review/stage5d_review_records.jsonl --revisions data_review/stage5d_revisions.jsonl --rejection-sampling-rows data_sft/stage5d_rejection_sampling_sft.jsonl --preference-rows data_pref/stage5d_same_plot_preference.jsonl --generation-records outputs/stage5d_generation_records.jsonl --output reports/stage5e_entry_check.json
python scripts/build_paired_eval_report.py --baseline-metrics outputs/stage5e/baseline_metrics.jsonl --candidate-metrics outputs/stage5e/candidate_metrics.jsonl --judgments data_review/stage5e_paired_judgments.jsonl --summary-output reports/stage5e_paired_eval_summary.json --report-output reports/stage5e_paired_eval_report.md
python scripts/run_experiment_matrix.py --manifest reports/stage5e_experiment_manifest.json --output reports/stage5e_experiment_commands.jsonl --dry-run
python -c "from pathlib import Path; b=Path('configs/sft_qlora_qwen3_4b_smoke_6144.yaml').read_text(encoding='utf-8'); c=Path('configs/stage5e_candidate_lr_probe.yaml').read_text(encoding='utf-8'); assert 'learning_rate: 3.0e-5' in b; assert 'learning_rate: 8e-5' in c; print('baseline and candidate learning rates verified')"
python -c "import json; rows=[json.loads(l) for l in open('reports/stage5e_experiment_commands.jsonl',encoding='utf-8')]; assert len(rows)==1; cmd=rows[0]['command']; assert rows[0]['dry_run'] is True; assert '--dry-run' in cmd; assert 'configs/stage5e_candidate_lr_probe.yaml' in cmd; print('stage5e matrix candidate config dry-run verified')"
python -m pytest -q
git diff --check
```

## Final Evidence

Pending Task 7 final verification.
