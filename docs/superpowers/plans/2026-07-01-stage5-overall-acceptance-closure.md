# Stage 5 Overall Acceptance Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close Stage 5 as an auditable whole by generating the missing 5B/5C/5E acceptance artifacts, correcting stale roadmap status, and producing a final closure report backed by fresh verification output.

**Architecture:** Use the existing Stage 5 scripts and tests as the source of truth, then add a small closure verifier only if manual checks remain ambiguous. The closure boundary is engineering/control-plane acceptance: it must not claim model-quality improvement, preference optimization, DPO/SimPO/ORPO/KTO training, or efficiency wins that were not actually run.

**Tech Stack:** Python 3.11, pytest, existing `scripts/` CLIs, JSON/JSONL artifacts, Markdown docs.

---

## File Structure

- Generate or refresh: `data_style/style_metrics_author_main_v1.json`
  - Stage 5B style metrics artifact required by public docs.
- Generate or refresh: `data_sft/sft_chapter_formal.jsonl`
  - Formal SFT dataset built from approved/frozen execution cards.
- Generate or refresh: `data_sft/dataset_info_formal.json`
  - LLaMA-Factory dataset metadata for the formal dataset.
- Generate or refresh: `data_sft/sft_chapter_formal_manifest.json`
  - Stage 5C dataset provenance manifest.
- Create: `outputs/stage5e/baseline_metrics.jsonl`
  - Minimal truthful paired-eval baseline metrics for the control-plane probe.
- Create: `outputs/stage5e/candidate_metrics.jsonl`
  - Minimal truthful paired-eval candidate metrics for the control-plane probe.
- Create: `data_review/stage5e_paired_judgments.jsonl`
  - Minimal paired judgment row, clearly scoped to control-plane verification.
- Generate: `reports/stage5e_paired_eval_summary.json`
  - Paired eval machine summary.
- Generate: `reports/stage5e_paired_eval_report.md`
  - Paired eval Markdown report.
- Generate: `reports/stage5e_experiment_manifest.json`
  - Stage 5E one-primary-variable experiment manifest.
- Generate: `reports/stage5e_experiment_commands.jsonl`
  - Dry-run experiment matrix command output.
- Create: `reports/stage5_closure_report.md`
  - Human-readable Stage 5 closure decision and acceptance boundary.
- Modify: `docs/superpowers/plans/2026-06-23-small-model-train-full-roadmap.md`
  - Replace stale Stage 5 status lines with actual closure status and artifact pointers.
- Optional create: `scripts/check_stage5_closure.py`
  - Only add this if the closure checklist needs repeatable machine verification beyond existing CLI/test coverage.
- Optional test: `tests/test_stage5_closure.py`
  - Only add this with the optional closure checker.

---

### Task 1: Refresh Stage 5B Style Metrics

**Files:**
- Generate: `data_style/style_contract_author_main_v1.json`
- Generate: `style_contract.md`
- Generate: `data_style/style_metrics_author_main_v1.json`
- Verify: `tests/test_style_profile.py`

- [ ] **Step 1: Run the existing Stage 5B style contract command**

```powershell
python scripts/build_style_contract.py --chapters data_clean/chapters_split.jsonl --contract-json-output data_style/style_contract_author_main_v1.json --contract-output style_contract.md --metrics-output data_style/style_metrics_author_main_v1.json --style-contract-id author_main_v1
```

Expected: command exits `0` and prints that the StyleContract, Markdown contract, and metrics were written.

- [ ] **Step 2: Verify required files exist**

```powershell
Test-Path data_style/style_contract_author_main_v1.json
Test-Path style_contract.md
Test-Path data_style/style_metrics_author_main_v1.json
```

Expected:

```text
True
True
True
```

- [ ] **Step 3: Run the focused style tests**

```powershell
python -m pytest tests/test_style_profile.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```powershell
git add -f data_style/style_contract_author_main_v1.json style_contract.md data_style/style_metrics_author_main_v1.json docs/superpowers/plans/2026-07-01-stage5-overall-acceptance-closure.md
git commit -m "chore: refresh stage5b style metrics artifacts"
```

Expected: commit succeeds.

---

### Task 2: Generate Stage 5C Formal Dataset Manifest

**Files:**
- Generate: `data_sft/sft_chapter_formal.jsonl`
- Generate: `data_sft/dataset_info_formal.json`
- Generate: `data_sft/sft_chapter_formal_manifest.json`
- Verify: `tests/test_sft_builder.py`, `tests/test_dataset_manifest.py`

- [ ] **Step 1: Run the formal SFT build command**

```powershell
python scripts/build_sft_dataset.py --cards data_cards/chapter_execution_cards_approved.jsonl --chapters data_clean/chapters_split.jsonl --output data_sft/sft_chapter_formal.jsonl --dataset-info-output data_sft/dataset_info_formal.json --style-contract-json data_style/style_contract_author_main_v1.json --dataset-manifest-output data_sft/sft_chapter_formal_manifest.json
```

Expected: command exits `0` and writes formal dataset rows plus manifest.

- [ ] **Step 2: Verify required files exist**

```powershell
Test-Path data_sft/sft_chapter_formal.jsonl
Test-Path data_sft/dataset_info_formal.json
Test-Path data_sft/sft_chapter_formal_manifest.json
```

Expected:

```text
True
True
True
```

- [ ] **Step 3: Inspect manifest boundary fields**

```powershell
python -c "import json; p='data_sft/sft_chapter_formal_manifest.json'; o=json.load(open(p,encoding='utf-8')); print(o.keys()); print(o.get('style_contract')); print(o.get('split_counts'))"
```

Expected: output includes manifest keys, a non-empty style contract section, and split counts.

- [ ] **Step 4: Run focused formal dataset tests**

```powershell
python -m pytest tests/test_sft_builder.py tests/test_dataset_manifest.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add -f data_sft/sft_chapter_formal.jsonl data_sft/dataset_info_formal.json data_sft/sft_chapter_formal_manifest.json
git commit -m "chore: generate stage5c formal dataset manifest"
```

Expected: commit succeeds.

---

### Task 3: Create Truthful Stage 5E Paired-Eval Probe Inputs

**Files:**
- Create: `outputs/stage5e/baseline_metrics.jsonl`
- Create: `outputs/stage5e/candidate_metrics.jsonl`
- Create: `data_review/stage5e_paired_judgments.jsonl`
- Generate: `reports/stage5e_paired_eval_summary.json`
- Generate: `reports/stage5e_paired_eval_report.md`
- Verify: `tests/test_paired_eval.py`

- [ ] **Step 1: Create the output directories**

```powershell
New-Item -ItemType Directory -Force outputs/stage5e | Out-Null
New-Item -ItemType Directory -Force data_review | Out-Null
```

Expected: command exits `0`.

- [ ] **Step 2: Create minimal paired metrics and judgment rows**

Use `apply_patch` or an equivalent non-destructive editor operation to create:

```jsonl
{"id":"stage5e-control-plane-probe-001","hard_gate_pass":true,"failure_types":[]}
```

in `outputs/stage5e/baseline_metrics.jsonl`.

Create:

```jsonl
{"id":"stage5e-control-plane-probe-001","hard_gate_pass":true,"failure_types":[]}
```

in `outputs/stage5e/candidate_metrics.jsonl`.

Create:

```jsonl
{"id":"stage5e-control-plane-probe-001","winner":"tie","source":"control_plane_probe","note":"Synthetic paired row used only to verify Stage 5E paired-eval reporting plumbing; it does not claim model-quality improvement."}
```

in `data_review/stage5e_paired_judgments.jsonl`.

- [ ] **Step 3: Generate the paired eval report**

```powershell
python scripts/build_paired_eval_report.py --baseline-metrics outputs/stage5e/baseline_metrics.jsonl --candidate-metrics outputs/stage5e/candidate_metrics.jsonl --judgments data_review/stage5e_paired_judgments.jsonl --summary-output reports/stage5e_paired_eval_summary.json --report-output reports/stage5e_paired_eval_report.md
```

Expected: `wrote Stage 5E paired eval report to reports\stage5e_paired_eval_report.md`.

- [ ] **Step 4: Verify the report boundary**

```powershell
Select-String -Path reports/stage5e_paired_eval_report.md -Pattern "paired_eval_no_training","Candidate wins: 0","Ties: 1"
```

Expected: all three patterns are present.

- [ ] **Step 5: Run focused paired eval tests**

```powershell
python -m pytest tests/test_paired_eval.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add -f outputs/stage5e/baseline_metrics.jsonl outputs/stage5e/candidate_metrics.jsonl data_review/stage5e_paired_judgments.jsonl reports/stage5e_paired_eval_summary.json reports/stage5e_paired_eval_report.md
git commit -m "chore: add stage5e paired eval probe artifacts"
```

Expected: commit succeeds.

---

### Task 4: Generate Stage 5E Experiment Manifest And Dry-Run Matrix

**Files:**
- Input: `reports/stage5e_entry_check.json`
- Input: `configs/sft_qlora_qwen3_4b_smoke_6144.yaml`
- Input: `data_sft/stage5d_rejection_sampling_sft.jsonl`
- Input: `data_cards/eval_cards_50.jsonl`
- Input: `reports/stage5e_paired_eval_summary.json`
- Generate: `reports/stage5e_experiment_manifest.json`
- Generate: `reports/stage5e_experiment_commands.jsonl`
- Verify: `tests/test_stage5e_experiment_manifest.py`, `tests/test_run_experiment_matrix.py`

- [ ] **Step 1: Re-run the Stage 5E entry gate**

```powershell
python scripts/check_stage5e_entry.py --summary reports/stage5d_review_summary.json --review-records data_review/stage5d_review_records.jsonl --revisions data_review/stage5d_revisions.jsonl --rejection-sampling-rows data_sft/stage5d_rejection_sampling_sft.jsonl --preference-rows data_pref/stage5d_same_plot_preference.jsonl --generation-records outputs/stage5d_generation_records.jsonl --output reports/stage5e_entry_check.json
```

Expected: `Stage 5E entry gate passed; wrote reports\stage5e_entry_check.json`.

- [ ] **Step 2: Build the experiment manifest**

```powershell
python scripts/build_stage5e_experiment_manifest.py --experiment-id stage5e_control_plane_probe_001 --baseline-run-id stage5d_baseline --candidate-run-id stage5e_candidate_lr_probe --primary-variable-name learning_rate --primary-baseline-value 1e-4 --primary-candidate-value 8e-5 --controlled-variable seed=20260628=20260628 --stage5e-entry-check reports/stage5e_entry_check.json --artifact config=configs/sft_qlora_qwen3_4b_smoke_6144.yaml --artifact sft_dataset=data_sft/stage5d_rejection_sampling_sft.jsonl --artifact eval_cards=data_cards/eval_cards_50.jsonl --artifact paired_eval_summary=reports/stage5e_paired_eval_summary.json --paired-eval-json "{\"summary\":\"reports/stage5e_paired_eval_summary.json\",\"report\":\"reports/stage5e_paired_eval_report.md\",\"boundary\":\"paired_eval_no_training\",\"scope\":\"control_plane_probe_no_training_claim\",\"eval_artifact\":\"data_cards/eval_cards_50.jsonl\"}" --output reports/stage5e_experiment_manifest.json
```

Expected: `wrote Stage 5E experiment manifest to reports\stage5e_experiment_manifest.json`.

- [ ] **Step 3: Generate the dry-run experiment matrix**

```powershell
python scripts/run_experiment_matrix.py --manifest reports/stage5e_experiment_manifest.json --output reports/stage5e_experiment_commands.jsonl --dry-run
```

Expected: `wrote 1 Stage 5E experiment commands to reports\stage5e_experiment_commands.jsonl`.

- [ ] **Step 4: Verify dry-run command stays dry-run**

```powershell
python -c "import json; rows=[json.loads(l) for l in open('reports/stage5e_experiment_commands.jsonl',encoding='utf-8')]; assert len(rows)==1; cmd=rows[0]['command']; assert rows[0]['dry_run'] is True; assert '--dry-run' in cmd; assert '--config' in cmd; assert '--sft-dataset' in cmd; assert '--eval-cards' in cmd; print('stage5e matrix dry-run verified')"
```

Expected: `stage5e matrix dry-run verified`.

- [ ] **Step 5: Run focused Stage 5E tests**

```powershell
python -m pytest tests/test_stage5e_entry.py tests/test_stage5e_experiment_manifest.py tests/test_run_experiment_matrix.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add -f reports/stage5e_entry_check.json reports/stage5e_experiment_manifest.json reports/stage5e_experiment_commands.jsonl
git commit -m "chore: generate stage5e control-plane closure artifacts"
```

Expected: commit succeeds.

---

### Task 5: Add A Stage 5 Closure Report

**Files:**
- Create: `reports/stage5_closure_report.md`

- [ ] **Step 1: Create the closure report**

Use `apply_patch` or an equivalent non-destructive editor operation to create `reports/stage5_closure_report.md`:

```markdown
# Stage 5 Closure Report

## Decision

Stage 5 is accepted as an engineering and control-plane closure after all listed verification commands pass.

This closure does not claim model-quality improvement, efficiency win, preference optimization, DPO, SimPO, ORPO, KTO, reward-model training, sealed-eval success, or author-acceptance improvement beyond the artifacts listed here.

## Required Artifacts

- Stage 5A evidence reports:
  - `reports/stage5a_review_model_check_report.json`
  - `reports/stage5a_review_training_env_report.json`
- Stage 5B style artifacts:
  - `data_style/style_contract_author_main_v1.json`
  - `data_style/style_metrics_author_main_v1.json`
  - `style_contract.md`
- Stage 5C formal data artifacts:
  - `data_cards/chapter_execution_cards_approved.jsonl`
  - `data_sft/sft_chapter_formal.jsonl`
  - `data_sft/dataset_info_formal.json`
  - `data_sft/sft_chapter_formal_manifest.json`
- Stage 5D review/candidate artifacts:
  - `data_review/stage5d_review_records.jsonl`
  - `data_review/stage5d_revisions.jsonl`
  - `data_sft/stage5d_rejection_sampling_sft.jsonl`
  - `data_pref/stage5d_same_plot_preference.jsonl`
  - `reports/stage5d_review_summary.json`
  - `reports/stage5d_review_report.md`
  - `outputs/stage5d_generation_records.jsonl`
- Stage 5E control-plane artifacts:
  - `reports/stage5e_entry_check.json`
  - `outputs/stage5e/baseline_metrics.jsonl`
  - `outputs/stage5e/candidate_metrics.jsonl`
  - `data_review/stage5e_paired_judgments.jsonl`
  - `reports/stage5e_paired_eval_summary.json`
  - `reports/stage5e_paired_eval_report.md`
  - `reports/stage5e_experiment_manifest.json`
  - `reports/stage5e_experiment_commands.jsonl`

The Stage 5E manifest records `data_cards/eval_cards_50.jsonl` as a fixed eval artifact for the control-plane probe. This does not claim the stricter `eval_execution_cards_50.jsonl` execution-card schema has been prepared or validated.

## Acceptance Commands

```powershell
python scripts/check_stage5e_entry.py --summary reports/stage5d_review_summary.json --review-records data_review/stage5d_review_records.jsonl --revisions data_review/stage5d_revisions.jsonl --rejection-sampling-rows data_sft/stage5d_rejection_sampling_sft.jsonl --preference-rows data_pref/stage5d_same_plot_preference.jsonl --generation-records outputs/stage5d_generation_records.jsonl --output reports/stage5e_entry_check.json
python scripts/build_paired_eval_report.py --baseline-metrics outputs/stage5e/baseline_metrics.jsonl --candidate-metrics outputs/stage5e/candidate_metrics.jsonl --judgments data_review/stage5e_paired_judgments.jsonl --summary-output reports/stage5e_paired_eval_summary.json --report-output reports/stage5e_paired_eval_report.md
python scripts/run_experiment_matrix.py --manifest reports/stage5e_experiment_manifest.json --output reports/stage5e_experiment_commands.jsonl --dry-run
python -m pytest -q
git diff --check
```

## Final Evidence

Record the exact final command output here during Task 7.
```

- [ ] **Step 2: Commit**

```powershell
git add -f reports/stage5_closure_report.md
git commit -m "docs: add stage5 closure report"
```

Expected: commit succeeds.

---

### Task 6: Update The Full Roadmap Status

**Files:**
- Modify: `docs/superpowers/plans/2026-06-23-small-model-train-full-roadmap.md`
- Optionally modify: `docs/index.zh.md`

- [ ] **Step 1: Replace stale Stage 5 status language**

In `docs/superpowers/plans/2026-06-23-small-model-train-full-roadmap.md`, update the Stage 5 status lines so they no longer claim:

```markdown
**Status:** Current executable stage.
**Status:** Forward index only. Do not implement until Stage 5A exits.
**Status:** Forward index only. Do not implement until Stage 5B style contract assets exist.
**Status:** Blocked until `reports/stage5e_entry_check.json` exists with `"passed": true`.
- Type consistency: Stage naming follows `5A` through `5E`, and Stage 5A remains the only current executable implementation plan.
```

Use replacement wording like:

```markdown
**Status:** Implemented and covered by Stage 5 closure artifacts.
```

For Stage 5E, use:

```markdown
**Status:** Control-plane implemented. Closure requires `reports/stage5e_entry_check.json`, `reports/stage5e_experiment_manifest.json`, `reports/stage5e_experiment_commands.jsonl`, `reports/stage5e_paired_eval_summary.json`, and full pytest passing.
```

Replace the final type-consistency bullet with:

```markdown
- Type consistency: Stage naming follows `5A` through `5E`; overall Stage 5 closure is recorded in `reports/stage5_closure_report.md`.
```

- [ ] **Step 2: Add closure report link to docs index if absent**

If `docs/index.zh.md` does not mention `reports/stage5_closure_report.md`, add a short link under the Stage 5 section:

```markdown
- Stage 5 closure report: `reports/stage5_closure_report.md`
```

- [ ] **Step 3: Verify stale status strings are gone**

```powershell
rg -n "Stage 5A remains the only current executable|Forward index only|Blocked until `reports/stage5e_entry_check.json`|Current executable stage" docs/superpowers/plans/2026-06-23-small-model-train-full-roadmap.md
```

Expected: no output and exit code `1`.

- [ ] **Step 4: Commit**

```powershell
git add docs/superpowers/plans/2026-06-23-small-model-train-full-roadmap.md docs/index.zh.md
git commit -m "docs: mark stage5 roadmap closure status"
```

Expected: commit succeeds. If `docs/index.zh.md` did not change, omit it from `git add`.

---

### Task 7: Final Verification And Closure Decision

**Files:**
- Update: `reports/stage5_closure_report.md`

- [ ] **Step 1: Verify all required artifacts exist**

```powershell
$paths = @(
  'reports/stage5a_review_model_check_report.json',
  'reports/stage5a_review_training_env_report.json',
  'data_style/style_contract_author_main_v1.json',
  'data_style/style_metrics_author_main_v1.json',
  'style_contract.md',
  'data_cards/chapter_execution_cards_approved.jsonl',
  'data_sft/sft_chapter_formal.jsonl',
  'data_sft/dataset_info_formal.json',
  'data_sft/sft_chapter_formal_manifest.json',
  'data_review/stage5d_review_records.jsonl',
  'data_review/stage5d_revisions.jsonl',
  'data_sft/stage5d_rejection_sampling_sft.jsonl',
  'data_pref/stage5d_same_plot_preference.jsonl',
  'reports/stage5d_review_summary.json',
  'reports/stage5d_review_report.md',
  'outputs/stage5d_generation_records.jsonl',
  'reports/stage5e_entry_check.json',
  'outputs/stage5e/baseline_metrics.jsonl',
  'outputs/stage5e/candidate_metrics.jsonl',
  'data_review/stage5e_paired_judgments.jsonl',
  'reports/stage5e_paired_eval_summary.json',
  'reports/stage5e_paired_eval_report.md',
  'reports/stage5e_experiment_manifest.json',
  'reports/stage5e_experiment_commands.jsonl',
  'data_cards/eval_cards_50.jsonl'
)
foreach ($p in $paths) { if (-not (Test-Path $p)) { throw "missing required Stage 5 closure artifact: $p" } }
"all required Stage 5 closure artifacts exist"
```

Expected: `all required Stage 5 closure artifacts exist`.

- [ ] **Step 2: Re-run Stage 5E gate**

```powershell
python scripts/check_stage5e_entry.py --summary reports/stage5d_review_summary.json --review-records data_review/stage5d_review_records.jsonl --revisions data_review/stage5d_revisions.jsonl --rejection-sampling-rows data_sft/stage5d_rejection_sampling_sft.jsonl --preference-rows data_pref/stage5d_same_plot_preference.jsonl --generation-records outputs/stage5d_generation_records.jsonl --output reports/stage5e_entry_check.json
```

Expected: `Stage 5E entry gate passed; wrote reports\stage5e_entry_check.json`.

- [ ] **Step 3: Re-run Stage 5E artifact commands**

```powershell
python scripts/build_paired_eval_report.py --baseline-metrics outputs/stage5e/baseline_metrics.jsonl --candidate-metrics outputs/stage5e/candidate_metrics.jsonl --judgments data_review/stage5e_paired_judgments.jsonl --summary-output reports/stage5e_paired_eval_summary.json --report-output reports/stage5e_paired_eval_report.md
python scripts/run_experiment_matrix.py --manifest reports/stage5e_experiment_manifest.json --output reports/stage5e_experiment_commands.jsonl --dry-run
```

Expected: both commands exit `0`.

- [ ] **Step 4: Run full test suite**

```powershell
python -m pytest -q
```

Expected: full suite passes. Current baseline before this plan was `596 passed`.

- [ ] **Step 5: Run whitespace/checksum safety checks**

```powershell
git diff --check
rg -n "Stage 5A remains the only current executable|Forward index only|Blocked until `reports/stage5e_entry_check.json`|Current executable stage" docs/superpowers/plans/2026-06-23-small-model-train-full-roadmap.md
```

Expected: `git diff --check` exits `0`; `rg` prints no stale status lines.

- [ ] **Step 6: Record final evidence in the closure report**

Append exact final outputs under `## Final Evidence` in `reports/stage5_closure_report.md`, including:

```markdown
- Stage 5E entry gate: passed.
- Paired eval report generation: passed.
- Dry-run experiment matrix generation: passed.
- Full pytest: `596 passed` or the exact current count.
- `git diff --check`: passed.
- Roadmap stale-status scan: no matches.
```

- [ ] **Step 7: Final commit**

```powershell
git add -f reports/stage5_closure_report.md
git commit -m "docs: record stage5 final closure evidence"
```

Expected: commit succeeds.

---

## Self-Review Checklist

- Spec coverage: This plan closes the previously observed gaps: missing 5B metrics, missing 5C formal manifest, missing 5E manifest/matrix/paired-eval artifacts, stale roadmap status, and final verification evidence.
- Boundary honesty: The plan only accepts Stage 5 as engineering/control-plane closure. It explicitly does not claim model-quality improvement, preference optimization, real training, efficiency wins, or sealed-eval success.
- Repeatability: The final closure decision depends on existing scripts plus fresh command output, not on memory from earlier turns.
- Risk: The paired-eval probe rows are synthetic plumbing evidence. The report must say this clearly so readers do not interpret it as real model comparison.
