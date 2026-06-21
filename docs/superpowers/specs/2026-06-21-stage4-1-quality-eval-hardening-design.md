# Stage 4.1 Quality Eval Hardening Design

## Approval Context

The Stage 4 summary recommended `Stage 4.1 Quality Eval Hardening`, and the user approved starting it on 2026-06-21 with "可以，开始4.1吧". This spec turns that approval into a concrete, bounded first slice.

## Goal

Stage 4.1 should make the next quality-eval loop reproducible before any 100/500-sample expansion. It should preserve the repaired 50-sample control set, create a smaller fixed quality subset, run budget scans against the real smoke adapter, summarize long-generation evidence, and make the expansion decision auditable.

## Current Evidence

- Stage 3 readiness is `ready_for_stage4_smoke_training`.
- Stage 4 real smoke training produced `outputs/sft_smoke` and adapter check passed.
- The 50-card budgetized eval completed with `max_new_tokens=256`.
- Current quality gates do not pass: `hard_gate_pass=0/50`, `length_short=50`, `outline_leak=12`.
- A successful 6144-cutoff smoke retry depends on ignored `outputs/sft_smoke_retry_6144.yaml`, so the equivalent config must move to a tracked config path.

## Approaches Considered

### Recommended: Small Tooling Slice First

Add a tracked 6144 smoke config, a deterministic quality-subset builder, and a Stage 4.1 budget-report builder. This creates stable commands and reports before spending more GPU time. It is the least risky path because every long-generation run leaves comparable artifacts.

### Alternative: Run Full 50 Long Eval Immediately

Run `max_new_tokens=5120` or higher against all 50 eval cards. This may eventually answer the quality question, but Stage 4 already showed this can run for minutes with zero completed rows, which makes failures harder to interpret.

### Alternative: Expand Training Data First

Generate 100 training cards and retrain. This is premature because the existing 50-sample adapter has not passed a long-generation subset gate, and outline leak still lacks a diagnosis.

## Chosen Design

Stage 4.1 starts with reproducibility and diagnosis:

1. Add `configs/sft_qlora_qwen3_4b_smoke_6144.yaml` so smoke retry no longer depends on ignored `outputs/` files.
2. Add `scripts/build_eval_quality_subset.py` backed by `small_model_train.stage4_quality`, selecting a fixed subset from `eval_cards_50` and prioritizing known `outline_leak` samples when prior metrics are available.
3. Add `scripts/build_stage4_quality_report.py`, summarizing generated rows, metrics rows, generation budgets, character counts, hard-gate pass rate, failure counts, and outline-leak IDs/markers without copying generated chapter text.
4. Add `docs/stage4-1-quality-eval-guide.zh.md` and update README with the exact 4.1 command sequence.
5. Generate local ignored artifacts for the current 256-token baseline and, when GPU budget allows, use the same commands for 1024/2048/4096 subset scans.

## Components

- `src/small_model_train/stage4_quality.py`
  - Selects fixed subset cards.
  - Summarizes budget eval metrics.
  - Detects outline-leak markers in generated outputs.
  - Renders Markdown reports without including raw generated prose.

- `scripts/build_eval_quality_subset.py`
  - CLI wrapper for deterministic subset creation.
  - Inputs: eval cards, optional metrics, output path, count.

- `scripts/build_stage4_quality_report.py`
  - CLI wrapper for report generation from cards, generated JSONL, metrics JSONL, and optional event log.

- `configs/sft_qlora_qwen3_4b_smoke_6144.yaml`
  - Checked-in smoke retry config with `cutoff_len: 6144`.

## Data Flow

`eval_cards_50.jsonl` plus optional prior `metrics.jsonl` creates `eval_cards_quality_subset.jsonl`. Each token-budget run writes `generated_subset_<tokens>.jsonl`, then `score_outputs.py` writes matching metrics. The Stage 4.1 report script reads those artifacts and writes a Markdown budget report. Promotion to full 50 long eval requires complete subset rows, no OOM/process failure, length budget evidence near the 2000-2500 target, and outline leak explained or reduced.

## Error Handling

- Missing cards or empty inputs should fail with clear CLI errors.
- Metrics are optional for subset creation; without them, the first N eval cards are used.
- Reports should clearly label missing generated/metrics rows instead of implying quality evidence exists.
- Outline leak analysis reports markers and IDs only, not raw generated prose.

## Testing

Tests should cover subset prioritization, fallback selection, budget summary counts, outline marker detection, Markdown report content, and both CLI wrappers. Existing full-suite tests remain the regression gate.

## Out Of Scope

- No automatic 100/500-sample expansion.
- No prompt rewrite unless outline-leak analysis proves it is needed.
- No full 50 long eval until subset budget evidence is complete.
- No generated model output text committed to docs.

## Self-Review

- Placeholder scan: no TBD/TODO placeholders remain.
- Consistency check: the design uses the existing JSONL, scoring, and reporting patterns.
- Scope check: this is a single implementation slice focused on reproducible Stage 4.1 quality evaluation.
- Ambiguity check: promotion gates are explicit and do not treat 256-token smoke eval as quality evidence.
