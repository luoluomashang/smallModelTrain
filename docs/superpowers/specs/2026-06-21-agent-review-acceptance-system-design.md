# Agent Review Acceptance System Design

## Context

Stage 4.1 full50 exposed a gap in the current acceptance system. The old rule-based hard gate reported `50/50` pass, but manual review found problems that the scorer did not capture:

- eval schema mismatch: `data_cards/eval_cards_50.jsonl` is not an execution-card file.
- semantic repetition: outputs repeat the same relationship, emotion, or conclusion in different wording to fill length.
- non-prose residue: markdown blockquotes, disclaimers, final confirmations, and internal structure notes.
- unnatural endings: outputs often run to the 2500-character cap instead of naturally closing.
- generic phrase overuse and style drift.

The acceptance system must become a layered gate. Rule metrics remain necessary, but they cannot be the only source of truth. Agent reading review must become a required middle layer before human approval or expansion decisions.

## Goal

Build a two-stage acceptance workflow:

1. Use Codex subagents now to perform structured reading review with majority voting.
2. After the rubric stabilizes, implement a project CLI that runs the same review contract and writes reproducible review artifacts.

The immediate deliverable is a design and later an implementation plan. No new training expansion should proceed until this acceptance layer is in place and has blocked the known bad full50 artifact.

## Non-Goals

- Do not expand to 100 or 500 samples in this work.
- Do not rely on one human read-through as the only quality signal.
- Do not treat exact n-gram repetition as sufficient coverage for semantic repetition.
- Do not put generated chapter prose into Markdown reports; reports may include IDs, labels, short issue summaries, and brief compliant snippets only when necessary.
- Do not build a general-purpose literary judge. The system is an acceptance gate for this project’s chapter-generation pipeline.

## Acceptance Layers

### Layer 1: Rule Gate

The deterministic scorer remains the first gate. The first implementation should cover the current known blind spots:

- card schema validity.
- row completeness.
- Chinese character length.
- exact repetition.
- semantic or chunk-level repetition diagnostics.
- non-prose residue.
- suspicious ending/truncation.
- generic phrase overuse.
- must-include and forbidden coverage when execution cards provide those fields.

Rule-gate status values should distinguish:

- `blocked_by_rule_gate`
- `rules_pass_agent_pending`

### Layer 2: Agent Reading Gate

Agent reading review is a majority-vote hard gate. For each eval batch, three reviewer roles read the generated output against the card and metrics.

Reviewer roles:

1. **Structure Reviewer**
   - Checks whether the output is a coherent chapter-like unit.
   - Detects padding-by-repetition, repeated information points, missing progression, weak conflict movement, and unnatural closure.

2. **Style Reviewer**
   - Checks style drift, generic AI phrasing, tonal mismatch, mechanical paragraph rhythm, over-explanation, and simplified/traditional style mixing.

3. **Compliance Reviewer**
   - Checks whether the output is only prose.
   - Detects markdown, disclaimers, final confirmations, internal prompt language, residual outline/meta text, truncation, and execution-card requirement failures.

Majority rule:

- `3/3 pass`: agent gate passes.
- `2/3 pass`: agent gate passes, but the sample is added to the human spot-check pool.
- `1/3 pass` or `0/3 pass`: agent gate blocks.
- Any reviewer may set `severity = blocker`; a blocker sends the sample to human arbitration and prevents automatic expansion.

Batch-level decision:

- A batch cannot pass if any sample has `agent_gate_pass = false`.
- A batch cannot advance automatically if any sample has `requires_human_arbitration = true`.

### Layer 3: Human Arbitration

Human review becomes an arbitration and calibration layer, not the only reading gate.

Human review is required when:

- reviewers disagree and at least one marks `blocker`.
- agent review identifies a new issue category not represented in the rubric.
- the batch is being used as promotion evidence for a larger expansion.

Manual override must be explicit and recorded:

- `manual_override: pass | fail | none`
- `manual_reason`
- reviewer or operator name may be omitted for local runs, but the reason cannot be empty when an override is present.

## Stage 1: Codex Subagent Review Flow

Stage 1 uses Codex subagents directly from the current workflow. It is intentionally process-first, because the rubric needs calibration before being codified.

Inputs:

- execution cards JSONL, preferably a real execution-card file.
- generated outputs JSONL.
- rule metrics JSONL.
- reviewer rubric prompt.

Outputs:

- `outputs/.../agent_reviews.jsonl`
- `reports/...agent_review_report.md`
- optional manual review notes in `docs/` or `reports/`.

Per-sample review JSON schema:

```json
{
  "id": "sample-id",
  "reviewer": "structure",
  "pass": false,
  "severity": "blocker",
  "issues": ["semantic_repetition", "padding_to_length"],
  "evidence": [
    {
      "type": "summary",
      "location": "middle-to-late output",
      "note": "The same relationship explanation is restated several times without new plot movement."
    }
  ],
  "recommendation": "reject_or_retry",
  "confidence": "high"
}
```

Allowed reviewer names:

- `structure`
- `style`
- `compliance`

Allowed severity values:

- `none`
- `minor`
- `major`
- `blocker`

Common issue labels:

- `semantic_repetition`
- `padding_to_length`
- `generic_ai_phrase`
- `style_drift`
- `tone_mismatch`
- `non_prose_residue`
- `markdown_residue`
- `disclaimer_residue`
- `meta_evaluation_residue`
- `truncation`
- `unnatural_ending`
- `schema_mismatch`
- `must_include_missing`
- `forbidden_violation`
- `weak_plot_progression`

Stage 1 orchestration:

1. Rule gate runs first.
2. If rule gate blocks, do not run full agent review unless the purpose is diagnostic.
3. If rules pass, dispatch three reviewer subagents.
4. Each reviewer writes a JSONL review for every sample or for a defined subset.
5. A coordinator merges reviewer results and computes majority vote.
6. The final report includes rule summary, agent vote summary, blocker IDs, and next action.

Stage 1 can be run manually in this Codex thread while the CLI does not exist. The coordinator must record enough information in Markdown so the result is auditable.

## Stage 2: Project CLI

After Stage 1 produces stable rubric examples, implement a CLI:

```powershell
python scripts/run_agent_review.py `
  --cards data_cards/eval_execution_cards_50.jsonl `
  --outputs outputs/sft_smoke/generated.jsonl `
  --metrics outputs/sft_smoke/metrics.jsonl `
  --output outputs/sft_smoke/agent_reviews.jsonl `
  --report reports/stage4_agent_review_report.md
```

The CLI should be backend-pluggable:

- `--backend codex-subagent` for Codex-mediated review when available.
- `--backend mock` for tests.
- Additional local or API model backends are out of scope for the first implementation.

The first implementation may only include `mock` plus a file-based import mode if Codex subagent execution cannot be called directly from repo code. The important part is to freeze the schemas, vote aggregation, and report decision logic.

CLI outputs:

- per-review JSONL.
- per-sample majority results JSONL.
- Markdown report.
- non-zero exit code when the agent gate blocks.

## Data Model

### Per-Review Row

Required fields:

- `id`
- `reviewer`
- `pass`
- `severity`
- `issues`
- `evidence`
- `recommendation`
- `confidence`

### Per-Sample Vote Row

Required fields:

- `id`
- `review_count`
- `pass_votes`
- `fail_votes`
- `blocker_votes`
- `agent_gate_pass`
- `requires_human_arbitration`
- `issues`
- `reviewers`

### Batch Summary

Required fields:

- `expected_rows`
- `reviewed_rows`
- `missing_review_ids`
- `agent_gate_pass`
- `blocked_ids`
- `arbitration_ids`
- `issue_counts`
- `decision`

Decision values:

- `blocked_incomplete_agent_review`
- `blocked_by_agent_review`
- `blocked_by_human_arbitration`
- `rules_pass_agent_pending`
- `ready_for_human_spot_check`
- `ready_for_next_expansion`

## Error Handling

Schema mismatch is a blocking error. If cards do not contain execution-card fields required for the selected eval mode, the pipeline must fail before generation or review.

Missing reviews are blocking. If a sample has fewer than three reviewer rows, its `agent_gate_pass` is false and the batch decision is `blocked_incomplete_agent_review`.

Malformed reviewer output is blocking for that reviewer and sample. The coordinator should preserve the malformed row path or parse error in the report, but it must not silently ignore it.

Reviewer disagreement is expected. Disagreement is handled by majority vote unless a blocker vote is present, in which case human arbitration is required.

## Testing Strategy

Unit tests:

- validate per-review row schema.
- aggregate 3/3, 2/3, 1/3, and 0/3 vote cases.
- mark human arbitration when any reviewer sets `severity = blocker`.
- block incomplete reviews.
- block schema-mismatched eval cards.
- report issue counts without copying full generated text.

CLI tests:

- mock backend writes deterministic review rows.
- import mode reads prewritten review JSONL and produces majority results.
- CLI exits non-zero on blocked agent gate.
- CLI exits zero on agent gate pass.

Regression fixtures:

- Include a small synthetic output with semantic repetition that exact 4-gram repetition does not catch.
- Include a synthetic output with markdown/meta residue.
- Include a synthetic output that passes all three reviewers.

## Reporting

The Stage 4 quality report should no longer decide promotion from rule metrics alone. It should include:

- rule-gate decision.
- agent-review decision.
- human-arbitration decision.
- final promotion decision.

A final expansion decision can only be `ready_for_next_expansion` when:

- rule gate passes.
- agent gate passes.
- no blocker requires unresolved human arbitration.
- any required human spot-check is recorded.

## Migration Plan

1. Mark the current full50 merged result as invalid for promotion evidence.
2. Add schema guard for eval execution cards.
3. Add deterministic quality rules for the known blind spots.
4. Run Stage 1 subagent review on a small subset to calibrate the rubric.
5. Implement the CLI once the rubric stabilizes.
6. Re-run Stage 4.1 quality subset.
7. Re-run full50 only after subset passes rule gate and agent gate.

## Fixed First-Version Policy Decisions

These policy decisions are fixed for the first implementation:

- Majority mode is required: two out of three reviewer passes are needed.
- Any blocker vote requires human arbitration.
- Human arbitration can override agent review only with a recorded reason.
- The system should prefer blocking over false promotion.
