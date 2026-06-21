# Male Webnovel Agent Review Acceptance System Design

## Context

Stage 4.1 full50 exposed a gap in the current acceptance system. The old rule-based hard gate reported `50/50` pass, but manual review found problems that the scorer did not capture:

- eval schema mismatch: `data_cards/eval_cards_50.jsonl` is not an execution-card file.
- semantic repetition: outputs repeat the same relationship, emotion, or conclusion in different wording to fill length.
- non-prose residue: markdown blockquotes, disclaimers, final confirmations, and internal structure notes.
- unnatural endings: outputs often run to the 2500-character cap instead of naturally closing.
- generic phrase overuse and style drift.

The acceptance system must become a layered gate. Rule metrics remain necessary, but they cannot be the only source of truth. Agent reading review must become a required middle layer before human approval or expansion decisions.

The user later clarified that the target is not traditional prose or general literary fiction. The acceptance standard must be aligned to male-frequency commercial webnovel chapters in the direction of Fanqie Novel and Qidian: clear hook, fast readable progression, protagonist agency, genre payoff, and a concrete reason to continue reading.

## Goal

Build a two-stage acceptance workflow for male-frequency webnovel chapter acceptance:

1. Use Codex subagents now to perform structured reading review with majority voting.
2. After the rubric stabilizes, implement a project CLI that runs the same review contract and writes reproducible review artifacts.

The immediate deliverable is a design and later an implementation plan. No new training expansion should proceed until this acceptance layer is in place and has blocked the known bad full50 artifact. Passing means "ready to serve as male webnovel training evidence", not "beautiful literary writing".

## Non-Goals

- Do not expand to 100 or 500 samples in this work.
- Do not rely on one human read-through as the only quality signal.
- Do not treat exact n-gram repetition as sufficient coverage for semantic repetition.
- Do not put generated chapter prose into Markdown reports; reports may include IDs, labels, short issue summaries, and brief compliant snippets only when necessary.
- Do not build a general-purpose literary judge. The system is an acceptance gate for this project’s male webnovel chapter-generation pipeline.
- Do not reward traditional literary polish when it weakens pace, hook, agency, or payoff.
- Do not encode or claim any non-public platform algorithm. Platform profiles are rubric presets based on public-facing genre direction and the project's target style.

## Target Platform Profile

Every review run must declare a `target_platform`:

- `fanqie`: favors direct readability, fast setup, high conflict frequency, clear emotional payoff, and low-friction continuation.
- `qidian`: favors genre contract, power or resource progression, setting continuity, arc-level setup, and payoff that supports long serialization.
- `hybrid_fanqie_qidian`: default first-version profile. It uses Fanqie's pace and hook bias while retaining Qidian-style genre consistency and progression logic.

The review should also carry lightweight `genre_tags` when known, such as `urban`, `xuanhuan`, `xianxia`, `system`, `fantasy`, `sci_fi`, or `mystery`. Unknown tags do not block by themselves, but the profile cannot be empty.

The first version should optimize for chapter-level evidence:

- Is there an opening hook or immediate pressure?
- Does the protagonist have a visible goal, decision, or action?
- Does the chapter create or escalate conflict?
- Is there at least one concrete payoff, reveal, gain, reversal, or pressure increase?
- Does the ending create a specific reason to read the next chapter?

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
- platform profile presence.
- execution-card fields for chapter goal, conflict beat, payoff beat, and ending hook when the selected eval mode requires them.

Rule-gate status values should distinguish:

- `blocked_by_rule_gate`
- `rules_pass_agent_pending`

### Layer 2: Agent Reading Gate

Agent reading review is a majority-vote hard gate. For each eval batch, three reviewer roles read the generated output against the card, metrics, and target platform profile. The reviewer question is not "is this good literature?" The question is "does this behave like an acceptable male webnovel chapter for the selected platform profile?"

Reviewer roles:

1. **Readthrough Structure Reviewer**
   - Checks whether the chapter creates continued-reading momentum.
   - Looks for opening hook, clear chapter goal, scene progression, conflict escalation, rhythm, and ending hook.
   - Detects padding-by-repetition, repeated information points, missing progression, weak conflict movement, and unnatural closure.

2. **Male-Genre Payoff Reviewer**
   - Checks whether the chapter delivers or advances male-frequency genre satisfaction.
   - Looks for protagonist agency, concrete action, status/resource/power gain, reveal, reversal, face-slapping, pressure increase, or setup that clearly promises payoff.
   - Detects passive rumination, setup without payoff, empty "the protagonist understood/changed" claims, and chapters where nothing materially changes.

3. **Platform Style and Compliance Reviewer**
   - Checks whether the output matches the selected `target_platform` profile and genre tags.
   - For `fanqie`, watches pace, low-friction readability, direct conflict, and quick emotional payoff.
   - For `qidian`, watches genre contract, world/rule consistency, progression logic, and serialized arc coherence.
   - Checks whether the output is only prose.
   - Detects style drift, generic AI phrasing, tonal mismatch, mechanical paragraph rhythm, over-explanation, simplified/traditional style mixing, markdown, disclaimers, final confirmations, internal prompt language, residual outline/meta text, truncation, and execution-card requirement failures.

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
- target platform profile and genre tags.
- reviewer rubric prompt.

Outputs:

- `outputs/.../agent_reviews.jsonl`
- `reports/...agent_review_report.md`
- optional manual review notes in `docs/` or `reports/`.

Per-sample review JSON schema:

```json
{
  "id": "sample-id",
  "target_platform": "hybrid_fanqie_qidian",
  "genre_tags": ["urban", "system"],
  "rubric_version": "male_webnovel_v1",
  "reviewer": "readthrough_structure",
  "pass": false,
  "severity": "blocker",
  "issues": ["semantic_repetition", "weak_plot_progression", "weak_ending_hook"],
  "evidence": [
    {
      "type": "summary",
      "location": "middle-to-late output",
      "note": "The chapter restates the same relationship explanation several times without a new action, payoff, or next-chapter pressure."
    }
  ],
  "recommendation": "reject_or_retry",
  "confidence": "high"
}
```

Allowed reviewer names:

- `readthrough_structure`
- `male_genre_payoff`
- `platform_style_compliance`

Allowed severity values:

- `none`
- `minor`
- `major`
- `blocker`

Common issue labels:

- `missing_opening_hook`
- `missing_chapter_goal`
- `weak_conflict_escalation`
- `weak_ending_hook`
- `weak_protagonist_agency`
- `no_visible_payoff`
- `setup_without_payoff`
- `passive_rumination`
- `empty_powerup_claim`
- `male_frequency_contract_miss`
- `platform_tone_mismatch`
- `over_literary_prose`
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
  --target-platform hybrid_fanqie_qidian `
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
- `target_platform`
- `genre_tags`
- `rubric_version`
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
- `target_platform`
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

- `target_platform`
- `rubric_version`
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

Unknown or empty `target_platform` is a blocking error. `genre_tags` may be empty only when the eval card is explicitly genre-agnostic.

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
- block empty or unknown target platform values.
- report issue counts without copying full generated text.

CLI tests:

- mock backend writes deterministic review rows.
- import mode reads prewritten review JSONL and produces majority results.
- CLI exits non-zero on blocked agent gate.
- CLI exits zero on agent gate pass.

Regression fixtures:

- Include a small synthetic output with semantic repetition that exact 4-gram repetition does not catch.
- Include a synthetic output with markdown/meta residue.
- Include a synthetic output with no protagonist action or payoff despite adequate length.
- Include a synthetic output with a literary but slow passage that lacks hook and next-chapter pressure.
- Include a synthetic output that passes all three male-webnovel reviewers.

## Reporting

The Stage 4 quality report should no longer decide promotion from rule metrics alone. It should include:

- target platform profile.
- genre tags when available.
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
2. Build or convert eval execution cards with `target_platform`, `genre_tags`, chapter goal, conflict beat, payoff beat, and ending hook.
3. Add schema guard for eval execution cards.
4. Add deterministic quality rules for the known blind spots.
5. Run Stage 1 subagent review on a small subset to calibrate the male-webnovel rubric.
6. Implement the CLI once the rubric stabilizes.
7. Re-run Stage 4.1 quality subset.
8. Re-run full50 only after subset passes rule gate and agent gate.

## Fixed First-Version Policy Decisions

These policy decisions are fixed for the first implementation:

- Majority mode is required: two out of three reviewer passes are needed.
- Any blocker vote requires human arbitration.
- Human arbitration can override agent review only with a recorded reason.
- The system should prefer blocking over false promotion.
- The default `target_platform` is `hybrid_fanqie_qidian`.
- The first version treats missing hook, missing protagonist agency, no visible payoff, and weak ending hook as male-webnovel acceptance risks even if prose quality looks fluent.
