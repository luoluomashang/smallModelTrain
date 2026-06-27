# Stage 5D Author Feedback And AI-Taste Reduction Design

## Goal

Stage 5D reduces AI-like prose using formal cards, raw generation evidence, and same-plot author feedback, while first closing the remaining Stage 5C formal admission integrity gap.

The outcome is not larger training and not an experiment matrix. The outcome is a trustworthy author-feedback loop: every accepted improvement can be traced to one formal `ChapterExecutionCard`, one `StyleContract`, one generated output, one author or reviewer decision, and one dataset row candidate.

## Current Context

Stage 5A made evidence raw-first and stopped rule projections from pretending to be independent review. Stage 5B introduced approved or frozen `StyleContract` assets. Stage 5C introduced formal `ChapterExecutionCard` records, formal-card rendering, grouped split helpers, leakage checks, near-duplicate checks, and dataset manifests.

One Stage 5C admission issue must be fixed before Stage 5D can safely consume formal datasets:

- `validate_formal_card_batch()` currently indexes chapters by `chapter_id`, which collapses duplicate chapter ids.
- `build_formal_sft_rows()` then iterates all trainable chapter rows and can reuse the same formal card for multiple rows with the same `chapter_id`.
- A duplicate `train/A` chapter id can therefore produce multiple SFT rows from one card, including rows whose output text does not match the card's `source_chapter_sha256`.
- Dataset manifest helpers also use hash maps keyed by chapter id or card id, so duplicate keys can silently overwrite provenance.

This is not a separate Stage 5C.1. It is Stage 5D Task 0 because Stage 5D depends on a clean formal admission contract.

## Selected Approach

Use a single Stage 5D plan with two layers:

1. **Formal admission repair gate.** Before adding author-feedback workflows, reject duplicate trainable chapter ids, reject duplicate manifest hash keys, and add regression tests for the one-card-multiple-rows failure mode.
2. **Author feedback and AI-taste reduction.** Build structured defect labels, evidence-spanned review records, same-plot author revision records, rejection-sampling SFT rows, and preference candidate rows.

Rejected alternatives:

- Separate Stage 5C.1 plan: technically clean, but it adds process overhead for a small gate fix that Stage 5D must perform before doing useful work.
- Fold the fix invisibly into a late Stage 5D task: too risky, because author-feedback records would inherit a weak card-to-chapter contract.
- Jump to Stage 5E experiments: premature. Controlled experiments need stable same-card, same-style, same-seed records from Stage 5D first.

## Scope

In scope:

- Reject duplicate `train/A` chapter ids in formal SFT admission.
- Reject duplicate chapter or card hash keys when writing formal dataset manifests.
- Preserve the explicit draft-card smoke/dev path behind `--allow-draft-cards`.
- Define an AI-taste defect taxonomy for male webnovel prose.
- Store defect evidence spans against raw generated output, not sanitized-only text.
- Store same-plot author revisions tied to formal cards and StyleContract hashes.
- Build rejection-sampling SFT rows only from accepted same-plot candidates.
- Build optional same-plot preference candidate rows when chosen/rejected evidence exists.
- Track author acceptance rate, edit burden, defect density, and plan-execution regressions.
- Add a Stage 5D Chinese runbook that explains the operating sequence and boundaries.

Out of scope:

- Expanding to 100 or 500 samples.
- DPO, SimPO, ORPO, KTO, reward-model training, or any preference optimization run.
- Stage 5E experiment matrices.
- Automatic approval of formal cards, author revisions, or StyleContract assets.
- Letting a large model replace the author as final prose judge.
- Treating rule projection or deterministic defect labels as independent literary review.

## Formal Admission Repair

Formal SFT admission must reject ambiguous chapter identity before any Stage 5D records are built.

Rules:

- A `train/A` chapter id must appear at most once in the formal chapters input.
- Every `train/A` chapter row must have exactly one approved or frozen formal card.
- Every approved or frozen formal card must point to exactly one existing chapter row.
- A formal card's `source_chapter_sha256` must match that exact chapter row's text.
- Manifest `chapter_hashes` must not overwrite duplicate chapter ids.
- Manifest `card_hashes` must not overwrite duplicate card ids.
- Error messages must name the duplicate id and the affected rows or cards when possible.

The regression case is explicit: two `train/A` rows with the same `id` and one approved card must fail before `build_formal_sft_rows()` writes any rows.

## AI-Taste Defect Taxonomy

Stage 5D introduces a small, stable taxonomy rather than free-form review labels.

Initial labels:

- `generic_phrase`: generic webnovel or assistant-like phrasing.
- `explanation_voice`: over-explaining motives, stakes, or rules instead of dramatizing them.
- `summary_narration`: summary-style narration where scene execution is expected.
- `empty_intensity`: claims of tension or payoff without concrete action.
- `repeated_psychology`: repeated internal-state explanation without new beat movement.
- `dialogue_flatness`: dialogue that states information without character pressure or subtext.
- `payoff_blur`: payoff beat is present but weak, vague, or not tied to the card.
- `hook_blur`: ending hook lacks a specific next-chapter pressure.
- `style_contract_drift`: prose violates approved StyleContract constraints.
- `plan_execution_regression`: AI-taste repair harms card execution.

Each defect record has a severity:

- `minor`: noticeable but not blocking.
- `major`: likely requires revision before reuse.
- `blocker`: cannot enter accepted Stage 5D training candidates.

## Evidence-Spanned Review Records

Defect records must be evidence-backed. A record stores:

- `record_id`
- `schema_version`
- `card_id`
- `chapter_id`
- `style_contract_id`
- `style_contract_sha256`
- `source_output_id`
- `raw_output_sha256`
- `reviewer`
- `reviewed_at`
- `defects`
- `overall_acceptance`
- `notes`

Each defect stores:

- `label`
- `severity`
- `evidence_text`
- `evidence_start`
- `evidence_end`
- `suggested_fix`

Evidence spans are Python character offsets into the raw generated output. If an external reviewer cannot provide offsets, the import path may accept `evidence_text` and resolve the first exact occurrence. If the text cannot be found exactly, import fails instead of storing ungrounded evidence.

## Same-Plot Author Revision Records

Stage 5D's core data unit is a same-plot revision record. It compares a small-model output with an author or reviewer revision under the same formal card and StyleContract.

Required fields:

- `revision_id`
- `schema_version`
- `card_id`
- `chapter_id`
- `style_contract_id`
- `style_contract_sha256`
- `prompt_sha256`
- `raw_output_sha256`
- `model_output`
- `revised_output`
- `revision_status`
- `revision_author`
- `revised_at`
- `edit_summary`
- `defect_record_ids`
- `acceptance_reason`

Allowed `revision_status` values:

- `accepted`
- `accepted_with_minor_edits`
- `rejected`
- `needs_rewrite`

Only `accepted` and `accepted_with_minor_edits` records can feed rejection-sampling SFT rows.

## Rejection-Sampling SFT Rows

The rejection-sampling SFT builder consumes accepted same-plot revision records and formal cards.

Rules:

- The row input must be rendered from the formal card and StyleContract, not copied from the revision record.
- The row output is `revised_output`.
- The builder must validate card hash, StyleContract hash, prompt hash, and source output hash.
- The builder must reject records with `revision_status` other than `accepted` or `accepted_with_minor_edits`.
- The builder must preserve dataset manifest provenance for revision ids and source output hashes.
- The builder must not claim model improvement; it only produces candidate rows.

## Preference Candidate Rows

Preference candidates are optional Stage 5D artifacts. They prepare evidence for later preference optimization but do not run preference training.

Rules:

- `chosen` is an accepted or lightly edited revision.
- `rejected` is the original small-model output or a rejected same-plot candidate.
- `chosen` and `rejected` must share the same `card_id`, `chapter_id`, `style_contract_sha256`, and prompt hash.
- Preference rows must include defect labels explaining why the rejected side is weaker.
- If the evidence is not paired and same-plot, no preference row is written.

## Metrics And Reports

Stage 5D reports should track:

- Number of reviewed outputs.
- Defect counts by label and severity.
- Defects per 10k Chinese characters.
- Author acceptance rate.
- Mean and median edit distance or changed-character count.
- Accepted revision count.
- Rejection-sampling SFT row count.
- Preference candidate row count.
- Plan-execution regression count.
- Whether any sealed or validation data was consumed by training candidates.

Reports must separate deterministic labels from human or author review decisions.

## Data Flow

1. Formal SFT admission repair validates cards, chapters, StyleContract, manifests, and duplicate ids.
2. Existing inference/eval flow produces raw generation rows with prompt and generation metadata.
3. Review import builds defect records with evidence spans.
4. Author or reviewer same-plot revisions are imported as revision records.
5. Rejection-sampling SFT builder writes candidate SFT rows from accepted revisions.
6. Preference candidate builder writes chosen/rejected rows only for valid same-plot pairs.
7. Stage 5D report summarizes defects, acceptance, edit burden, and candidate dataset provenance.

## File Map

Likely creates:

- `src/small_model_train/review/__init__.py`
- `src/small_model_train/review/style_defects.py`
- `src/small_model_train/review/evidence.py`
- `src/small_model_train/review/revision_records.py`
- `src/small_model_train/review/rejection_sampling.py`
- `scripts/build_rejection_sampling_sft.py`
- `scripts/build_same_plot_preference_dataset.py`
- `scripts/build_stage5d_review_report.py`
- `tests/test_style_defects.py`
- `tests/test_review_evidence.py`
- `tests/test_revision_records.py`
- `tests/test_rejection_sampling_sft.py`
- `docs/stage5d-author-feedback-ai-taste-reduction.zh.md`

Likely modifies:

- `src/small_model_train/cards/card_validator.py`
- `src/small_model_train/sft_builder.py`
- `scripts/build_sft_dataset.py`
- `src/small_model_train/data/dataset_manifest.py`
- `tests/test_card_validator.py`
- `tests/test_sft_builder.py`
- `tests/test_dataset_manifest.py`
- `src/small_model_train/preference_builder.py`
- `tests/test_preference_builder.py`
- `docs/superpowers/plans/2026-06-23-small-model-train-full-roadmap.md`

## Testing Strategy

Formal admission repair tests:

- Duplicate `train/A` chapter ids fail formal batch validation.
- Duplicate non-train ids do not block formal SFT unless they are referenced by an approved or frozen formal card.
- One formal card cannot generate two SFT rows for duplicate trainable chapter ids.
- Manifest helpers reject duplicate chapter hash keys.
- Manifest helpers reject duplicate card hash keys.

Defect taxonomy tests:

- Known labels and severities validate.
- Unknown labels fail.
- Unknown severities fail.
- Defect summaries count labels and severities deterministically.

Evidence tests:

- Evidence offsets must point to the exact raw output substring.
- Evidence-text import resolves the first exact occurrence.
- Missing evidence text fails.
- Sanitized-only review records fail.

Revision tests:

- Same-plot revision records require matching card, chapter, StyleContract, prompt, and raw output hashes.
- Accepted statuses can feed rejection-sampling SFT.
- Rejected and needs-rewrite statuses cannot feed SFT.

Builder tests:

- Rejection-sampling SFT rows use formal card rendering for input and revised output for output.
- Builder rejects hash mismatches.
- Preference candidate builder writes rows only for same-card, same-style, same-prompt pairs.
- CLI commands reject missing input files without Python tracebacks.

Regression tests:

- Existing Stage 5A, 5B, and 5C tests continue to pass.
- Draft-card smoke/dev path remains explicit and unaffected.
- Formal SFT still refuses draft cards, reviewed cards, pending StyleContract assets, source hash mismatches, and leakage.

## Documentation

Create `docs/stage5d-author-feedback-ai-taste-reduction.zh.md` covering:

- Why Stage 5D starts with a formal admission repair gate.
- What AI-taste defect labels mean.
- How to import evidence-spanned review records.
- How to import same-plot author revisions.
- How accepted revisions become rejection-sampling SFT candidates.
- Why preference rows are only candidates and do not mean DPO/SimPO has run.
- What Stage 5D does not prove.

Update the roadmap so Stage 5D explicitly includes the merged formal admission repair instead of pointing to a separate Stage 5C.1.

## Entry Criteria

- Stage 5C formal cards, StyleContract binding, grouped split helpers, leakage checks, near-duplicate checks, and dataset manifests exist.
- Raw generation outputs preserve raw text and prompt metadata.
- The duplicate chapter id admission issue is known and reproduced.

## Exit Criteria

Stage 5D exits when all are true:

- Full pytest suite passes.
- Formal SFT rejects duplicate trainable chapter ids before writing rows.
- Formal dataset manifests reject duplicate chapter or card hash keys.
- AI-taste defect records validate labels, severity, evidence spans, and raw output provenance.
- Same-plot revision records validate card, StyleContract, prompt, and raw output provenance.
- Rejection-sampling SFT candidate rows can be built from accepted revisions.
- Preference candidate rows can be built only from valid same-plot chosen/rejected pairs.
- Stage 5D reports summarize defect density, acceptance rate, edit burden, candidate row counts, and plan-execution regressions.
- No Stage 5D artifact claims larger-scale model quality improvement without later paired experiments.

## Follow-On

Stage 5E should consume Stage 5D dataset manifests, revision records, defect summaries, and same-card paired comparisons for controlled experimentation. It should remain blocked until Stage 5D can show stable same-card, same-style, same-seed evidence.
