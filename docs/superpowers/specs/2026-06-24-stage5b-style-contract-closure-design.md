# Stage 5B Style Contract Closure Design

## Goal

Stage 5B turns the current inline Markdown style contract into a versioned, reviewable, hash-bound asset that can be traced through formal SFT data, training manifests, inference records, and later formal execution cards.

The stage also closes one Stage 5A evidence-chain gap before building on top of it: training dry-runs must not accept raw eval cards or write misleading successful manifests when the eval-card schema is wrong.

## Current Context

The project currently has a lightweight style path:

- `src/small_model_train/style_profile.py` computes `chapter_count`, average Chinese characters, average paragraph length, and average dialogue ratio.
- `scripts/build_style_contract.py` writes `style_profile.json` and `style_contract.md`.
- Stage 5A formal SFT gates already require cards to contain `style_contract_id` and `style_contract_sha256`.
- Stage 5A run manifests record preflight and adapter evidence, but the current training input validation only checks whether `--eval-cards` exists and is non-empty.

Stage 5B must preserve the existing simple entrypoints while making formal evidence stricter.

## Selected Approach

Use **Contract Asset First + Strict Gate**.

The implementation should first harden the Stage 5A training input gate, then introduce a real `StyleContract` asset package. The first generated contract asset should default to `approval_status: pending_review`, not `approved`, so it can be inspected before becoming formal training evidence.

Rejected alternatives:

- Metrics-first only: cheaper, but it would leave the project with better reports rather than a formal style asset.
- Full governance all at once: too broad, because card approval lifecycle and sealed data belong to Stage 5C.

## Scope

In scope:

- Require execution-card schema validation for `run_sft_train.py` and `run_sft_smoke.py` eval-card inputs.
- Record `sft_dataset`, `eval_cards`, file hashes, and schema validation results in `run_manifest.json`.
- Create a structured StyleContract JSON asset.
- Generate StyleContract JSON, author-review Markdown, and metrics JSON from the same source corpus.
- Expand style metrics beyond the current four averages.
- Make formal SFT require an approved or frozen StyleContract asset.
- Bind formal SFT cards to the selected contract id and hash.
- Record StyleContract provenance in training manifests.
- Add a Stage 5B Chinese runbook and update README/docs index.

Out of scope:

- Automatically approving the generated StyleContract.
- Rewriting existing `chapter_cards.jsonl` into formal approved cards.
- Building the Stage 5C ChapterExecutionCard compiler.
- Implementing sealed tests, group split, or near-duplicate detection.
- Running larger training or declaring expansion readiness.
- Implementing author feedback, rejection sampling, DPO, or experiment matrices.

## StyleContract Asset

The canonical machine asset is a JSON object with these fields:

- `schema_version`: integer, initially `1`.
- `style_contract_id`: stable id such as `author_main_v1`.
- `approval_status`: one of `draft`, `pending_review`, `approved`, `frozen`, `rejected`.
- `contract_sha256`: SHA-256 over a canonical JSON projection of the contract contents excluding `contract_sha256` itself.
- `created_at`: UTC ISO timestamp.
- `source_corpus`: object with chapter source path, row count, quality filter, corpus sha256, and split summary.
- `profile_metrics`: expanded style metrics generated from accepted source chapters.
- `prompt_rules`: prompt-facing style rules used by SFT and inference.
- `ai_taste_guardrails`: banned or discouraged phrases/patterns.
- `author_notes`: optional human notes, defaulting to an empty string.
- `review`: object with reviewer, reviewed_at, and review_notes; empty for `pending_review`.

The JSON is the source of truth for formal gates. Markdown is only for review.

## Metrics

`style_profile.py` should grow from averages into diagnostic distributions while staying deterministic and cheap to compute.

Required metrics:

- chapter count and Chinese-character min/max/average/p50/p90.
- paragraph character min/max/average/p50/p90.
- dialogue ratio average/p50/p90.
- punctuation density for common Chinese punctuation.
- sentence-length distribution where sentence boundaries can be inferred.
- AI-taste phrase hit counts and rates using the existing phrase list from scoring or a shared constant.
- source filter summary: total rows, selected rows, skipped rows, and skip reasons where practical.

These metrics diagnose drift. They are not the only release gate and must not replace human style review.

## CLI Behavior

`scripts/build_style_contract.py` remains the public entrypoint and should support:

- `--chapters`
- `--contract-json-output`
- `--contract-output`
- `--metrics-output`
- `--style-contract-id`
- `--approval-status`, default `pending_review`
- `--author-notes`, default empty

The command must reject output paths that resolve to the same file. It should write all three artifacts from one in-memory asset so the Markdown and metrics match the JSON.

Expected default output paths in docs:

- `data_style/style_contract_author_main_v1.json`
- `style_contract.md`
- `data_style/style_metrics_author_main_v1.json`

`data_style/` should be treated as a generated local artifact directory and ignored by git, like `data_sft/` and `outputs/`.

## Formal SFT Gate

`scripts/build_sft_dataset.py` should add `--style-contract-json`.

Smoke/dev behavior:

- If `--allow-draft-cards` is present, draft cards may be used.
- A pending StyleContract can be referenced for inspection or smoke/dev prompt building.
- Generated metadata must not describe this path as formal evidence.

Formal behavior:

- `--allow-draft-cards` is absent.
- `--style-contract-json` is required.
- The contract must validate.
- `approval_status` must be `approved` or `frozen`.
- Each card entering formal SFT must contain `style_contract_id` and `style_contract_sha256`.
- Each formal card's id/hash must match the selected contract JSON.
- Missing contract fields, pending status, inline-only style text, or hash mismatch must fail loudly.
- Non-train or non-A chapters may still be skipped according to existing split rules, but no candidate that enters formal SFT may bypass the contract gate.

## Manifest Provenance

`run_sft_train.py` should include style contract provenance in `run_manifest.json` when a style contract is supplied.

Manifest additions:

- `sft_dataset`: path, sha256, row count when readable.
- `eval_cards`: path, sha256, row count, schema validation result.
- `style_contract`: path, `style_contract_id`, `contract_sha256`, `approval_status`, schema validation result.
- `formal_evidence`: boolean. It is true only for non-dry-run training with valid preflight, valid execution cards, approved/frozen style contract, zero training exit, and passing adapter check.

Dry-run manifests may pass command-construction verification, but they must not imply adapter success or formal evidence.

## Stage 5A Gate Fix

Before Stage 5B relies on manifests, `run_sft_train.py` and `run_sft_smoke.py` must validate eval cards with `validate_execution_cards()`, not only file existence.

This specifically prevents `data_cards/eval_cards_50.jsonl` from passing as `data_cards/eval_execution_cards_50.jsonl`.

The failure should happen before command construction and before writing a misleading successful manifest.

## Error Handling

Errors should be explicit and user-actionable:

- Invalid StyleContract JSON: name the missing or invalid field.
- Pending contract in formal mode: explain that `approved` or `frozen` is required.
- Hash mismatch: print card id, card hash, and selected contract hash.
- Raw eval-card schema mismatch: report the execution-card validation error.
- Duplicate output paths in contract generation: reject with argparse error.

No path should silently downgrade formal mode into smoke/dev mode.

## Testing Strategy

Unit tests:

- StyleContract schema validation accepts valid pending/approved/frozen contracts.
- Invalid approval status fails.
- Canonical contract hash is stable and excludes `contract_sha256`.
- Contract hash mismatch fails validation.
- Expanded style metrics are deterministic for small fixtures.
- Empty inputs produce explicit metrics without division errors.

CLI tests:

- `build_style_contract.py` writes JSON, Markdown, and metrics.
- Default generated contract status is `pending_review`.
- Duplicate output paths fail.
- Recomputed contract hash matches the stored hash.

SFT gate tests:

- Pending contract blocks formal SFT.
- Approved contract allows formal SFT.
- Frozen contract allows formal SFT.
- Missing `--style-contract-json` blocks formal SFT.
- Card style contract id/hash mismatch blocks formal SFT.
- Draft cards remain allowed only with `--allow-draft-cards`.

Manifest tests:

- `run_sft_train.py --dry-run` records `sft_dataset`, `eval_cards`, and style contract provenance.
- Raw `eval_cards_50.jsonl` fails training input validation.
- Dry-run manifest does not set `formal_evidence: true`.

Regression tests:

- Existing Stage 5A prompt, raw-output, preflight, rule-projection, and draft-card tests continue to pass.

## Documentation

Create `docs/stage5b-style-contract-closure.zh.md` covering:

- What StyleContract means in this project.
- How to generate the three artifact files.
- Why the default status is `pending_review`.
- How approved/frozen differ from pending.
- How formal SFT consumes a StyleContract.
- What 5B does not prove: no card compiler, no sealed test, no data expansion.

Update:

- `README.md`
- `docs/index.zh.md`
- `docs/project-map.zh.md`
- `docs/pipeline-flow.zh.md` where the style contract flow is described.

## Entry Criteria

- Stage 5A raw generation records and rule-projection isolation are present.
- Formal SFT already rejects draft-only cards by default.
- The Stage 5A training input/schema manifest gap is accepted as the first 5B precondition to fix.

## Exit Criteria

Stage 5B exits when all are true:

- Full pytest suite passes.
- `run_sft_train.py` and `run_sft_smoke.py` reject raw eval-card files that do not match execution-card schema.
- `build_style_contract.py` writes JSON, Markdown, and metrics from the same source data.
- Generated StyleContract defaults to `pending_review`.
- Formal SFT refuses pending contracts.
- Formal SFT accepts approved/frozen contracts with matching card id/hash.
- `run_manifest.json` records dataset, eval-card, and StyleContract provenance.
- Docs describe the Stage 5B operating sequence.

## Follow-On

Stage 5C should consume the approved/frozen StyleContract id and hash when compiling formal ChapterExecutionCard assets.

Stage 5D should use the StyleContract and formal cards to separate plan-execution failures from prose-style failures.

Stage 5E should use StyleContract hashes in experiment manifests so paired comparisons never mix incompatible style assets.
