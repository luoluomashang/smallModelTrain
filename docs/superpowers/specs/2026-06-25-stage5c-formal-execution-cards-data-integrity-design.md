# Stage 5C Formal Execution Cards And Data Integrity Design

## Goal

Stage 5C turns chapter execution cards from generic draft prompts into approved, hash-bound, data-integrity-checked assets. It also closes the remaining Stage 5B documentation mismatch so user-facing smoke/dev commands match the current StyleContract gate.

The outcome is not larger training. The outcome is that formal SFT can prove every training row came from exactly one approved chapter execution card, one approved or frozen StyleContract, one known chapter source, and one non-overlapping dataset split.

## Current Context

Stage 5A made evidence stricter: prompt rendering is shared, generation keeps raw outputs, rule projection is isolated, run manifests are written, and draft cards cannot enter formal SFT by default.

Stage 5B added structured StyleContract assets. Formal SFT now requires `--style-contract-json`, cards must bind `style_contract_id` and `style_contract_sha256`, training manifests record StyleContract provenance, and `formal_evidence` stays false for dry-runs.

One documentation mismatch remains: several older guides still show `build_sft_dataset.py` without `--allow-draft-cards` or `--style-contract-json`. Those commands now fail because the CLI correctly treats missing StyleContract JSON as formal mode.

Existing card code is still lightweight:

- `chapter_cards.py` creates generic draft cards with broad goals and `draft_only: true`.
- `execution_cards.py` validates a small execution-card shape for eval and training gate checks.
- `dataset_split.py` only assigns train/eval by seeded sampling.
- `sft_builder.py` blocks draft cards and checks StyleContract id/hash in formal mode, but it does not yet require one formal card per intended chapter or write a dataset manifest.

## Selected Approach

Use **Documentation Closure + Formal Card Asset First**.

First, repair the user-facing 5B command mismatch. Smoke/dev examples must use `--allow-draft-cards`; formal examples must use `--style-contract-json`. This is intentionally small and does not change runtime behavior.

Second, introduce `ChapterExecutionCard` as a new formal schema with its own compiler, validator, renderer, and dataset manifest. Existing draft cards remain usable for smoke/dev, but formal SFT must consume approved or frozen `ChapterExecutionCard` records.

Rejected alternatives:

- Documentation-only follow-up: leaves formal cards generic and still unable to prove data integrity.
- Full Stage 5C plus author feedback: too broad; AI-taste feedback and same-plot revisions belong to Stage 5D.
- Embedding/vector near-duplicate detection immediately: useful later, but a deterministic text-fingerprint gate is cheaper, reproducible, and enough for the first sealed-data boundary.

## Scope

In scope:

- Update older docs so draft SFT examples use `--allow-draft-cards` and formal examples use `--style-contract-json`.
- Define a structured `ChapterExecutionCard` schema.
- Add card status lifecycle: `draft`, `reviewed`, `approved`, `frozen`, `rejected`.
- Add a Card Compiler / Plan Normalizer that transforms draft cards into formal card candidates without auto-approving them.
- Separate hard constraints from model freedom.
- Require each formal SFT target chapter to have exactly one approved or frozen formal card.
- Validate StyleContract id/hash binding on formal cards.
- Add stable train/validation/sealed group assignment and group hashes.
- Add source, prompt, future-context, reference-fragment, and near-duplicate checks.
- Add a dataset manifest for source hashes, card hashes, split/group provenance, and StyleContract provenance.
- Add a Stage 5C Chinese runbook.

Out of scope:

- Automatically approving cards.
- Generating high-quality plot plans from scratch.
- Expanding to 100/500 samples.
- Author feedback collection, same-plot revision datasets, rejection sampling, DPO, or experiment matrices.
- Replacing human review with rule checks.
- Vector database or semantic dedup infrastructure.

## Stage 5B Documentation Closure

The documentation patch is part of this stage because the implementation already changed the behavior.

Smoke/dev commands that build from `data_cards/chapter_cards.jsonl` should use:

```powershell
python scripts/build_sft_dataset.py --cards data_cards/chapter_cards.jsonl --chapters data_clean/chapters_split.jsonl --output data_sft/sft_chapter_v1.jsonl --dataset-info-output data_sft/dataset_info.json --allow-draft-cards
```

Formal commands should use:

```powershell
python scripts/build_sft_dataset.py --cards data_cards/chapter_execution_cards_approved.jsonl --chapters data_clean/chapters_split.jsonl --output data_sft/sft_chapter_formal.jsonl --dataset-info-output data_sft/dataset_info_formal.json --style-contract-json data_style/style_contract_author_main_v1.json
```

Docs that mention old commands should explain the difference between draft smoke/dev data and formal training data.

## ChapterExecutionCard Schema

The new formal card is a JSON object with these top-level fields:

- `schema_version`: integer, initially `1`.
- `card_id`: stable id for this card asset.
- `chapter_id`: id of the target chapter row.
- `card_status`: one of `draft`, `reviewed`, `approved`, `frozen`, `rejected`.
- `style_contract_id`: StyleContract id.
- `style_contract_sha256`: StyleContract canonical hash.
- `source_chapter_sha256`: hash of the target source chapter text.
- `card_sha256`: hash of the canonical card projection excluding `card_sha256`.
- `target_platform`: existing platform enum such as `hybrid_fanqie_qidian`.
- `genre_tags`: non-empty list of tags.
- `hard_constraints`: object containing required facts and hard bans.
- `execution_plan`: object containing chapter goal, conflict beat, payoff beat, chapter structure, character states, ending hook, and target word count.
- `creative_space`: object containing allowed improvisation areas.
- `provenance`: object with source card id, compiler version, created_at, reviewer, reviewed_at, review_notes, group id, and split assignment.

Formal SFT accepts only `approved` and `frozen` cards. `draft` and `reviewed` cards may be inspected, rendered, and used for smoke/dev only when an explicit override is present.

## Hard Constraints And Creative Space

`hard_constraints` holds facts that the model must not violate:

- `must_include`: concrete entities, events, objects, or reveals.
- `must_not_include`: banned events, prose forms, spoilers, and non正文 artifacts.
- `continuity_facts`: facts from previous chapters that must remain true.
- `forbidden_future_facts`: facts from later chapters or sealed material that must not appear.
- `style_bans`: style or output restrictions inherited from StyleContract and card review.

`creative_space` holds allowed freedom:

- `optional_sensory_details`
- `optional_dialogue_moves`
- `optional_micro_conflicts`
- `allowed_scene_expansion`

The renderer should show both sections clearly. Formal cards must not collapse creative freedom into hard constraints, because that would turn the small model into a mechanical copier rather than an executor.

## Card Compiler / Plan Normalizer

The compiler converts existing draft cards or human/Agent card notes into `ChapterExecutionCard` candidates.

Compiler rules:

- Preserve executable facts: chapter goal, character state, conflict, payoff, must include, must not include, ending hook.
- Normalize `chapter_structure` into ordered beats with positive step numbers and estimated character ranges.
- Move abstract prose advice into `creative_space` only if it names a usable freedom.
- Reject cards whose core goal is only abstract advice such as "节奏紧凑", "写得爽一点", or "减少 AI 味".
- Bind the selected StyleContract id and hash.
- Compute `source_chapter_sha256` from the matched chapter row.
- Output `card_status: reviewed` by default unless the input explicitly requests `draft`.
- Never output `approved` or `frozen`; those states require a human edit or explicit approval command in a later workflow.

This keeps automation useful without letting it silently approve formal training data.

## Formal SFT Admission

Formal SFT must validate all of these before writing rows:

- The StyleContract JSON validates and has `approval_status` `approved` or `frozen`.
- Every candidate card validates as `ChapterExecutionCard`.
- Every trainable chapter selected for formal SFT has exactly one approved or frozen card.
- No approved or frozen card points to a missing chapter.
- No two approved or frozen cards point to the same chapter.
- Card `style_contract_id` and `style_contract_sha256` match the supplied StyleContract.
- Card `source_chapter_sha256` matches the target chapter text.
- The rendered prompt contains no target answer text, source text fragment, future context fragment, or disallowed reference fragment.
- The output dataset writes a manifest with all source, card, style, and split hashes.

Non-train rows may still be skipped according to split rules. What cannot happen is silent skipping of intended train chapters in formal mode.

## Split And Sealed Data Integrity

Stage 5C extends the existing seeded split into stable groups:

- `train`: used for SFT.
- `validation`: used for development comparison and budget tuning.
- `sealed`: reserved for later claims and should not influence training or tuning.

The split should be deterministic from chapter ids, source hashes, and a seed. It should produce a `split_manifest` with:

- `split_seed`
- `split_strategy`
- `group_id` per chapter
- `group_sha256` per group
- counts per split
- source chapter file sha256

No chapter id or group id may appear in more than one split. Formal training must not include validation or sealed groups.

## Leakage Checks

Stage 5C should expand current source-text leakage checks into a reusable data-integrity module.

Checks:

- Target-text leakage: rendered prompt must not include long Chinese fragments from the target chapter text.
- Source-text leakage: existing `source_text` fragments must not be rendered into the prompt.
- Future-context leakage: prompt must not include fragments from later chapter rows or sealed split rows.
- Reference-fragment leakage: if a card includes reference snippets, long verbatim fragments must be disallowed unless explicitly marked as allowed short facts.
- Near-duplicate detection: deterministic fingerprinting should flag high overlap between train, validation, and sealed texts. The first implementation can use normalized Chinese character shingles and Jaccard overlap.

Errors should name card id, chapter id, split, and the fragment or overlap reason when possible.

## Dataset Manifest

Formal SFT output should include a manifest alongside the JSONL dataset.

Required fields:

- `schema_version`
- `created_at`
- `sft_dataset_path`
- `sft_dataset_sha256`
- `row_count`
- `chapters_path`
- `chapters_sha256`
- `cards_path`
- `cards_sha256`
- `style_contract_id`
- `style_contract_sha256`
- `style_contract_path`
- `split_manifest`
- `card_hashes`
- `chapter_hashes`
- `leakage_report`
- `near_duplicate_report`
- `formal_dataset`: boolean

This manifest is separate from `run_manifest.json`. The dataset manifest proves what data was built; the run manifest proves what training command consumed.

## File Map

Likely creates:

- `src/small_model_train/schemas/chapter_execution_card.py`: schema constants, canonical hash, validation helpers, status rules.
- `src/small_model_train/cards/card_compiler.py`: draft-card to formal-card candidate conversion.
- `src/small_model_train/cards/card_validator.py`: batch validation, one-card-per-chapter checks, StyleContract binding checks.
- `src/small_model_train/cards/card_renderer.py`: formal prompt rendering for ChapterExecutionCard.
- `src/small_model_train/data/dedup.py`: deterministic text fingerprint and overlap checks.
- `src/small_model_train/data/dataset_manifest.py`: formal SFT manifest builder.
- `docs/stage5c-formal-execution-cards-data-integrity.zh.md`: operator runbook.

Likely modifies:

- `src/small_model_train/chapter_cards.py`: clarify that generated cards are draft-only inputs to the compiler.
- `src/small_model_train/execution_cards.py`: preserve lightweight eval schema or route to formal schema where appropriate.
- `src/small_model_train/dataset_split.py`: add stable grouped train/validation/sealed split.
- `src/small_model_train/sft_builder.py`: accept formal cards, enforce one-card-per-chapter, write manifest inputs.
- `scripts/build_sft_dataset.py`: add formal-card manifest output and formal-card validation wiring.
- `README.md`, `docs/index.zh.md`, `docs/project-map.zh.md`, `docs/pipeline-flow.zh.md`, `docs/stage1-pipeline-guide.zh.md`, `docs/stage3-data-bring-up-guide.zh.md`, `docs/stage4-smoke-eval-guide.zh.md`, `docs/zero-start.zh.md`: update command language.

Likely tests:

- `tests/test_chapter_execution_card.py`
- `tests/test_card_compiler.py`
- `tests/test_card_validator.py`
- `tests/test_dataset_split.py`
- `tests/test_dataset_manifest.py`
- updates to `tests/test_sft_builder.py`

## CLI Behavior

The existing `build_sft_dataset.py` remains the public entrypoint.

Smoke/dev behavior:

- `--allow-draft-cards` keeps the current draft-card path.
- Smoke/dev output should not write `formal_dataset: true`.

Formal behavior:

- `--style-contract-json` is required.
- `--cards` should point to formal `ChapterExecutionCard` records.
- A new `--dataset-manifest-output` should write the formal dataset manifest.
- A new `--require-all-train-chapters` should make missing train chapter cards fatal. This should be the formal default.

Optional helper CLI:

- `scripts/compile_chapter_execution_cards.py` can turn draft cards plus chapters plus StyleContract JSON into reviewed formal-card candidates.

Approval workflow can remain manual in Stage 5C: users may edit `card_status` from `reviewed` to `approved` after inspection, with hash recomputation handled by a helper or documented command.

## Error Handling

Errors should be explicit and actionable:

- Missing formal card: name the `chapter_id`.
- Duplicate cards: name the conflicting `card_id` values and `chapter_id`.
- StyleContract mismatch: show card id, card hash, and selected contract hash.
- Source chapter hash mismatch: show `chapter_id`, card hash, and recomputed hash.
- Abstract-only card: name the abstract phrase and explain that executable facts are required.
- Leakage: show card id, chapter id, leakage type, and fragment.
- Split overlap: show the duplicate group or chapter id.

No command should silently downgrade formal mode into smoke/dev mode.

## Testing Strategy

Unit tests:

- Valid `ChapterExecutionCard` passes schema validation.
- Missing required fields fail with field names.
- Invalid lifecycle status fails.
- Approved/frozen cards require valid StyleContract id/hash.
- Canonical card hash is stable and excludes `card_sha256`.
- Compiler converts a draft card into a reviewed formal candidate.
- Compiler rejects abstract-only cards.
- Renderer excludes `source_text` and target text.
- Group split is deterministic and non-overlapping.
- Dedup flags high-overlap train/sealed samples.
- Dataset manifest records row count, hashes, split summary, and formal flag.

CLI tests:

- Old smoke/dev command succeeds only with `--allow-draft-cards`.
- Formal command without StyleContract fails.
- Formal command with draft cards fails.
- Formal command with one missing train chapter card fails.
- Formal command with approved cards and matching StyleContract writes JSONL and manifest.

Regression tests:

- Existing Stage 5A and Stage 5B tests continue to pass.
- `run_sft_train.py` still rejects raw eval cards.
- Pending StyleContract still blocks formal SFT.

## Documentation

Create `docs/stage5c-formal-execution-cards-data-integrity.zh.md` covering:

- What makes a card formal.
- How formal cards differ from draft cards.
- How to compile reviewed candidates.
- How humans approve or freeze cards.
- How formal SFT checks one-card-per-chapter and StyleContract binding.
- What the dataset manifest proves.
- Why sealed data must not be used for training or tuning.
- What Stage 5C does not prove.

Update existing docs so users can still run smoke/dev paths after Stage 5B:

- Draft smoke/dev examples use `--allow-draft-cards`.
- Formal examples use approved/frozen StyleContract JSON and approved/frozen formal cards.

## Entry Criteria

- Stage 5B StyleContract JSON assets can be generated and validated.
- Formal SFT already blocks missing or unapproved StyleContract JSON.
- Draft cards can still be used explicitly for smoke/dev.

## Exit Criteria

Stage 5C exits when all are true:

- Full pytest suite passes.
- Old draft-card commands in public docs are no longer misleading.
- Formal SFT refuses generic draft cards.
- Formal SFT refuses missing, duplicate, unapproved, or hash-mismatched formal cards.
- Every formal SFT row can be traced to exactly one formal card, one source chapter, and one StyleContract hash.
- Train, validation, and sealed groups are deterministic and non-overlapping.
- Target/source/future/reference leakage checks are machine-verifiable.
- Near-duplicate checks flag obvious split contamination.
- Formal SFT writes a dataset manifest before training.
- Stage 5C docs explain the operating sequence and boundaries.

## Follow-On

Stage 5D should consume formal cards and sealed evaluation to separate plan-execution failures from prose-style failures, then collect same-plot author revisions and AI-taste defect evidence.

Stage 5E should consume dataset manifests, card hashes, StyleContract hashes, and sealed split metadata for controlled paired experiments.
