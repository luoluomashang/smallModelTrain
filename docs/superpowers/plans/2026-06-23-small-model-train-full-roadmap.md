# smallModelTrain Full Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Map every phase from the uploaded design review into a complete project roadmap and record the current Stage 5 closure state.

**Architecture:** Treat the uploaded review as the source-of-truth strategy and split it into sequential delivery stages. Stage 5A-5E now have implemented engineering, control-plane, and data-integrity closure artifacts. Overall Stage 5 engineering/control-plane/data-integrity closure is accepted in `reports/stage5_closure_report.md` after the final verification evidence recorded there; this does not imply model-quality acceptance, production-scale formal training, real preference optimization, or an efficiency win.

**Tech Stack:** Python 3.10+, pytest, JSON/JSONL, Markdown docs, Transformers/PEFT worker path, LLaMA-Factory configs, current `small_model_train` package layout.

---

## Source Mapping

This roadmap maps the uploaded `smallModelTrain_项目设计评审与优化落地方案.md` into repo-native stages:

- Review Phase A `修正证据链` -> **Stage 5A Evidence Chain Hardening**.
- Review Phase B `完成风格契约闭环` -> **Stage 5B Style Contract Closure**.
- Review Phase C `重做正式章节执行卡` -> **Stage 5C Formal Execution Cards And Data Integrity**.
- Review Phase D `消除 AI 味` -> **Stage 5D Author Feedback And AI-Taste Reduction**.
- Review Phase E `参数与效率实验` -> **Stage 5E Controlled Experimentation And Efficiency**.

The project should not expand to larger formal training until Stage 5A proves that evidence, raw outputs, preflight state, adapter state, and review conclusions are trustworthy.

---

## Stage Index

### Stage 5A: Evidence Chain Hardening

**Status:** Implemented and covered by Stage 5 closure artifacts.

**Detailed plan:** `docs/superpowers/plans/2026-06-23-stage5a-evidence-chain-hardening.md`

**Purpose:** Make training, inference, scoring, review, and formal SFT admission produce trustworthy, replayable evidence before expanding data volume or changing training technique.

**Primary deliverables:**

- Shared prompt renderer for SFT and inference.
- Raw-first generation records with sanitizer events.
- Scoring that defaults to `raw_output`.
- Machine-readable model and environment preflight reports.
- Training `run_manifest.json`.
- Rule projection isolated from real review decisions.
- Formal SFT gates that reject draft-only cards.

**Exit criteria:**

- Full pytest suite passes.
- Same prompt renderer is used across SFT and inference.
- Eval outputs retain `raw_output`, `sanitized_output`, `prompt_sha256`, `generated_tokens`, and generation params.
- Failed JSON preflight reports block formal training.
- Zero-exit training with invalid adapter is treated as failed.
- Projection-only review cannot produce `ready_for_next_expansion`.
- Draft cards require an explicit smoke-only override.

---

### Stage 5B: Style Contract Closure

**Status:** Implemented as full-corpus StyleContract artifacts; formal SFT closure uses the separate approved Stage 5C closure probe recorded in `reports/stage5_closure_report.md`.

**Plan file:** `docs/superpowers/plans/2026-06-23-stage5b-style-contract-closure.md`

**Purpose:** Turn author style from inline prompt text into a versioned, approved, traceable asset shared by training, inference, adapter manifests, and evaluation.

**Planned scope:**

- Create structured `StyleContract` JSON schema with `schema_version`, `style_contract_id`, `approved_by_author`, `contract_hash`, source corpus metadata, and author notes.
- Expand `style_profile.py` beyond averages into sentence, paragraph, dialogue, punctuation, rhythm, and AI-taste diagnostic distributions.
- Emit three style artifacts: machine JSON, author-review Markdown, and metrics JSON.
- Add author approval state and manual override fields.
- Bind every formal card and SFT row to `style_contract_id` and `style_contract_sha256`.
- Add adapter manifest fields for style contract provenance.
- Add style-metric reporting that diagnoses drift without becoming the sole release gate.
- Add an author blind-review table template.

**Likely files:**

- Create: `src/small_model_train/schemas/style_contract.py`
- Create: `src/small_model_train/style_contract.py`
- Modify: `src/small_model_train/style_profile.py`
- Modify: `scripts/build_style_contract.py`
- Modify: `src/small_model_train/sft_builder.py`
- Modify: `src/small_model_train/stage2_inference.py`
- Modify: `src/small_model_train/stage4_quality.py`
- Create: `tests/test_style_contract.py`
- Modify: `tests/test_style_profile.py`
- Create: `docs/stage5b-style-contract-closure.zh.md`

**Entry criteria:**

- Stage 5A raw generation and manifest fields are present.
- Formal SFT can distinguish smoke-only draft data from approved training data.

**Exit criteria:**

- Formal training data cannot be built with an unapproved style contract.
- Adapter and eval records can be traced to a specific contract hash.
- Style metrics can identify concrete drift categories.
- Author blind-review artifacts exist but remain human-governed.

---

### Stage 5C: Formal Execution Cards And Data Integrity

**Status:** Implemented as formal data-integrity gates plus a minimal approved closure-probe dataset and manifest recorded in `reports/stage5_closure_report.md`.

**Plan file:** `docs/superpowers/plans/2026-06-23-stage5c-formal-execution-cards-data-integrity.md`

**Purpose:** Replace generic draft chapter cards with concrete, approved `ChapterExecutionCard` records that tell the small model exactly what to execute while preserving freedom in how to write.

**Planned scope:**

- Define `ChapterExecutionCard` schema with hard constraints and free-space fields.
- Add approval lifecycle: `draft -> reviewed -> approved -> frozen`.
- Add a Card Compiler / Plan Normalizer that removes abstract writing advice and keeps executable facts.
- Separate hard constraints from allowed creative space.
- Require one card per target chapter with no silent skipping.
- Add work/group split with stable hash and sealed test support.
- Add leakage checks for target text, future context, reference fragments, and near duplicates.
- Add dataset manifest with source hashes and split provenance.

**Likely files:**

- Create: `src/small_model_train/schemas/chapter_execution_card.py`
- Create: `src/small_model_train/cards/card_compiler.py`
- Create: `src/small_model_train/cards/card_validator.py`
- Create: `src/small_model_train/cards/card_renderer.py`
- Modify: `src/small_model_train/chapter_cards.py`
- Modify: `src/small_model_train/execution_cards.py`
- Modify: `src/small_model_train/dataset_split.py`
- Create: `src/small_model_train/data/dedup.py`
- Create: `src/small_model_train/data/dataset_manifest.py`
- Modify: `src/small_model_train/sft_builder.py`
- Create: `tests/test_chapter_execution_card.py`
- Create: `tests/test_card_compiler.py`
- Modify: `tests/test_dataset_split.py`
- Create: `docs/stage5c-formal-execution-cards-data-integrity.zh.md`

**Entry criteria:**

- Stage 5B style contracts can be referenced by ID and hash.
- Formal SFT already blocks draft-only data by default.

**Exit criteria:**

- No generic placeholder card can enter formal training.
- Each approved card maps to exactly one target chapter.
- Train/validation/sealed-test groups do not overlap.
- Source and prompt leakage checks are machine-verifiable.
- Card text avoids abstract LLM writing advice.

---

### Stage 5D: Author Feedback And AI-Taste Reduction

**Status:** Implemented as Stage 5D docs and data-candidate tooling after merging the Stage 5C.1 formal admission repair into this stage.

**Plan file:** `docs/superpowers/plans/2026-06-27-stage5d-author-feedback-ai-taste-reduction.md`

**Purpose:** Reduce AI-like prose using same-plot production feedback without letting the large model take over final writing.

**Planned scope:**

- First repair Stage 5C formal admission gaps: duplicate trainable chapter id gates, and checks for duplicate manifest card_id/chapter_id keys that would silently overwrite card_hashes/chapter_hashes entries.
- Create an AI-taste defect taxonomy from the review document.
- Add evidence-spanned defect records for generated outputs.
- Collect small-model outputs and author same-plot revisions with card, StyleContract, prompt, and raw output provenance.
- Build rejection-sampling SFT rows from accepted candidates.
- Add local rewrite record format for targeted small-model rewrites.
- Build same-plot preference candidate rows from valid accepted revisions.
- Track author acceptance rate, major-edit character count, defect rate per thousand characters, and regression samples.
- Do not run DPO, SimPO, ORPO, KTO, reward model training, or preference optimization in Stage 5D; preference rows are candidate data only.

**Likely files:**

- Create: `src/small_model_train/review/style_defects.py`
- Create: `src/small_model_train/review/evidence.py`
- Modify: `src/small_model_train/agent_review.py`
- Modify: `src/small_model_train/preference_builder.py`
- Create: `src/small_model_train/inference/local_reviser.py`
- Create: `scripts/build_rejection_sampling_sft.py`
- Create: `scripts/build_same_plot_preference_dataset.py`
- Create: `tests/test_style_defects.py`
- Create: `tests/test_rejection_sampling_sft.py`
- Modify: `tests/test_preference_builder.py`
- Create: `docs/stage5d-author-feedback-ai-taste-reduction.zh.md`

**Entry criteria:**

- Sealed test and group split are available.
- Formal execution cards can reliably distinguish plan failures from prose failures.
- Raw outputs and sanitizer events are preserved.

**Exit criteria:**

- Formal admission rejects duplicate trainable chapter ids, and duplicate manifest card_id/chapter_id keys that would silently overwrite card_hashes/chapter_hashes entries.
- AI-taste defects are recorded with labels and evidence spans.
- Same-plot author revisions can be transformed into training data.
- Style improvements do not reduce plan execution pass rate.
- Author acceptance rate and edit burden are tracked across runs.

---

### Stage 5E: Controlled Experimentation And Efficiency

**Status:** Control-plane implemented. Overall Stage 5 engineering/control-plane/data-integrity closure is accepted in `reports/stage5_closure_report.md` after the final verification evidence recorded there.

**Plan file:** `docs/superpowers/plans/2026-06-28-stage5e-controlled-experimentation-efficiency.md`

**Purpose:** Run controlled model, data, PEFT, generation, and efficiency experiments only after the evidence system can prove whether each change helps.

**Planned scope:**

- Define paired experiment manifests.
- Compare base model, current SFT, style-contract-enhanced SFT, formal-card SFT, rejection-sampling SFT, optional CPT, optional later preference-optimization methods, and candidate PEFT variants.
- Record seeds, model revisions, dataset hashes, style contract hashes, card set hashes, generation params, and adapter hashes.
- Add paired eval report with win/loss/tie and regression samples.
- Experiment with learning rate, rank, rsLoRA, PiSSA, DoRA, packing, FlashAttention, and alternative base models one major variable at a time.

**Likely files:**

- Create: `src/small_model_train/evaluation/paired_eval.py`
- Create: `src/small_model_train/training/run_manifest.py`
- Modify: `src/small_model_train/run_manifest.py`
- Create: `scripts/build_paired_eval_report.py`
- Create: `scripts/run_experiment_matrix.py`
- Create: `tests/test_paired_eval.py`
- Create: `docs/stage5e-controlled-experimentation-efficiency.zh.md`

**Entry criteria:**

- `scripts/check_stage5e_entry.py` exits 0 and writes `reports/stage5e_entry_check.json` with `"passed": true`.
- Full pytest passes after the Stage 5D closure plan.

**Exit criteria:**

- Every experiment changes one primary variable.
- Reports include paired comparisons, variance, regression samples, and author acceptance signals.
- Efficiency gains do not mask prose or execution regressions.

---

## Global Gates

These gates apply to every stage after Stage 5A:

- No stage can promote an artifact using only sanitized text.
- No stage can treat rule projection as independent literary review.
- No formal SFT can use draft cards or unapproved style contracts.
- No expansion can proceed without run, dataset, style, adapter, and eval manifests.
- No model-technique experiment can skip base-model and previous-best paired baselines.

---

## Current Execution Decision

Stage 5E gate and control-plane artifacts exist. Overall Stage 5 engineering/control-plane/data-integrity closure is accepted in `reports/stage5_closure_report.md` after the final verification evidence recorded there.

This roadmap status does not imply model-quality acceptance, real preference optimization, production-scale formal training, or an efficiency win.

---

## Self-Review

- Spec coverage: all uploaded review phases A-E have a repo-native stage, plan or closure reference, purpose, scope, entry criteria, and exit criteria.
- Placeholder scan: Stage 5A-5E now have repo-native docs, closure artifacts, and verification gates; any future model-quality, production-scale training, preference-optimization, or efficiency-win work must be planned separately.
- Type consistency: Stage naming follows `5A` through `5E`; overall Stage 5 closure is recorded in `reports/stage5_closure_report.md`.
