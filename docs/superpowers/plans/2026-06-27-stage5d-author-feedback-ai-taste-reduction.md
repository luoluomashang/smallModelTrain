# Stage 5D Author Feedback And AI-Taste Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Stage 5D so author feedback and AI-taste reduction can produce traceable same-plot revision data, while first repairing the Stage 5C formal admission duplicate-id gap.

**Architecture:** Keep formal admission checks in the existing card/SFT/manifest path. Add a focused `small_model_train.review` package for defect taxonomy, evidence-spanned review records, same-plot revision records, rejection-sampling SFT rows, and Stage 5D reporting. Keep preference optimization out of scope: Stage 5D may build same-plot preference candidates, but it must not run DPO/SimPO/ORPO/KTO.

**Tech Stack:** Python 3.10+, pytest, JSON/JSONL, Markdown docs, stdlib `hashlib`, current `small_model_train` package, existing `scripts/` CLI style.

---

## Scope Check

This plan implements the approved spec:

- `docs/superpowers/specs/2026-06-27-stage5d-author-feedback-ai-taste-reduction-design.md`

It intentionally folds the Stage 5C duplicate chapter id repair into Stage 5D Task 1. There is no separate Stage 5C.1 plan.

Out of scope:

- Expanding to 100 or 500 samples.
- DPO, SimPO, ORPO, KTO, reward-model training, or preference optimization runs.
- Stage 5E experiment matrix.
- Automatic approval of cards, revisions, or StyleContract assets.

Expected baseline before this plan:

```powershell
python -m pytest -q
```

Expected current result: `476 passed`.

---

## File Map

- Modify: `src/small_model_train/cards/card_validator.py`
  - Reject duplicate trainable chapter ids before formal SFT can write rows.
- Modify: `scripts/build_sft_dataset.py`
  - Reject duplicate `chapter_hashes` and `card_hashes` keys before manifest creation.
- Modify: `tests/test_card_validator.py`
  - Cover duplicate trainable ids and allowed non-train duplicates.
- Modify: `tests/test_sft_builder.py`
  - Cover the one-card-multiple-rows regression and manifest duplicate-key rejection.
- Modify: `tests/test_dataset_manifest.py`
  - Cover manifest duplicate-key protection through the build script helpers.
- Create: `src/small_model_train/review/__init__.py`
  - Package marker for Stage 5D review helpers.
- Create: `src/small_model_train/review/style_defects.py`
  - AI-taste labels, severities, validation, and summaries.
- Create: `tests/test_style_defects.py`
  - Unit coverage for taxonomy validation and summaries.
- Create: `src/small_model_train/review/evidence.py`
  - Evidence-spanned review record validation and JSONL helpers.
- Create: `tests/test_review_evidence.py`
  - Unit coverage for span validation and text-to-offset resolution.
- Create: `src/small_model_train/review/revision_records.py`
  - Same-plot author revision validation and JSONL helpers.
- Create: `tests/test_revision_records.py`
  - Unit coverage for revision provenance and accepted-status filters.
- Create: `src/small_model_train/review/rejection_sampling.py`
  - Rejection-sampling SFT and same-plot preference candidate builders.
- Create: `tests/test_rejection_sampling_sft.py`
  - Unit and CLI coverage for accepted revisions and preference candidates.
- Create: `scripts/build_rejection_sampling_sft.py`
  - CLI for Stage 5D candidate SFT rows.
- Create: `scripts/build_same_plot_preference_dataset.py`
  - CLI for Stage 5D same-plot preference candidates.
- Create: `src/small_model_train/review/stage5d_report.py`
  - Stage 5D report summarizer and Markdown renderer.
- Create: `scripts/build_stage5d_review_report.py`
  - CLI for Stage 5D JSON and Markdown report.
- Create: `tests/test_stage5d_report.py`
  - Unit and CLI coverage for report metrics.
- Modify: `src/small_model_train/preference_builder.py`
  - Export Stage 5D same-plot preference builder or delegate to `review.rejection_sampling`.
- Modify: `tests/test_preference_builder.py`
  - Preserve legacy failed-eval candidate behavior and add Stage 5D delegation coverage.
- Create: `docs/stage5d-author-feedback-ai-taste-reduction.zh.md`
  - Chinese operator runbook.
- Modify: `README.md`
  - Add Stage 5D guide link.
- Modify: `docs/index.zh.md`
  - Add Stage 5D guide link.
- Modify: `docs/project-map.zh.md`
  - Document Stage 5D artifacts.
- Modify: `docs/pipeline-flow.zh.md`
  - Add Stage 5D data flow.
- Modify: `docs/superpowers/plans/2026-06-23-small-model-train-full-roadmap.md`
  - Update Stage 5D forward index to include the merged formal admission repair.

---

## Task 1: Repair Formal Admission Duplicate-Key Gates

**Files:**
- Modify: `src/small_model_train/cards/card_validator.py`
- Modify: `scripts/build_sft_dataset.py`
- Modify: `tests/test_card_validator.py`
- Modify: `tests/test_sft_builder.py`
- Modify: `tests/test_dataset_manifest.py`

- [ ] **Step 1: Add failing validator tests for duplicate trainable chapter ids**

Append to `tests/test_card_validator.py`:

```python

def test_validate_formal_card_batch_rejects_duplicate_train_chapter_ids():
    from small_model_train.cards.card_validator import validate_formal_card_batch

    text = "这一章用于计算来源哈希。"
    chapters = [
        {"id": "c1", "text": "第一条重复章节正文。", "split": "train", "quality_tag": "A"},
        {"id": "c1", "text": text, "split": "train", "quality_tag": "A"},
    ]
    result = validate_formal_card_batch([_formal_card("c1", text=text)], chapters, _style_contract())

    assert result["passed"] is False
    assert "duplicate trainable chapter id: c1" in "\n".join(result["errors"])


def test_validate_formal_card_batch_allows_duplicate_non_train_ids_when_unreferenced():
    from small_model_train.cards.card_validator import validate_formal_card_batch

    train_text = "这一章用于计算来源哈希。"
    chapters = [
        {"id": "c1", "text": train_text, "split": "train", "quality_tag": "A"},
        {"id": "dup", "text": "验证集重复一。", "split": "validation", "quality_tag": "A"},
        {"id": "dup", "text": "验证集重复二。", "split": "validation", "quality_tag": "A"},
    ]
    result = validate_formal_card_batch([_formal_card("c1", text=train_text)], chapters, _style_contract())

    assert result["passed"] is True
    assert result["errors"] == []
```

- [ ] **Step 2: Add the one-card-multiple-rows regression test**

Append to `tests/test_sft_builder.py`:

```python

def test_build_formal_sft_rows_rejects_duplicate_train_chapter_id_before_rows():
    contract = _formal_style_contract_for_stage5c()
    chapters = [
        {"id": "c1", "text": "第一条重复章节正文。", "split": "train", "quality_tag": "A"},
        {"id": "c1", "text": "这一章用于计算来源哈希。", "split": "train", "quality_tag": "A"},
    ]
    card = _formal_card_for_stage5c(text="这一章用于计算来源哈希。")

    with pytest.raises(ValueError, match="duplicate trainable chapter id: c1"):
        build_formal_sft_rows([card], chapters, contract)
```

- [ ] **Step 3: Add failing manifest duplicate-key tests**

Append to `tests/test_dataset_manifest.py`:

```python

def test_build_sft_dataset_chapter_hashes_reject_duplicate_ids():
    import importlib.util
    from pathlib import Path

    script_path = Path("scripts/build_sft_dataset.py")
    spec = importlib.util.spec_from_file_location("build_sft_dataset", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with pytest.raises(ValueError, match="duplicate chapter id for manifest: c1"):
        module._chapter_hashes(
            [
                {"id": "c1", "text": "第一条正文"},
                {"id": "c1", "text": "第二条正文"},
            ]
        )


def test_build_sft_dataset_card_hashes_reject_duplicate_ids():
    import importlib.util
    from pathlib import Path

    script_path = Path("scripts/build_sft_dataset.py")
    spec = importlib.util.spec_from_file_location("build_sft_dataset", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with pytest.raises(ValueError, match="duplicate card id for manifest: card-c1-v1"):
        module._card_hashes(
            [
                {"card_id": "card-c1-v1", "card_sha256": "a" * 64},
                {"card_id": "card-c1-v1", "card_sha256": "b" * 64},
            ]
        )
```

- [ ] **Step 4: Run focused tests and verify failure**

Run:

```powershell
python -m pytest tests/test_card_validator.py::test_validate_formal_card_batch_rejects_duplicate_train_chapter_ids tests/test_sft_builder.py::test_build_formal_sft_rows_rejects_duplicate_train_chapter_id_before_rows tests/test_dataset_manifest.py::test_build_sft_dataset_chapter_hashes_reject_duplicate_ids tests/test_dataset_manifest.py::test_build_sft_dataset_card_hashes_reject_duplicate_ids -q
```

Expected: fail because duplicate chapter ids and duplicate manifest keys are not rejected yet.

- [ ] **Step 5: Implement trainable duplicate-id detection**

Modify `src/small_model_train/cards/card_validator.py`.

Add below `FUTURE_CONTEXT_SPLITS`:

```python

def _is_trainable_chapter(chapter: dict[str, Any]) -> bool:
    return chapter.get("split") == "train" and chapter.get("quality_tag") == "A"
```

Inside `validate_formal_card_batch()`, immediately after the StyleContract approval-status check and before `chapter_by_id = _chapter_by_id(chapters)`, add:

```python
    errors.extend(_duplicate_trainable_chapter_id_errors(chapters))
```

Replace the `required_train_chapter_ids` comprehension with:

```python
    required_train_chapter_ids = {
        str(chapter.get("id"))
        for chapter in chapters
        if isinstance(chapter, dict)
        and _is_trainable_chapter(chapter)
        and chapter.get("id") is not None
    }
```

Add below `_chapter_by_id()`:

```python

def _duplicate_trainable_chapter_id_errors(chapters: list[dict[str, Any]]) -> list[str]:
    first_index_by_id: dict[str, int] = {}
    errors: list[str] = []
    for index, chapter in enumerate(chapters, start=1):
        if not isinstance(chapter, dict) or not _is_trainable_chapter(chapter):
            continue
        if chapter.get("id") is None:
            continue
        chapter_id = str(chapter["id"])
        previous_index = first_index_by_id.get(chapter_id)
        if previous_index is None:
            first_index_by_id[chapter_id] = index
        else:
            errors.append(
                "duplicate trainable chapter id: "
                f"{chapter_id} rows {previous_index}, {index}"
            )
    return errors
```

- [ ] **Step 6: Implement manifest duplicate-key rejection**

Replace `_card_hashes()` and `_chapter_hashes()` in `scripts/build_sft_dataset.py` with:

```python
def _card_hashes(cards: list[dict]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for index, card in enumerate(cards, start=1):
        if "card_id" not in card or "card_sha256" not in card:
            continue
        card_id = str(card["card_id"])
        if card_id in hashes:
            raise ValueError(f"duplicate card id for manifest: {card_id} row {index}")
        hashes[card_id] = str(card["card_sha256"])
    return hashes


def _chapter_hashes(chapters: list[dict]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for index, chapter in enumerate(chapters, start=1):
        if "id" not in chapter:
            continue
        chapter_id = str(chapter["id"])
        if chapter_id in hashes:
            raise ValueError(f"duplicate chapter id for manifest: {chapter_id} row {index}")
        hashes[chapter_id] = text_sha256(str(chapter.get("text") or ""))
    return hashes
```

- [ ] **Step 7: Run focused tests and verify pass**

Run:

```powershell
python -m pytest tests/test_card_validator.py tests/test_sft_builder.py tests/test_dataset_manifest.py -q
```

Expected: pass.

- [ ] **Step 8: Commit Task 1**

Run:

```bash
git add src/small_model_train/cards/card_validator.py scripts/build_sft_dataset.py tests/test_card_validator.py tests/test_sft_builder.py tests/test_dataset_manifest.py
git commit -m "fix: reject ambiguous formal admission ids"
```

---

## Task 2: Add AI-Taste Defect Taxonomy

**Files:**
- Create: `src/small_model_train/review/__init__.py`
- Create: `src/small_model_train/review/style_defects.py`
- Create: `tests/test_style_defects.py`

- [ ] **Step 1: Write failing taxonomy tests**

Create `tests/test_style_defects.py`:

```python
from __future__ import annotations

import pytest


def test_validate_style_defect_accepts_known_label_and_severity():
    from small_model_train.review.style_defects import validate_style_defect

    defect = validate_style_defect(
        {
            "label": "generic_phrase",
            "severity": "major",
            "evidence_text": "他知道，这一刻已经没有退路。",
            "evidence_start": 2,
            "evidence_end": 12,
            "suggested_fix": "改成具体动作承压。",
        }
    )

    assert defect["label"] == "generic_phrase"
    assert defect["severity"] == "major"


@pytest.mark.parametrize("label", ["unknown", "", 123])
def test_validate_style_defect_rejects_unknown_label(label):
    from small_model_train.review.style_defects import validate_style_defect

    with pytest.raises(ValueError, match="defects\\[0\\].label"):
        validate_style_defect(
            {
                "label": label,
                "severity": "minor",
                "evidence_text": "文本",
                "evidence_start": 0,
                "evidence_end": 2,
                "suggested_fix": "",
            }
        )


def test_summarize_style_defects_counts_labels_and_severity():
    from small_model_train.review.style_defects import summarize_style_defects

    summary = summarize_style_defects(
        [
            {"label": "generic_phrase", "severity": "minor"},
            {"label": "generic_phrase", "severity": "major"},
            {"label": "hook_blur", "severity": "blocker"},
        ]
    )

    assert summary["total_defects"] == 3
    assert summary["by_label"] == {"generic_phrase": 2, "hook_blur": 1}
    assert summary["by_severity"] == {"blocker": 1, "major": 1, "minor": 1}
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
python -m pytest tests/test_style_defects.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'small_model_train.review'`.

- [ ] **Step 3: Create review package marker**

Create `src/small_model_train/review/__init__.py`:

```python
"""Stage 5D review helpers."""
```

- [ ] **Step 4: Implement taxonomy module**

Create `src/small_model_train/review/style_defects.py`:

```python
from __future__ import annotations

from typing import Any


DEFECT_LABELS = {
    "generic_phrase",
    "explanation_voice",
    "summary_narration",
    "empty_intensity",
    "repeated_psychology",
    "dialogue_flatness",
    "payoff_blur",
    "hook_blur",
    "style_contract_drift",
    "plan_execution_regression",
}
DEFECT_SEVERITIES = {"minor", "major", "blocker"}


def validate_style_defect(defect: dict[str, Any], *, index: int = 0) -> dict[str, Any]:
    if not isinstance(defect, dict):
        raise ValueError(f"defects[{index}] must be a JSON object")
    label = defect.get("label")
    if label not in DEFECT_LABELS:
        raise ValueError(f"defects[{index}].label must be one of: {', '.join(sorted(DEFECT_LABELS))}")
    severity = defect.get("severity")
    if severity not in DEFECT_SEVERITIES:
        raise ValueError(
            f"defects[{index}].severity must be one of: {', '.join(sorted(DEFECT_SEVERITIES))}"
        )
    for field in ("evidence_text", "suggested_fix"):
        if not isinstance(defect.get(field), str):
            raise ValueError(f"defects[{index}].{field} must be a string")
    for field in ("evidence_start", "evidence_end"):
        value = defect.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"defects[{index}].{field} must be an int >= 0")
    if defect["evidence_end"] < defect["evidence_start"]:
        raise ValueError(f"defects[{index}].evidence_end must be >= evidence_start")
    return defect


def validate_style_defects(defects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(defects, list):
        raise ValueError("defects must be a list")
    return [validate_style_defect(defect, index=index) for index, defect in enumerate(defects)]


def summarize_style_defects(defects: list[dict[str, Any]]) -> dict[str, Any]:
    by_label: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    for defect in defects:
        label = str(defect.get("label"))
        severity = str(defect.get("severity"))
        by_label[label] = by_label.get(label, 0) + 1
        by_severity[severity] = by_severity.get(severity, 0) + 1
    return {
        "total_defects": len(defects),
        "by_label": dict(sorted(by_label.items())),
        "by_severity": dict(sorted(by_severity.items())),
    }
```

- [ ] **Step 5: Run taxonomy tests**

Run:

```powershell
python -m pytest tests/test_style_defects.py -q
```

Expected: pass.

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add src/small_model_train/review/__init__.py src/small_model_train/review/style_defects.py tests/test_style_defects.py
git commit -m "feat: define stage5d defect taxonomy"
```

---

## Task 3: Add Evidence-Spanned Review Records

**Files:**
- Create: `src/small_model_train/review/evidence.py`
- Create: `tests/test_review_evidence.py`

- [ ] **Step 1: Write failing evidence tests**

Create `tests/test_review_evidence.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest


RAW_OUTPUT = "林默没有解释，只把合同推到桌面。岳家的人第一次停住。"


def _record(**overrides) -> dict:
    from small_model_train.schemas.chapter_execution_card import text_sha256

    record = {
        "record_id": "review-c1-001",
        "schema_version": 1,
        "card_id": "card-c1-v1",
        "chapter_id": "c1",
        "style_contract_id": "contract-v1",
        "style_contract_sha256": "a" * 64,
        "source_output_id": "gen-c1",
        "raw_output_sha256": text_sha256(RAW_OUTPUT),
        "reviewer": "author",
        "reviewed_at": "2026-06-27T00:00:00Z",
        "defects": [
            {
                "label": "dialogue_flatness",
                "severity": "minor",
                "evidence_text": "岳家的人第一次停住",
                "evidence_start": RAW_OUTPUT.index("岳家的人第一次停住"),
                "evidence_end": RAW_OUTPUT.index("岳家的人第一次停住") + len("岳家的人第一次停住"),
                "suggested_fix": "补一处更具体的压迫反应。",
            }
        ],
        "overall_acceptance": "accepted_with_minor_edits",
        "notes": "可用。",
    }
    record.update(overrides)
    return record


def test_validate_review_record_accepts_matching_span():
    from small_model_train.review.evidence import validate_review_record

    record = validate_review_record(_record(), raw_output=RAW_OUTPUT)

    assert record["defects"][0]["evidence_text"] == "岳家的人第一次停住"


def test_validate_review_record_rejects_sanitized_only_without_raw_text():
    from small_model_train.review.evidence import validate_review_record

    with pytest.raises(ValueError, match="raw_output is required"):
        validate_review_record(_record(), raw_output="")


def test_validate_review_record_rejects_mismatched_span():
    from small_model_train.review.evidence import validate_review_record

    record = _record()
    record["defects"][0]["evidence_start"] = 0
    record["defects"][0]["evidence_end"] = 2

    with pytest.raises(ValueError, match="evidence span does not match evidence_text"):
        validate_review_record(record, raw_output=RAW_OUTPUT)


def test_resolve_evidence_text_fills_offsets():
    from small_model_train.review.evidence import resolve_evidence_text

    defect = resolve_evidence_text(
        {
            "label": "generic_phrase",
            "severity": "major",
            "evidence_text": "合同推到桌面",
            "suggested_fix": "换成动作链。",
        },
        raw_output=RAW_OUTPUT,
        index=0,
    )

    assert RAW_OUTPUT[defect["evidence_start"] : defect["evidence_end"]] == "合同推到桌面"


def test_read_write_review_records_round_trip(tmp_path: Path):
    from small_model_train.review.evidence import read_review_records, write_review_records

    path = tmp_path / "review_records.jsonl"
    write_review_records(path, [_record()], raw_outputs={"gen-c1": RAW_OUTPUT})

    assert read_review_records(path, raw_outputs={"gen-c1": RAW_OUTPUT})[0]["record_id"] == "review-c1-001"
```

- [ ] **Step 2: Run evidence tests and verify failure**

Run:

```powershell
python -m pytest tests/test_review_evidence.py -q
```

Expected: fail because `small_model_train.review.evidence` does not exist.

- [ ] **Step 3: Implement evidence module**

Create `src/small_model_train/review/evidence.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from small_model_train.review.style_defects import validate_style_defect, validate_style_defects
from small_model_train.schemas.chapter_execution_card import text_sha256


SCHEMA_VERSION = 1
ACCEPTANCE_STATUSES = {"accepted", "accepted_with_minor_edits", "rejected", "needs_rewrite"}
REQUIRED_FIELDS = (
    "record_id",
    "schema_version",
    "card_id",
    "chapter_id",
    "style_contract_id",
    "style_contract_sha256",
    "source_output_id",
    "raw_output_sha256",
    "reviewer",
    "reviewed_at",
    "defects",
    "overall_acceptance",
    "notes",
)


def resolve_evidence_text(defect: dict[str, Any], *, raw_output: str, index: int) -> dict[str, Any]:
    if not raw_output:
        raise ValueError("raw_output is required for evidence span resolution")
    if not isinstance(defect.get("evidence_text"), str) or not defect["evidence_text"]:
        raise ValueError(f"defects[{index}].evidence_text must be a non-empty string")
    start = raw_output.find(defect["evidence_text"])
    if start < 0:
        raise ValueError(f"defects[{index}].evidence_text not found in raw_output")
    resolved = dict(defect)
    resolved["evidence_start"] = start
    resolved["evidence_end"] = start + len(defect["evidence_text"])
    return validate_style_defect(resolved, index=index)


def validate_review_record(record: dict[str, Any], *, raw_output: str) -> dict[str, Any]:
    if not raw_output:
        raise ValueError("raw_output is required for review record validation")
    if not isinstance(record, dict):
        raise ValueError("review record must be a JSON object")
    for field in REQUIRED_FIELDS:
        if field not in record:
            raise ValueError(f"{field} is required")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    for field in (
        "record_id",
        "card_id",
        "chapter_id",
        "style_contract_id",
        "style_contract_sha256",
        "source_output_id",
        "raw_output_sha256",
        "reviewer",
        "reviewed_at",
        "notes",
    ):
        if not isinstance(record[field], str):
            raise ValueError(f"{field} must be a string")
    if record["overall_acceptance"] not in ACCEPTANCE_STATUSES:
        raise ValueError("overall_acceptance must be one of: " + ", ".join(sorted(ACCEPTANCE_STATUSES)))
    if record["raw_output_sha256"] != text_sha256(raw_output):
        raise ValueError("raw_output_sha256 mismatch")
    defects = validate_style_defects(record["defects"])
    for index, defect in enumerate(defects):
        evidence = raw_output[defect["evidence_start"] : defect["evidence_end"]]
        if evidence != defect["evidence_text"]:
            raise ValueError(f"defects[{index}] evidence span does not match evidence_text")
    return record


def validate_review_records(
    records: list[dict[str, Any]],
    *,
    raw_outputs: dict[str, str],
) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        output_id = str(record.get("source_output_id", ""))
        raw_output = raw_outputs.get(output_id, "")
        try:
            validated.append(validate_review_record(record, raw_output=raw_output))
        except ValueError as exc:
            raise ValueError(f"review record {index}: {exc}") from exc
    return validated


def write_review_records(
    path: str | Path,
    records: list[dict[str, Any]],
    *,
    raw_outputs: dict[str, str],
) -> None:
    validated = validate_review_records(records, raw_outputs=raw_outputs)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in validated:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_review_records(
    path: str | Path,
    *,
    raw_outputs: dict[str, str],
) -> list[dict[str, Any]]:
    input_path = Path(path)
    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{input_path}:{line_number} is not valid JSON") from exc
            rows.append(row)
    return validate_review_records(rows, raw_outputs=raw_outputs)
```

- [ ] **Step 4: Run evidence tests**

Run:

```powershell
python -m pytest tests/test_review_evidence.py -q
```

Expected: pass.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add src/small_model_train/review/evidence.py tests/test_review_evidence.py
git commit -m "feat: validate stage5d review evidence"
```

---

## Task 4: Add Same-Plot Revision Records

**Files:**
- Create: `src/small_model_train/review/revision_records.py`
- Create: `tests/test_revision_records.py`

- [ ] **Step 1: Write failing revision tests**

Create `tests/test_revision_records.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest


MODEL_OUTPUT = "林默把合同推过去，对方沉默。"
REVISED_OUTPUT = "林默没有解释，只把合同推到桌面。岳家的人第一次停住。"


def _revision(**overrides) -> dict:
    from small_model_train.schemas.chapter_execution_card import text_sha256

    record = {
        "revision_id": "rev-c1-001",
        "schema_version": 1,
        "card_id": "card-c1-v1",
        "chapter_id": "c1",
        "style_contract_id": "contract-v1",
        "style_contract_sha256": "a" * 64,
        "prompt_sha256": "b" * 64,
        "raw_output_sha256": text_sha256(MODEL_OUTPUT),
        "model_output": MODEL_OUTPUT,
        "revised_output": REVISED_OUTPUT,
        "revision_status": "accepted_with_minor_edits",
        "revision_author": "author",
        "revised_at": "2026-06-27T01:00:00Z",
        "edit_summary": "把解释改成动作和反应。",
        "defect_record_ids": ["review-c1-001"],
        "acceptance_reason": "同剧情更像作者正文。",
    }
    record.update(overrides)
    return record


def test_validate_revision_record_accepts_same_plot_revision():
    from small_model_train.review.revision_records import (
        is_revision_accepted_for_rejection_sampling,
        validate_revision_record,
    )

    record = validate_revision_record(_revision())

    assert record["revision_id"] == "rev-c1-001"
    assert is_revision_accepted_for_rejection_sampling(record) is True


@pytest.mark.parametrize("status", ["rejected", "needs_rewrite"])
def test_rejected_revision_is_not_sft_candidate(status):
    from small_model_train.review.revision_records import (
        is_revision_accepted_for_rejection_sampling,
        validate_revision_record,
    )

    record = validate_revision_record(_revision(revision_status=status))

    assert is_revision_accepted_for_rejection_sampling(record) is False


def test_validate_revision_record_rejects_raw_hash_mismatch():
    from small_model_train.review.revision_records import validate_revision_record

    with pytest.raises(ValueError, match="raw_output_sha256 mismatch"):
        validate_revision_record(_revision(raw_output_sha256="c" * 64))


def test_validate_revision_record_rejects_empty_revised_output():
    from small_model_train.review.revision_records import validate_revision_record

    with pytest.raises(ValueError, match="revised_output"):
        validate_revision_record(_revision(revised_output=""))


def test_read_write_revision_records_round_trip(tmp_path: Path):
    from small_model_train.review.revision_records import read_revision_records, write_revision_records

    path = tmp_path / "revisions.jsonl"
    write_revision_records(path, [_revision()])

    assert read_revision_records(path)[0]["revision_id"] == "rev-c1-001"
```

- [ ] **Step 2: Run revision tests and verify failure**

Run:

```powershell
python -m pytest tests/test_revision_records.py -q
```

Expected: fail because `small_model_train.review.revision_records` does not exist.

- [ ] **Step 3: Implement revision records module**

Create `src/small_model_train/review/revision_records.py`:

```python
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from small_model_train.schemas.chapter_execution_card import text_sha256


SCHEMA_VERSION = 1
REVISION_STATUSES = {"accepted", "accepted_with_minor_edits", "rejected", "needs_rewrite"}
ACCEPTED_REVISION_STATUSES = {"accepted", "accepted_with_minor_edits"}
LOWER_HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_FIELDS = (
    "revision_id",
    "schema_version",
    "card_id",
    "chapter_id",
    "style_contract_id",
    "style_contract_sha256",
    "prompt_sha256",
    "raw_output_sha256",
    "model_output",
    "revised_output",
    "revision_status",
    "revision_author",
    "revised_at",
    "edit_summary",
    "defect_record_ids",
    "acceptance_reason",
)


def validate_revision_record(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("revision record must be a JSON object")
    for field in REQUIRED_FIELDS:
        if field not in record:
            raise ValueError(f"{field} is required")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    for field in (
        "revision_id",
        "card_id",
        "chapter_id",
        "style_contract_id",
        "model_output",
        "revised_output",
        "revision_author",
        "revised_at",
        "edit_summary",
        "acceptance_reason",
    ):
        if not isinstance(record[field], str) or not record[field].strip():
            raise ValueError(f"{field} must be a non-empty string")
    for field in ("style_contract_sha256", "prompt_sha256", "raw_output_sha256"):
        if not isinstance(record[field], str) or LOWER_HEX_SHA256_RE.fullmatch(record[field]) is None:
            raise ValueError(f"{field} must be a 64-character lowercase hex string")
    if record["revision_status"] not in REVISION_STATUSES:
        raise ValueError("revision_status must be one of: " + ", ".join(sorted(REVISION_STATUSES)))
    if record["raw_output_sha256"] != text_sha256(record["model_output"]):
        raise ValueError("raw_output_sha256 mismatch")
    defect_ids = record["defect_record_ids"]
    if not isinstance(defect_ids, list) or not all(isinstance(item, str) and item.strip() for item in defect_ids):
        raise ValueError("defect_record_ids must be a list of non-empty strings")
    return record


def is_revision_accepted_for_rejection_sampling(record: dict[str, Any]) -> bool:
    validated = validate_revision_record(record)
    return validated["revision_status"] in ACCEPTED_REVISION_STATUSES


def validate_revision_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        try:
            validated.append(validate_revision_record(record))
        except ValueError as exc:
            raise ValueError(f"revision record {index}: {exc}") from exc
    return validated


def write_revision_records(path: str | Path, records: list[dict[str, Any]]) -> None:
    validated = validate_revision_records(records)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in validated:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_revision_records(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{input_path}:{line_number} is not valid JSON") from exc
            rows.append(row)
    return validate_revision_records(rows)
```

- [ ] **Step 4: Run revision tests**

Run:

```powershell
python -m pytest tests/test_revision_records.py -q
```

Expected: pass.

- [ ] **Step 5: Commit Task 4**

Run:

```bash
git add src/small_model_train/review/revision_records.py tests/test_revision_records.py
git commit -m "feat: validate same plot revision records"
```

---

## Task 5: Build Rejection-Sampling SFT Rows And CLI

**Files:**
- Create: `src/small_model_train/review/rejection_sampling.py`
- Create: `scripts/build_rejection_sampling_sft.py`
- Create: `tests/test_rejection_sampling_sft.py`

- [ ] **Step 1: Write failing rejection-sampling tests**

Create `tests/test_rejection_sampling_sft.py`:

```python
from __future__ import annotations

import subprocess
import sys

import pytest

from small_model_train.io_utils import read_jsonl, write_jsonl


def _style_contract() -> dict:
    from small_model_train.style_contract import build_style_contract_asset

    return build_style_contract_asset(
        style_contract_id="contract-v1",
        approval_status="approved",
        source_corpus={
            "path": "chapters.jsonl",
            "sha256": "b" * 64,
            "quality_filter": "quality_tag=A",
            "row_count": 1,
            "selected_rows": 1,
            "split_summary": {"train": 1},
        },
        profile_metrics={
            "chapter_count": 1,
            "avg_dialogue_ratio": 0.1,
            "avg_paragraph_chars": 20,
            "ai_taste": {"phrase_hits": {}, "total_hits": 0, "hits_per_10k_chars": 0},
        },
    )


def _formal_card(text: str = "来源正文") -> dict:
    from small_model_train.schemas.chapter_execution_card import build_chapter_execution_card

    contract = _style_contract()
    return build_chapter_execution_card(
        card_id="card-c1-v1",
        chapter_id="c1",
        card_status="approved",
        style_contract_id=contract["style_contract_id"],
        style_contract_sha256=contract["contract_sha256"],
        source_chapter_text=text,
        target_platform="local",
        genre_tags=["都市"],
        hard_constraints={
            "must_include": ["合同证据"],
            "must_not_include": ["作者说明"],
            "continuity_facts": ["林默刚拿到关键证据。"],
            "forbidden_future_facts": [],
            "style_bans": [],
        },
        execution_plan={
            "chapter_goal": "林默用证据稳住局面。",
            "conflict_beat": "岳家当众施压。",
            "payoff_beat": "林默亮出备份。",
            "chapter_structure": [{"step": 1, "name": "压迫", "goal": "推进", "estimated_chars": "300"}],
            "character_states": [{"name": "林默", "state": "冷静", "speech_style": "短句"}],
            "ending_hook": "门外响起真正买家的声音。",
            "target_word_count": "2000-2500中文汉字",
        },
        creative_space={
            "optional_sensory_details": [],
            "optional_dialogue_moves": [],
            "optional_micro_conflicts": [],
            "allowed_scene_expansion": [],
        },
        provenance={
            "source_card_id": "draft-c1",
            "compiler_version": "test",
            "created_at": "2026-06-27T00:00:00Z",
            "reviewer": "qa",
            "reviewed_at": "2026-06-27T00:00:00Z",
            "review_notes": "",
            "group_id": "group-c1",
            "split": "train",
        },
    )


def _revision(status: str = "accepted_with_minor_edits") -> dict:
    from small_model_train.schemas.chapter_execution_card import text_sha256

    model_output = "林默把合同推过去，对方沉默。"
    return {
        "revision_id": "rev-c1-001",
        "schema_version": 1,
        "card_id": "card-c1-v1",
        "chapter_id": "c1",
        "style_contract_id": "contract-v1",
        "style_contract_sha256": _style_contract()["contract_sha256"],
        "prompt_sha256": "b" * 64,
        "raw_output_sha256": text_sha256(model_output),
        "model_output": model_output,
        "revised_output": "林默没有解释，只把合同推到桌面。",
        "revision_status": status,
        "revision_author": "author",
        "revised_at": "2026-06-27T01:00:00Z",
        "edit_summary": "改成动作。",
        "defect_record_ids": ["review-c1-001"],
        "acceptance_reason": "可用。",
    }


def test_build_rejection_sampling_sft_rows_uses_formal_prompt_and_revised_output():
    from small_model_train.review.rejection_sampling import build_rejection_sampling_sft_rows

    rows = build_rejection_sampling_sft_rows([_revision()], [_formal_card()], _style_contract())

    assert rows[0]["output"] == "林默没有解释，只把合同推到桌面。"
    assert "【本章目标】" in rows[0]["input"]
    assert rows[0]["revision_id"] == "rev-c1-001"


def test_build_rejection_sampling_sft_rows_rejects_unaccepted_revision():
    from small_model_train.review.rejection_sampling import build_rejection_sampling_sft_rows

    with pytest.raises(ValueError, match="revision_status must be accepted"):
        build_rejection_sampling_sft_rows([_revision("rejected")], [_formal_card()], _style_contract())


def test_build_rejection_sampling_sft_cli_writes_jsonl(tmp_path):
    from small_model_train.style_contract import write_style_contract_asset

    revisions_path = tmp_path / "revisions.jsonl"
    cards_path = tmp_path / "cards.jsonl"
    contract_path = tmp_path / "style_contract.json"
    output_path = tmp_path / "rs_sft.jsonl"
    write_jsonl(revisions_path, [_revision()])
    write_jsonl(cards_path, [_formal_card()])
    write_style_contract_asset(contract_path, _style_contract())

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_rejection_sampling_sft.py",
            "--revisions",
            str(revisions_path),
            "--cards",
            str(cards_path),
            "--style-contract-json",
            str(contract_path),
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert read_jsonl(output_path)[0]["revision_id"] == "rev-c1-001"
```

- [ ] **Step 2: Run rejection-sampling tests and verify failure**

Run:

```powershell
python -m pytest tests/test_rejection_sampling_sft.py -q
```

Expected: fail because `review.rejection_sampling` and the CLI do not exist.

- [ ] **Step 3: Implement rejection-sampling builder**

Create `src/small_model_train/review/rejection_sampling.py`:

```python
from __future__ import annotations

from typing import Any

from small_model_train.cards.card_renderer import render_chapter_execution_input
from small_model_train.schemas.chapter_execution_card import validate_chapter_execution_card
from small_model_train.sft_builder import INSTRUCTION
from small_model_train.style_contract import validate_style_contract_asset
from small_model_train.review.revision_records import (
    is_revision_accepted_for_rejection_sampling,
    validate_revision_record,
)


def build_rejection_sampling_sft_rows(
    revisions: list[dict[str, Any]],
    cards: list[dict[str, Any]],
    style_contract: dict[str, Any],
) -> list[dict[str, str]]:
    contract = validate_style_contract_asset(style_contract)
    card_by_id = {card["card_id"]: validate_chapter_execution_card(card) for card in cards}
    rows: list[dict[str, str]] = []
    for revision in revisions:
        validated_revision = validate_revision_record(revision)
        if not is_revision_accepted_for_rejection_sampling(validated_revision):
            raise ValueError(
                "revision_status must be accepted or accepted_with_minor_edits: "
                f"{validated_revision['revision_id']}"
            )
        card = card_by_id.get(validated_revision["card_id"])
        if card is None:
            raise ValueError(f"formal card not found for revision: {validated_revision['card_id']}")
        _require_revision_matches_card_and_contract(validated_revision, card, contract)
        rows.append(
            {
                "instruction": INSTRUCTION,
                "input": render_chapter_execution_input(card, contract),
                "output": validated_revision["revised_output"],
                "revision_id": validated_revision["revision_id"],
                "card_id": validated_revision["card_id"],
                "chapter_id": validated_revision["chapter_id"],
                "style_contract_sha256": validated_revision["style_contract_sha256"],
                "raw_output_sha256": validated_revision["raw_output_sha256"],
            }
        )
    return rows


def build_same_plot_preference_rows(revisions: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for revision in revisions:
        validated = validate_revision_record(revision)
        if not is_revision_accepted_for_rejection_sampling(validated):
            continue
        rows.append(
            {
                "id": validated["revision_id"],
                "prompt_sha256": validated["prompt_sha256"],
                "card_id": validated["card_id"],
                "chapter_id": validated["chapter_id"],
                "style_contract_sha256": validated["style_contract_sha256"],
                "chosen": validated["revised_output"],
                "rejected": validated["model_output"],
                "reject_type": ",".join(validated["defect_record_ids"]),
                "source": "stage5d_same_plot_revision",
            }
        )
    return rows


def _require_revision_matches_card_and_contract(
    revision: dict[str, Any],
    card: dict[str, Any],
    style_contract: dict[str, Any],
) -> None:
    if revision["card_id"] != card["card_id"]:
        raise ValueError("revision card_id mismatch")
    if revision["chapter_id"] != card["chapter_id"]:
        raise ValueError("revision chapter_id mismatch")
    if revision["style_contract_id"] != style_contract["style_contract_id"]:
        raise ValueError("revision style_contract_id mismatch")
    if revision["style_contract_sha256"] != style_contract["contract_sha256"]:
        raise ValueError("revision style_contract_sha256 mismatch")
    if card["style_contract_sha256"] != style_contract["contract_sha256"]:
        raise ValueError("card style_contract_sha256 mismatch")
```

- [ ] **Step 4: Implement rejection-sampling CLI**

Create `scripts/build_rejection_sampling_sft.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.io_utils import read_jsonl, write_jsonl
from small_model_train.review.rejection_sampling import build_rejection_sampling_sft_rows
from small_model_train.style_contract import read_style_contract_asset


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Stage 5D rejection-sampling SFT candidates.")
    parser.add_argument("--revisions", required=True)
    parser.add_argument("--cards", required=True)
    parser.add_argument("--style-contract-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        revisions = read_jsonl(args.revisions)
        cards = read_jsonl(args.cards)
        style_contract = read_style_contract_asset(args.style_contract_json)
        rows = build_rejection_sampling_sft_rows(revisions, cards, style_contract)
        write_jsonl(args.output, rows)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"wrote {len(rows)} rejection-sampling SFT rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run rejection-sampling tests**

Run:

```powershell
python -m pytest tests/test_rejection_sampling_sft.py -q
```

Expected: pass.

- [ ] **Step 6: Commit Task 5**

Run:

```bash
git add src/small_model_train/review/rejection_sampling.py scripts/build_rejection_sampling_sft.py tests/test_rejection_sampling_sft.py
git commit -m "feat: build rejection sampling sft candidates"
```

---

## Task 6: Add Same-Plot Preference Candidate CLI

**Files:**
- Modify: `src/small_model_train/preference_builder.py`
- Modify: `tests/test_preference_builder.py`
- Create: `scripts/build_same_plot_preference_dataset.py`

- [ ] **Step 1: Add failing preference candidate tests**

Append to `tests/test_preference_builder.py`:

```python

def test_build_same_plot_preference_candidates_from_accepted_revisions():
    from small_model_train.preference_builder import build_same_plot_preference_candidates
    from small_model_train.schemas.chapter_execution_card import text_sha256

    model_output = "林默把合同推过去，对方沉默。"
    revisions = [
        {
            "revision_id": "rev-c1-001",
            "schema_version": 1,
            "card_id": "card-c1-v1",
            "chapter_id": "c1",
            "style_contract_id": "contract-v1",
            "style_contract_sha256": "a" * 64,
            "prompt_sha256": "b" * 64,
            "raw_output_sha256": text_sha256(model_output),
            "model_output": model_output,
            "revised_output": "林默没有解释，只把合同推到桌面。",
            "revision_status": "accepted",
            "revision_author": "author",
            "revised_at": "2026-06-27T01:00:00Z",
            "edit_summary": "改成动作。",
            "defect_record_ids": ["review-c1-001"],
            "acceptance_reason": "可用。",
        }
    ]

    rows = build_same_plot_preference_candidates(revisions)

    assert rows == [
        {
            "id": "rev-c1-001",
            "prompt_sha256": "b" * 64,
            "card_id": "card-c1-v1",
            "chapter_id": "c1",
            "style_contract_sha256": "a" * 64,
            "chosen": "林默没有解释，只把合同推到桌面。",
            "rejected": model_output,
            "reject_type": "review-c1-001",
            "source": "stage5d_same_plot_revision",
        }
    ]


def test_build_same_plot_preference_dataset_cli_writes_jsonl(tmp_path):
    from small_model_train.schemas.chapter_execution_card import text_sha256

    revisions_path = tmp_path / "revisions.jsonl"
    output_path = tmp_path / "pref.jsonl"
    model_output = "林默把合同推过去，对方沉默。"
    write_jsonl(
        revisions_path,
        [
            {
                "revision_id": "rev-c1-001",
                "schema_version": 1,
                "card_id": "card-c1-v1",
                "chapter_id": "c1",
                "style_contract_id": "contract-v1",
                "style_contract_sha256": "a" * 64,
                "prompt_sha256": "b" * 64,
                "raw_output_sha256": text_sha256(model_output),
                "model_output": model_output,
                "revised_output": "林默没有解释，只把合同推到桌面。",
                "revision_status": "accepted",
                "revision_author": "author",
                "revised_at": "2026-06-27T01:00:00Z",
                "edit_summary": "改成动作。",
                "defect_record_ids": ["review-c1-001"],
                "acceptance_reason": "可用。",
            }
        ],
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/build_same_plot_preference_dataset.py",
            "--revisions",
            str(revisions_path),
            "--output",
            str(output_path),
        ],
        check=True,
    )

    assert read_jsonl(output_path)[0]["source"] == "stage5d_same_plot_revision"
```

- [ ] **Step 2: Run preference tests and verify failure**

Run:

```powershell
python -m pytest tests/test_preference_builder.py::test_build_same_plot_preference_candidates_from_accepted_revisions tests/test_preference_builder.py::test_build_same_plot_preference_dataset_cli_writes_jsonl -q
```

Expected: fail because `build_same_plot_preference_candidates` and the CLI do not exist.

- [ ] **Step 3: Export Stage 5D preference builder**

Append to `src/small_model_train/preference_builder.py`:

```python

def build_same_plot_preference_candidates(revisions: list[dict]) -> list[dict]:
    from small_model_train.review.rejection_sampling import build_same_plot_preference_rows

    return build_same_plot_preference_rows(revisions)
```

- [ ] **Step 4: Implement same-plot preference CLI**

Create `scripts/build_same_plot_preference_dataset.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.io_utils import read_jsonl, write_jsonl
from small_model_train.preference_builder import build_same_plot_preference_candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Stage 5D same-plot preference candidates.")
    parser.add_argument("--revisions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        revisions = read_jsonl(args.revisions)
        rows = build_same_plot_preference_candidates(revisions)
        write_jsonl(args.output, rows)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"wrote {len(rows)} same-plot preference rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run preference tests**

Run:

```powershell
python -m pytest tests/test_preference_builder.py -q
```

Expected: pass.

- [ ] **Step 6: Commit Task 6**

Run:

```bash
git add src/small_model_train/preference_builder.py scripts/build_same_plot_preference_dataset.py tests/test_preference_builder.py
git commit -m "feat: build same plot preference candidates"
```

---

## Task 7: Add Stage 5D Review Report

**Files:**
- Create: `src/small_model_train/review/stage5d_report.py`
- Create: `scripts/build_stage5d_review_report.py`
- Create: `tests/test_stage5d_report.py`

- [ ] **Step 1: Write failing report tests**

Create `tests/test_stage5d_report.py`:

```python
from __future__ import annotations

import json
import subprocess
import sys

from small_model_train.io_utils import read_jsonl, write_jsonl


def test_build_stage5d_summary_counts_defects_and_candidates():
    from small_model_train.review.stage5d_report import build_stage5d_summary

    summary = build_stage5d_summary(
        review_records=[
            {
                "defects": [
                    {"label": "generic_phrase", "severity": "major"},
                    {"label": "hook_blur", "severity": "blocker"},
                ],
                "overall_acceptance": "needs_rewrite",
            }
        ],
        revision_records=[
            {"revision_status": "accepted", "model_output": "坏正文", "revised_output": "好正文"}
        ],
        rejection_sampling_rows=[{"revision_id": "rev-1"}],
        preference_rows=[{"id": "rev-1"}],
    )

    assert summary["reviewed_outputs"] == 1
    assert summary["defects"]["by_label"]["generic_phrase"] == 1
    assert summary["accepted_revisions"] == 1
    assert summary["rejection_sampling_sft_rows"] == 1
    assert summary["preference_candidate_rows"] == 1


def test_render_stage5d_report_contains_boundaries():
    from small_model_train.review.stage5d_report import build_stage5d_summary, render_stage5d_report

    summary = build_stage5d_summary([], [], [], [])
    report = render_stage5d_report(summary)

    assert "Stage 5D Review Report" in report
    assert "不代表已经运行 DPO/SimPO" in report


def test_build_stage5d_review_report_cli_writes_outputs(tmp_path):
    review_path = tmp_path / "review.jsonl"
    revisions_path = tmp_path / "revisions.jsonl"
    rs_path = tmp_path / "rs.jsonl"
    pref_path = tmp_path / "pref.jsonl"
    json_path = tmp_path / "summary.json"
    report_path = tmp_path / "report.md"
    write_jsonl(review_path, [{"defects": [{"label": "generic_phrase", "severity": "major"}], "overall_acceptance": "needs_rewrite"}])
    write_jsonl(revisions_path, [{"revision_status": "accepted", "model_output": "坏正文", "revised_output": "好正文"}])
    write_jsonl(rs_path, [{"revision_id": "rev-1"}])
    write_jsonl(pref_path, [{"id": "rev-1"}])

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_stage5d_review_report.py",
            "--review-records",
            str(review_path),
            "--revisions",
            str(revisions_path),
            "--rejection-sampling-rows",
            str(rs_path),
            "--preference-rows",
            str(pref_path),
            "--summary-output",
            str(json_path),
            "--report-output",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(json_path.read_text(encoding="utf-8"))["reviewed_outputs"] == 1
    assert "Stage 5D Review Report" in report_path.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run report tests and verify failure**

Run:

```powershell
python -m pytest tests/test_stage5d_report.py -q
```

Expected: fail because report module and CLI do not exist.

- [ ] **Step 3: Implement report module**

Create `src/small_model_train/review/stage5d_report.py`:

```python
from __future__ import annotations

from typing import Any

from small_model_train.review.style_defects import summarize_style_defects
from small_model_train.text_utils import count_chinese_chars


def build_stage5d_summary(
    review_records: list[dict[str, Any]],
    revision_records: list[dict[str, Any]],
    rejection_sampling_rows: list[dict[str, Any]],
    preference_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    defects = [defect for record in review_records for defect in record.get("defects", [])]
    accepted_revisions = [
        record
        for record in revision_records
        if record.get("revision_status") in {"accepted", "accepted_with_minor_edits"}
    ]
    total_model_chars = sum(count_chinese_chars(str(row.get("model_output") or "")) for row in revision_records)
    total_revised_chars = sum(count_chinese_chars(str(row.get("revised_output") or "")) for row in revision_records)
    changed_char_delta = abs(total_revised_chars - total_model_chars)
    return {
        "reviewed_outputs": len(review_records),
        "defects": summarize_style_defects(defects),
        "revision_records": len(revision_records),
        "accepted_revisions": len(accepted_revisions),
        "author_acceptance_rate": round(len(accepted_revisions) / len(revision_records), 4)
        if revision_records
        else 0.0,
        "changed_char_delta": changed_char_delta,
        "rejection_sampling_sft_rows": len(rejection_sampling_rows),
        "preference_candidate_rows": len(preference_rows),
        "plan_execution_regressions": sum(1 for defect in defects if defect.get("label") == "plan_execution_regression"),
        "boundary": "candidate_data_only_no_preference_training",
    }


def render_stage5d_report(summary: dict[str, Any]) -> str:
    defects = summary["defects"]
    lines = [
        "# Stage 5D Review Report",
        "",
        f"- reviewed_outputs: {summary['reviewed_outputs']}",
        f"- total_defects: {defects['total_defects']}",
        f"- accepted_revisions: {summary['accepted_revisions']}",
        f"- author_acceptance_rate: {summary['author_acceptance_rate']}",
        f"- changed_char_delta: {summary['changed_char_delta']}",
        f"- rejection_sampling_sft_rows: {summary['rejection_sampling_sft_rows']}",
        f"- preference_candidate_rows: {summary['preference_candidate_rows']}",
        f"- plan_execution_regressions: {summary['plan_execution_regressions']}",
        "",
        "## Boundary",
        "",
        "这些 preference rows 只是候选数据，不代表已经运行 DPO/SimPO/ORPO/KTO。",
    ]
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Implement report CLI**

Create `scripts/build_stage5d_review_report.py`:

```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.io_utils import read_jsonl
from small_model_train.review.stage5d_report import build_stage5d_summary, render_stage5d_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Stage 5D review summary and Markdown report.")
    parser.add_argument("--review-records", required=True)
    parser.add_argument("--revisions", required=True)
    parser.add_argument("--rejection-sampling-rows", required=True)
    parser.add_argument("--preference-rows", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--report-output", required=True)
    args = parser.parse_args()

    try:
        summary = build_stage5d_summary(
            read_jsonl(args.review_records),
            read_jsonl(args.revisions),
            read_jsonl(args.rejection_sampling_rows),
            read_jsonl(args.preference_rows),
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_stage5d_report(summary), encoding="utf-8")
    print(f"wrote Stage 5D summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run report tests**

Run:

```powershell
python -m pytest tests/test_stage5d_report.py -q
```

Expected: pass.

- [ ] **Step 6: Commit Task 7**

Run:

```bash
git add src/small_model_train/review/stage5d_report.py scripts/build_stage5d_review_report.py tests/test_stage5d_report.py
git commit -m "feat: report stage5d review metrics"
```

---

## Task 8: Add Stage 5D Docs And Roadmap Updates

**Files:**
- Create: `docs/stage5d-author-feedback-ai-taste-reduction.zh.md`
- Modify: `README.md`
- Modify: `docs/index.zh.md`
- Modify: `docs/project-map.zh.md`
- Modify: `docs/pipeline-flow.zh.md`
- Modify: `docs/superpowers/plans/2026-06-23-small-model-train-full-roadmap.md`

- [ ] **Step 1: Create the Stage 5D runbook**

Create `docs/stage5d-author-feedback-ai-taste-reduction.zh.md`:

````markdown
# Stage 5D 作者反馈与 AI 味收敛指南

## 目标

Stage 5D 在 Stage 5C formal cards 的基础上整理作者反馈、AI 味缺陷和 same-plot 修订数据。它不扩样，不运行 DPO/SimPO/ORPO/KTO，也不进入 Stage 5E 实验矩阵。

## 先修 formal admission

Stage 5D 的第一步是修复 formal SFT admission 的重复 id 风险：

- `train/A` 章节 id 不能重复。
- 一张 formal card 不能为多个重复章节行生成多条 SFT row。
- dataset manifest 不能静默覆盖重复 chapter/card hash key。

这一步通过后，作者反馈记录才有可信的 card/chapter/source hash 归属。

## AI 味缺陷标签

Stage 5D 使用固定标签记录问题，例如：

- `generic_phrase`
- `explanation_voice`
- `summary_narration`
- `empty_intensity`
- `repeated_psychology`
- `dialogue_flatness`
- `payoff_blur`
- `hook_blur`
- `style_contract_drift`
- `plan_execution_regression`

缺陷必须带原始输出片段和位置。只看 sanitized text 的记录不能作为 Stage 5D 证据。

## Same-Plot 作者修订

same-plot revision 记录同一张正式卡、同一个 StyleContract、同一个 prompt 下的小模型输出和作者修订。只有 `accepted` 和 `accepted_with_minor_edits` 可以进入 rejection-sampling SFT 候选。

## 构建候选数据

rejection-sampling SFT:

```powershell
python scripts/build_rejection_sampling_sft.py --revisions data_review/stage5d_revisions.jsonl --cards data_cards/chapter_execution_cards_approved.jsonl --style-contract-json data_style/style_contract_author_main_v1.json --output data_sft/stage5d_rejection_sampling_sft.jsonl
```

same-plot preference candidates:

```powershell
python scripts/build_same_plot_preference_dataset.py --revisions data_review/stage5d_revisions.jsonl --output data_pref/stage5d_same_plot_preference.jsonl
```

这些 preference rows 只是候选数据，不代表已经运行 DPO/SimPO。

## 报告

```powershell
python scripts/build_stage5d_review_report.py --review-records data_review/stage5d_review_records.jsonl --revisions data_review/stage5d_revisions.jsonl --rejection-sampling-rows data_sft/stage5d_rejection_sampling_sft.jsonl --preference-rows data_pref/stage5d_same_plot_preference.jsonl --summary-output reports/stage5d_review_summary.json --report-output reports/stage5d_review_report.md
```

报告关注缺陷密度、作者接受率、修订负担、候选 row 数和 plan execution regression。

## Stage 5D 不证明什么

- 不证明模型质量已经提升。
- 不证明可以扩到 100/500。
- 不证明偏好训练已经完成。
- 不替代作者或人工审阅。
- 不允许 sealed 数据进入训练候选。
````

- [ ] **Step 2: Update public indexes**

In `README.md`, add the Stage 5D guide next to the other stage docs:

```markdown
- [Stage 5D 作者反馈与 AI 味收敛指南](docs/stage5d-author-feedback-ai-taste-reduction.zh.md)
```

In `docs/index.zh.md`, add:

```markdown
- [Stage 5D 作者反馈与 AI 味收敛指南](stage5d-author-feedback-ai-taste-reduction.zh.md)：解释 AI 味缺陷标签、same-plot 作者修订、rejection-sampling SFT 候选和偏好候选边界。
```

- [ ] **Step 3: Update project map and pipeline flow**

In `docs/project-map.zh.md`, add entries for:

```markdown
- `data_review/stage5d_review_records.jsonl`：Stage 5D AI 味缺陷和 evidence spans。
- `data_review/stage5d_revisions.jsonl`：same-plot 作者修订记录。
- `data_sft/stage5d_rejection_sampling_sft.jsonl`：accepted revisions 生成的 SFT 候选。
- `data_pref/stage5d_same_plot_preference.jsonl`：same-plot chosen/rejected 偏好候选。
```

In `docs/pipeline-flow.zh.md`, add a Stage 5D subsection after Stage 5C:

```markdown
Stage 5D 先修 formal admission 的重复 id 门禁，然后记录 AI 味缺陷、same-plot 作者修订，并生成 rejection-sampling SFT 候选和 same-plot preference 候选。Stage 5D 不运行 DPO/SimPO，也不扩样。
```

- [ ] **Step 4: Update the full roadmap Stage 5D section**

In `docs/superpowers/plans/2026-06-23-small-model-train-full-roadmap.md`, update Stage 5D planned scope to include:

```markdown
- First repair the Stage 5C formal admission duplicate chapter id and manifest overwrite gap.
```

Keep Stage 5E blocked until Stage 5D produces stable same-card, same-style, same-prompt/raw-output evidence with generation seed provenance.

- [ ] **Step 5: Run docs scan**

Run:

```powershell
rg -n "Stage 5D|AI 味|same-plot|rejection-sampling|DPO|SimPO|duplicate" README.md docs --glob "!docs/superpowers/specs/**"
```

Expected: public docs mention the Stage 5D flow and clearly say preference rows are candidates only.

- [ ] **Step 6: Commit Task 8**

Run:

```bash
git add docs/stage5d-author-feedback-ai-taste-reduction.zh.md README.md docs/index.zh.md docs/project-map.zh.md docs/pipeline-flow.zh.md docs/superpowers/plans/2026-06-23-small-model-train-full-roadmap.md
git commit -m "docs: add stage5d author feedback runbook"
```

---

## Task 9: Final Verification

**Files:**
- No new files.

- [ ] **Step 1: Run focused Stage 5D tests**

Run:

```powershell
python -m pytest tests/test_card_validator.py tests/test_sft_builder.py tests/test_dataset_manifest.py tests/test_style_defects.py tests/test_review_evidence.py tests/test_revision_records.py tests/test_rejection_sampling_sft.py tests/test_preference_builder.py tests/test_stage5d_report.py -q
```

Expected: pass.

- [ ] **Step 2: Run full suite**

Run:

```powershell
python -m pytest -q
```

Expected: pass.

- [ ] **Step 3: Verify stale public draft commands are still clean**

Run:

```powershell
rg --pcre2 -n "build_sft_dataset.py(?!.*(--allow-draft-cards|--style-contract-json))" README.md docs --glob "!docs/superpowers/**"
```

Expected: no runnable stale draft commands. File-name references or historical audit table rows are acceptable only if they are not runnable commands.

- [ ] **Step 4: Verify red-flag scan**

Run:

```powershell
$patterns = @("TB" + "D", "TO" + "DO", "not implemented", "pass$"); rg -n -i ($patterns -join "|") src scripts tests docs/stage5d-author-feedback-ai-taste-reduction.zh.md
```

Expected: no production red flags. Test doubles with names such as `fake_*` are acceptable if they are in tests and clearly mocks.

- [ ] **Step 5: Verify diff and status**

Run:

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors and no uncommitted tracked changes after task commits.

---

## Stage 5D Exit Criteria

Stage 5D is complete only when all of these are true:

- Full pytest suite passes.
- Formal SFT rejects duplicate trainable chapter ids before writing rows.
- Formal dataset manifests reject duplicate chapter or card hash keys.
- AI-taste defect records validate labels, severity, evidence spans, and raw output provenance.
- Same-plot revision records validate card, StyleContract, prompt, and raw output provenance.
- Rejection-sampling SFT candidate rows can be built from accepted revisions.
- Preference candidate rows can be built only from valid same-plot chosen/rejected pairs.
- Stage 5D reports summarize defect density, acceptance rate, edit burden, candidate row counts, and plan-execution regressions.
- Docs clearly state that Stage 5D does not run DPO/SimPO and does not prove larger-scale model quality improvement.

---

## Self-Review

- Spec coverage: Task 1 covers the merged formal admission repair. Task 2 covers taxonomy. Task 3 covers evidence-spanned review records. Task 4 covers same-plot revision records. Task 5 covers rejection-sampling SFT rows. Task 6 covers same-plot preference candidates without preference training. Task 7 covers metrics and reports. Task 8 covers docs and roadmap updates. Task 9 covers verification.
- Red-flag scan: no task uses unresolved markers or future-work labels as a substitute for implementation detail.
- Type consistency: `card_id`, `chapter_id`, `style_contract_id`, `style_contract_sha256`, `prompt_sha256`, `raw_output_sha256`, `revision_status`, and `defect_record_ids` are used consistently across modules, CLIs, and tests.
