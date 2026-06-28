# Stage 5D Closure And Stage 5E Entry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair Stage 5D so it satisfies the approved Stage 5D exit criteria, produce a verifiable Stage 5E entry gate, and only then begin Stage 5E controlled experimentation.

**Architecture:** Keep Stage 5D as data-candidate tooling, not training or experiment execution. Add missing report metrics, make preference candidates self-describing, preserve train/eval split provenance on candidate rows, and add a dedicated Stage 5E entry checker that consumes Stage 5D artifacts. Stage 5E starts only after the checker passes against real local artifacts.

**Tech Stack:** Python 3.10+, pytest, JSONL files, existing `small_model_train` modules and `scripts/` CLI pattern.

---

## Scope Check

This plan closes Stage 5D and prepares Stage 5E entry. It does not implement Stage 5E experiments, DPO, SimPO, ORPO, KTO, reward-model training, or preference optimization.

The authoritative requirements are:

- `docs/superpowers/specs/2026-06-27-stage5d-author-feedback-ai-taste-reduction-design.md`
- `docs/superpowers/plans/2026-06-27-stage5d-author-feedback-ai-taste-reduction.md`
- Stage 5E entry criteria in `docs/superpowers/plans/2026-06-23-small-model-train-full-roadmap.md`

Stage 5E is blocked until:

- Stage 5D report metrics include defect density, author acceptance, edit burden, candidate row counts, plan-execution regressions, and non-train candidate detection.
- Same-plot preference rows include defect labels, not only defect record ids.
- Rejection-sampling SFT rows preserve source split provenance.
- Stage 5D artifacts exist locally and pass a Stage 5E entry checker.
- Full pytest passes.

---

## File Map

- Modify: `src/small_model_train/review/stage5d_report.py`
  - Add defect density, edit burden mean/median, candidate split checks, and review decision source counts.
- Modify: `scripts/build_stage5d_review_report.py`
  - Add required `--raw-outputs` input so density has a real denominator.
- Modify: `tests/test_stage5d_report.py`
  - Add failing tests for the missing report metrics and CLI raw-output requirement.
- Modify: `src/small_model_train/preference_builder.py`
  - Build same-plot preference rows with `defect_labels` and `defect_record_ids`.
- Modify: `scripts/build_same_plot_preference_dataset.py`
  - Require `--review-records` so defect labels can be resolved.
- Modify: `tests/test_preference_builder.py`
  - Add tests for defect label resolution and missing review evidence failure.
- Modify: `src/small_model_train/review/rejection_sampling.py`
  - Preserve `source_split` from formal card provenance on rejection-sampling SFT rows.
- Modify: `tests/test_rejection_sampling_sft.py`
  - Assert candidate rows include `source_split: train`.
- Create: `src/small_model_train/review/stage5e_entry.py`
  - Validate Stage 5D artifacts against Stage 5E entry criteria.
- Create: `scripts/check_stage5e_entry.py`
  - CLI wrapper for the entry checker.
- Create: `tests/test_stage5e_entry.py`
  - Unit and CLI tests for pass/fail gate behavior.
- Modify: `docs/stage5d-author-feedback-ai-taste-reduction.zh.md`
  - Update commands and explain Stage 5E gate.
- Modify: `docs/superpowers/plans/2026-06-23-small-model-train-full-roadmap.md`
  - Fix stale Stage 5D/Stage 5E status text after the gate passes.

---

## Task 1: Complete Stage 5D Report Metrics

**Files:**
- Modify: `tests/test_stage5d_report.py`
- Modify: `src/small_model_train/review/stage5d_report.py`
- Modify: `scripts/build_stage5d_review_report.py`

- [ ] **Step 1: Add failing summary metric tests**

Append this test to `tests/test_stage5d_report.py`:

```python
def test_build_stage5d_summary_tracks_density_edit_burden_and_split_risk():
    from small_model_train.review.stage5d_report import build_stage5d_summary

    review_records = [
        {
            "record_id": "review-1",
            "source_output_id": "gen-1",
            "review_source": "author",
            "defects": [
                {"label": "generic_phrase", "severity": "major"},
                {"label": "plan_execution_regression", "severity": "blocker"},
            ],
        },
        {
            "record_id": "review-2",
            "source_output_id": "gen-2",
            "review_source": "deterministic",
            "defects": [],
        },
    ]
    raw_outputs = {
        "gen-1": "林默没有解释，只把合同推到桌面。岳家的人第一次停住。",
        "gen-2": "门外响起脚步声。",
    }
    revision_records = [
        {
            "revision_status": "accepted",
            "model_output": "林默把合同推过去，对方沉默。",
            "revised_output": "林默没有解释，只把合同推到桌面。",
        },
        {
            "revision_status": "accepted_with_minor_edits",
            "model_output": "门外有人来了。",
            "revised_output": "门外响起第二个人的脚步声。",
        },
    ]
    rejection_sampling_rows = [
        {"revision_id": "rev-1", "source_split": "train"},
        {"revision_id": "rev-2", "source_split": "sealed"},
    ]

    summary = build_stage5d_summary(
        review_records,
        revision_records,
        rejection_sampling_rows=rejection_sampling_rows,
        preference_rows=[{"id": "pref-1"}],
        raw_outputs=raw_outputs,
    )

    assert summary["reviewed_output_chars"] == 33
    assert summary["defect_density_per_10k_chars"] == round(2 / 33 * 10000, 4)
    assert summary["edit_burden"]["mean_changed_chars"] == 4.5
    assert summary["edit_burden"]["median_changed_chars"] == 4.5
    assert summary["candidate_split_counts"] == {"sealed": 1, "train": 1}
    assert summary["non_train_rejection_sampling_rows"] == ["rev-2"]
    assert summary["review_source_counts"] == {"author": 1, "deterministic": 1}
```

- [ ] **Step 2: Add failing CLI raw-output test**

Append this test to `tests/test_stage5d_report.py`:

```python
def test_cli_requires_raw_outputs_for_density_metrics(tmp_path: Path):
    review_records = tmp_path / "review.jsonl"
    revisions = tmp_path / "revisions.jsonl"
    rejection_sampling_rows = tmp_path / "rs.jsonl"
    preference_rows = tmp_path / "pref.jsonl"
    summary_output = tmp_path / "summary.json"
    report_output = tmp_path / "report.md"
    _write_jsonl(review_records, [{"source_output_id": "gen-1", "defects": []}])
    _write_jsonl(revisions, [])
    _write_jsonl(rejection_sampling_rows, [])
    _write_jsonl(preference_rows, [])

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--review-records",
            str(review_records),
            "--revisions",
            str(revisions),
            "--rejection-sampling-rows",
            str(rejection_sampling_rows),
            "--preference-rows",
            str(preference_rows),
            "--summary-output",
            str(summary_output),
            "--report-output",
            str(report_output),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--raw-outputs is required" in result.stderr
    assert not summary_output.exists()
    assert not report_output.exists()
```

- [ ] **Step 3: Run tests and verify failure**

Run:

```powershell
python -m pytest tests/test_stage5d_report.py::test_build_stage5d_summary_tracks_density_edit_burden_and_split_risk tests/test_stage5d_report.py::test_cli_requires_raw_outputs_for_density_metrics -q
```

Expected: both tests fail because `raw_outputs`, density metrics, edit-burden summary, split checks, and CLI raw-output input are absent.

- [ ] **Step 4: Replace Stage 5D report implementation**

Replace `src/small_model_train/review/stage5d_report.py` with:

```python
from __future__ import annotations

from statistics import mean, median
from typing import Any

from small_model_train.review.style_defects import summarize_style_defects
from small_model_train.text_utils import count_chinese_chars


BOUNDARY = "candidate_data_only_no_preference_training"
BOUNDARY_TEXT = "这些 preference rows 只是候选数据，不代表已经运行 DPO/SimPO/ORPO/KTO。"
ACCEPTED_STATUSES = {"accepted", "accepted_with_minor_edits"}
TRAIN_SPLIT = "train"


def build_stage5d_summary(
    review_records: list[dict[str, Any]],
    revision_records: list[dict[str, Any]],
    rejection_sampling_rows: list[dict[str, Any]],
    preference_rows: list[dict[str, Any]],
    *,
    raw_outputs: dict[str, str],
) -> dict[str, Any]:
    defects = _collect_defects(review_records)
    accepted_revisions = sum(
        1 for row in revision_records if row.get("revision_status") in ACCEPTED_STATUSES
    )
    reviewed_output_chars = _reviewed_output_chars(review_records, raw_outputs)
    changed_chars = _changed_chars_by_revision(revision_records)
    split_counts = _candidate_split_counts(rejection_sampling_rows)
    non_train_rows = _non_train_rejection_sampling_rows(rejection_sampling_rows)
    revision_count = len(revision_records)
    return {
        "reviewed_outputs": len(review_records),
        "reviewed_output_chars": reviewed_output_chars,
        "defects": summarize_style_defects(defects),
        "defect_density_per_10k_chars": round(len(defects) / reviewed_output_chars * 10000, 4)
        if reviewed_output_chars
        else 0.0,
        "revision_records": revision_count,
        "accepted_revisions": accepted_revisions,
        "author_acceptance_rate": round(accepted_revisions / revision_count, 4)
        if revision_count
        else 0.0,
        "changed_char_delta": sum(changed_chars),
        "edit_burden": _edit_burden(changed_chars),
        "rejection_sampling_sft_rows": len(rejection_sampling_rows),
        "preference_candidate_rows": len(preference_rows),
        "candidate_split_counts": split_counts,
        "non_train_rejection_sampling_rows": non_train_rows,
        "review_source_counts": _review_source_counts(review_records),
        "plan_execution_regressions": sum(
            1 for defect in defects if defect.get("label") == "plan_execution_regression"
        ),
        "boundary": BOUNDARY,
    }


def render_stage5d_report(summary: dict[str, Any]) -> str:
    defects = summary.get("defects", {})
    edit_burden = summary.get("edit_burden", {})
    lines = [
        "# Stage 5D Review Report",
        "",
        "## Summary",
        "",
        f"- Reviewed outputs: {summary.get('reviewed_outputs', 0)}",
        f"- Reviewed output Chinese chars: {summary.get('reviewed_output_chars', 0)}",
        f"- Total defects: {defects.get('total_defects', 0)}",
        f"- Defect density per 10k chars: {summary.get('defect_density_per_10k_chars', 0.0)}",
        f"- Revision records: {summary.get('revision_records', 0)}",
        f"- Accepted revisions: {summary.get('accepted_revisions', 0)}",
        f"- Author acceptance rate: {summary.get('author_acceptance_rate', 0.0)}",
        f"- Changed Chinese char delta: {summary.get('changed_char_delta', 0)}",
        f"- Mean changed chars: {edit_burden.get('mean_changed_chars', 0.0)}",
        f"- Median changed chars: {edit_burden.get('median_changed_chars', 0.0)}",
        f"- Rejection-sampling SFT rows: {summary.get('rejection_sampling_sft_rows', 0)}",
        f"- Preference candidate rows: {summary.get('preference_candidate_rows', 0)}",
        f"- Plan execution regressions: {summary.get('plan_execution_regressions', 0)}",
        f"- Boundary: {summary.get('boundary', BOUNDARY)}",
        "",
        "## Defects By Label",
        "",
        *_render_count_lines(defects.get("by_label", {})),
        "",
        "## Defects By Severity",
        "",
        *_render_count_lines(defects.get("by_severity", {})),
        "",
        "## Candidate Splits",
        "",
        *_render_count_lines(summary.get("candidate_split_counts", {})),
        "",
        "## Review Sources",
        "",
        *_render_count_lines(summary.get("review_source_counts", {})),
        "",
        "## Boundary",
        "",
        BOUNDARY_TEXT,
        "",
    ]
    non_train_rows = summary.get("non_train_rejection_sampling_rows", [])
    if non_train_rows:
        lines.extend(
            [
                "## Stage 5E Blocker",
                "",
                "Non-train rejection-sampling rows were found: " + ", ".join(non_train_rows),
                "",
            ]
        )
    return "\n".join(lines)


def _collect_defects(review_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    defects: list[dict[str, Any]] = []
    for record in review_records:
        record_defects = record.get("defects", [])
        if isinstance(record_defects, list):
            defects.extend(defect for defect in record_defects if isinstance(defect, dict))
    return defects


def _reviewed_output_chars(
    review_records: list[dict[str, Any]], raw_outputs: dict[str, str]
) -> int:
    seen_output_ids: set[str] = set()
    total = 0
    for record in review_records:
        source_output_id = record.get("source_output_id")
        if not isinstance(source_output_id, str) or not source_output_id:
            continue
        if source_output_id in seen_output_ids:
            continue
        seen_output_ids.add(source_output_id)
        total += count_chinese_chars(str(raw_outputs.get(source_output_id, "")))
    return total


def _changed_chars_by_revision(revision_records: list[dict[str, Any]]) -> list[int]:
    return [
        abs(
            count_chinese_chars(str(row.get("revised_output", "")))
            - count_chinese_chars(str(row.get("model_output", "")))
        )
        for row in revision_records
    ]


def _edit_burden(changed_chars: list[int]) -> dict[str, float]:
    if not changed_chars:
        return {"mean_changed_chars": 0.0, "median_changed_chars": 0.0}
    return {
        "mean_changed_chars": round(mean(changed_chars), 4),
        "median_changed_chars": round(median(changed_chars), 4),
    }


def _candidate_split_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        split = str(row.get("source_split") or "unknown")
        counts[split] = counts.get(split, 0) + 1
    return dict(sorted(counts.items()))


def _non_train_rejection_sampling_rows(rows: list[dict[str, Any]]) -> list[str]:
    blocked: list[str] = []
    for index, row in enumerate(rows, start=1):
        if row.get("source_split") != TRAIN_SPLIT:
            blocked.append(str(row.get("revision_id") or row.get("id") or f"row-{index}"))
    return blocked


def _review_source_counts(review_records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in review_records:
        source = str(record.get("review_source") or record.get("reviewer") or "unknown")
        counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


def _render_count_lines(counts: Any) -> list[str]:
    if not isinstance(counts, dict) or not counts:
        return ["- None"]
    return [f"- {key}: {counts[key]}" for key in sorted(counts)]
```

- [ ] **Step 5: Update the report CLI**

In `scripts/build_stage5d_review_report.py`, add:

```python
parser.add_argument("--raw-outputs", required=True)
```

Add this helper below `_read_required_jsonl`:

```python
def _raw_outputs_by_id(rows: list[dict]) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for index, row in enumerate(rows, start=1):
        raw_id = row.get("id") or row.get("source_output_id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise ValueError(f"raw output row {index} missing id")
        text = row.get("output", row.get("text", row.get("raw_output", "")))
        if not isinstance(text, str) or not text:
            raise ValueError(f"raw output row {index} missing output text")
        if raw_id in outputs:
            raise ValueError(f"duplicate raw output id: {raw_id}")
        outputs[raw_id] = text
    return outputs
```

Then load raw outputs before `build_stage5d_summary`:

```python
raw_outputs = _raw_outputs_by_id(_read_required_jsonl(args.raw_outputs))
```

And call:

```python
summary = build_stage5d_summary(
    review_records,
    revision_records,
    rejection_sampling_rows,
    preference_rows,
    raw_outputs=raw_outputs,
)
```

- [ ] **Step 6: Preserve current tests**

Update existing `tests/test_stage5d_report.py` calls to `build_stage5d_summary()` by passing `raw_outputs={}` when the test expects zero density, or a populated map when the test asserts density.

Update the existing CLI happy-path test command with:

```python
"--raw-outputs",
str(raw_outputs),
```

where `raw_outputs` is a JSONL file containing:

```python
_write_jsonl(raw_outputs, [{"id": "out-1", "output": "山河故人"}])
```

- [ ] **Step 7: Run Stage 5D report tests**

Run:

```powershell
python -m pytest tests/test_stage5d_report.py -q
```

Expected: all Stage 5D report tests pass.

- [ ] **Step 8: Commit Task 1**

```powershell
git add src/small_model_train/review/stage5d_report.py scripts/build_stage5d_review_report.py tests/test_stage5d_report.py
git commit -m "fix: complete stage5d report metrics"
```

---

## Task 2: Add Defect Labels To Same-Plot Preference Rows

**Files:**
- Modify: `tests/test_preference_builder.py`
- Modify: `src/small_model_train/preference_builder.py`
- Modify: `scripts/build_same_plot_preference_dataset.py`

- [ ] **Step 1: Add failing preference-label tests**

Append this test to `tests/test_preference_builder.py`:

```python
def test_build_same_plot_preference_candidates_includes_defect_labels():
    review_records = [
        {
            "record_id": "review-c1-001",
            "defects": [
                {"label": "generic_phrase", "severity": "major"},
                {"label": "dialogue_flatness", "severity": "minor"},
            ],
        }
    ]

    rows = build_same_plot_preference_candidates(
        [_same_plot_revision(defect_record_ids=["review-c1-001"])],
        review_records=review_records,
    )

    assert rows[0]["defect_record_ids"] == ["review-c1-001"]
    assert rows[0]["defect_labels"] == ["dialogue_flatness", "generic_phrase"]
    assert rows[0]["reject_type"] == "dialogue_flatness,generic_phrase"
```

Append this test too:

```python
def test_build_same_plot_preference_candidates_rejects_missing_defect_record():
    with pytest.raises(ValueError, match="defect record not found: review-c1-404"):
        build_same_plot_preference_candidates(
            [_same_plot_revision(defect_record_ids=["review-c1-404"])],
            review_records=[],
        )
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
python -m pytest tests/test_preference_builder.py::test_build_same_plot_preference_candidates_includes_defect_labels tests/test_preference_builder.py::test_build_same_plot_preference_candidates_rejects_missing_defect_record -q
```

Expected: fail because `build_same_plot_preference_candidates()` does not accept `review_records` and rows lack `defect_labels`.

- [ ] **Step 3: Replace same-plot preference builder**

In `src/small_model_train/preference_builder.py`, replace `build_same_plot_preference_candidates()` and add the helper below it:

```python
def build_same_plot_preference_candidates(
    revisions: list[dict[str, Any]],
    *,
    review_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    labels_by_record_id = _defect_labels_by_record_id(review_records)
    rows: list[dict[str, Any]] = []
    for revision in revisions:
        validated_revision = validate_revision_record(revision)
        if not is_revision_accepted_for_rejection_sampling(validated_revision):
            continue

        defect_record_ids = list(validated_revision["defect_record_ids"])
        defect_labels: set[str] = set()
        for record_id in defect_record_ids:
            labels = labels_by_record_id.get(record_id)
            if labels is None:
                raise ValueError(f"defect record not found: {record_id}")
            defect_labels.update(labels)
        sorted_labels = sorted(defect_labels)
        if not sorted_labels:
            raise ValueError(f"defect labels are required for revision: {validated_revision['revision_id']}")

        rows.append(
            {
                "id": validated_revision["revision_id"],
                "prompt_sha256": validated_revision["prompt_sha256"],
                "card_id": validated_revision["card_id"],
                "chapter_id": validated_revision["chapter_id"],
                "style_contract_sha256": validated_revision["style_contract_sha256"],
                "chosen": validated_revision["revised_output"],
                "rejected": validated_revision["model_output"],
                "defect_record_ids": defect_record_ids,
                "defect_labels": sorted_labels,
                "reject_type": ",".join(sorted_labels),
                "source": "stage5d_same_plot_revision",
            }
        )
    return rows


def _defect_labels_by_record_id(review_records: list[dict[str, Any]]) -> dict[str, list[str]]:
    labels_by_id: dict[str, list[str]] = {}
    for index, record in enumerate(review_records, start=1):
        record_id = record.get("record_id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"review record {index} missing record_id")
        defects = record.get("defects")
        if not isinstance(defects, list):
            raise ValueError(f"review record {index} defects must be a list")
        labels: list[str] = []
        for defect_index, defect in enumerate(defects, start=1):
            if not isinstance(defect, dict):
                raise ValueError(f"review record {index} defect {defect_index} must be an object")
            label = defect.get("label")
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"review record {index} defect {defect_index} missing label")
            labels.append(label)
        labels_by_id[record_id] = sorted(set(labels))
    return labels_by_id
```

- [ ] **Step 4: Update tests that call the builder**

Every existing `build_same_plot_preference_candidates([...])` call in `tests/test_preference_builder.py` must pass `review_records=[...]`.

For tests that expect a row, use:

```python
review_records = [
    {
        "record_id": "review-c1-001",
        "defects": [{"label": "generic_phrase", "severity": "major"}],
    },
    {
        "record_id": "review-c1-002",
        "defects": [{"label": "dialogue_flatness", "severity": "minor"}],
    },
]
```

Expected rows should include:

```python
"defect_record_ids": ["review-c1-001", "review-c1-002"],
"defect_labels": ["dialogue_flatness", "generic_phrase"],
"reject_type": "dialogue_flatness,generic_phrase",
```

- [ ] **Step 5: Update same-plot preference CLI**

In `scripts/build_same_plot_preference_dataset.py`, add:

```python
parser.add_argument("--review-records", required=True)
```

Load review records and pass them to the builder:

```python
review_records_path = Path(args.review_records)
if not review_records_path.exists():
    raise ValueError(f"review records JSONL not found: {review_records_path}")

rows = build_same_plot_preference_candidates(
    read_jsonl(revisions_path),
    review_records=read_jsonl(review_records_path),
)
```

- [ ] **Step 6: Update CLI test command**

In `tests/test_preference_builder.py`, update `test_build_same_plot_preference_dataset_cli_writes_jsonl` to create a `review_records_path` file and pass:

```python
"--review-records",
str(review_records_path),
```

Write the file with:

```python
write_jsonl(
    review_records_path,
    [
        {
            "record_id": "review-c1-001",
            "defects": [{"label": "generic_phrase", "severity": "major"}],
        },
        {
            "record_id": "review-c1-002",
            "defects": [{"label": "dialogue_flatness", "severity": "minor"}],
        },
    ],
)
```

- [ ] **Step 7: Run preference tests**

Run:

```powershell
python -m pytest tests/test_preference_builder.py -q
```

Expected: all preference builder tests pass.

- [ ] **Step 8: Commit Task 2**

```powershell
git add src/small_model_train/preference_builder.py scripts/build_same_plot_preference_dataset.py tests/test_preference_builder.py
git commit -m "fix: include defect labels in stage5d preferences"
```

---

## Task 3: Preserve Source Split On Rejection-Sampling SFT Rows

**Files:**
- Modify: `tests/test_rejection_sampling_sft.py`
- Modify: `src/small_model_train/review/rejection_sampling.py`

- [ ] **Step 1: Add failing split-provenance assertion**

In `tests/test_rejection_sampling_sft.py::test_build_rejection_sampling_sft_rows_uses_formal_prompt_and_revised_output`, add this expected field:

```python
"source_split": "train",
```

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```powershell
python -m pytest tests/test_rejection_sampling_sft.py::test_build_rejection_sampling_sft_rows_uses_formal_prompt_and_revised_output -q
```

Expected: fail because `source_split` is absent.

- [ ] **Step 3: Add source split to candidate rows**

In `src/small_model_train/review/rejection_sampling.py`, inside the appended row in `build_rejection_sampling_sft_rows()`, add:

```python
"source_split": str(card["provenance"].get("split") or "unknown"),
```

The row block should contain:

```python
rows.append(
    {
        "instruction": INSTRUCTION,
        "input": prompt,
        "output": validated_revision["revised_output"],
        "revision_id": validated_revision["revision_id"],
        "card_id": validated_revision["card_id"],
        "chapter_id": validated_revision["chapter_id"],
        "style_contract_sha256": validated_revision["style_contract_sha256"],
        "raw_output_sha256": validated_revision["raw_output_sha256"],
        "source_split": str(card["provenance"].get("split") or "unknown"),
    }
)
```

- [ ] **Step 4: Run rejection-sampling tests**

Run:

```powershell
python -m pytest tests/test_rejection_sampling_sft.py -q
```

Expected: all rejection-sampling tests pass.

- [ ] **Step 5: Commit Task 3**

```powershell
git add src/small_model_train/review/rejection_sampling.py tests/test_rejection_sampling_sft.py
git commit -m "fix: record source split on stage5d sft rows"
```

---

## Task 4: Add Stage 5E Entry Gate

**Files:**
- Create: `src/small_model_train/review/stage5e_entry.py`
- Create: `scripts/check_stage5e_entry.py`
- Create: `tests/test_stage5e_entry.py`

- [ ] **Step 1: Write failing gate tests**

Create `tests/test_stage5e_entry.py`:

```python
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from small_model_train.io_utils import write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_stage5e_entry.py"


def _summary(**overrides):
    summary = {
        "reviewed_outputs": 2,
        "reviewed_output_chars": 40,
        "defects": {"total_defects": 3},
        "defect_density_per_10k_chars": 750.0,
        "revision_records": 2,
        "accepted_revisions": 1,
        "author_acceptance_rate": 0.5,
        "edit_burden": {"mean_changed_chars": 4.0, "median_changed_chars": 4.0},
        "rejection_sampling_sft_rows": 1,
        "preference_candidate_rows": 1,
        "non_train_rejection_sampling_rows": [],
        "plan_execution_regressions": 0,
        "boundary": "candidate_data_only_no_preference_training",
    }
    summary.update(overrides)
    return summary


def _revision(**overrides):
    row = {
        "revision_id": "rev-1",
        "revision_status": "accepted",
        "card_id": "card-c1-v1",
        "chapter_id": "c1",
        "style_contract_sha256": "a" * 64,
        "prompt_sha256": "b" * 64,
        "raw_output_sha256": "c" * 64,
    }
    row.update(overrides)
    return row


def _generation(**overrides):
    row = {
        "id": "gen-1",
        "card_id": "card-c1-v1",
        "chapter_id": "c1",
        "style_contract_sha256": "a" * 64,
        "prompt_sha256": "b" * 64,
        "raw_output_sha256": "c" * 64,
        "seed": 42,
        "model_role": "stage5d_candidate",
        "generation_params_sha256": "d" * 64,
    }
    row.update(overrides)
    return row


def test_stage5e_entry_gate_passes_complete_stage5d_evidence():
    from small_model_train.review.stage5e_entry import check_stage5e_entry

    result = check_stage5e_entry(
        summary=_summary(),
        review_records=[{"record_id": "review-1", "review_source": "author"}],
        revision_records=[_revision()],
        rejection_sampling_rows=[{"revision_id": "rev-1", "source_split": "train"}],
        preference_rows=[{"id": "pref-1", "defect_labels": ["generic_phrase"]}],
        generation_records=[_generation()],
    )

    assert result["passed"] is True
    assert result["errors"] == []


def test_stage5e_entry_gate_rejects_non_train_candidate():
    from small_model_train.review.stage5e_entry import check_stage5e_entry

    result = check_stage5e_entry(
        summary=_summary(non_train_rejection_sampling_rows=["rev-2"]),
        review_records=[{"record_id": "review-1", "review_source": "author"}],
        revision_records=[_revision()],
        rejection_sampling_rows=[{"revision_id": "rev-2", "source_split": "sealed"}],
        preference_rows=[{"id": "pref-1", "defect_labels": ["generic_phrase"]}],
        generation_records=[_generation()],
    )

    assert result["passed"] is False
    assert "non-train rejection-sampling rows block Stage 5E: rev-2" in result["errors"]


def test_stage5e_entry_gate_rejects_missing_seeded_generation_link():
    from small_model_train.review.stage5e_entry import check_stage5e_entry

    result = check_stage5e_entry(
        summary=_summary(),
        review_records=[{"record_id": "review-1", "review_source": "author"}],
        revision_records=[_revision()],
        rejection_sampling_rows=[{"revision_id": "rev-1", "source_split": "train"}],
        preference_rows=[{"id": "pref-1", "defect_labels": ["generic_phrase"]}],
        generation_records=[],
    )

    assert result["passed"] is False
    assert "accepted revision lacks same-card same-style same-seed generation record: rev-1" in result["errors"]


def test_stage5e_entry_cli_writes_json_report(tmp_path: Path):
    summary_path = tmp_path / "summary.json"
    review_records = tmp_path / "review.jsonl"
    revisions = tmp_path / "revisions.jsonl"
    rs_rows = tmp_path / "rs.jsonl"
    pref_rows = tmp_path / "pref.jsonl"
    generation_records = tmp_path / "generations.jsonl"
    output = tmp_path / "stage5e_entry.json"
    summary_path.write_text(json.dumps(_summary(), ensure_ascii=False), encoding="utf-8")
    write_jsonl(review_records, [{"record_id": "review-1", "review_source": "author"}])
    write_jsonl(revisions, [_revision()])
    write_jsonl(rs_rows, [{"revision_id": "rev-1", "source_split": "train"}])
    write_jsonl(pref_rows, [{"id": "pref-1", "defect_labels": ["generic_phrase"]}])
    write_jsonl(generation_records, [_generation()])

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--summary",
            str(summary_path),
            "--review-records",
            str(review_records),
            "--revisions",
            str(revisions),
            "--rejection-sampling-rows",
            str(rs_rows),
            "--preference-rows",
            str(pref_rows),
            "--generation-records",
            str(generation_records),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True
```

- [ ] **Step 2: Run gate tests and verify failure**

Run:

```powershell
python -m pytest tests/test_stage5e_entry.py -q
```

Expected: fail because the module and script do not exist.

- [ ] **Step 3: Implement Stage 5E entry checker**

Create `src/small_model_train/review/stage5e_entry.py`:

```python
from __future__ import annotations

from typing import Any


ACCEPTED_STATUSES = {"accepted", "accepted_with_minor_edits"}
BOUNDARY = "candidate_data_only_no_preference_training"


def check_stage5e_entry(
    *,
    summary: dict[str, Any],
    review_records: list[dict[str, Any]],
    revision_records: list[dict[str, Any]],
    rejection_sampling_rows: list[dict[str, Any]],
    preference_rows: list[dict[str, Any]],
    generation_records: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    _require_summary_metrics(summary, errors)
    _require_author_or_blind_review(review_records, errors)
    _require_candidate_rows(summary, rejection_sampling_rows, preference_rows, errors)
    _require_seeded_generation_links(revision_records, generation_records, errors)
    return {
        "passed": not errors,
        "errors": errors,
        "entry": "stage5e_controlled_experimentation",
    }


def _require_summary_metrics(summary: dict[str, Any], errors: list[str]) -> None:
    required_fields = (
        "reviewed_outputs",
        "reviewed_output_chars",
        "defect_density_per_10k_chars",
        "author_acceptance_rate",
        "edit_burden",
        "rejection_sampling_sft_rows",
        "preference_candidate_rows",
        "plan_execution_regressions",
        "boundary",
    )
    for field in required_fields:
        if field not in summary:
            errors.append(f"summary missing required Stage 5D metric: {field}")

    if summary.get("reviewed_outputs", 0) <= 0:
        errors.append("Stage 5D review records are required before Stage 5E")
    if summary.get("reviewed_output_chars", 0) <= 0:
        errors.append("reviewed output character count is required before Stage 5E")
    if summary.get("accepted_revisions", 0) <= 0:
        errors.append("at least one accepted author revision is required before Stage 5E")
    edit_burden = summary.get("edit_burden")
    if not isinstance(edit_burden, dict):
        errors.append("summary edit_burden must be an object")
    elif "mean_changed_chars" not in edit_burden or "median_changed_chars" not in edit_burden:
        errors.append("summary edit_burden must include mean and median changed chars")
    if summary.get("boundary") != BOUNDARY:
        errors.append("Stage 5D boundary marker is missing or changed")
    non_train_rows = summary.get("non_train_rejection_sampling_rows", [])
    if non_train_rows:
        errors.append(
            "non-train rejection-sampling rows block Stage 5E: "
            + ", ".join(str(row_id) for row_id in non_train_rows)
        )


def _require_author_or_blind_review(
    review_records: list[dict[str, Any]], errors: list[str]
) -> None:
    accepted_sources = {"author", "human", "blind_review"}
    has_accepted_source = any(
        str(record.get("review_source") or record.get("reviewer") or "") in accepted_sources
        for record in review_records
    )
    if not has_accepted_source:
        errors.append("author, human, or blind-review acceptance data is required before Stage 5E")


def _require_candidate_rows(
    summary: dict[str, Any],
    rejection_sampling_rows: list[dict[str, Any]],
    preference_rows: list[dict[str, Any]],
    errors: list[str],
) -> None:
    if summary.get("rejection_sampling_sft_rows", 0) <= 0 or not rejection_sampling_rows:
        errors.append("rejection-sampling SFT candidate rows are required before Stage 5E")
    for index, row in enumerate(rejection_sampling_rows, start=1):
        if row.get("source_split") != "train":
            errors.append(f"rejection-sampling row {index} must come from train split")

    for index, row in enumerate(preference_rows, start=1):
        labels = row.get("defect_labels")
        if not isinstance(labels, list) or not all(isinstance(label, str) and label for label in labels):
            errors.append(f"preference row {index} must include defect_labels")


def _require_seeded_generation_links(
    revision_records: list[dict[str, Any]],
    generation_records: list[dict[str, Any]],
    errors: list[str],
) -> None:
    generation_keys = {
        (
            row.get("card_id"),
            row.get("style_contract_sha256"),
            row.get("prompt_sha256"),
            row.get("raw_output_sha256"),
        )
        for row in generation_records
        if isinstance(row.get("seed"), int) and not isinstance(row.get("seed"), bool)
    }
    accepted_revisions = [
        row for row in revision_records if row.get("revision_status") in ACCEPTED_STATUSES
    ]
    if not accepted_revisions:
        errors.append("accepted revision records are required before Stage 5E")
        return

    for revision in accepted_revisions:
        key = (
            revision.get("card_id"),
            revision.get("style_contract_sha256"),
            revision.get("prompt_sha256"),
            revision.get("raw_output_sha256"),
        )
        if key not in generation_keys:
            errors.append(
                "accepted revision lacks same-card same-style same-seed generation record: "
                + str(revision.get("revision_id", "<missing revision_id>"))
            )
```

- [ ] **Step 4: Implement Stage 5E entry CLI**

Create `scripts/check_stage5e_entry.py`:

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
from small_model_train.review.stage5e_entry import check_stage5e_entry


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether Stage 5E may start.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--review-records", required=True)
    parser.add_argument("--revisions", required=True)
    parser.add_argument("--rejection-sampling-rows", required=True)
    parser.add_argument("--preference-rows", required=True)
    parser.add_argument("--generation-records", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
        result = check_stage5e_entry(
            summary=summary,
            review_records=read_jsonl(args.review_records),
            revision_records=read_jsonl(args.revisions),
            rejection_sampling_rows=read_jsonl(args.rejection_sampling_rows),
            preference_rows=read_jsonl(args.preference_rows),
            generation_records=read_jsonl(args.generation_records),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not result["passed"]:
        for error in result["errors"]:
            print(error, file=sys.stderr)
        return 1
    print(f"Stage 5E entry gate passed; wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run gate tests**

Run:

```powershell
python -m pytest tests/test_stage5e_entry.py -q
```

Expected: all Stage 5E entry tests pass.

- [ ] **Step 6: Commit Task 4**

```powershell
git add src/small_model_train/review/stage5e_entry.py scripts/check_stage5e_entry.py tests/test_stage5e_entry.py
git commit -m "feat: add stage5e entry gate"
```

---

## Task 5: Update Stage 5D Runbook And Roadmap Gate

**Files:**
- Modify: `docs/stage5d-author-feedback-ai-taste-reduction.zh.md`
- Modify: `docs/pipeline-flow.zh.md`
- Modify: `docs/superpowers/plans/2026-06-23-small-model-train-full-roadmap.md`

- [ ] **Step 1: Update Stage 5D report command**

In `docs/stage5d-author-feedback-ai-taste-reduction.zh.md`, replace the report command with:

```powershell
python scripts/build_stage5d_review_report.py --review-records data_review/stage5d_review_records.jsonl --raw-outputs outputs/stage5d_generation_records.jsonl --revisions data_review/stage5d_revisions.jsonl --rejection-sampling-rows data_sft/stage5d_rejection_sampling_sft.jsonl --preference-rows data_pref/stage5d_same_plot_preference.jsonl --summary-output reports/stage5d_review_summary.json --report-output reports/stage5d_review_report.md
```

- [ ] **Step 2: Update same-plot preference command**

In `docs/stage5d-author-feedback-ai-taste-reduction.zh.md`, replace the preference command with:

```powershell
python scripts/build_same_plot_preference_dataset.py --revisions data_review/stage5d_revisions.jsonl --review-records data_review/stage5d_review_records.jsonl --output data_pref/stage5d_same_plot_preference.jsonl
```

- [ ] **Step 3: Add Stage 5E gate command**

Add this section to `docs/stage5d-author-feedback-ai-taste-reduction.zh.md` after the report section:

```markdown
## Stage 5E 入场检查

Stage 5E 只能在 Stage 5D 证据通过入场检查后开始：

```powershell
python scripts/check_stage5e_entry.py --summary reports/stage5d_review_summary.json --review-records data_review/stage5d_review_records.jsonl --revisions data_review/stage5d_revisions.jsonl --rejection-sampling-rows data_sft/stage5d_rejection_sampling_sft.jsonl --preference-rows data_pref/stage5d_same_plot_preference.jsonl --generation-records outputs/stage5d_generation_records.jsonl --output reports/stage5e_entry_check.json
```

通过条件：报告包含缺陷密度、作者接受率、编辑负担、候选行数和 plan-execution regression；rejection-sampling SFT 候选只来自 train split；accepted 作者修订能追溯到同 card、同 StyleContract、同 prompt hash、同 seed 的生成记录；存在作者、人审或盲审接受数据。
```
```

- [ ] **Step 4: Fix roadmap stale status**

In `docs/superpowers/plans/2026-06-23-small-model-train-full-roadmap.md`, replace the Stage 5D plan pointer line with:

```markdown
**Plan file:** `docs/superpowers/plans/2026-06-27-stage5d-author-feedback-ai-taste-reduction.md`
```

Replace the Stage 5E status line with:

```markdown
**Status:** Blocked until `reports/stage5e_entry_check.json` exists with `"passed": true`.
```

Replace the current execution decision with:

```markdown
Do not execute Stage 5E until the Stage 5D closure plan passes:

```powershell
python scripts/check_stage5e_entry.py --summary reports/stage5d_review_summary.json --review-records data_review/stage5d_review_records.jsonl --revisions data_review/stage5d_revisions.jsonl --rejection-sampling-rows data_sft/stage5d_rejection_sampling_sft.jsonl --preference-rows data_pref/stage5d_same_plot_preference.jsonl --generation-records outputs/stage5d_generation_records.jsonl --output reports/stage5e_entry_check.json
```

After that command exits 0 and full pytest passes, Stage 5E planning may begin.
```
```

- [ ] **Step 5: Run docs scan**

Run:

```powershell
rg -n "build_stage5d_review_report.py|build_same_plot_preference_dataset.py|check_stage5e_entry.py|Stage 5E|Stage 5D" docs README.md --glob "!docs/superpowers/specs/**"
```

Expected: public docs show the new `--raw-outputs`, `--review-records`, and `check_stage5e_entry.py` commands.

- [ ] **Step 6: Commit Task 5**

```powershell
git add docs/stage5d-author-feedback-ai-taste-reduction.zh.md docs/pipeline-flow.zh.md docs/superpowers/plans/2026-06-23-small-model-train-full-roadmap.md
git commit -m "docs: add stage5e entry gate workflow"
```

---

## Task 6: Produce Local Stage 5D Evidence Artifacts

**Files:**
- Generate: `outputs/stage5d_generation_records.jsonl`
- Generate: `data_review/stage5d_review_records.jsonl`
- Generate: `data_review/stage5d_revisions.jsonl`
- Generate: `data_sft/stage5d_rejection_sampling_sft.jsonl`
- Generate: `data_pref/stage5d_same_plot_preference.jsonl`
- Generate: `reports/stage5d_review_summary.json`
- Generate: `reports/stage5d_review_report.md`
- Generate: `reports/stage5e_entry_check.json`

- [ ] **Step 1: Confirm required source artifacts exist**

Run:

```powershell
Test-Path data_cards/chapter_execution_cards_approved.jsonl
Test-Path data_style/style_contract_author_main_v1.json
Test-Path data_review/stage5d_review_records.jsonl
Test-Path data_review/stage5d_revisions.jsonl
Test-Path outputs/stage5d_generation_records.jsonl
```

Expected: all commands print `True`. If any print `False`, create or collect that artifact before continuing.

- [ ] **Step 2: Build rejection-sampling SFT candidates**

Run:

```powershell
python scripts/build_rejection_sampling_sft.py --revisions data_review/stage5d_revisions.jsonl --cards data_cards/chapter_execution_cards_approved.jsonl --style-contract-json data_style/style_contract_author_main_v1.json --output data_sft/stage5d_rejection_sampling_sft.jsonl
```

Expected: stdout includes `wrote` and `rejection-sampling SFT rows`.

- [ ] **Step 3: Build same-plot preference candidates**

Run:

```powershell
python scripts/build_same_plot_preference_dataset.py --revisions data_review/stage5d_revisions.jsonl --review-records data_review/stage5d_review_records.jsonl --output data_pref/stage5d_same_plot_preference.jsonl
```

Expected: stdout includes `wrote` and `same-plot preference rows`.

- [ ] **Step 4: Build Stage 5D report**

Run:

```powershell
python scripts/build_stage5d_review_report.py --review-records data_review/stage5d_review_records.jsonl --raw-outputs outputs/stage5d_generation_records.jsonl --revisions data_review/stage5d_revisions.jsonl --rejection-sampling-rows data_sft/stage5d_rejection_sampling_sft.jsonl --preference-rows data_pref/stage5d_same_plot_preference.jsonl --summary-output reports/stage5d_review_summary.json --report-output reports/stage5d_review_report.md
```

Expected: `reports/stage5d_review_summary.json` contains `defect_density_per_10k_chars`, `edit_burden`, `candidate_split_counts`, and `non_train_rejection_sampling_rows`.

- [ ] **Step 5: Run Stage 5E entry gate**

Run:

```powershell
python scripts/check_stage5e_entry.py --summary reports/stage5d_review_summary.json --review-records data_review/stage5d_review_records.jsonl --revisions data_review/stage5d_revisions.jsonl --rejection-sampling-rows data_sft/stage5d_rejection_sampling_sft.jsonl --preference-rows data_pref/stage5d_same_plot_preference.jsonl --generation-records outputs/stage5d_generation_records.jsonl --output reports/stage5e_entry_check.json
```

Expected: exit code 0 and stdout includes `Stage 5E entry gate passed`.

- [ ] **Step 6: Inspect gate JSON**

Run:

```powershell
Get-Content reports/stage5e_entry_check.json
```

Expected: JSON contains:

```json
{
  "passed": true,
  "errors": [],
  "entry": "stage5e_controlled_experimentation"
}
```

---

## Task 7: Final Verification And Stage 5E Handoff

**Files:**
- Existing files changed by Tasks 1-5.
- Generated local artifacts from Task 6.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
python -m pytest tests/test_stage5d_report.py tests/test_preference_builder.py tests/test_rejection_sampling_sft.py tests/test_stage5e_entry.py -q
```

Expected: all focused tests pass.

- [ ] **Step 2: Run full suite**

Run:

```powershell
python -m pytest -q
```

Expected: full suite passes.

- [ ] **Step 3: Run whitespace check**

Run:

```powershell
git diff --check
```

Expected: no output.

- [ ] **Step 4: Run red-flag scan**

Run:

```powershell
$patterns = @("to" + "do", "tb" + "d", "place" + "holder", "fa" + "ke", "st" + "ub", "not" + " implemented", "pass$")
rg -n -i ($patterns -join "|") src scripts tests docs README.md --glob "!docs/superpowers/plans/**" --glob "!docs/superpowers/specs/**"
```

Expected: no production-code matches. Test-double matches are acceptable only when they are inside tests.

- [ ] **Step 5: Verify Stage 5E gate result**

Run:

```powershell
python scripts/check_stage5e_entry.py --summary reports/stage5d_review_summary.json --review-records data_review/stage5d_review_records.jsonl --revisions data_review/stage5d_revisions.jsonl --rejection-sampling-rows data_sft/stage5d_rejection_sampling_sft.jsonl --preference-rows data_pref/stage5d_same_plot_preference.jsonl --generation-records outputs/stage5d_generation_records.jsonl --output reports/stage5e_entry_check.json
```

Expected: exit code 0.

- [ ] **Step 6: Commit final code and docs**

```powershell
git status --short
git add src scripts tests docs
git commit -m "fix: close stage5d entry criteria"
```

- [ ] **Step 7: Begin Stage 5E planning only after gate pass**

Create a Stage 5E implementation plan only after Step 5 exits 0 and Step 2 passes:

```powershell
Test-Path reports/stage5e_entry_check.json
python -m pytest -q
```

Expected: the path check prints `True`, and pytest exits 0. At that point, create `docs/superpowers/plans/2026-06-28-stage5e-controlled-experimentation-efficiency.md` using the existing Stage 5E roadmap section as scope.

---

## Stage 5D Closure Exit Criteria

Stage 5D closure is complete only when all of these are true:

- Full pytest passes.
- Stage 5D report contains defect density, author acceptance rate, edit burden mean and median, candidate row counts, candidate split counts, and plan-execution regression count.
- Preference candidate rows contain `defect_labels` and `defect_record_ids`.
- Rejection-sampling SFT rows contain `source_split`.
- `reports/stage5e_entry_check.json` exists and contains `"passed": true`.
- Docs show the new Stage 5D report and Stage 5E entry commands.
- No code or docs claim that Stage 5D proves larger-scale model quality improvement.

## Self-Review

- Spec coverage: Tasks 1-3 close the Stage 5D metric, preference-label, and candidate provenance gaps. Task 4 creates the Stage 5E gate required by the roadmap. Task 6 produces local artifacts and blocks Stage 5E until evidence exists.
- Scope control: The plan does not implement model-technique experiments or preference optimization.
- Type consistency: `record_id`, `source_output_id`, `source_split`, `defect_labels`, `defect_record_ids`, `prompt_sha256`, `raw_output_sha256`, and `style_contract_sha256` are used consistently across tests, modules, CLIs, docs, and gate checks.
