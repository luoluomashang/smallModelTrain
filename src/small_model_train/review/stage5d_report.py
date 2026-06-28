from __future__ import annotations

from collections import Counter
from statistics import mean, median
from typing import Any

from small_model_train.review.style_defects import summarize_style_defects
from small_model_train.text_utils import count_chinese_chars


BOUNDARY = "candidate_data_only_no_preference_training"
BOUNDARY_TEXT = "这些 preference rows 只是候选数据，不代表已经运行 DPO/SimPO/ORPO/KTO。"
ACCEPTED_STATUSES = {"accepted", "accepted_with_minor_edits"}


def build_stage5d_summary(
    review_records: list[dict[str, Any]],
    revision_records: list[dict[str, Any]],
    rejection_sampling_rows: list[dict[str, Any]],
    preference_rows: list[dict[str, Any]],
    *,
    raw_outputs: dict[str, str],
) -> dict[str, Any]:
    defects = _collect_defects(review_records)
    defect_summary = summarize_style_defects(defects)
    accepted_revisions = sum(
        1 for row in revision_records if row.get("revision_status") in ACCEPTED_STATUSES
    )
    revision_count = len(revision_records)
    changed_char_deltas = [_revision_changed_char_delta(row) for row in revision_records]
    accepted_changed_char_deltas = [
        _revision_changed_char_delta(row)
        for row in revision_records
        if row.get("revision_status") in ACCEPTED_STATUSES
    ]
    reviewed_output_chars = _reviewed_output_chars(review_records, raw_outputs)
    candidate_split_counts, non_train_rejection_sampling_rows = _candidate_split_summary(
        rejection_sampling_rows
    )
    return {
        "reviewed_outputs": len(review_records),
        "reviewed_output_chars": reviewed_output_chars,
        "defects": defect_summary,
        "defect_density_per_10k_chars": _defect_density_per_10k_chars(
            defect_summary.get("total_defects", 0),
            reviewed_output_chars,
        ),
        "revision_records": revision_count,
        "accepted_revisions": accepted_revisions,
        "author_acceptance_rate": round(accepted_revisions / revision_count, 4)
        if revision_count
        else 0.0,
        "changed_char_delta": sum(changed_char_deltas),
        "edit_burden": _edit_burden(accepted_changed_char_deltas),
        "rejection_sampling_sft_rows": len(rejection_sampling_rows),
        "preference_candidate_rows": len(preference_rows),
        "candidate_split_counts": candidate_split_counts,
        "non_train_rejection_sampling_rows": non_train_rejection_sampling_rows,
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
        f"- Defect density per 10k reviewed Chinese chars: {summary.get('defect_density_per_10k_chars', 0.0)}",
        f"- Revision records: {summary.get('revision_records', 0)}",
        f"- Accepted revisions: {summary.get('accepted_revisions', 0)}",
        f"- Author acceptance rate: {summary.get('author_acceptance_rate', 0.0)}",
        f"- Changed Chinese char delta: {summary.get('changed_char_delta', 0)}",
        f"- Mean changed Chinese chars: {edit_burden.get('mean_changed_chars', 0.0)}",
        f"- Median changed Chinese chars: {edit_burden.get('median_changed_chars', 0.0)}",
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
        "## Candidate Split Counts",
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
    blocker_rows = summary.get("non_train_rejection_sampling_rows", [])
    if blocker_rows:
        lines.extend(
            [
                "## Stage 5E Blocker",
                "",
                "Non-train rejection-sampling rows must be removed before Stage 5E training:",
                "",
                *[f"- {row_id}" for row_id in blocker_rows],
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


def _revision_changed_char_delta(row: dict[str, Any]) -> int:
    model_chars = count_chinese_chars(str(row.get("model_output", "")))
    revised_chars = count_chinese_chars(str(row.get("revised_output", "")))
    return abs(model_chars - revised_chars)


def _reviewed_output_chars(
    review_records: list[dict[str, Any]],
    raw_outputs: dict[str, str],
) -> int:
    reviewed_output_ids: list[str] = []
    seen_output_ids: set[str] = set()
    for record in review_records:
        output_id = record.get("source_output_id") or record.get("id")
        if output_id is None:
            raise ValueError("review record is missing source_output_id/id")
        output_key = str(output_id)
        if output_key not in seen_output_ids:
            reviewed_output_ids.append(output_key)
            seen_output_ids.add(output_key)

    total = 0
    for output_id in reviewed_output_ids:
        if output_id not in raw_outputs:
            raise ValueError(f"missing raw output for reviewed output id: {output_id}")
        raw_output = raw_outputs[output_id]
        if raw_output == "":
            raise ValueError(f"empty raw output for reviewed output id: {output_id}")
        total += count_chinese_chars(raw_output)
    return total


def _defect_density_per_10k_chars(total_defects: int, reviewed_output_chars: int) -> float:
    if reviewed_output_chars == 0:
        return 0.0
    return round(total_defects / reviewed_output_chars * 10000, 4)


def _edit_burden(changed_char_deltas: list[int]) -> dict[str, float]:
    if not changed_char_deltas:
        return {"mean_changed_chars": 0.0, "median_changed_chars": 0.0}
    return {
        "mean_changed_chars": round(float(mean(changed_char_deltas)), 4),
        "median_changed_chars": round(float(median(changed_char_deltas)), 4),
    }


def _candidate_split_summary(
    rejection_sampling_rows: list[dict[str, Any]],
) -> tuple[dict[str, int], list[str]]:
    counts: Counter[str] = Counter()
    non_train_rows: list[str] = []
    for index, row in enumerate(rejection_sampling_rows, start=1):
        split = str(row.get("source_split") or "unknown")
        counts[split] += 1
        if split != "train":
            non_train_rows.append(_candidate_row_id(row, index))
    return dict(sorted(counts.items())), non_train_rows


def _candidate_row_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("revision_id") or row.get("id") or f"row-{index}")


def _review_source_counts(review_records: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in review_records:
        counts[str(record.get("review_source") or record.get("reviewer") or "unknown")] += 1
    return dict(sorted(counts.items()))


def _render_count_lines(counts: Any) -> list[str]:
    if not isinstance(counts, dict) or not counts:
        return ["- None"]
    return [f"- {key}: {counts[key]}" for key in sorted(counts)]
