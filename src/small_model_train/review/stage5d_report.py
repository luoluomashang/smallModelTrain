from __future__ import annotations

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
) -> dict[str, Any]:
    defects = _collect_defects(review_records)
    accepted_revisions = sum(
        1 for row in revision_records if row.get("revision_status") in ACCEPTED_STATUSES
    )
    revision_count = len(revision_records)
    return {
        "reviewed_outputs": len(review_records),
        "defects": summarize_style_defects(defects),
        "revision_records": revision_count,
        "accepted_revisions": accepted_revisions,
        "author_acceptance_rate": round(accepted_revisions / revision_count, 4)
        if revision_count
        else 0.0,
        "changed_char_delta": _changed_char_delta(revision_records),
        "rejection_sampling_sft_rows": len(rejection_sampling_rows),
        "preference_candidate_rows": len(preference_rows),
        "plan_execution_regressions": sum(
            1 for defect in defects if defect.get("label") == "plan_execution_regression"
        ),
        "boundary": BOUNDARY,
    }


def render_stage5d_report(summary: dict[str, Any]) -> str:
    defects = summary.get("defects", {})
    lines = [
        "# Stage 5D Review Report",
        "",
        "## Summary",
        "",
        f"- Reviewed outputs: {summary.get('reviewed_outputs', 0)}",
        f"- Total defects: {defects.get('total_defects', 0)}",
        f"- Revision records: {summary.get('revision_records', 0)}",
        f"- Accepted revisions: {summary.get('accepted_revisions', 0)}",
        f"- Author acceptance rate: {summary.get('author_acceptance_rate', 0.0)}",
        f"- Changed Chinese char delta: {summary.get('changed_char_delta', 0)}",
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
        "## Boundary",
        "",
        BOUNDARY_TEXT,
        "",
    ]
    return "\n".join(lines)


def _collect_defects(review_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    defects: list[dict[str, Any]] = []
    for record in review_records:
        record_defects = record.get("defects", [])
        if isinstance(record_defects, list):
            defects.extend(defect for defect in record_defects if isinstance(defect, dict))
    return defects


def _changed_char_delta(revision_records: list[dict[str, Any]]) -> int:
    model_chars = sum(count_chinese_chars(str(row.get("model_output", ""))) for row in revision_records)
    revised_chars = sum(
        count_chinese_chars(str(row.get("revised_output", ""))) for row in revision_records
    )
    return abs(model_chars - revised_chars)


def _render_count_lines(counts: Any) -> list[str]:
    if not isinstance(counts, dict) or not counts:
        return ["- None"]
    return [f"- {key}: {counts[key]}" for key in sorted(counts)]
