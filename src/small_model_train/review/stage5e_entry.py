from __future__ import annotations

from typing import Any


ENTRY = "stage5e_controlled_experimentation"
BOUNDARY = "candidate_data_only_no_preference_training"
ACCEPTED_REVISION_STATUSES = {"accepted", "accepted_with_minor_edits"}
HUMAN_REVIEW_SOURCES = {"author", "human", "blind_review"}
REQUIRED_SUMMARY_FIELDS = (
    "reviewed_outputs",
    "reviewed_output_chars",
    "defect_density_per_10k_chars",
    "revision_records",
    "accepted_revisions",
    "author_acceptance_rate",
    "edit_burden",
    "rejection_sampling_sft_rows",
    "preference_candidate_rows",
    "plan_execution_regressions",
    "boundary",
)
GENERATION_LINK_FIELDS = (
    "card_id",
    "style_contract_sha256",
    "prompt_sha256",
    "raw_output_sha256",
)


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

    _check_summary(
        summary,
        review_records=review_records,
        revision_records=revision_records,
        rejection_sampling_rows=rejection_sampling_rows,
        preference_rows=preference_rows,
        errors=errors,
    )
    _check_review_records(review_records, errors)
    _check_rejection_sampling_rows(summary, rejection_sampling_rows, errors)
    _check_preference_rows(summary, preference_rows, errors)
    _check_accepted_revisions(revision_records, generation_records, errors)

    return {"passed": not errors, "errors": errors, "entry": ENTRY}


def _check_summary(
    summary: dict[str, Any],
    *,
    review_records: list[dict[str, Any]],
    revision_records: list[dict[str, Any]],
    rejection_sampling_rows: list[dict[str, Any]],
    preference_rows: list[dict[str, Any]],
    errors: list[str],
) -> None:
    if not isinstance(summary, dict):
        errors.append("Stage 5D summary must be a JSON object")
        return

    for field in REQUIRED_SUMMARY_FIELDS:
        if field not in summary:
            errors.append(f"Stage 5D summary is missing required field: {field}")

    if _number(summary.get("reviewed_outputs")) <= 0:
        errors.append("reviewed_outputs must be greater than 0 before Stage 5E")
    if _number(summary.get("reviewed_output_chars")) <= 0:
        errors.append("reviewed_output_chars must be greater than 0 before Stage 5E")
    if _number(summary.get("accepted_revisions")) <= 0:
        errors.append("accepted_revisions must be greater than 0 before Stage 5E")

    _check_summary_count(
        summary,
        "reviewed_outputs",
        _row_count(review_records),
        "review records",
        errors,
    )
    _check_summary_count(
        summary,
        "revision_records",
        _row_count(revision_records),
        "revision records",
        errors,
    )
    _check_summary_count(
        summary,
        "accepted_revisions",
        _accepted_revision_count(revision_records),
        "accepted revisions",
        errors,
    )
    _check_summary_count(
        summary,
        "rejection_sampling_sft_rows",
        _row_count(rejection_sampling_rows),
        "rejection-sampling rows",
        errors,
    )
    _check_summary_count(
        summary,
        "preference_candidate_rows",
        _row_count(preference_rows),
        "preference rows",
        errors,
    )

    edit_burden = summary.get("edit_burden")
    if not isinstance(edit_burden, dict):
        errors.append("edit_burden must include mean_changed_chars and median_changed_chars")
    else:
        for field in ("mean_changed_chars", "median_changed_chars"):
            if field not in edit_burden:
                errors.append(f"edit_burden is missing required field: {field}")

    if summary.get("boundary") != BOUNDARY:
        errors.append(f"boundary must equal {BOUNDARY}")

    blocker_rows = summary.get("non_train_rejection_sampling_rows", [])
    if blocker_rows:
        blocker_ids = ", ".join(str(row_id) for row_id in blocker_rows)
        errors.append(f"non-train rejection-sampling rows block Stage 5E: {blocker_ids}")


def _check_review_records(review_records: list[dict[str, Any]], errors: list[str]) -> None:
    if not isinstance(review_records, list):
        errors.append("review records must be a list")
        return

    for record in review_records:
        if not isinstance(record, dict):
            continue
        review_source = str(record.get("review_source") or "").strip()
        reviewer = str(record.get("reviewer") or "").strip()
        acceptance = record.get("overall_acceptance")
        if (
            (review_source in HUMAN_REVIEW_SOURCES or reviewer in HUMAN_REVIEW_SOURCES)
            and acceptance in ACCEPTED_REVISION_STATUSES
        ):
            return

    errors.append("accepted author, human, or blind-review evidence is required before Stage 5E")


def _check_rejection_sampling_rows(
    summary: dict[str, Any],
    rejection_sampling_rows: list[dict[str, Any]],
    errors: list[str],
) -> None:
    if _number(_summary_value(summary, "rejection_sampling_sft_rows")) <= 0:
        errors.append("rejection_sampling_sft_rows must be greater than 0 before Stage 5E")

    if not isinstance(rejection_sampling_rows, list):
        errors.append("rejection-sampling rows must be a list")
        return
    if not rejection_sampling_rows:
        errors.append("rejection-sampling rows are required before Stage 5E")
        return

    for index, row in enumerate(rejection_sampling_rows, start=1):
        if not isinstance(row, dict):
            errors.append(f"rejection-sampling row must be a JSON object: row-{index}")
            continue
        if row.get("source_split") != "train":
            errors.append(
                "rejection-sampling row source_split must be train: "
                + _row_id(row, index)
            )


def _check_preference_rows(
    summary: dict[str, Any],
    preference_rows: list[dict[str, Any]],
    errors: list[str],
) -> None:
    missing_candidate_error_added = False
    if _number(_summary_value(summary, "preference_candidate_rows")) <= 0:
        errors.append("preference candidate rows are required before Stage 5E")
        missing_candidate_error_added = True

    if not isinstance(preference_rows, list):
        errors.append("preference rows must be a list")
        return
    if not preference_rows:
        if not missing_candidate_error_added:
            errors.append("preference candidate rows are required before Stage 5E")
        return

    for index, row in enumerate(preference_rows, start=1):
        if not isinstance(row, dict):
            errors.append(f"preference row requires non-empty defect_labels: row-{index}")
            continue
        defect_labels = row.get("defect_labels")
        if not _non_empty_string_list(defect_labels):
            errors.append(
                "preference row requires non-empty defect_labels: "
                + _row_id(row, index)
            )


def _check_accepted_revisions(
    revision_records: list[dict[str, Any]],
    generation_records: list[dict[str, Any]],
    errors: list[str],
) -> None:
    if not isinstance(revision_records, list):
        errors.append("revision records must be a list")
        return

    accepted_revisions = _accepted_revisions(revision_records)
    if not accepted_revisions:
        errors.append("at least one accepted revision is required before Stage 5E")
        return

    generation_keys = _validated_generation_keys(generation_records, errors)
    for index, revision in enumerate(accepted_revisions, start=1):
        if not _accepted_revision_has_link_fields(revision, index, errors):
            continue
        if _generation_key(revision) not in generation_keys:
            errors.append(
                "accepted revision lacks same-card same-style same-seed generation record: "
                + _row_id(revision, index)
            )


def _validated_generation_keys(
    generation_records: list[dict[str, Any]],
    errors: list[str],
) -> set[tuple[Any, ...]]:
    if not isinstance(generation_records, list):
        errors.append("generation records must be a list")
        return set()

    keys: set[tuple[Any, ...]] = set()
    for index, row in enumerate(generation_records, start=1):
        if not isinstance(row, dict):
            errors.append(f"generation record must be a JSON object: row-{index}")
            continue

        has_link_fields = True
        row_id = _row_id(row, index)
        for field in GENERATION_LINK_FIELDS:
            if not _non_empty_string(row.get(field)):
                has_link_fields = False
                errors.append(f"generation record missing link field: {row_id} {field}")

        has_seed = _is_int_seed(row.get("seed"))
        if not has_seed:
            errors.append(f"generation record missing integer seed: {row_id}")

        if has_link_fields and has_seed:
            keys.add(_generation_key(row))
    return keys


def _accepted_revision_has_link_fields(
    revision: dict[str, Any],
    index: int,
    errors: list[str],
) -> bool:
    has_link_fields = True
    revision_id = _row_id(revision, index)
    for field in GENERATION_LINK_FIELDS:
        if not _non_empty_string(revision.get(field)):
            has_link_fields = False
            errors.append(f"accepted revision missing generation link field: {revision_id} {field}")
    return has_link_fields


def _generation_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("card_id"),
        row.get("style_contract_sha256"),
        row.get("prompt_sha256"),
        row.get("raw_output_sha256"),
    )


def _row_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("revision_id") or row.get("id") or f"row-{index}")


def _row_count(rows: Any) -> int:
    if isinstance(rows, list):
        return len(rows)
    return 0


def _accepted_revisions(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    return [
        row
        for row in rows
        if isinstance(row, dict) and row.get("revision_status") in ACCEPTED_REVISION_STATUSES
    ]


def _accepted_revision_count(rows: Any) -> int:
    return len(_accepted_revisions(rows))


def _check_summary_count(
    summary: dict[str, Any],
    field: str,
    actual: int,
    actual_label: str,
    errors: list[str],
) -> None:
    summary_value = summary.get(field)
    if summary_value != actual:
        errors.append(f"{field} does not match {actual_label}: summary={summary_value} actual={actual}")


def _summary_value(summary: Any, field: str) -> Any:
    if isinstance(summary, dict):
        return summary.get(field)
    return None


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _non_empty_string_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and bool(item.strip()) for item in value)
    )


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_int_seed(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
