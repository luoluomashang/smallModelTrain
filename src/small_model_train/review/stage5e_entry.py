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
    "author_acceptance_rate",
    "edit_burden",
    "rejection_sampling_sft_rows",
    "preference_candidate_rows",
    "plan_execution_regressions",
    "boundary",
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

    _check_summary(summary, errors)
    _check_review_records(review_records, errors)
    _check_rejection_sampling_rows(summary, rejection_sampling_rows, errors)
    _check_preference_rows(preference_rows, errors)
    _check_accepted_revisions(revision_records, generation_records, errors)

    return {"passed": not errors, "errors": errors, "entry": ENTRY}


def _check_summary(summary: dict[str, Any], errors: list[str]) -> None:
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
        if review_source in HUMAN_REVIEW_SOURCES or reviewer in HUMAN_REVIEW_SOURCES:
            return

    errors.append("author, human, or blind-review acceptance data is required before Stage 5E")


def _check_rejection_sampling_rows(
    summary: dict[str, Any],
    rejection_sampling_rows: list[dict[str, Any]],
    errors: list[str],
) -> None:
    if _number(summary.get("rejection_sampling_sft_rows")) <= 0:
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


def _check_preference_rows(preference_rows: list[dict[str, Any]], errors: list[str]) -> None:
    if not isinstance(preference_rows, list):
        errors.append("preference rows must be a list")
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

    accepted_revisions = [
        row
        for row in revision_records
        if isinstance(row, dict) and row.get("revision_status") in ACCEPTED_REVISION_STATUSES
    ]
    if not accepted_revisions:
        errors.append("at least one accepted revision is required before Stage 5E")
        return

    generation_keys = _seeded_generation_keys(generation_records)
    for index, revision in enumerate(accepted_revisions, start=1):
        if _generation_key(revision) not in generation_keys:
            errors.append(
                "accepted revision lacks same-card same-style same-seed generation record: "
                + _row_id(revision, index)
            )


def _seeded_generation_keys(generation_records: list[dict[str, Any]]) -> set[tuple[Any, ...]]:
    if not isinstance(generation_records, list):
        return set()

    keys: set[tuple[Any, ...]] = set()
    for row in generation_records:
        if isinstance(row, dict) and _is_int_seed(row.get("seed")):
            keys.add(_generation_key(row))
    return keys


def _generation_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("card_id"),
        row.get("style_contract_sha256"),
        row.get("prompt_sha256"),
        row.get("raw_output_sha256"),
    )


def _row_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("revision_id") or row.get("id") or f"row-{index}")


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


def _is_int_seed(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
