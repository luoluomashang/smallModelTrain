from __future__ import annotations

import re
from typing import Any

from small_model_train.review.evidence import validate_review_record
from small_model_train.review.revision_records import (
    ACCEPTED_REVISION_STATUSES,
    validate_revision_record,
)
from small_model_train.schemas.chapter_execution_card import text_sha256


ENTRY = "stage5e_controlled_experimentation"
BOUNDARY = "candidate_data_only_no_preference_training"
HUMAN_REVIEW_SOURCES = {"author", "human", "blind_review"}
LOWER_HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
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
REVISION_GENERATION_REQUIRED_FIELDS = (
    "card_id",
    "style_contract_sha256",
    "prompt_sha256",
    "raw_output_sha256",
)
REJECTION_SAMPLING_REQUIRED_FIELDS = (
    "instruction",
    "input",
    "output",
    "revision_id",
    "card_id",
    "chapter_id",
    "style_contract_sha256",
    "raw_output_sha256",
    "source_split",
)
REJECTION_SAMPLING_REVISION_FIELDS = (
    ("revision_id", "revision_id"),
    ("card_id", "card_id"),
    ("chapter_id", "chapter_id"),
    ("style_contract_sha256", "style_contract_sha256"),
    ("raw_output_sha256", "raw_output_sha256"),
    ("output", "revised_output"),
)
PREFERENCE_REQUIRED_FIELDS = (
    "id",
    "prompt_sha256",
    "card_id",
    "chapter_id",
    "style_contract_sha256",
    "chosen",
    "rejected",
    "defect_record_ids",
    "defect_labels",
    "reject_type",
    "source",
)
PREFERENCE_REVISION_FIELDS = (
    ("id", "revision_id"),
    ("prompt_sha256", "prompt_sha256"),
    ("card_id", "card_id"),
    ("chapter_id", "chapter_id"),
    ("style_contract_sha256", "style_contract_sha256"),
    ("chosen", "revised_output"),
    ("rejected", "model_output"),
    ("defect_record_ids", "defect_record_ids"),
)
SAME_PLOT_PREFERENCE_SOURCE = "stage5d_same_plot_revision"
REVIEW_REVISION_PROVENANCE_FIELDS = (
    "card_id",
    "chapter_id",
    "style_contract_id",
    "style_contract_sha256",
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

    generations = _validate_generation_records(generation_records, errors)
    revisions = _validate_revision_records(revision_records, errors)
    accepted_revisions = [
        revision
        for revision in revisions
        if revision["revision_status"] in ACCEPTED_REVISION_STATUSES
    ]
    accepted_by_id = _accepted_revisions_by_id(accepted_revisions, errors)
    reviews = _validate_review_records(
        review_records,
        raw_outputs=_raw_outputs_by_output_id(generations),
        errors=errors,
    )
    reviews_by_id = _reviews_by_record_id(reviews, errors)
    review_links_by_revision_id = _accepted_revision_review_links(
        accepted_revisions,
        reviews_by_id=reviews_by_id,
        errors=errors,
    )

    _check_summary(
        summary,
        review_records=review_records,
        revision_records=revision_records,
        accepted_revision_count=len(accepted_revisions),
        rejection_sampling_rows=rejection_sampling_rows,
        preference_rows=preference_rows,
        errors=errors,
    )
    _check_accepted_revision_review_evidence(review_links_by_revision_id, errors)
    _check_rejection_sampling_rows(
        summary,
        rejection_sampling_rows,
        accepted_by_id=accepted_by_id,
        errors=errors,
    )
    _check_preference_rows(
        summary,
        preference_rows,
        accepted_by_id=accepted_by_id,
        review_links_by_revision_id=review_links_by_revision_id,
        errors=errors,
    )
    _check_accepted_generation_links(accepted_revisions, generations, errors)

    return {"passed": not errors, "errors": errors, "entry": ENTRY}


def _check_summary(
    summary: dict[str, Any],
    *,
    review_records: list[dict[str, Any]],
    revision_records: list[dict[str, Any]],
    accepted_revision_count: int,
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
        accepted_revision_count,
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


def _validate_revision_records(
    revision_records: list[dict[str, Any]],
    errors: list[str],
) -> list[dict[str, Any]]:
    if not isinstance(revision_records, list):
        errors.append("revision records must be a list")
        return []

    revisions: list[dict[str, Any]] = []
    for index, row in enumerate(revision_records, start=1):
        row_id = _row_id(row, index)
        if not isinstance(row, dict):
            errors.append(f"revision record invalid: {row_id} revision record must be a JSON object")
            continue
        if row.get("revision_status") in ACCEPTED_REVISION_STATUSES:
            _check_accepted_revision_link_fields(row, index, errors)
        try:
            revisions.append(validate_revision_record(row))
        except ValueError as exc:
            errors.append(f"revision record invalid: {row_id} {exc}")
    return revisions


def _check_accepted_revision_link_fields(
    revision: dict[str, Any],
    index: int,
    errors: list[str],
) -> None:
    revision_id = _row_id(revision, index)
    for field in REVISION_GENERATION_REQUIRED_FIELDS:
        if not _non_empty_string(revision.get(field)):
            errors.append(f"accepted revision missing generation link field: {revision_id} {field}")


def _accepted_revisions_by_id(
    accepted_revisions: list[dict[str, Any]],
    errors: list[str],
) -> dict[str, dict[str, Any]]:
    accepted_by_id: dict[str, dict[str, Any]] = {}
    for revision in accepted_revisions:
        revision_id = revision["revision_id"]
        if revision_id in accepted_by_id:
            errors.append(f"duplicate accepted revision id: {revision_id}")
            continue
        accepted_by_id[revision_id] = revision
    return accepted_by_id


def _validate_generation_records(
    generation_records: list[dict[str, Any]],
    errors: list[str],
) -> list[dict[str, Any]]:
    if not isinstance(generation_records, list):
        errors.append("generation records must be a list")
        return []

    generations: list[dict[str, Any]] = []
    for index, row in enumerate(generation_records, start=1):
        if not isinstance(row, dict):
            errors.append(f"generation record must be a JSON object: row-{index}")
            continue

        output_id = _generation_output_id(row)
        card_id = _generation_card_id(row, output_id)
        row_id = _generation_row_id(row, index)
        prompt_sha256 = row.get("prompt_sha256")
        raw_output = _generation_raw_output(row)
        seed = _generation_seed(row)
        style_contract_sha256 = row.get("style_contract_sha256")
        raw_output_sha256 = _generation_raw_output_sha256(
            row,
            row_id=row_id,
            raw_output=raw_output,
            errors=errors,
        )

        valid = True
        if output_id is None:
            valid = False
            errors.append(f"generation record missing output id: {row_id}")
        if card_id is None:
            valid = False
            errors.append(f"generation record missing card id: {row_id}")
        if not _is_lower_hex_sha256(prompt_sha256):
            valid = False
            if _non_empty_string(prompt_sha256):
                errors.append(f"generation record invalid prompt_sha256: {row_id}")
            else:
                errors.append(f"generation record missing prompt_sha256: {row_id}")
        if raw_output is None:
            valid = False
            errors.append(f"generation record missing raw output: {row_id}")
        if raw_output_sha256 is None:
            valid = False
        if not _is_int_seed(seed):
            valid = False
            errors.append(f"generation record missing integer seed: {row_id}")
        if style_contract_sha256 is not None and not _is_lower_hex_sha256(style_contract_sha256):
            valid = False
            errors.append(f"generation record invalid style_contract_sha256: {row_id}")

        if valid:
            if any(generation["output_id"] == output_id for generation in generations):
                errors.append(f"duplicate generation output id: {output_id}")
                continue
            generations.append(
                {
                    "row_id": row_id,
                    "output_id": output_id,
                    "card_id": card_id,
                    "prompt_sha256": prompt_sha256,
                    "raw_output": raw_output,
                    "raw_output_sha256": raw_output_sha256,
                    "seed": seed,
                    "style_contract_sha256": style_contract_sha256,
                }
            )
    return generations


def _generation_raw_output_sha256(
    row: dict[str, Any],
    *,
    row_id: str,
    raw_output: str | None,
    errors: list[str],
) -> str | None:
    explicit_hash = row.get("raw_output_sha256")
    if explicit_hash is not None and not _is_lower_hex_sha256(explicit_hash):
        errors.append(f"generation record invalid raw_output_sha256: {row_id}")
        return None
    if raw_output is None:
        return None

    computed_hash = text_sha256(raw_output)
    if explicit_hash is None:
        return computed_hash
    if explicit_hash != computed_hash:
        errors.append(f"generation record raw_output_sha256 mismatch: {row_id}")
        return None
    return explicit_hash


def _generation_raw_output(row: dict[str, Any]) -> str | None:
    for field in ("raw_output", "output"):
        value = row.get(field)
        if isinstance(value, str) and value:
            return value
    return None


def _generation_seed(row: dict[str, Any]) -> Any:
    if "seed" in row:
        return row.get("seed")
    params = row.get("params")
    if isinstance(params, dict) and "seed" in params:
        return params.get("seed")
    return None


def _generation_output_id(row: dict[str, Any]) -> str | None:
    return _first_non_empty_string(row, ("source_output_id", "id"))


def _generation_card_id(row: dict[str, Any], output_id: str | None) -> str | None:
    return _first_non_empty_string(row, ("card_id",)) or output_id


def _raw_outputs_by_output_id(generations: list[dict[str, Any]]) -> dict[str, str]:
    raw_outputs: dict[str, str] = {}
    for generation in generations:
        raw_outputs[generation["output_id"]] = generation["raw_output"]
    return raw_outputs


def _validate_review_records(
    review_records: list[dict[str, Any]],
    *,
    raw_outputs: dict[str, str],
    errors: list[str],
) -> list[dict[str, Any]]:
    if not isinstance(review_records, list):
        errors.append("review records must be a list")
        return []

    reviews: list[dict[str, Any]] = []
    for index, record in enumerate(review_records, start=1):
        record_id = _review_row_id(record, index)
        if not isinstance(record, dict):
            errors.append(f"review record invalid: {record_id} review record must be a JSON object")
            continue

        source_output_id = record.get("source_output_id")
        raw_output = raw_outputs.get(source_output_id)
        if raw_output is None:
            errors.append(
                f"review record invalid: {record_id} raw output not found for source_output_id: {source_output_id}"
            )
            continue
        try:
            reviews.append(validate_review_record(record, raw_output=raw_output))
        except ValueError as exc:
            errors.append(f"review record invalid: {record_id} {exc}")
    return reviews


def _reviews_by_record_id(
    review_records: list[dict[str, Any]],
    errors: list[str],
) -> dict[str, dict[str, Any]]:
    reviews_by_id: dict[str, dict[str, Any]] = {}
    for record in review_records:
        record_id = record["record_id"]
        if record_id in reviews_by_id:
            errors.append(f"duplicate review record id: {record_id}")
            continue
        reviews_by_id[record_id] = record
    return reviews_by_id


def _accepted_revision_review_links(
    accepted_revisions: list[dict[str, Any]],
    *,
    reviews_by_id: dict[str, dict[str, Any]],
    errors: list[str],
) -> dict[str, dict[str, Any]]:
    links_by_revision_id: dict[str, dict[str, Any]] = {}
    for revision in accepted_revisions:
        revision_id = revision["revision_id"]
        labels: set[str] = set()
        has_accepted_human_evidence = False

        for record_id in revision["defect_record_ids"]:
            review = reviews_by_id.get(record_id)
            if review is None:
                errors.append(f"accepted revision referenced review missing: {revision_id} {record_id}")
                continue

            for field in REVIEW_REVISION_PROVENANCE_FIELDS:
                if review.get(field) != revision[field]:
                    errors.append(
                        "accepted revision referenced review provenance mismatch: "
                        f"{revision_id} {record_id} {field}"
                    )

            for defect in review.get("defects", []):
                label = defect.get("label")
                if _non_empty_string(label):
                    labels.add(str(label))

            if _is_accepted_human_review(review):
                has_accepted_human_evidence = True

        defect_labels = sorted(labels)
        if not defect_labels:
            errors.append(f"accepted revision referenced reviews have no defect labels: {revision_id}")

        links_by_revision_id[revision_id] = {
            "defect_labels": defect_labels,
            "has_accepted_human_evidence": has_accepted_human_evidence,
        }
    return links_by_revision_id


def _check_accepted_revision_review_evidence(
    review_links_by_revision_id: dict[str, dict[str, Any]],
    errors: list[str],
) -> None:
    for link in review_links_by_revision_id.values():
        if link["has_accepted_human_evidence"]:
            return

    errors.append("accepted author, human, or blind-review evidence is required before Stage 5E")


def _is_accepted_human_review(record: dict[str, Any]) -> bool:
    review_source = str(record.get("review_source") or "").strip()
    reviewer = str(record.get("reviewer") or "").strip()
    acceptance = record.get("overall_acceptance")
    return (
        (review_source in HUMAN_REVIEW_SOURCES or reviewer in HUMAN_REVIEW_SOURCES)
        and acceptance in ACCEPTED_REVISION_STATUSES
    )


def _check_rejection_sampling_rows(
    summary: dict[str, Any],
    rejection_sampling_rows: list[dict[str, Any]],
    *,
    accepted_by_id: dict[str, dict[str, Any]],
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

        row_id = _row_id(row, index)
        for field in REJECTION_SAMPLING_REQUIRED_FIELDS:
            if not _non_empty_string(row.get(field)):
                errors.append(f"rejection-sampling row missing required field: {row_id} {field}")

        if row.get("source_split") != "train":
            errors.append("rejection-sampling row source_split must be train: " + row_id)

        revision = accepted_by_id.get(row.get("revision_id"))
        if revision is None:
            errors.append(f"rejection-sampling row not linked to accepted revision: {row_id}")
            continue

        for row_field, revision_field in REJECTION_SAMPLING_REVISION_FIELDS:
            if row.get(row_field) != revision[revision_field]:
                errors.append(
                    f"rejection-sampling row mismatch accepted revision: {row_id} {row_field}"
                )


def _check_preference_rows(
    summary: dict[str, Any],
    preference_rows: list[dict[str, Any]],
    *,
    accepted_by_id: dict[str, dict[str, Any]],
    review_links_by_revision_id: dict[str, dict[str, Any]],
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
            errors.append(f"preference row must be a JSON object: row-{index}")
            continue

        row_id = _row_id(row, index)
        for field in PREFERENCE_REQUIRED_FIELDS:
            if field in ("defect_record_ids", "defect_labels"):
                if not _non_empty_string_list(row.get(field)):
                    errors.append(f"preference row requires non-empty {field}: {row_id}")
            elif not _non_empty_string(row.get(field)):
                errors.append(f"preference row missing required field: {row_id} {field}")

        if row.get("source") != SAME_PLOT_PREFERENCE_SOURCE:
            errors.append(f"preference row source must be {SAME_PLOT_PREFERENCE_SOURCE}: {row_id}")

        revision = accepted_by_id.get(row.get("id"))
        if revision is None:
            errors.append(f"preference row not linked to accepted revision: {row_id}")
            continue

        for row_field, revision_field in PREFERENCE_REVISION_FIELDS:
            if row.get(row_field) != revision[revision_field]:
                errors.append(f"preference row mismatch accepted revision: {row_id} {row_field}")

        expected_labels = review_links_by_revision_id.get(revision["revision_id"], {}).get(
            "defect_labels",
            [],
        )
        defect_labels = row.get("defect_labels")
        if _non_empty_string_list(defect_labels) and defect_labels != expected_labels:
            errors.append(
                f"preference row defect_labels do not match referenced review defects: {row_id}"
            )
        expected_reject_type = ",".join(expected_labels)
        if expected_reject_type and row.get("reject_type") != expected_reject_type:
            errors.append(
                f"preference row reject_type does not match referenced review defects: {row_id}"
            )


def _check_accepted_generation_links(
    accepted_revisions: list[dict[str, Any]],
    generations: list[dict[str, Any]],
    errors: list[str],
) -> None:
    if not accepted_revisions:
        errors.append("at least one accepted revision is required before Stage 5E")
        return

    for revision in accepted_revisions:
        revision_id = revision["revision_id"]
        linked = False
        for generation in generations:
            if not _generation_matches_revision_core(generation, revision):
                continue
            generation_style = generation.get("style_contract_sha256")
            if generation_style is not None and generation_style != revision["style_contract_sha256"]:
                errors.append(
                    "generation record style_contract_sha256 mismatch accepted revision: "
                    f"{generation['row_id']} {revision_id}"
                )
                continue
            linked = True
            break

        if not linked:
            errors.append(
                "accepted revision lacks same-card same-style same-seed generation record: "
                + revision_id
            )


def _generation_matches_revision_core(
    generation: dict[str, Any],
    revision: dict[str, Any],
) -> bool:
    return (
        generation["card_id"] == revision["card_id"]
        and generation["prompt_sha256"] == revision["prompt_sha256"]
        and generation["raw_output_sha256"] == revision["raw_output_sha256"]
    )


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


def _row_count(rows: Any) -> int:
    if isinstance(rows, list):
        return len(rows)
    return 0


def _row_id(row: Any, index: int) -> str:
    if isinstance(row, dict):
        for field in ("revision_id", "id", "record_id"):
            value = row.get(field)
            if _non_empty_string(value):
                return str(value)
    return f"row-{index}"


def _review_row_id(row: Any, index: int) -> str:
    if isinstance(row, dict):
        value = row.get("record_id")
        if _non_empty_string(value):
            return str(value)
    return _row_id(row, index)


def _generation_row_id(row: dict[str, Any], index: int) -> str:
    value = _first_non_empty_string(row, ("source_output_id", "id", "card_id"))
    if value is not None:
        return value
    return f"row-{index}"


def _first_non_empty_string(row: dict[str, Any], fields: tuple[str, ...]) -> str | None:
    for field in fields:
        value = row.get(field)
        if _non_empty_string(value):
            return str(value)
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


def _is_lower_hex_sha256(value: Any) -> bool:
    return isinstance(value, str) and LOWER_HEX_SHA256_RE.fullmatch(value) is not None


def _is_int_seed(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
