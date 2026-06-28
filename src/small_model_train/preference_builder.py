"""Preference-candidate construction from failed scoring rows.

Stage 1 only prepares candidate pairs for later preference work. It does not
pretend that a reward model or DPO training loop has already been run.
"""

from __future__ import annotations

from typing import Any

from small_model_train.review.revision_records import (
    is_revision_accepted_for_rejection_sampling,
    validate_revision_record,
)
from small_model_train.review.style_defects import DEFECT_LABELS
from small_model_train.sft_builder import render_sft_input


REVIEW_RECORD_PROVENANCE_FIELDS = (
    "card_id",
    "chapter_id",
    "style_contract_id",
    "style_contract_sha256",
    "raw_output_sha256",
)


def build_preference_candidates(
    cards: list[dict],
    outputs: list[dict],
    scores: list[dict],
) -> list[dict]:
    cards_by_id = {row["id"]: row for row in cards}
    outputs_by_id = {row["id"]: row for row in outputs}
    rows: list[dict] = []
    for score in scores:
        failure_types = score.get("failure_types", [])
        if not failure_types:
            continue
        sample_id = score["id"]
        card = cards_by_id.get(sample_id, {})
        output = outputs_by_id.get(sample_id, {})
        reject_type = failure_types[0] if failure_types else "unknown"
        rows.append(
            {
                "id": sample_id,
                "prompt": card["prompt"] if "prompt" in card else render_sft_input(card),
                "rejected": output.get("output", output.get("text", "")),
                "reject_type": reject_type,
                "chosen": "",
                "source": "failed_eval",
            }
        )
    return rows


def build_same_plot_preference_candidates(
    revisions: list[dict[str, Any]],
    *,
    review_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    review_records_by_id = _review_records_by_id(review_records)

    rows: list[dict[str, Any]] = []
    for revision in revisions:
        validated_revision = validate_revision_record(revision)
        if not is_revision_accepted_for_rejection_sampling(validated_revision):
            continue

        defect_labels = _resolve_defect_labels(
            validated_revision,
            review_records_by_id=review_records_by_id,
        )
        rows.append(
            {
                "id": validated_revision["revision_id"],
                "prompt_sha256": validated_revision["prompt_sha256"],
                "card_id": validated_revision["card_id"],
                "chapter_id": validated_revision["chapter_id"],
                "style_contract_sha256": validated_revision["style_contract_sha256"],
                "chosen": validated_revision["revised_output"],
                "rejected": validated_revision["model_output"],
                "defect_record_ids": list(validated_revision["defect_record_ids"]),
                "defect_labels": defect_labels,
                "reject_type": ",".join(defect_labels),
                "source": "stage5d_same_plot_revision",
            }
        )
    return rows


def _review_records_by_id(
    review_records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    review_records_by_id: dict[str, dict[str, Any]] = {}
    for record_index, record in enumerate(review_records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"review record {record_index}: must be a JSON object")

        record_id = record.get("record_id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(
                f"review record {record_index}: record_id must be a non-empty string"
            )
        if record_id in review_records_by_id:
            raise ValueError(f"duplicate review record id: {record_id}")

        defects = record.get("defects")
        if not isinstance(defects, list):
            raise ValueError(f"review record {record_index}: defects must be a list")

        labels: set[str] = set()
        for defect_index, defect in enumerate(defects):
            if not isinstance(defect, dict):
                raise ValueError(
                    f"review record {record_index}: defects[{defect_index}] must be a JSON object"
                )

            label = defect.get("label")
            if not isinstance(label, str) or not label.strip():
                raise ValueError(
                    f"review record {record_index}: defects[{defect_index}].label must be a non-empty string"
                )
            if label not in DEFECT_LABELS:
                raise ValueError(f"defect label is not recognized: {label}")
            labels.add(label)

        review_records_by_id[record_id] = {"record": record, "labels": labels}

    return review_records_by_id


def _resolve_defect_labels(
    revision: dict[str, Any],
    *,
    review_records_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    labels: set[str] = set()
    for record_id in revision["defect_record_ids"]:
        review_record = review_records_by_id.get(record_id)
        if review_record is None:
            raise ValueError(f"defect record not found: {record_id}")
        _validate_review_record_provenance(
            revision,
            review_record=review_record["record"],
        )
        labels.update(review_record["labels"])

    sorted_labels = sorted(labels)
    if not sorted_labels:
        raise ValueError(
            f"defect labels not found for revision: {revision['revision_id']}"
        )
    return sorted_labels


def _validate_review_record_provenance(
    revision: dict[str, Any],
    *,
    review_record: dict[str, Any],
) -> None:
    record_id = review_record["record_id"]
    for field in REVIEW_RECORD_PROVENANCE_FIELDS:
        if field not in review_record:
            raise ValueError(f"review record provenance missing: {record_id} {field}")
        if review_record[field] != revision[field]:
            raise ValueError(f"review record provenance mismatch: {record_id} {field}")
