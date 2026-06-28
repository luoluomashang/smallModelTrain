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
from small_model_train.sft_builder import render_sft_input


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
    labels_by_record_id = _defect_labels_by_record_id(review_records)

    rows: list[dict[str, Any]] = []
    for revision in revisions:
        validated_revision = validate_revision_record(revision)
        if not is_revision_accepted_for_rejection_sampling(validated_revision):
            continue

        defect_labels = _resolve_defect_labels(
            validated_revision,
            labels_by_record_id=labels_by_record_id,
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


def _defect_labels_by_record_id(
    review_records: list[dict[str, Any]],
) -> dict[str, set[str]]:
    labels_by_record_id: dict[str, set[str]] = {}
    for record_index, record in enumerate(review_records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"review record {record_index}: must be a JSON object")

        record_id = record.get("record_id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(
                f"review record {record_index}: record_id must be a non-empty string"
            )

        defects = record.get("defects")
        if not isinstance(defects, list):
            raise ValueError(f"review record {record_index}: defects must be a list")

        labels = labels_by_record_id.setdefault(record_id, set())
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
            labels.add(label)

    return labels_by_record_id


def _resolve_defect_labels(
    revision: dict[str, Any],
    *,
    labels_by_record_id: dict[str, set[str]],
) -> list[str]:
    labels: set[str] = set()
    for record_id in revision["defect_record_ids"]:
        if record_id not in labels_by_record_id:
            raise ValueError(f"defect record not found: {record_id}")
        labels.update(labels_by_record_id[record_id])

    sorted_labels = sorted(labels)
    if not sorted_labels:
        raise ValueError(
            f"defect labels not found for revision: {revision['revision_id']}"
        )
    return sorted_labels
