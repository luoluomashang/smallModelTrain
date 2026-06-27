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
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for revision in revisions:
        validated_revision = validate_revision_record(revision)
        if not is_revision_accepted_for_rejection_sampling(validated_revision):
            continue

        rows.append(
            {
                "id": validated_revision["revision_id"],
                "prompt_sha256": validated_revision["prompt_sha256"],
                "card_id": validated_revision["card_id"],
                "chapter_id": validated_revision["chapter_id"],
                "style_contract_sha256": validated_revision["style_contract_sha256"],
                "chosen": validated_revision["revised_output"],
                "rejected": validated_revision["model_output"],
                "reject_type": ",".join(validated_revision["defect_record_ids"]),
                "source": "stage5d_same_plot_revision",
            }
        )
    return rows
