"""Build SFT rows from accepted same-plot author revisions."""

from __future__ import annotations

from typing import Any

from small_model_train.cards.card_renderer import render_chapter_execution_input
from small_model_train.review.revision_records import (
    is_revision_accepted_for_rejection_sampling,
    validate_revision_record_provenance,
)
from small_model_train.schemas.chapter_execution_card import (
    is_card_approved_for_formal_sft,
    text_sha256,
    validate_chapter_execution_card,
)
from small_model_train.sft_builder import INSTRUCTION
from small_model_train.style_contract import validate_style_contract_asset


def build_rejection_sampling_sft_rows(
    revisions: list[dict[str, Any]],
    cards: list[dict[str, Any]],
    style_contract: dict[str, Any],
) -> list[dict[str, str]]:
    contract = validate_style_contract_asset(style_contract)
    card_by_id = _validated_card_by_id(cards)

    rows: list[dict[str, str]] = []
    for index, revision in enumerate(revisions, start=1):
        if not is_revision_accepted_for_rejection_sampling(revision):
            revision_id = revision.get("revision_id", f"row {index}") if isinstance(revision, dict) else f"row {index}"
            raise ValueError(f"revision_status must be accepted for rejection sampling: {revision_id}")

        card_id = revision["card_id"]
        card = card_by_id.get(card_id)
        if card is None:
            raise ValueError(f"matching formal card is required for revision: {card_id}")

        prompt = render_chapter_execution_input(card, contract)
        prompt_sha256 = text_sha256(prompt)
        validated_revision = validate_revision_record_provenance(
            revision,
            card=card,
            style_contract_id=contract["style_contract_id"],
            style_contract_sha256=contract["contract_sha256"],
            prompt_sha256=prompt_sha256,
        )

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
            }
        )
    return rows


def _validated_card_by_id(cards: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not isinstance(cards, list):
        raise ValueError("chapter execution cards must be a list")

    card_by_id: dict[str, dict[str, Any]] = {}
    for index, card in enumerate(cards, start=1):
        try:
            validated_card = validate_chapter_execution_card(card)
        except ValueError as exc:
            raise ValueError(f"card {index}: {exc}") from exc

        card_id = validated_card["card_id"]
        if not is_card_approved_for_formal_sft(validated_card):
            raise ValueError(f"formal card status must be approved or frozen: {card_id}")
        if card_id in card_by_id:
            raise ValueError(f"duplicate formal card id: {card_id}")
        card_by_id[card_id] = validated_card
    return card_by_id
