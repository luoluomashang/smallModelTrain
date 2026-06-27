"""Batch integrity checks for formal ChapterExecutionCard assets."""

from __future__ import annotations

from typing import Any

from small_model_train.cards.card_renderer import render_chapter_execution_input
from small_model_train.schemas.chapter_execution_card import (
    FORMAL_CARD_STATUSES,
    text_sha256,
    validate_chapter_execution_card,
)
from small_model_train.style_contract import (
    APPROVED_FORMAL_STATUSES,
    validate_style_contract_asset,
)
from small_model_train.text_utils import count_chinese_chars


LEAK_MIN_CHARS = 12
FUTURE_CONTEXT_SPLITS = {"validation", "sealed", "eval"}


def _is_trainable_chapter(chapter: dict[str, Any]) -> bool:
    return chapter.get("split") == "train" and chapter.get("quality_tag") == "A"


def validate_formal_card_batch(
    cards: list[dict[str, Any]],
    chapters: list[dict[str, Any]],
    style_contract: dict[str, Any],
    *,
    require_all_train_chapters: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    card_by_chapter_id: dict[str, dict[str, Any]] = {}
    card_by_card_id: dict[str, dict[str, Any]] = {}

    try:
        contract = validate_style_contract_asset(style_contract)
    except ValueError as exc:
        return {
            "passed": False,
            "errors": [f"style contract invalid: {exc}"],
            "card_by_chapter_id": card_by_chapter_id,
        }

    if contract["approval_status"] not in APPROVED_FORMAL_STATUSES:
        errors.append(
            "style contract must be approved or frozen for formal batch: "
            f"{contract['style_contract_id']} status={contract['approval_status']}"
        )

    errors.extend(_duplicate_trainable_chapter_id_errors(chapters))
    chapter_by_id = _chapter_by_id(chapters)
    required_train_chapter_ids = {
        str(chapter.get("id"))
        for chapter in chapters
        if isinstance(chapter, dict)
        and _is_trainable_chapter(chapter)
        and chapter.get("id") is not None
    }

    for index, raw_card in enumerate(cards, start=1):
        try:
            card = validate_chapter_execution_card(raw_card)
        except ValueError as exc:
            errors.append(f"card {index}: {exc}")
            continue

        if card["card_status"] not in FORMAL_CARD_STATUSES:
            errors.append(f"formal card must be approved or frozen: {card['card_id']}")
            continue

        chapter_id = card["chapter_id"]
        card_id = card["card_id"]
        contract_matches = True
        if card["style_contract_id"] != contract["style_contract_id"]:
            errors.append(f"style_contract_id mismatch: {card_id} chapter {chapter_id}")
            contract_matches = False
        if card["style_contract_sha256"] != contract["contract_sha256"]:
            errors.append(f"style_contract_sha256 mismatch: {card_id} chapter {chapter_id}")
            contract_matches = False

        chapter = chapter_by_id.get(chapter_id)
        if chapter is None:
            errors.append(f"formal card points to missing chapter: {chapter_id} card {card_id}")
            continue

        expected_source_hash = text_sha256(str(chapter.get("text") or ""))
        if card["source_chapter_sha256"] != expected_source_hash:
            errors.append(f"source_chapter_sha256 mismatch: {chapter_id} card {card_id}")

        if card_id in card_by_card_id:
            previous_card = card_by_card_id[card_id]
            errors.append(
                "duplicate formal card id: "
                f"{card_id} chapters {previous_card['chapter_id']}, {chapter_id}"
            )
        else:
            card_by_card_id[card_id] = card

        if chapter_id in card_by_chapter_id:
            errors.append(
                "duplicate formal cards for chapter "
                f"{chapter_id}: {card_by_chapter_id[chapter_id]['card_id']}, {card_id}"
            )
        else:
            card_by_chapter_id[chapter_id] = card

        if contract_matches:
            errors.extend(_leakage_errors(card, chapter, chapters, contract))

    if require_all_train_chapters:
        for chapter_id in sorted(required_train_chapter_ids - set(card_by_chapter_id)):
            errors.append(f"missing formal card for train chapter: {chapter_id}")

    return {
        "passed": not errors,
        "errors": errors,
        "card_by_chapter_id": card_by_chapter_id,
    }


def _chapter_by_id(chapters: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(chapter["id"]): chapter
        for chapter in chapters
        if isinstance(chapter, dict) and chapter.get("id") is not None
    }


def _duplicate_trainable_chapter_id_errors(chapters: list[dict[str, Any]]) -> list[str]:
    first_index_by_id: dict[str, int] = {}
    errors: list[str] = []
    for index, chapter in enumerate(chapters, start=1):
        if not isinstance(chapter, dict) or not _is_trainable_chapter(chapter):
            continue
        if chapter.get("id") is None:
            continue
        chapter_id = str(chapter["id"])
        previous_index = first_index_by_id.get(chapter_id)
        if previous_index is None:
            first_index_by_id[chapter_id] = index
        else:
            errors.append(
                "duplicate trainable chapter id: "
                f"{chapter_id} rows {previous_index}, {index}"
            )
    return errors


def _leakage_errors(
    card: dict[str, Any],
    chapter: dict[str, Any],
    chapters: list[dict[str, Any]],
    style_contract: dict[str, Any],
) -> list[str]:
    rendered = render_chapter_execution_input(card, style_contract)
    errors: list[str] = []

    target_fragment = _find_leakage_fragment(rendered, str(chapter.get("text") or ""))
    if target_fragment:
        errors.append(
            f"target-text leakage: {card['card_id']} "
            f"chapter {chapter.get('id')}: {target_fragment}"
        )

    target_chapter_id = chapter.get("id")
    target_chapter_index = _chapter_index(chapters, chapter)
    for other_index, other in enumerate(chapters):
        if not isinstance(other, dict):
            continue
        if other.get("id") == target_chapter_id:
            continue
        if (
            other.get("split") not in FUTURE_CONTEXT_SPLITS
            and not _is_later_chapter(other_index, target_chapter_index)
        ):
            continue
        future_fragment = _find_leakage_fragment(rendered, str(other.get("text") or ""))
        if future_fragment:
            errors.append(
                "future-context leakage: "
                f"{card['card_id']} chapter {target_chapter_id}: "
                f"source chapter {other.get('id')}: {future_fragment}"
            )

    return errors


def _find_leakage_fragment(rendered_input: str, source_text: str) -> str | None:
    if count_chinese_chars(source_text) < LEAK_MIN_CHARS:
        return None

    for run in _chinese_runs(source_text):
        if len(run) < LEAK_MIN_CHARS:
            continue
        for start in range(0, len(run) - LEAK_MIN_CHARS + 1):
            fragment = run[start : start + LEAK_MIN_CHARS]
            if fragment in rendered_input:
                return fragment
    return None


def _chapter_index(chapters: list[dict[str, Any]], chapter: dict[str, Any]) -> int | None:
    for index, candidate in enumerate(chapters):
        if candidate is chapter:
            return index
    return None


def _is_later_chapter(other_index: int, target_chapter_index: int | None) -> bool:
    return target_chapter_index is not None and other_index > target_chapter_index


def _chinese_runs(text: str) -> list[str]:
    runs: list[str] = []
    current: list[str] = []
    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            current.append(char)
        elif current:
            runs.append("".join(current))
            current = []
    if current:
        runs.append("".join(current))
    return runs
