"""Batch integrity checks for formal ChapterExecutionCard assets."""

from __future__ import annotations

from typing import Any

from small_model_train.cards.card_renderer import render_chapter_execution_input
from small_model_train.schemas.chapter_execution_card import (
    FORMAL_CARD_STATUSES,
    text_sha256,
    validate_chapter_execution_card,
)
from small_model_train.style_contract import validate_style_contract_asset
from small_model_train.text_utils import count_chinese_chars


LEAK_MIN_CHARS = 12
FUTURE_CONTEXT_SPLITS = {"validation", "sealed", "eval"}


def validate_formal_card_batch(
    cards: list[dict[str, Any]],
    chapters: list[dict[str, Any]],
    style_contract: dict[str, Any],
    *,
    require_all_train_chapters: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    card_by_chapter_id: dict[str, dict[str, Any]] = {}

    try:
        contract = validate_style_contract_asset(style_contract)
    except ValueError as exc:
        return {
            "passed": False,
            "errors": [f"style contract invalid: {exc}"],
            "card_by_chapter_id": card_by_chapter_id,
        }

    chapter_by_id = _chapter_by_id(chapters)
    required_train_chapter_ids = {
        str(chapter.get("id"))
        for chapter in chapters
        if chapter.get("split") == "train"
        and chapter.get("quality_tag") == "A"
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
        contract_matches = True
        if card["style_contract_id"] != contract["style_contract_id"]:
            errors.append(f"style_contract_id mismatch: {card['card_id']}")
            contract_matches = False
        if card["style_contract_sha256"] != contract["contract_sha256"]:
            errors.append(f"style_contract_sha256 mismatch: {card['card_id']}")
            contract_matches = False

        chapter = chapter_by_id.get(chapter_id)
        if chapter is None:
            errors.append(f"formal card points to missing chapter: {chapter_id}")
            continue

        expected_source_hash = text_sha256(str(chapter.get("text") or ""))
        if card["source_chapter_sha256"] != expected_source_hash:
            errors.append(f"source_chapter_sha256 mismatch: {chapter_id}")

        if chapter_id in card_by_chapter_id:
            errors.append(
                "duplicate formal cards for chapter "
                f"{chapter_id}: {card_by_chapter_id[chapter_id]['card_id']}, {card['card_id']}"
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
        errors.append(f"target-text leakage: {card['card_id']}: {target_fragment}")

    target_chapter_id = chapter.get("id")
    for other in chapters:
        if not isinstance(other, dict):
            continue
        if other.get("id") == target_chapter_id:
            continue
        if other.get("split") not in FUTURE_CONTEXT_SPLITS:
            continue
        future_fragment = _find_leakage_fragment(rendered, str(other.get("text") or ""))
        if future_fragment:
            errors.append(
                "future-context leakage: "
                f"{card['card_id']}: {other.get('id')}: {future_fragment}"
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
