"""SFT prompt construction from chapter cards and cleaned chapters.

The builder joins explicit planning cards with target chapter text. It also
guards against copying source_text into the prompt, because that would turn
training into answer leakage rather than instruction following.
"""

from __future__ import annotations

import re

from small_model_train.prompt_renderer import SYSTEM_PROMPT, render_execution_input


INSTRUCTION = SYSTEM_PROMPT
SOURCE_LEAK_MIN_CHARS = 12
SOURCE_LEAK_ERROR_PREFIX = "SFT input contains source_text fragment"
APPROVED_CARD_STATUSES = {"approved", "frozen"}


def is_source_text_leak_error(error: ValueError) -> bool:
    return str(error).startswith(SOURCE_LEAK_ERROR_PREFIX)


def _find_source_text_leak(rendered_input: str, source_text: str, min_chars: int = SOURCE_LEAK_MIN_CHARS) -> str | None:
    # This check catches long verbatim spans before SFT rows are written, preventing answer leakage into prompts.
    if not source_text:
        return None
    for match in re.finditer(r"[\u4e00-\u9fff]+", source_text):
        chinese_run = match.group(0)
        if len(chinese_run) < min_chars:
            continue
        for start in range(0, len(chinese_run) - min_chars + 1):
            fragment = chinese_run[start : start + min_chars]
            if fragment in rendered_input:
                return fragment
    return None


def render_sft_input(card: dict) -> str:
    rendered_input = render_execution_input(card)
    leak = _find_source_text_leak(rendered_input, card.get("source_text", ""))
    if leak:
        raise ValueError(f"{SOURCE_LEAK_ERROR_PREFIX}: {leak}")
    return rendered_input


def _is_trainable_chapter(chapter: dict) -> bool:
    return chapter.get("split") == "train" and chapter.get("quality_tag") == "A"


def _require_approved_card(card: dict) -> None:
    card_id = card.get("id", "<missing id>")
    if card.get("draft_only") is True:
        raise ValueError(f"draft card cannot enter formal SFT: {card_id}")
    if card.get("approval_status") not in APPROVED_CARD_STATUSES:
        raise ValueError(f"approval_status must be approved or frozen for formal SFT: {card_id}")
    if not card.get("style_contract_id"):
        raise ValueError(f"style_contract_id is required for formal SFT: {card_id}")
    style_contract_sha256 = card.get("style_contract_sha256")
    if not isinstance(style_contract_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", style_contract_sha256):
        raise ValueError(f"style_contract_sha256 must be a 64-character hex digest for formal SFT: {card_id}")


def build_sft_rows(cards: list[dict], chapters: list[dict], require_approved_cards: bool = False) -> list[dict]:
    chapter_by_id = {chapter["id"]: chapter for chapter in chapters}
    rows: list[dict] = []
    for card in cards:
        chapter = chapter_by_id.get(card["id"])
        # Non-train rows are skipped deliberately; eval rows must stay unseen so later adapter scores mean something.
        if not chapter or not _is_trainable_chapter(chapter):
            continue
        if require_approved_cards:
            _require_approved_card(card)
        rows.append(
            {
                "instruction": INSTRUCTION,
                "input": render_sft_input(card),
                "output": chapter.get("text", ""),
            }
        )
    return rows
