from __future__ import annotations

from typing import Any


DEFAULT_TARGET_PLATFORM = "hybrid_fanqie_qidian"
VALID_TARGET_PLATFORMS = {"fanqie", "qidian", DEFAULT_TARGET_PLATFORM}
RUBRIC_VERSION = "male_webnovel_v1"

REQUIRED_EXECUTION_FIELDS = (
    "id",
    "target_platform",
    "genre_tags",
    "style_contract",
    "chapter_goal",
    "chapter_structure",
    "conflict_beat",
    "payoff_beat",
    "must_include",
    "must_not_include",
    "ending_hook",
    "target_word_count",
)


def validate_execution_card(card: dict[str, Any]) -> dict[str, Any]:
    missing = [
        field
        for field in REQUIRED_EXECUTION_FIELDS
        if field not in card or card.get(field) in (None, "")
    ]
    if missing:
        raise ValueError(
            "missing execution-card fields: " + ", ".join(sorted(missing))
        )

    target_platform = card.get("target_platform")
    if target_platform not in VALID_TARGET_PLATFORMS:
        raise ValueError(f"unknown target_platform: {target_platform}")

    genre_tags = card.get("genre_tags")
    if not isinstance(genre_tags, list) or not genre_tags:
        raise ValueError("genre_tags must be a non-empty list")
    if not all(isinstance(tag, str) and tag.strip() for tag in genre_tags):
        raise ValueError("genre_tags must contain non-empty strings")

    chapter_structure = card.get("chapter_structure")
    if not isinstance(chapter_structure, list) or not chapter_structure:
        raise ValueError("chapter_structure must be a non-empty list")
    for item in chapter_structure:
        if not isinstance(item, dict):
            raise ValueError("chapter_structure items must be dicts")
        step = item.get("step")
        if not isinstance(step, int) or step <= 0:
            raise ValueError("chapter_structure step must be a positive integer")
        for field in ("name", "goal", "estimated_chars"):
            value = item.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"chapter_structure {field} must be a non-empty string")

    for field in ("must_include", "must_not_include"):
        values = card.get(field)
        if not isinstance(values, list):
            raise ValueError(f"{field} must be a list")
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise ValueError(f"{field} must contain non-empty strings")

    return card


def validate_execution_cards(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for index, row in enumerate(rows, start=1):
        try:
            validate_execution_card(row)
        except ValueError as exc:
            raise ValueError(f"row {index}: {exc}") from exc
    return rows
