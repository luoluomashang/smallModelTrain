"""ChapterExecutionCard schema helpers for formal SFT assets."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
CARD_STATUSES = {"draft", "reviewed", "approved", "frozen", "rejected"}
FORMAL_CARD_STATUSES = {"approved", "frozen"}

LOWER_HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TOP_LEVEL_FIELDS = (
    "schema_version",
    "card_id",
    "chapter_id",
    "card_status",
    "style_contract_id",
    "style_contract_sha256",
    "source_chapter_sha256",
    "card_sha256",
    "target_platform",
    "genre_tags",
    "hard_constraints",
    "execution_plan",
    "creative_space",
    "provenance",
)
HARD_CONSTRAINT_FIELDS = (
    "must_include",
    "must_not_include",
    "continuity_facts",
    "forbidden_future_facts",
    "style_bans",
)
EXECUTION_PLAN_FIELDS = (
    "chapter_goal",
    "conflict_beat",
    "payoff_beat",
    "chapter_structure",
    "character_states",
    "ending_hook",
    "target_word_count",
)
CREATIVE_SPACE_FIELDS = (
    "optional_sensory_details",
    "optional_dialogue_moves",
    "optional_micro_conflicts",
    "allowed_scene_expansion",
)
PROVENANCE_FIELDS = (
    "source_card_id",
    "compiler_version",
    "created_at",
    "reviewer",
    "reviewed_at",
    "review_notes",
    "group_id",
    "split",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def text_sha256(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_card_sha256(card: dict[str, Any]) -> str:
    canonical_card = copy.deepcopy(card)
    canonical_card.pop("card_sha256", None)
    payload = json.dumps(
        canonical_card,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_chapter_execution_card(
    *,
    card_id: str,
    chapter_id: str,
    card_status: str,
    style_contract_id: str,
    style_contract_sha256: str,
    source_chapter_text: str,
    target_platform: str,
    genre_tags: list[str],
    hard_constraints: dict[str, Any],
    execution_plan: dict[str, Any],
    creative_space: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    card: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "card_id": card_id,
        "chapter_id": chapter_id,
        "card_status": card_status,
        "style_contract_id": style_contract_id,
        "style_contract_sha256": style_contract_sha256,
        "source_chapter_sha256": text_sha256(source_chapter_text),
        "card_sha256": "",
        "target_platform": target_platform,
        "genre_tags": copy.deepcopy(genre_tags),
        "hard_constraints": copy.deepcopy(hard_constraints),
        "execution_plan": copy.deepcopy(execution_plan),
        "creative_space": copy.deepcopy(creative_space),
        "provenance": copy.deepcopy(provenance),
    }
    card["card_sha256"] = canonical_card_sha256(card)
    return validate_chapter_execution_card(card)


def validate_chapter_execution_card(card: Any) -> dict[str, Any]:
    if not isinstance(card, dict):
        raise ValueError("chapter execution card must be a JSON object")

    missing = [field for field in TOP_LEVEL_FIELDS if field not in card]
    if missing:
        raise ValueError(f"{missing[0]} is required")

    if card["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    for field in (
        "card_id",
        "chapter_id",
        "style_contract_id",
        "target_platform",
    ):
        _require_non_empty_string("", card, field)

    if card["card_status"] not in CARD_STATUSES:
        raise ValueError(
            "card_status must be one of: " + ", ".join(sorted(CARD_STATUSES))
        )

    for field in ("style_contract_sha256", "source_chapter_sha256", "card_sha256"):
        if not _is_lower_hex_sha256(card[field]):
            raise ValueError(f"{field} must be a 64-character lowercase hex string")

    _validate_genre_tags(card["genre_tags"])

    for field in ("hard_constraints", "execution_plan", "creative_space", "provenance"):
        if not isinstance(card[field], dict):
            raise ValueError(f"{field} must be a JSON object")

    _validate_hard_constraints(card["hard_constraints"])
    _validate_execution_plan(card["execution_plan"])
    _validate_creative_space(card["creative_space"])
    _validate_provenance(card["provenance"])

    expected_sha256 = canonical_card_sha256(card)
    if card["card_sha256"] != expected_sha256:
        raise ValueError("card_sha256 mismatch")

    return card


def validate_chapter_execution_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(cards, list):
        raise ValueError("chapter execution cards must be a list")
    for index, card in enumerate(cards, start=1):
        try:
            validate_chapter_execution_card(card)
        except ValueError as exc:
            raise ValueError(f"card {index}: {exc}") from exc
    return cards


def is_card_approved_for_formal_sft(card: dict[str, Any]) -> bool:
    validated = validate_chapter_execution_card(card)
    return validated["card_status"] in FORMAL_CARD_STATUSES


def read_chapter_execution_cards(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    file_path = Path(path)
    if not file_path.exists():
        raise ValueError(f"chapter execution cards JSONL not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{file_path}:{line_number} is not valid JSON") from exc
            try:
                rows.append(validate_chapter_execution_card(row))
            except ValueError as exc:
                raise ValueError(f"{file_path}:{line_number}: {exc}") from exc
    return rows


def write_chapter_execution_cards(path: str | Path, cards: list[dict[str, Any]]) -> None:
    validated_cards = validate_chapter_execution_cards(cards)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for card in validated_cards:
            handle.write(json.dumps(_ordered_card(card), ensure_ascii=False) + "\n")


def _is_lower_hex_sha256(value: object) -> bool:
    return isinstance(value, str) and LOWER_HEX_SHA256_RE.fullmatch(value) is not None


def _require_fields(section_name: str, section: dict[str, Any], fields: tuple[str, ...]) -> None:
    missing = [field for field in fields if field not in section]
    if missing:
        raise ValueError(f"{section_name}.{missing[0]} is required")


def _require_non_empty_string(section_name: str, section: dict[str, Any], field: str) -> None:
    value = section.get(field)
    if not isinstance(value, str) or not value.strip():
        prefix = f"{section_name}." if section_name else ""
        raise ValueError(f"{prefix}{field} must be a non-empty string")


def _validate_string_list(section_name: str, section: dict[str, Any], field: str) -> None:
    values = section.get(field)
    if not isinstance(values, list):
        raise ValueError(f"{section_name}.{field} must be a list")
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise ValueError(f"{section_name}.{field} must contain non-empty strings")


def _validate_genre_tags(genre_tags: object) -> None:
    if not isinstance(genre_tags, list) or not genre_tags:
        raise ValueError("genre_tags must be a non-empty list")
    if not all(isinstance(tag, str) and tag.strip() for tag in genre_tags):
        raise ValueError("genre_tags must contain non-empty strings")


def _validate_hard_constraints(hard_constraints: dict[str, Any]) -> None:
    _require_fields("hard_constraints", hard_constraints, HARD_CONSTRAINT_FIELDS)
    for field in HARD_CONSTRAINT_FIELDS:
        _validate_string_list("hard_constraints", hard_constraints, field)


def _validate_execution_plan(execution_plan: dict[str, Any]) -> None:
    _require_fields("execution_plan", execution_plan, EXECUTION_PLAN_FIELDS)
    for field in (
        "chapter_goal",
        "conflict_beat",
        "payoff_beat",
        "ending_hook",
        "target_word_count",
    ):
        _require_non_empty_string("execution_plan", execution_plan, field)
    _validate_chapter_structure(execution_plan["chapter_structure"])
    _validate_character_states(execution_plan["character_states"])


def _validate_chapter_structure(chapter_structure: object) -> None:
    if not isinstance(chapter_structure, list) or not chapter_structure:
        raise ValueError("execution_plan.chapter_structure must be a non-empty list")
    for index, item in enumerate(chapter_structure, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"execution_plan.chapter_structure[{index}] must be a JSON object")
        _require_fields(
            "execution_plan.chapter_structure",
            item,
            ("step", "name", "goal", "estimated_chars"),
        )
        step = item["step"]
        if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
            raise ValueError("execution_plan.chapter_structure.step must be a positive integer")
        for field in ("name", "goal", "estimated_chars"):
            _require_non_empty_string("execution_plan.chapter_structure", item, field)


def _validate_character_states(character_states: object) -> None:
    if not isinstance(character_states, list) or not character_states:
        raise ValueError("execution_plan.character_states must be a non-empty list")
    for index, item in enumerate(character_states, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"execution_plan.character_states[{index}] must be a JSON object")
        _require_fields(
            "execution_plan.character_states",
            item,
            ("name", "state", "speech_style"),
        )
        for field in ("name", "state", "speech_style"):
            _require_non_empty_string("execution_plan.character_states", item, field)


def _validate_creative_space(creative_space: dict[str, Any]) -> None:
    _require_fields("creative_space", creative_space, CREATIVE_SPACE_FIELDS)
    for field in CREATIVE_SPACE_FIELDS:
        _validate_string_list("creative_space", creative_space, field)


def _validate_provenance(provenance: dict[str, Any]) -> None:
    _require_fields("provenance", provenance, PROVENANCE_FIELDS)
    for field in PROVENANCE_FIELDS:
        if not isinstance(provenance[field], str):
            raise ValueError(f"provenance.{field} must be a string")


def _ordered_card(card: dict[str, Any]) -> dict[str, Any]:
    ordered: dict[str, Any] = {}
    for field in TOP_LEVEL_FIELDS:
        if field in card:
            ordered[field] = card[field]
    for field in sorted(key for key in card if key not in ordered):
        ordered[field] = card[field]
    return ordered
