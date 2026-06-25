"""Compile draft chapter cards into formal ChapterExecutionCard assets."""

from __future__ import annotations

from typing import Any

from small_model_train.execution_cards import DEFAULT_TARGET_PLATFORM
from small_model_train.schemas.chapter_execution_card import (
    build_chapter_execution_card,
    utc_now_iso,
)
from small_model_train.style_contract import validate_style_contract_asset


COMPILER_VERSION = "stage5c_v1"
ABSTRACT_ONLY_GOALS = ("节奏紧凑", "写得爽一点", "减少 AI 味")
FORMAL_APPROVAL_STATUSES = {"approved", "frozen"}


def compile_chapter_execution_card(
    draft_card: dict[str, Any],
    chapter: dict[str, Any],
    style_contract: dict[str, Any],
    group_id: str,
    split: str,
    card_status: str = "reviewed",
) -> dict[str, Any]:
    validated_contract = validate_style_contract_asset(style_contract)
    if card_status in FORMAL_APPROVAL_STATUSES:
        raise ValueError("compiler must not auto-approve chapter execution cards")
    _require_matching_chapter(draft_card, chapter)
    _reject_abstract_only_card(draft_card)

    chapter_id = _required_string(chapter, "id")
    draft_id = _required_string(draft_card, "id")
    source_chapter_text = _required_string(chapter, "text")

    return build_chapter_execution_card(
        card_id=f"card-{chapter_id}-v1",
        chapter_id=chapter_id,
        card_status=card_status,
        style_contract_id=validated_contract["style_contract_id"],
        style_contract_sha256=validated_contract["contract_sha256"],
        source_chapter_text=source_chapter_text,
        target_platform=str(draft_card.get("target_platform") or DEFAULT_TARGET_PLATFORM),
        genre_tags=_string_list_or_default(draft_card.get("genre_tags"), ["male_webnovel"]),
        hard_constraints={
            "must_include": _string_list(draft_card.get("must_include")),
            "must_not_include": _string_list(draft_card.get("must_not_include")),
            "continuity_facts": _continuity_facts(draft_card),
            "forbidden_future_facts": _string_list(draft_card.get("forbidden_future_facts")),
            "style_bans": _style_bans(draft_card, validated_contract),
        },
        execution_plan={
            "chapter_goal": _required_string(draft_card, "chapter_goal"),
            "conflict_beat": _required_string(draft_card, "conflict_beat"),
            "payoff_beat": _required_string(draft_card, "payoff_beat"),
            "chapter_structure": list(draft_card.get("chapter_structure") or []),
            "character_states": list(draft_card.get("character_states") or []),
            "ending_hook": _required_string(draft_card, "ending_hook"),
            "target_word_count": _required_string(draft_card, "target_word_count"),
        },
        creative_space=_creative_space(draft_card),
        provenance={
            "source_card_id": draft_id,
            "compiler_version": COMPILER_VERSION,
            "created_at": utc_now_iso(),
            "reviewer": "",
            "reviewed_at": "",
            "review_notes": "",
            "group_id": group_id,
            "split": split,
        },
    )


def _require_matching_chapter(draft_card: dict[str, Any], chapter: dict[str, Any]) -> None:
    draft_id = _required_string(draft_card, "id")
    chapter_id = _required_string(chapter, "id")
    if draft_id != chapter_id:
        raise ValueError(f"draft card id {draft_id!r} does not match chapter id {chapter_id!r}")


def _reject_abstract_only_card(draft_card: dict[str, Any]) -> None:
    goal = str(draft_card.get("chapter_goal") or "")
    has_only_abstract_goal = bool(goal.strip()) and _only_abstract_advice(goal)
    has_concrete_execution = any(
        [
            _non_empty_string(draft_card.get("conflict_beat")),
            _non_empty_string(draft_card.get("payoff_beat")),
            bool(_string_list(draft_card.get("must_include"))),
            _non_empty_string(draft_card.get("ending_hook")),
        ]
    )
    if has_only_abstract_goal and not has_concrete_execution:
        raise ValueError("abstract-only chapter execution card")


def _only_abstract_advice(goal: str) -> bool:
    normalized_terms = {_normalize_text(term) for term in ABSTRACT_ONLY_GOALS}
    phrases = [
        _normalize_text(phrase)
        for phrase in goal.replace("，", ",").replace("。", ",").split(",")
        if _normalize_text(phrase)
    ]
    return bool(phrases) and all(phrase in normalized_terms for phrase in phrases)


def _normalize_text(value: str) -> str:
    return value.replace(" ", "").replace("AI", "ai").strip()


def _required_string(values: dict[str, Any], field: str) -> str:
    value = values.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("expected a list of strings")
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _string_list_or_default(value: Any, default: list[str]) -> list[str]:
    values = _string_list(value)
    return values or list(default)


def _continuity_facts(draft_card: dict[str, Any]) -> list[str]:
    values = _string_list(draft_card.get("continuity_facts"))
    previous_summary = draft_card.get("previous_summary")
    if isinstance(previous_summary, str) and previous_summary.strip():
        values.insert(0, previous_summary.strip())
    return values


def _style_bans(draft_card: dict[str, Any], style_contract: dict[str, Any]) -> list[str]:
    bans = _string_list(draft_card.get("style_bans"))
    bans.extend(_string_list(style_contract["ai_taste_guardrails"].get("banned_phrases")))
    return list(dict.fromkeys(bans))


def _creative_space(draft_card: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "optional_sensory_details": _string_list_or_default(
            draft_card.get("optional_sensory_details"),
            ["按场景补足气味、声响、光线等具体感官细节"],
        ),
        "optional_dialogue_moves": _string_list_or_default(
            draft_card.get("optional_dialogue_moves"),
            ["用短对白、停顿和反问推进试探"],
        ),
        "optional_micro_conflicts": _string_list_or_default(
            draft_card.get("optional_micro_conflicts"),
            ["允许加入不改变主线的小阻碍"],
        ),
        "allowed_scene_expansion": _string_list_or_default(
            draft_card.get("allowed_scene_expansion"),
            ["允许扩写动作过程和场景反应，但不得改变硬约束"],
        ),
    }
