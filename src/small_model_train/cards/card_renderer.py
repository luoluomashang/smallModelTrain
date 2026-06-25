"""Render formal ChapterExecutionCard assets into model prompts."""

from __future__ import annotations

from typing import Any

from small_model_train.prompt_renderer import render_execution_input
from small_model_train.schemas.chapter_execution_card import validate_chapter_execution_card
from small_model_train.style_contract import validate_style_contract_asset


def formal_card_to_prompt_card(card: dict[str, Any], style_contract: dict[str, Any]) -> dict[str, Any]:
    validated_card = validate_chapter_execution_card(card)
    validated_contract = validate_style_contract_asset(style_contract)
    _require_matching_style_contract(validated_card, validated_contract)

    execution_plan = validated_card["execution_plan"]
    hard_constraints = validated_card["hard_constraints"]
    return {
        "style_contract": validated_contract["prompt_rules"]["style_contract_text"],
        "previous_summary": "\n".join(hard_constraints["continuity_facts"]),
        "chapter_goal": execution_plan["chapter_goal"],
        "chapter_structure": execution_plan["chapter_structure"],
        "character_states": execution_plan["character_states"],
        "conflict_beat": execution_plan["conflict_beat"],
        "payoff_beat": execution_plan["payoff_beat"],
        "must_include": hard_constraints["must_include"],
        "must_not_include": hard_constraints["must_not_include"],
        "ending_hook": execution_plan["ending_hook"],
        "target_word_count": execution_plan["target_word_count"],
    }


def render_chapter_execution_input(card: dict[str, Any], style_contract: dict[str, Any]) -> str:
    validated_card = validate_chapter_execution_card(card)
    prompt_card = formal_card_to_prompt_card(validated_card, style_contract)
    return "\n".join(
        [
            render_execution_input(prompt_card),
            _format_creative_space(validated_card["creative_space"]),
        ]
    )


def _require_matching_style_contract(
    card: dict[str, Any], style_contract: dict[str, Any]
) -> None:
    if card["style_contract_id"] != style_contract["style_contract_id"]:
        raise ValueError("style_contract_id mismatch")
    if card["style_contract_sha256"] != style_contract["contract_sha256"]:
        raise ValueError("style_contract_sha256 mismatch")


def _format_creative_space(creative_space: dict[str, list[str]]) -> str:
    lines = ["【创作自由】"]
    for title, field in (
        ("可选感官细节", "optional_sensory_details"),
        ("可选对白动作", "optional_dialogue_moves"),
        ("可选微冲突", "optional_micro_conflicts"),
        ("允许扩写", "allowed_scene_expansion"),
    ):
        values = creative_space[field]
        if values:
            lines.append(f"- {title}：" + "；".join(values))
    return "\n".join(lines)
