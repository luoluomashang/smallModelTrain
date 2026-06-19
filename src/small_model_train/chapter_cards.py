"""Chapter card generation and schema validation."""

from __future__ import annotations

from typing import Any


STYLE_CONTRACT = (
    "只输出正文。以动作、场景反应和短对白推进章节，不输出提纲、小标题、分析或创作说明。"
    "保持节奏清晰，承接上一章压力，在本章结尾留下自然余波或下一步悬念。"
)

STRUCTURE_TEMPLATE = [
    ("承接", "承接上一章留下的压力，交代本章开场状态。", 0.15),
    ("入局", "让核心视角人物进入新的场景、关系变化或任务压力。", 0.20),
    ("试探", "通过行动和短对白推进矛盾，不急于解释设定。", 0.20),
    ("加压", "让局势出现阻碍、误判、代价或反向压力。", 0.20),
    ("转折", "完成本章最关键的信息变化、人物选择或关系推进。", 0.15),
    ("收束", "用余波、未完成动作或新压力把读者带向下一章。", 0.10),
]

REQUIRED_CARD_FIELDS = (
    "id",
    "style_contract",
    "previous_summary",
    "chapter_goal",
    "chapter_structure",
    "character_states",
    "must_include",
    "must_not_include",
    "ending_hook",
    "target_word_count",
)


def normalize_chapter_structure(items: list[dict[str, Any]]) -> list[dict[str, str | int]]:
    normalized = []
    for index, item in enumerate(items, start=1):
        normalized.append(
            {
                "step": int(item.get("step") or index),
                "name": str(item.get("name") or item.get("beat") or "").strip(),
                "goal": str(item.get("goal") or "").strip(),
                "estimated_chars": str(item.get("estimated_chars") or "").strip(),
            }
        )
    return normalized


def validate_chapter_card(card: dict[str, Any]) -> None:
    for field in REQUIRED_CARD_FIELDS:
        if field not in card:
            raise ValueError(f"missing required field: {field}")
        if card[field] in ("", None, []):
            raise ValueError(f"empty required field: {field}")

    for index, item in enumerate(card["chapter_structure"]):
        if not item.get("step"):
            raise ValueError(f"chapter_structure[{index}].step is required")
        if not item.get("name"):
            raise ValueError(f"chapter_structure[{index}].name is required")
        if not item.get("goal"):
            raise ValueError(f"chapter_structure[{index}].goal is required")
        if not item.get("estimated_chars"):
            raise ValueError(f"chapter_structure[{index}].estimated_chars is required")

    for index, item in enumerate(card["character_states"]):
        if not item.get("name"):
            raise ValueError(f"character_states[{index}].name is required")
        if not item.get("state"):
            raise ValueError(f"character_states[{index}].state is required")
        if not item.get("speech_style"):
            raise ValueError(f"character_states[{index}].speech_style is required")


def build_draft_chapter_cards(
    chapters: list[dict[str, Any]],
    count: int,
    min_chars: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    candidates = [
        chapter
        for chapter in chapters
        if chapter.get("split") == "train"
        and chapter.get("quality_tag") == "A"
        and min_chars <= int(chapter.get("char_count_zh", 0)) <= max_chars
    ]
    cards = [_build_card(chapter) for chapter in candidates[:count]]
    for card in cards:
        validate_chapter_card(card)
    return cards


def _build_card(chapter: dict[str, Any]) -> dict[str, Any]:
    char_count = int(chapter.get("char_count_zh", 0))
    card = {
        "id": chapter["id"],
        "card_version": "draft-v2",
        "source_title": chapter.get("chapter_title", ""),
        "style_contract": STYLE_CONTRACT,
        "previous_summary": "上一章事件留下新的压力，本章从既有关系、目标和未解决冲突继续推进。",
        "chapter_goal": "围绕本章既定剧情推进主要冲突，让核心人物在观察、试探、行动和选择中完成阶段性变化。",
        "chapter_structure": _structure_for(char_count),
        "character_states": [
            {
                "name": "核心视角人物",
                "state": "带着上一章留下的压力进入本章，在观察、试探和行动中推进目标。",
                "speech_style": "短句优先，少解释，多用反应、动作和停顿承接情绪。",
            },
            {
                "name": "关键对手或阻力方",
                "state": "制造误判、压力或选择代价，迫使核心人物调整行动。",
                "speech_style": "保持信息克制，避免直接替作者解释设定。",
            },
        ],
        "must_include": ["清楚的开场状态", "可感知的中段压力升级", "章末余波或下一步悬念"],
        "must_not_include": ["输出提纲或小标题", "跳出正文解释创作意图", "大段复述世界观设定", "直接照抄原文句段"],
        "ending_hook": "在本章余波中留下下一章继续推进的压力、疑问或动作方向。",
        "target_word_count": _target_word_count(char_count),
        "source_text": chapter.get("text", ""),
    }
    card["chapter_structure"] = normalize_chapter_structure(card["chapter_structure"])
    return card


def _structure_for(char_count: int) -> list[dict[str, str | int]]:
    return [
        {
            "step": index,
            "name": name,
            "goal": goal,
            "estimated_chars": str(int(round(char_count * ratio / 50) * 50)),
        }
        for index, (name, goal, ratio) in enumerate(STRUCTURE_TEMPLATE, start=1)
    ]


def _target_word_count(char_count: int) -> str:
    if char_count <= 2500:
        return "2000-2500中文汉字"
    if char_count <= 3000:
        return "2500-3000中文汉字"
    return "3000-4000中文汉字"
