"""Shared prompt rendering for SFT construction and stage 2 inference."""

from __future__ import annotations

import hashlib
from typing import Any


SYSTEM_PROMPT = "你是作者的正文执行器。严格执行章节卡，并保持指定作者风格。"


def _format_list(title: str, values: list[str]) -> str:
    body = "\n".join(f"- {value}" for value in values) if values else "- 无"
    return f"【{title}】\n{body}"


def _format_structure(items: list[dict[str, Any]]) -> str:
    if not items:
        return "【章节结构】\n- 无"
    lines = ["【章节结构】"]
    for index, item in enumerate(items):
        step = item.get("step")
        name = item.get("name")
        goal = item.get("goal")
        chars = item.get("estimated_chars")
        if type(step) is not int or step < 1:
            raise ValueError(f"chapter_structure[{index}].step must be a positive integer")
        if not name:
            raise ValueError(f"chapter_structure[{index}].name is required")
        if not goal:
            raise ValueError(f"chapter_structure[{index}].goal is required")
        if not isinstance(chars, str) or not chars.strip():
            raise ValueError(f"chapter_structure[{index}].estimated_chars must be a non-empty string")
        lines.append(f"- {step}. {name}：{goal}（建议 {chars}）")
    return "\n".join(lines)


def _format_characters(items: list[dict[str, Any]]) -> str:
    if not items:
        return "【人物状态】\n- 无"
    lines = ["【人物状态】"]
    for item in items:
        lines.append(
            f"- {item.get('name', '')}：{item.get('state', '')}；说话方式：{item.get('speech_style', '')}"
        )
    return "\n".join(lines)


def render_execution_input(card: dict[str, Any]) -> str:
    sections = [
        "【风格契约】",
        card.get("style_contract", ""),
        "【前情摘要】",
        card.get("previous_summary", ""),
        "【本章目标】",
        card.get("chapter_goal", ""),
        "【冲突推进】",
        card.get("conflict_beat", ""),
        "【爽点兑现】",
        card.get("payoff_beat", ""),
        _format_structure(card.get("chapter_structure", [])),
        _format_characters(card.get("character_states", [])),
        _format_list("必须出现", card.get("must_include", [])),
        _format_list("禁止事项", card.get("must_not_include", [])),
        "【章末钩子】",
        card.get("ending_hook", ""),
        "【目标字数】",
        card.get("target_word_count", "2000-2500中文汉字"),
        "【输出要求】",
        "只输出正文，不输出提纲、小标题、解释、分析或提示语。",
    ]
    return "\n".join(section for section in sections if section is not None)


def build_chat_messages(card: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_execution_input(card)},
    ]


def render_model_input_prefix(card: dict[str, Any], tokenizer: Any | None = None) -> str:
    messages = build_chat_messages(card)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{SYSTEM_PROMPT}\n\n{render_execution_input(card)}\n\n"


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
