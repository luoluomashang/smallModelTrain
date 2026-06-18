"""SFT 数据构造。

SFT 样本由“章节执行卡 prompt + 作者正文 output”组成。本模块只负责拼装
训练输入，不负责自动生成章节卡；章节卡通常来自人工或外部模型反推。
"""

from __future__ import annotations

import re


# LLaMA-Factory 常见数据格式里的 instruction 字段；具体细节放在 input 中。
INSTRUCTION = "你是作者的正文执行器。请严格根据章节执行卡，写出符合作者风格的一章正文。"
SOURCE_LEAK_MIN_CHARS = 12


def _find_source_text_leak(rendered_input: str, source_text: str, min_chars: int = SOURCE_LEAK_MIN_CHARS) -> str | None:
    """检查章节卡 prompt 是否泄漏了原文连续片段。

    章节卡可以来自原文反推，但不能把原文句子复制进 prompt；否则模型可能
    只是记忆/复读训练正文。这里按连续中文片段滑窗检查，默认 12 字触发。
    """

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


def _format_list(title: str, values: list[str]) -> str:
    """把 must_include/must_not_include 这类字符串列表渲染成章节卡段落。"""

    body = "\n".join(f"- {value}" for value in values) if values else "- 无"
    return f"【{title}】\n{body}"


def _format_structure(items: list[dict]) -> str:
    """渲染章节结构步骤。"""

    if not items:
        return "【章节结构】\n- 无"
    lines = ["【章节结构】"]
    for item in items:
        step = item.get("step", "")
        name = item.get("name", "")
        goal = item.get("goal", "")
        chars = item.get("estimated_chars", "")
        lines.append(f"- {step}. {name}：{goal}（建议 {chars}）")
    return "\n".join(lines)


def _format_characters(items: list[dict]) -> str:
    """渲染人物当前状态和说话方式。"""

    if not items:
        return "【人物状态】\n- 无"
    lines = ["【人物状态】"]
    for item in items:
        lines.append(
            f"- {item.get('name', '')}：{item.get('state', '')}；说话方式：{item.get('speech_style', '')}"
        )
    return "\n".join(lines)


def render_sft_input(card: dict) -> str:
    """把单张章节卡渲染成模型输入文本。"""

    sections = [
        "【风格契约】",
        card.get("style_contract", ""),
        "【前情摘要】",
        card.get("previous_summary", ""),
        "【本章目标】",
        card.get("chapter_goal", ""),
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
    rendered_input = "\n".join(section for section in sections if section is not None)
    # source_text 只允许作为离线溯源字段；一旦进入 prompt，就违背防泄漏要求。
    leak = _find_source_text_leak(rendered_input, card.get("source_text", ""))
    if leak:
        raise ValueError(f"SFT input contains source_text fragment: {leak}")
    return rendered_input


def _is_trainable_chapter(chapter: dict) -> bool:
    """第一阶段只让 train split 且 A 类质量的章节进入 SFT。"""

    return chapter.get("split") == "train" and chapter.get("quality_tag") == "A"


def build_sft_rows(cards: list[dict], chapters: list[dict]) -> list[dict]:
    """按 id 把章节卡和正文配对，输出 SFT JSONL 行。"""

    chapter_by_id = {chapter["id"]: chapter for chapter in chapters}
    rows: list[dict] = []
    for card in cards:
        chapter = chapter_by_id.get(card["id"])
        if not chapter or not _is_trainable_chapter(chapter):
            continue
        rows.append(
            {
                "instruction": INSTRUCTION,
                "input": render_sft_input(card),
                "output": chapter.get("text", ""),
            }
        )
    return rows
