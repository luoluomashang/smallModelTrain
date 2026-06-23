from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from small_model_train.execution_cards import validate_execution_cards
from small_model_train.io_utils import read_jsonl
from small_model_train.sft_builder import render_sft_input

OUTLINE_HEADING_RE = re.compile(r"【[^】]{1,80}】")
LIST_ITEM_RE = re.compile(r"^(?:[-*]\s+|\d+[.、]\s*)")
SEPARATOR_RE = re.compile(r"^[-—_]{3,}$")
PROSE_END_RE = re.compile(r"[。！？!?…’”）)]$")
CHINESE_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
META_DIRECTIVE_MARKERS = (
    "请严格遵循",
    "请根据以上",
    "已激活",
    "以下是正文",
    "检查清单",
    "输出格式",
    "章节结构",
    "风格契约",
    "前情摘要",
    "本章目标",
    "人物状态",
    "必须出现",
    "必须包含",
    "禁止事项",
    "章末钩子",
    "目标字数",
    "第一人称",
    "人称视角",
    "语言风格",
    "现实主义",
    "叙事节奏",
    "叙事视角",
    "主题锚点",
    "核心情绪",
    "关键意象",
    "情感结构",
    "最终效果",
)
DEFAULT_MAX_CHINESE_CHARS = 2500
MIN_SENTENCE_CAP_CHINESE_CHARS = 2000


def default_inference_params() -> dict[str, Any]:
    return {
        "max_new_tokens": 5120,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.05,
    }


def render_eval_prompt(card: dict) -> str:
    return render_sft_input(card)


def _count_chinese_chars(text: str) -> int:
    return len(CHINESE_CHAR_RE.findall(text))


def _cap_chinese_chars(
    text: str,
    max_chinese_chars: int | None = DEFAULT_MAX_CHINESE_CHARS,
) -> str:
    if max_chinese_chars is None:
        return text.strip()

    chinese_count = 0
    overflow = False
    kept: list[str] = []
    for char in text:
        if CHINESE_CHAR_RE.match(char):
            if chinese_count >= max_chinese_chars:
                overflow = True
                break
            chinese_count += 1
        kept.append(char)

    clipped = "".join(kept).rstrip()
    if not overflow:
        return clipped.strip()

    sentence_end = max(clipped.rfind(mark) for mark in "。！？!?")
    if sentence_end >= 0:
        sentence_clipped = clipped[: sentence_end + 1].rstrip()
        if _count_chinese_chars(sentence_clipped) >= MIN_SENTENCE_CAP_CHINESE_CHARS:
            return sentence_clipped.strip()
    return clipped.strip()


def sanitize_generated_output(
    text: str,
    max_chinese_chars: int | None = DEFAULT_MAX_CHINESE_CHARS,
) -> str:
    lines: list[str] = []
    in_meta_block = False

    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = raw_line.strip()
        if not stripped:
            if lines and lines[-1] != "":
                lines.append("")
            in_meta_block = False
            continue

        has_heading = bool(OUTLINE_HEADING_RE.search(stripped))
        has_outline_bracket = "【" in stripped or "】" in stripped
        has_meta_marker = any(marker in stripped for marker in META_DIRECTIVE_MARKERS)
        is_list_item = bool(LIST_ITEM_RE.match(stripped))
        is_separator = bool(SEPARATOR_RE.match(stripped))

        if has_heading or has_outline_bracket or has_meta_marker:
            in_meta_block = True
            continue
        if is_separator:
            in_meta_block = False
            continue
        if is_list_item:
            continue
        if in_meta_block and len(stripped) <= 60 and not PROSE_END_RE.search(stripped):
            continue

        in_meta_block = False
        lines.append(stripped)

    while lines and lines[-1] == "":
        lines.pop()
    return _cap_chinese_chars("\n".join(lines).strip(), max_chinese_chars)


def build_generation_row(
    sample_id: str,
    output: str,
    model: str,
    params: dict,
) -> dict[str, Any]:
    return {
        "id": sample_id,
        "output": output,
        "model": model,
        "params": dict(params),
    }


def load_eval_cards(path: str | Path) -> list[dict]:
    cards_path = Path(path)
    if not cards_path.exists():
        raise ValueError(f"cards file is missing: {cards_path}")

    rows = read_jsonl(cards_path)
    if not rows:
        raise ValueError(f"cards file has no rows: {cards_path}")

    return validate_execution_cards(rows)
