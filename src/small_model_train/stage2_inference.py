from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from small_model_train.execution_cards import validate_execution_cards
from small_model_train.io_utils import read_jsonl
from small_model_train.prompt_renderer import render_model_input_prefix
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


def render_eval_model_input(card: dict, tokenizer: Any | None = None) -> str:
    render_sft_input(card)
    return render_model_input_prefix(card, tokenizer)


def _count_chinese_chars(text: str) -> int:
    return len(CHINESE_CHAR_RE.findall(text))


def _event_preview(text: str, max_chars: int = 80) -> str:
    preview = text.strip().replace("\n", "\\n")
    if len(preview) <= max_chars:
        return preview
    return preview[: max_chars - 3] + "..."


def _line_event(
    event_type: str,
    reason: str,
    line_number: int,
    text: str,
) -> dict[str, Any]:
    return {
        "type": event_type,
        "reason": reason,
        "line_number": line_number,
        "preview": _event_preview(text),
    }


def _cap_chinese_chars_with_event(
    text: str,
    max_chinese_chars: int | None = DEFAULT_MAX_CHINESE_CHARS,
) -> tuple[str, dict[str, Any] | None]:
    if max_chinese_chars is None:
        return text.strip(), None

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
        return clipped.strip(), None

    sentence_end = max(clipped.rfind(mark) for mark in "。！？!?")
    if sentence_end >= 0:
        sentence_clipped = clipped[: sentence_end + 1].rstrip()
        if _count_chinese_chars(sentence_clipped) >= MIN_SENTENCE_CAP_CHINESE_CHARS:
            clipped = sentence_clipped
    capped = clipped.strip()
    return capped, {
        "type": "cap_chinese_chars",
        "reason": "max_chinese_chars",
        "preview": _event_preview(capped),
        "max_chinese_chars": max_chinese_chars,
        "chinese_chars_before": _count_chinese_chars(text),
        "chinese_chars_after": _count_chinese_chars(capped),
    }


def _cap_chinese_chars(
    text: str,
    max_chinese_chars: int | None = DEFAULT_MAX_CHINESE_CHARS,
) -> str:
    capped, _event = _cap_chinese_chars_with_event(text, max_chinese_chars)
    return capped


def sanitize_generated_output(
    text: str,
    max_chinese_chars: int | None = DEFAULT_MAX_CHINESE_CHARS,
) -> str:
    sanitized, _events = sanitize_generated_output_with_events(text, max_chinese_chars)
    return sanitized


def sanitize_generated_output_with_events(
    text: str,
    max_chinese_chars: int | None = DEFAULT_MAX_CHINESE_CHARS,
) -> tuple[str, list[dict[str, Any]]]:
    lines: list[str] = []
    events: list[dict[str, Any]] = []
    in_meta_block = False

    for line_number, raw_line in enumerate(
        text.replace("\r\n", "\n").replace("\r", "\n").split("\n"),
        start=1,
    ):
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
            if has_heading:
                reason = "outline_heading"
            elif has_outline_bracket:
                reason = "outline_bracket"
            else:
                reason = "meta_directive"
            events.append(
                _line_event("drop_meta_line", reason, line_number, stripped)
            )
            continue
        if is_separator:
            in_meta_block = False
            events.append(
                _line_event("drop_separator", "separator", line_number, stripped)
            )
            continue
        if is_list_item:
            events.append(
                _line_event("drop_list_line", "list_item", line_number, stripped)
            )
            continue
        if in_meta_block and len(stripped) <= 60 and not PROSE_END_RE.search(stripped):
            events.append(
                _line_event(
                    "drop_meta_continuation",
                    "meta_block_continuation",
                    line_number,
                    stripped,
                )
            )
            continue

        in_meta_block = False
        lines.append(stripped)

    while lines and lines[-1] == "":
        lines.pop()
    sanitized, cap_event = _cap_chinese_chars_with_event(
        "\n".join(lines).strip(),
        max_chinese_chars,
    )
    if cap_event is not None:
        events.append(cap_event)
    return sanitized, events


def build_generation_row(
    sample_id: str,
    output: str,
    model: str,
    params: dict,
    *,
    raw_output: str | None = None,
    sanitized_output: str | None = None,
    adapter_dir: str | None = "",
    finish_reason: str = "unknown",
    generated_tokens: int = 0,
    prompt_sha256: str = "",
    sanitizer_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    raw = output if raw_output is None else raw_output
    sanitized = raw if sanitized_output is None else sanitized_output
    return {
        "id": sample_id,
        "output": raw,
        "raw_output": raw,
        "sanitized_output": sanitized,
        "model": model,
        "adapter_dir": adapter_dir,
        "params": dict(params),
        "finish_reason": finish_reason,
        "generated_tokens": generated_tokens,
        "prompt_sha256": prompt_sha256,
        "sanitizer_events": list(sanitizer_events or []),
    }


def load_eval_cards(path: str | Path) -> list[dict]:
    cards_path = Path(path)
    if not cards_path.exists():
        raise ValueError(f"cards file is missing: {cards_path}")

    rows = read_jsonl(cards_path)
    if not rows:
        raise ValueError(f"cards file has no rows: {cards_path}")

    return validate_execution_cards(rows)
