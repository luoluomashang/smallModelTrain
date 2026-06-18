"""Raw novel cleanup and chapter extraction for Stage 1.

This module turns messy source text into chapter records. It does not infer
missing chapter cards; those remain an explicit human/agent-authored artifact
so the SFT prompt does not accidentally leak the answer.
"""

from __future__ import annotations

import re

from small_model_train.text_utils import count_chinese_chars, normalize_newlines


CHAPTER_TITLE_RE = re.compile(
    r"^[ \t\u3000]*(第\s*[零一二三四五六七八九十百千万0-9]+\s*[章节卷集部].*)$",
    re.MULTILINE,
)

AUTHOR_NOTE_RE = re.compile(
    r"^[ \t\u3000]*(?:作者有话说|作者的话|题外话)\s*[:：]?.*"
    r"(?:\n(?!\s*$)(?![ \t\u3000]*第\s*[零一二三四五六七八九十百千万0-9]+\s*[章节卷集部]).*)*",
    re.MULTILINE,
)
SEPARATOR_RE = re.compile(r"^\s*[-=*_]{3,}\s*$", re.MULTILINE)


def clean_raw_text(text: str) -> str:
    text = AUTHOR_NOTE_RE.sub("", text)
    text = SEPARATOR_RE.sub("", text)
    return normalize_newlines(text)


def split_chapters(
    text: str,
    work_id: str,
    quality_tag: str = "A",
    split: str = "train",
) -> list[dict]:
    cleaned = clean_raw_text(text)
    matches = list(CHAPTER_TITLE_RE.finditer(cleaned))
    if not matches:
        return [
            {
                "id": f"{work_id}_chapter_0001",
                "work_id": work_id,
                "chapter_title": "未命名章节",
                "text": cleaned,
                "char_count_zh": count_chinese_chars(cleaned),
                "quality_tag": quality_tag,
                "split": split,
            }
        ]

    chapters: list[dict] = []
    preface = normalize_newlines(cleaned[: matches[0].start()])
    if preface:
        chapters.append(
            {
                "id": f"{work_id}_chapter_0001",
                "work_id": work_id,
                "chapter_title": "前置正文",
                "text": preface,
                "char_count_zh": count_chinese_chars(preface),
                "quality_tag": quality_tag,
                "split": split,
            }
        )

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(cleaned)
        chapter_number = len(chapters) + 1
        title = match.group(1).strip()
        body = normalize_newlines(cleaned[start:end])
        chapters.append(
            {
                "id": f"{work_id}_chapter_{chapter_number:04d}",
                "work_id": work_id,
                "chapter_title": title,
                "text": body,
                "char_count_zh": count_chinese_chars(body),
                "quality_tag": quality_tag,
                "split": split,
            }
        )
    return chapters
