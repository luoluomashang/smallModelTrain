"""原始小说文本清洗与切章。

这一层负责把 txt/md 原稿变成统一的章节行数据。它不判断文学质量，只做
机械、可复现的清洗：去作者碎碎念、去分隔线、识别章节标题、计算字数。
"""

from __future__ import annotations

import re

from small_model_train.text_utils import count_chinese_chars, normalize_newlines


# 兼容“第1章”“第 1 章”“　 第十二章”等常见标题写法。
CHAPTER_TITLE_RE = re.compile(
    r"^[ \t\u3000]*(第\s*[零一二三四五六七八九十百千万0-9]+\s*[章节卷集部].*)$",
    re.MULTILINE,
)

# 作者话/题外话经常跨多行；正则会一直删到空行或下一章标题之前。
AUTHOR_NOTE_RE = re.compile(
    r"^[ \t\u3000]*(?:作者有话说|作者的话|题外话)\s*[:：]?.*"
    r"(?:\n(?!\s*$)(?![ \t\u3000]*第\s*[零一二三四五六七八九十百千万0-9]+\s*[章节卷集部]).*)*",
    re.MULTILINE,
)
SEPARATOR_RE = re.compile(r"^\s*[-=*_]{3,}\s*$", re.MULTILINE)


def clean_raw_text(text: str) -> str:
    """删除非正文噪音，并统一空行格式。"""

    text = AUTHOR_NOTE_RE.sub("", text)
    text = SEPARATOR_RE.sub("", text)
    return normalize_newlines(text)


def split_chapters(
    text: str,
    work_id: str,
    quality_tag: str = "A",
    split: str = "train",
) -> list[dict]:
    """把一篇作品切成章节字典列表。

    返回字段是后续步骤的公共契约：
    id 用于跨章节卡、SFT、评分结果关联；quality_tag 和 split 用于控制
    哪些章节进入训练，哪些章节只做评测。
    """

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
        # 有些小说在第一章前有楔子/序章，但没有标准标题；保留下来避免丢正文。
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
        # 如果前置正文已占用 0001，正式章节顺延编号，保证 id 不重复。
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
