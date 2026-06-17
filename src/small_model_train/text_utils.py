from __future__ import annotations

import re
from collections import Counter


CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")


def count_chinese_chars(text: str) -> int:
    return len(CHINESE_RE.findall(text))


def normalize_newlines(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def paragraphs(text: str) -> list[str]:
    normalized = normalize_newlines(text)
    if not normalized:
        return []
    return [part.strip() for part in normalized.split("\n\n") if part.strip()]


def paragraph_lengths(text: str) -> list[int]:
    return [count_chinese_chars(part) for part in paragraphs(text)]


def dialogue_ratio(text: str) -> float:
    parts = paragraphs(text)
    if not parts:
        return 0.0
    dialogue_count = sum(1 for part in parts if "“" in part or "”" in part or part.startswith('"'))
    return dialogue_count / len(parts)


def repeated_ngram_ratio(text: str, n: int = 4) -> float:
    chars = CHINESE_RE.findall(text)
    if len(chars) < n:
        return 0.0
    grams = ["".join(chars[index : index + n]) for index in range(len(chars) - n + 1)]
    if not grams:
        return 0.0
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(grams)
