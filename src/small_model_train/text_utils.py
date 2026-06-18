"""中文小说文本的基础统计工具。

这些函数故意保持轻量：只依赖标准库，能在数据清洗、评分、报告等步骤中
重复使用。第一阶段不追求复杂 NLP，只先建立稳定、可解释的规则指标。
"""

from __future__ import annotations

import re
from collections import Counter


# 只统计常用中日韩统一表意文字区间；数字、英文、标点不计入中文汉字数。
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")


def count_chinese_chars(text: str) -> int:
    """统计文本中的中文汉字数量。"""

    return len(CHINESE_RE.findall(text))


def normalize_newlines(text: str) -> str:
    """统一换行格式，并把过多空行压成一个段落分隔。

    小说原稿常混有 Windows 换行、尾随空格和多重空行。统一后，后续按
    空行切段、计算段落长度、切章都会更稳定。
    """

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def paragraphs(text: str) -> list[str]:
    """按空行切出段落，返回去掉首尾空白后的非空段落。"""

    normalized = normalize_newlines(text)
    if not normalized:
        return []
    return [part.strip() for part in normalized.split("\n\n") if part.strip()]


def paragraph_lengths(text: str) -> list[int]:
    """返回每个段落的中文汉字数。"""

    return [count_chinese_chars(part) for part in paragraphs(text)]


def dialogue_ratio(text: str) -> float:
    """估算对白段落比例。

    这里使用简单规则：段落里出现中文引号，或以英文双引号开头，就算对白。
    它不是文学判断，只是风格画像里的粗粒度参考指标。
    """

    parts = paragraphs(text)
    if not parts:
        return 0.0
    dialogue_count = sum(1 for part in parts if "“" in part or "”" in part or part.startswith('"'))
    return dialogue_count / len(parts)


def repeated_ngram_ratio(text: str, n: int = 4) -> float:
    """计算中文 n-gram 的重复比例，用来发现明显复读。

    例如连续重复“加钱加钱加钱”会让 2-gram/4-gram 重复率升高。评分阶段
    用这个指标作为“重复”硬门槛的一部分。
    """

    chars = CHINESE_RE.findall(text)
    if len(chars) < n:
        return 0.0
    grams = ["".join(chars[index : index + n]) for index in range(len(chars) - n + 1)]
    if not grams:
        return 0.0
    counts = Counter(grams)
    # 只把“第二次及以后出现”的部分算作重复，第一次出现是正常内容。
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(grams)
