"""Style profile and style contract generation for Stage 1.

The profile summarizes observable chapter statistics. The contract is a compact
prompt-facing description, not a learned model of authorial style.
"""

from __future__ import annotations

import re
from statistics import mean

from small_model_train.scoring import AI_TRACE_PHRASES
from small_model_train.text_utils import (
    count_chinese_chars,
    dialogue_ratio,
    paragraph_lengths,
)


SENTENCE_RE = re.compile(r"[^。！？!?]+[。！？!?]?")
PUNCTUATION_MARKS = ("。", "，", "、", "；", "：", "！", "？", "“", "”", "…")


def _percentile(values: list[float], percentile: float) -> float:
    """Return a linearly interpolated percentile for sorted-position stability."""
    if not values:
        return 0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    interpolated = ordered[lower_index] + (
        ordered[upper_index] - ordered[lower_index]
    ) * fraction
    return round(float(interpolated), 4)


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0, "max": 0, "avg": 0, "p50": 0, "p90": 0}
    return {
        "min": round(float(min(values)), 4),
        "max": round(float(max(values)), 4),
        "avg": round(mean(values), 4),
        "p50": _percentile(values, 0.5),
        "p90": _percentile(values, 0.9),
    }


def _sentence_lengths(text: str) -> list[int]:
    lengths = []
    for match in SENTENCE_RE.finditer(text.replace("\n", "")):
        sentence = match.group(0).strip()
        if sentence:
            lengths.append(count_chinese_chars(sentence))
    return [length for length in lengths if length > 0]


def _punctuation_density(texts: list[str]) -> dict[str, float]:
    total_chars = sum(count_chinese_chars(text) for text in texts)
    if total_chars <= 0:
        return {mark: 0 for mark in PUNCTUATION_MARKS}
    return {
        mark: round(sum(text.count(mark) for text in texts) / total_chars, 6)
        for mark in PUNCTUATION_MARKS
    }


def _ai_taste_metrics(texts: list[str]) -> dict:
    phrases = [
        phrase for phrase in AI_TRACE_PHRASES if isinstance(phrase, str) and phrase
    ]
    phrase_hits = {
        phrase: sum(text.count(phrase) for text in texts) for phrase in phrases
    }
    total_hits = sum(phrase_hits.values())
    total_chars = sum(count_chinese_chars(text) for text in texts)
    return {
        "phrase_hits": phrase_hits,
        "total_hits": total_hits,
        "hits_per_10k_chars": round(total_hits / total_chars * 10000, 4)
        if total_chars
        else 0,
    }


def _metric_value(metric: object, fallback: object = 0) -> object:
    if isinstance(metric, dict):
        return metric.get("avg", fallback)
    if metric is None:
        return fallback
    return metric


def _float_or_zero(value: object) -> float:
    if value is None:
        return 0
    return float(value)


def build_style_profile(rows: list[dict]) -> dict:
    texts = [row.get("text", "") for row in rows if row.get("text")]
    chapter_chars = [count_chinese_chars(text) for text in texts]
    paragraph_counts = [length for text in texts for length in paragraph_lengths(text)]
    dialogue_ratios = [dialogue_ratio(text) for text in texts]
    sentence_counts = [length for text in texts for length in _sentence_lengths(text)]
    return {
        "chapter_count": len(texts),
        "avg_chinese_chars": round(mean(chapter_chars), 2) if chapter_chars else 0,
        "avg_paragraph_chars": round(mean(paragraph_counts), 2) if paragraph_counts else 0,
        "avg_dialogue_ratio": round(mean(dialogue_ratios), 4) if dialogue_ratios else 0,
        "chinese_chars": _distribution([float(value) for value in chapter_chars]),
        "paragraph_chars": _distribution([float(value) for value in paragraph_counts]),
        "dialogue_ratio": _distribution([float(value) for value in dialogue_ratios]),
        "sentence_chars": _distribution([float(value) for value in sentence_counts]),
        "punctuation_density": _punctuation_density(texts),
        "ai_taste": _ai_taste_metrics(texts),
        "source_filter": {
            "total_rows": len(rows),
            "selected_rows": len(texts),
            "skipped_rows": len(rows) - len(texts),
            "quality_filter": "provided_rows",
        },
    }


def render_style_contract(profile: dict) -> str:
    dialogue_ratio = _metric_value(
        profile.get("dialogue_ratio"), profile.get("avg_dialogue_ratio", 0)
    )
    dialogue_percent = round(
        _float_or_zero(dialogue_ratio) * 100,
        1,
    )
    avg_paragraph_chars = _metric_value(
        profile.get("paragraph_chars"), profile.get("avg_paragraph_chars", 0)
    )
    if avg_paragraph_chars is None:
        avg_paragraph_chars = 0
    ai_taste = profile.get("ai_taste", {})
    if not isinstance(ai_taste, dict):
        ai_taste = {}
    return "\n".join(
        [
            "【角色】",
            "你是作者的正文执行器，只负责根据章节执行卡写正文。",
            "",
            "【叙述原则】",
            "1. 句子朴素直接，动作承接优先于心理解释。",
            "2. 情绪通过动作、对白和反应表现，不写总结式升华。",
            "3. 主角视角跟随，不随意跳到全知视角。",
            f"4. 段落长度参考：平均约 {avg_paragraph_chars} 个中文汉字。",
            "",
            "【对白原则】",
            f"1. 对话比例参考：约 {dialogue_percent}%。",
            "2. 对话短、准、自然，不用长篇对白解释世界观。",
            "3. 允许省略、打断和反问。",
            "",
            "【禁止风格】",
            "1. 不写空气仿佛凝固了。",
            "2. 不写难以言喻的情绪涌上心头。",
            "3. 不写命运的齿轮开始转动。",
            "4. 不写嘴角勾起一抹弧度。",
            "5. 不写眼神逐渐坚定起来。",
            (
                "6. 当前语料 AI 味短语命中约 "
                f"{ai_taste.get('hits_per_10k_chars', 0)} "
                "次/万字，生成时应继续压低。"
            ),
            "",
            "【输出要求】",
            "只输出正文。不要输出提纲、小标题、解释、分析或提示语。",
        ]
    )
