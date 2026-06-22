"""Deterministic quality rules for generated prose outputs."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from small_model_train.text_utils import count_chinese_chars


MARKDOWN_RE = re.compile(r"(^|\n)\s*(>|#{1,6}\s+|[-*]\s+|\d+[.、]\s+)")
DISCLAIMER_MARKERS = ("作为AI", "我无法", "不能保证", "仅供参考")
META_EVALUATION_MARKERS = ("以下是正文", "最终确认", "检查清单", "本章完成", "符合要求")
PROSE_END_RE = re.compile(r"[。！？!?…’”）)]$")
CHINESE_RUN_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
GENERIC_PHRASES = (
    "终于明白",
    "轻声说",
    "像是某种",
    "深吸一口气",
    "空气仿佛凝固",
)


def detect_quality_issues(card: dict[str, Any], text: str) -> dict[str, Any]:
    issues: list[str] = []
    details: dict[str, Any] = {}

    if MARKDOWN_RE.search(text):
        issues.append("markdown_residue")
    if any(marker in text for marker in DISCLAIMER_MARKERS):
        issues.append("disclaimer_residue")
    if any(marker in text for marker in META_EVALUATION_MARKERS):
        issues.append("meta_evaluation_residue")
    if text.strip() and not PROSE_END_RE.search(text.strip()):
        issues.append("unnatural_ending")

    repeated = _repeated_chinese_runs(text)
    if repeated:
        issues.append("semantic_repetition")
        details["repeated_runs"] = repeated[:5]

    generic_hits = [phrase for phrase in GENERIC_PHRASES if text.count(phrase) >= 2]
    if generic_hits:
        issues.append("generic_ai_phrase")
        details["generic_phrase_hits"] = generic_hits

    payoff_beat = str(card.get("payoff_beat", "")).strip()
    if payoff_beat and _coverage_terms(payoff_beat, text) < 0.34:
        issues.append("no_visible_payoff")

    ending_hook = str(card.get("ending_hook", "")).strip()
    if ending_hook and _coverage_terms(ending_hook, text[-200:]) < 0.34:
        issues.append("weak_ending_hook")

    if count_chinese_chars(text) >= 2450 and "semantic_repetition" in issues:
        issues.append("padding_to_length")

    return {"issues": sorted(set(issues)), "details": details}


def _repeated_chinese_runs(text: str) -> list[str]:
    runs = [run for run in CHINESE_RUN_RE.findall(text) if len(run) >= 5]
    windows: list[str] = []
    for run in runs:
        for index in range(0, max(len(run) - 7, 0) + 1):
            windows.append(run[index : index + 8])
    counts = Counter(windows)
    return [value for value, count in counts.items() if count >= 3]


def _coverage_terms(source: str, text: str) -> float:
    terms = _source_terms(source)
    if not terms:
        return 1.0
    hits = sum(1 for term in terms if term in text)
    return hits / len(terms)


def _source_terms(source: str) -> list[str]:
    terms: set[str] = set()
    for run in CHINESE_RUN_RE.findall(source):
        if len(run) <= 4:
            terms.add(run)
            continue
        for size in (2, 3, 4):
            for index in range(0, len(run) - size + 1):
                terms.add(run[index : index + size])
    return sorted(terms)
