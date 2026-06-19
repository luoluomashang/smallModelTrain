from __future__ import annotations

import pytest

from small_model_train.chapter_cards import (
    build_draft_chapter_cards,
    normalize_chapter_structure,
    validate_chapter_card,
)


def _chapter(sample_id: str = "chapter-1", char_count: int = 2800) -> dict:
    return {
        "id": sample_id,
        "text": "正文" * 1400,
        "split": "train",
        "quality_tag": "A",
        "char_count_zh": char_count,
        "chapter_title": "第一章 测试",
    }


def test_normalize_chapter_structure_writes_step_and_name():
    items = [
        {"beat": "承接", "goal": "承接上一章压力。", "estimated_chars": 400},
        {"name": "转折", "goal": "完成本章选择。", "estimated_chars": "500"},
    ]

    normalized = normalize_chapter_structure(items)

    assert normalized == [
        {"step": 1, "name": "承接", "goal": "承接上一章压力。", "estimated_chars": "400"},
        {"step": 2, "name": "转折", "goal": "完成本章选择。", "estimated_chars": "500"},
    ]


def test_validate_chapter_card_rejects_empty_structure_labels():
    card = {
        "id": "chapter-1",
        "style_contract": "只输出正文。",
        "previous_summary": "前情摘要。",
        "chapter_goal": "推进冲突。",
        "chapter_structure": [{"beat": "承接", "goal": "推进。", "estimated_chars": 400}],
        "character_states": [{"name": "核心视角人物", "state": "谨慎", "speech_style": "短句"}],
        "must_include": ["清楚的开场状态"],
        "must_not_include": ["输出提纲或小标题"],
        "ending_hook": "留下下一步压力。",
        "target_word_count": "2500-3000中文汉字",
        "source_text": "正文" * 100,
    }

    with pytest.raises(ValueError, match="chapter_structure\\[0\\].step"):
        validate_chapter_card(card)


def test_build_draft_chapter_cards_uses_train_a_chapters_only():
    chapters = [
        _chapter("train-a", 2800),
        {**_chapter("eval-a", 2800), "id": "eval-a", "split": "eval"},
        {**_chapter("train-b", 2800), "id": "train-b", "quality_tag": "B"},
    ]

    cards = build_draft_chapter_cards(chapters, count=1, min_chars=2000, max_chars=3000)

    assert [card["id"] for card in cards] == ["train-a"]
    assert cards[0]["chapter_structure"][0]["step"] == 1
    assert cards[0]["chapter_structure"][0]["name"] == "承接"
    assert cards[0]["source_text"].startswith("正文")
