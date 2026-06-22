from __future__ import annotations

import pytest

from small_model_train.execution_cards import (
    DEFAULT_TARGET_PLATFORM,
    validate_execution_card,
    validate_execution_cards,
)


def _valid_card() -> dict:
    return {
        "id": "case1",
        "target_platform": DEFAULT_TARGET_PLATFORM,
        "genre_tags": ["urban", "system"],
        "style_contract": "短句推进，口语化，男频爽文节奏。",
        "previous_summary": "林默刚拿到系统任务。",
        "chapter_goal": "林默必须在晚宴上证明自己。",
        "chapter_structure": [
            {
                "step": 1,
                "name": "开场压迫",
                "goal": "岳家众人当众羞辱林默。",
                "estimated_chars": "500-700",
            }
        ],
        "conflict_beat": "岳家要求林默认错，林默反手提出赌约。",
        "payoff_beat": "林默用系统奖励拿出证据，第一次压住对方。",
        "must_include": ["赌约", "证据"],
        "must_not_include": ["作者说明"],
        "ending_hook": "门外传来真正买家的声音。",
        "target_word_count": "2000-2500中文汉字",
    }


def test_validate_execution_card_accepts_complete_card():
    assert validate_execution_card(_valid_card()) == _valid_card()


def test_validate_execution_card_blocks_raw_eval_chapter():
    raw_card = {
        "id": "case1",
        "work_id": "book1",
        "chapter_title": "第1章",
        "text": "原文正文",
        "quality_tag": "A",
        "split": "eval",
    }

    with pytest.raises(ValueError) as excinfo:
        validate_execution_card(raw_card)

    assert "missing execution-card fields" in str(excinfo.value)
    assert "style_contract" in str(excinfo.value)


def test_validate_execution_card_blocks_unknown_platform():
    card = _valid_card()
    card["target_platform"] = "traditional_literary"

    with pytest.raises(ValueError) as excinfo:
        validate_execution_card(card)

    assert "unknown target_platform" in str(excinfo.value)


def test_validate_execution_card_blocks_empty_genre_tags():
    card = _valid_card()
    card["genre_tags"] = []

    with pytest.raises(ValueError) as excinfo:
        validate_execution_card(card)

    assert "genre_tags must be a non-empty list" in str(excinfo.value)


@pytest.mark.parametrize(
    ("chapter_step", "expected_message"),
    [
        ({"step": 0, "name": "开场压迫", "goal": "羞辱林默。", "estimated_chars": "500-700"}, "step"),
        ({"step": 1, "name": "", "goal": "羞辱林默。", "estimated_chars": "500-700"}, "name"),
        ({"step": 1, "name": "开场压迫", "goal": " ", "estimated_chars": "500-700"}, "goal"),
        ({"step": 1, "name": "开场压迫", "goal": "羞辱林默。", "estimated_chars": ""}, "estimated_chars"),
        ("开场压迫", "chapter_structure items must be dicts"),
    ],
)
def test_validate_execution_card_blocks_invalid_chapter_structure_items(
    chapter_step, expected_message
):
    card = _valid_card()
    card["chapter_structure"] = [chapter_step]

    with pytest.raises(ValueError) as excinfo:
        validate_execution_card(card)

    assert expected_message in str(excinfo.value)


@pytest.mark.parametrize("field", ["must_include", "must_not_include"])
def test_validate_execution_card_blocks_empty_include_items(field):
    card = _valid_card()
    card[field] = ["赌约", " "]

    with pytest.raises(ValueError) as excinfo:
        validate_execution_card(card)

    assert f"{field} must contain non-empty strings" in str(excinfo.value)


def test_validate_execution_cards_reports_row_number():
    rows = [_valid_card(), {"id": "bad"}]

    with pytest.raises(ValueError) as excinfo:
        validate_execution_cards(rows)

    assert "row 2" in str(excinfo.value)
