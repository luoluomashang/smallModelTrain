from __future__ import annotations

import hashlib
import re
import subprocess
import sys

import pytest

from small_model_train.chapter_cards import (
    STYLE_CONTRACT,
    build_draft_chapter_cards,
    normalize_chapter_structure,
    validate_chapter_card,
)
from small_model_train.io_utils import read_jsonl, write_jsonl


def _chapter(sample_id: str = "chapter-1", char_count: int = 2800) -> dict:
    return {
        "id": sample_id,
        "text": "正文" * 1400,
        "split": "train",
        "quality_tag": "A",
        "char_count_zh": char_count,
        "chapter_title": "第一章 测试",
    }


def _valid_card() -> dict:
    return {
        "id": "chapter-1",
        "style_contract": "只输出正文。",
        "previous_summary": "前情摘要。",
        "chapter_goal": "推进冲突。",
        "chapter_structure": [{"step": 1, "name": "承接", "goal": "推进。", "estimated_chars": "400"}],
        "character_states": [{"name": "核心视角人物", "state": "谨慎", "speech_style": "短句"}],
        "must_include": ["清楚的开场状态"],
        "must_not_include": ["输出提纲或小标题"],
        "ending_hook": "留下下一步压力。",
        "target_word_count": "2500-3000中文汉字",
        "source_text": "正文" * 100,
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
    card = _valid_card()
    card["chapter_structure"] = [{"beat": "承接", "goal": "推进。", "estimated_chars": 400}]

    with pytest.raises(ValueError, match="chapter_structure\\[0\\].step"):
        validate_chapter_card(card)


@pytest.mark.parametrize("chapter_structure", ["bad", {"step": 1}, None])
def test_validate_chapter_card_rejects_malformed_structure_container(chapter_structure):
    card = _valid_card()
    card["chapter_structure"] = chapter_structure

    with pytest.raises(ValueError, match="chapter_structure must be a non-empty list"):
        validate_chapter_card(card)


@pytest.mark.parametrize(
    ("item", "message"),
    [
        ("bad", "chapter_structure\\[0\\] must be a dict"),
        ({"step": 0, "name": "承接", "goal": "推进。", "estimated_chars": "400"}, "chapter_structure\\[0\\].step"),
        ({"step": -1, "name": "承接", "goal": "推进。", "estimated_chars": "400"}, "chapter_structure\\[0\\].step"),
        ({"step": "one", "name": "承接", "goal": "推进。", "estimated_chars": "400"}, "chapter_structure\\[0\\].step"),
    ],
)
def test_validate_chapter_card_rejects_malformed_structure_items(item, message):
    card = _valid_card()
    card["chapter_structure"] = [item]

    with pytest.raises(ValueError, match=message):
        validate_chapter_card(card)


@pytest.mark.parametrize("character_states", ["bad", {"name": "核心视角人物"}, None])
def test_validate_chapter_card_rejects_malformed_character_states_container(character_states):
    card = _valid_card()
    card["character_states"] = character_states

    with pytest.raises(ValueError, match="character_states must be a non-empty list"):
        validate_chapter_card(card)


def test_validate_chapter_card_rejects_malformed_character_state_item():
    card = _valid_card()
    card["character_states"] = ["bad"]

    with pytest.raises(ValueError, match="character_states\\[0\\] must be a dict"):
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


def test_build_draft_chapter_cards_marks_generated_cards_as_draft_only():
    cards = build_draft_chapter_cards([_chapter("train-a", 2800)], count=1, min_chars=2000, max_chars=3000)

    style_hash = cards[0]["style_contract_sha256"]
    assert cards[0]["draft_only"] is True
    assert cards[0]["approval_status"] == "draft"
    assert cards[0]["style_contract_id"] == "inline-draft-v0"
    assert style_hash == hashlib.sha256(STYLE_CONTRACT.encode("utf-8")).hexdigest()
    assert re.fullmatch(r"[0-9a-f]{64}", style_hash)


def test_build_draft_chapter_cards_filters_char_count_bounds_inclusively():
    chapters = [
        _chapter("too-short", 1999),
        _chapter("lower-bound", 2000),
        _chapter("upper-bound", 3000),
        _chapter("too-long", 3001),
    ]

    cards = build_draft_chapter_cards(chapters, count=10, min_chars=2000, max_chars=3000)

    assert [card["id"] for card in cards] == ["lower-bound", "upper-bound"]


def test_build_draft_chapter_cards_skips_malformed_char_count_rows():
    chapters = [
        {**_chapter("bad-count", 2800), "char_count_zh": "unknown"},
        _chapter("valid", 2800),
    ]

    cards = build_draft_chapter_cards(chapters, count=10, min_chars=2000, max_chars=3000)

    assert [card["id"] for card in cards] == ["valid"]


def test_build_draft_chapter_cards_rejects_negative_count():
    with pytest.raises(ValueError, match="count must be >= 0"):
        build_draft_chapter_cards([_chapter("valid", 2800)], count=-1, min_chars=2000, max_chars=3000)


def test_build_chapter_cards_cli_writes_fixed_cards(tmp_path):
    chapters_path = tmp_path / "chapters_split.jsonl"
    output_path = tmp_path / "chapter_cards.jsonl"
    write_jsonl(chapters_path, [_chapter("train-a", 2800)])

    subprocess.run(
        [
            sys.executable,
            "scripts/build_chapter_cards.py",
            "--chapters",
            str(chapters_path),
            "--output",
            str(output_path),
            "--count",
            "1",
            "--min-chars",
            "2000",
            "--max-chars",
            "3000",
        ],
        check=True,
    )

    rows = read_jsonl(output_path)
    assert len(rows) == 1
    assert rows[0]["chapter_structure"][0]["step"] == 1
    assert rows[0]["chapter_structure"][0]["name"] == "承接"
