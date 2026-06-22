from __future__ import annotations

import json
import subprocess
import sys

import pytest

from small_model_train.io_utils import read_jsonl, write_jsonl
from small_model_train.sft_builder import build_sft_rows, render_sft_input


def test_render_sft_input_excludes_source_text():
    card = {
        "style_contract": "风格契约",
        "previous_summary": "上一章摘要",
        "chapter_goal": "本章目标",
        "target_word_count": "2000-2500中文汉字",
        "chapter_structure": [
            {"step": 1, "name": "开场", "goal": "引出冲突", "estimated_chars": "300-400"}
        ],
        "character_states": [{"name": "林默", "state": "冷静", "speech_style": "短句"}],
        "must_include": ["加钱"],
        "must_not_include": ["提前揭露真相"],
        "ending_hook": "箱子响了一下",
        "source_text": "原文不能进入prompt",
    }
    text = render_sft_input(card)
    assert "原文不能进入prompt" not in text
    assert "只输出正文" in text
    assert "2000-2500中文汉字" in text


def test_render_sft_input_includes_external_control_beats():
    card = {
        "style_contract": "男频短句，强冲突。",
        "previous_summary": "林默被逼到晚宴角落。",
        "chapter_goal": "林默必须当场破局。",
        "chapter_structure": [
            {
                "step": 1,
                "name": "压迫",
                "goal": "岳家逼他低头。",
                "estimated_chars": "500-700",
            }
        ],
        "character_states": [],
        "conflict_beat": "岳家当众羞辱，林默提出反赌。",
        "payoff_beat": "林默拿出合同证据，让对方第一次失声。",
        "must_include": ["合同证据"],
        "must_not_include": ["作者说明"],
        "ending_hook": "门外响起真正买家的声音。",
        "target_word_count": "2000-2500中文汉字",
    }

    rendered = render_sft_input(card)

    assert "【冲突推进】\n岳家当众羞辱，林默提出反赌。" in rendered
    assert "【爽点兑现】\n林默拿出合同证据，让对方第一次失声。" in rendered


def test_render_sft_input_rejects_source_text_fragment_leakage():
    card = {
        "style_contract": "契约",
        "previous_summary": "他记得那句这是一段非常独特的原文句子，随后沉默。",
        "chapter_goal": "目标",
        "target_word_count": "2000-2500中文汉字",
        "chapter_structure": [],
        "character_states": [],
        "must_include": [],
        "must_not_include": [],
        "ending_hook": "",
        "source_text": "这是一段非常独特的原文句子，不能被复制进提示词。",
    }

    with pytest.raises(ValueError, match="source_text"):
        render_sft_input(card)


def test_render_sft_input_allows_nonleaking_source_text():
    card = {
        "style_contract": "契约",
        "previous_summary": "上一章只保留抽象摘要。",
        "chapter_goal": "目标",
        "target_word_count": "2000-2500中文汉字",
        "chapter_structure": [],
        "character_states": [],
        "must_include": ["加钱"],
        "must_not_include": [],
        "ending_hook": "",
        "source_text": "这是一段非常独特的原文句子，不能被复制进提示词。",
    }

    text = render_sft_input(card)

    assert "上一章只保留抽象摘要" in text
    assert "这是一段非常独特的原文句子" not in text


def test_render_sft_input_rejects_structure_without_step_or_name():
    card = {
        "style_contract": "契约",
        "previous_summary": "前情",
        "chapter_goal": "目标",
        "target_word_count": "2000-2500中文汉字",
        "chapter_structure": [{"beat": "承接", "goal": "推进", "estimated_chars": "300"}],
        "character_states": [{"name": "林默", "state": "冷静", "speech_style": "短句"}],
        "must_include": ["加钱"],
        "must_not_include": ["提前揭露真相"],
        "ending_hook": "箱子响了一下",
        "source_text": "",
    }

    with pytest.raises(ValueError, match="chapter_structure\\[0\\].step"):
        render_sft_input(card)


def test_render_sft_input_rejects_non_integer_structure_step():
    card = {
        "style_contract": "契约",
        "previous_summary": "前情",
        "chapter_goal": "目标",
        "target_word_count": "2000-2500中文汉字",
        "chapter_structure": [
            {"step": "1", "name": "开场", "goal": "推进", "estimated_chars": "300"}
        ],
        "character_states": [{"name": "林默", "state": "冷静", "speech_style": "短句"}],
        "must_include": ["加钱"],
        "must_not_include": ["提前揭露真相"],
        "ending_hook": "箱子响了一下",
        "source_text": "",
    }

    with pytest.raises(ValueError, match="chapter_structure\\[0\\].step must be a positive integer"):
        render_sft_input(card)


def test_build_sft_rows_rejects_non_integer_structure_step():
    cards = [
        {
            "id": "c1",
            "style_contract": "契约",
            "previous_summary": "",
            "chapter_goal": "目标",
            "target_word_count": "2000-2500中文汉字",
            "chapter_structure": [
                {"step": "1", "name": "开场", "goal": "推进", "estimated_chars": "300"}
            ],
            "character_states": [],
            "must_include": [],
            "must_not_include": [],
            "ending_hook": "",
        }
    ]
    chapters = [{"id": "c1", "text": "正文", "split": "train", "quality_tag": "A"}]

    with pytest.raises(ValueError, match="chapter_structure\\[0\\].step must be a positive integer"):
        build_sft_rows(cards, chapters)


def test_build_sft_rows_pairs_cards_with_chapters():
    cards = [
        {
            "id": "c1",
            "style_contract": "契约",
            "previous_summary": "",
            "chapter_goal": "",
            "target_word_count": "2000-2500中文汉字",
            "chapter_structure": [],
            "character_states": [],
            "must_include": [],
            "must_not_include": [],
            "ending_hook": "",
        }
    ]
    chapters = [{"id": "c1", "text": "正文", "split": "train", "quality_tag": "A"}]
    rows = build_sft_rows(cards, chapters)
    assert rows[0]["instruction"].startswith("你是作者的正文执行器")
    assert rows[0]["output"] == "正文"


def test_build_sft_rows_only_uses_train_a_chapters():
    cards = [
        {
            "id": "train_a",
            "style_contract": "契约",
            "previous_summary": "",
            "chapter_goal": "",
            "target_word_count": "2000-2500中文汉字",
            "chapter_structure": [],
            "character_states": [],
            "must_include": [],
            "must_not_include": [],
            "ending_hook": "",
        },
        {
            "id": "eval_a",
            "style_contract": "契约",
            "previous_summary": "",
            "chapter_goal": "",
            "target_word_count": "2000-2500中文汉字",
            "chapter_structure": [],
            "character_states": [],
            "must_include": [],
            "must_not_include": [],
            "ending_hook": "",
        },
        {
            "id": "train_b",
            "style_contract": "契约",
            "previous_summary": "",
            "chapter_goal": "",
            "target_word_count": "2000-2500中文汉字",
            "chapter_structure": [],
            "character_states": [],
            "must_include": [],
            "must_not_include": [],
            "ending_hook": "",
        },
    ]
    chapters = [
        {"id": "train_a", "text": "训练正文", "split": "train", "quality_tag": "A"},
        {"id": "eval_a", "text": "评估正文", "split": "eval", "quality_tag": "A"},
        {"id": "train_b", "text": "低质正文", "split": "train", "quality_tag": "B"},
    ]

    rows = build_sft_rows(cards, chapters)

    assert [row["output"] for row in rows] == ["训练正文"]


def test_build_sft_rows_skips_unmatched_card_id():
    cards = [
        {
            "id": "missing",
            "style_contract": "契约",
            "previous_summary": "",
            "chapter_goal": "",
            "target_word_count": "2000-2500中文汉字",
            "chapter_structure": [],
            "character_states": [],
            "must_include": [],
            "must_not_include": [],
            "ending_hook": "",
        }
    ]
    chapters = [{"id": "c1", "text": "正文"}]

    assert build_sft_rows(cards, chapters) == []


def test_build_sft_dataset_cli_writes_jsonl_without_source_text(tmp_path):
    cards_path = tmp_path / "cards.jsonl"
    chapters_path = tmp_path / "chapters.jsonl"
    output_path = tmp_path / "sft.jsonl"
    write_jsonl(
        cards_path,
        [
            {
                "id": "c1",
                "style_contract": "契约",
                "previous_summary": "前情",
                "chapter_goal": "目标",
                "target_word_count": "2000-2500中文汉字",
                "chapter_structure": [],
                "character_states": [],
                "must_include": [],
                "must_not_include": [],
                "ending_hook": "",
                "source_text": "泄漏文本",
            }
        ],
    )
    write_jsonl(
        chapters_path,
        [{"id": "c1", "text": "正文", "split": "train", "quality_tag": "A"}],
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/build_sft_dataset.py",
            "--cards",
            str(cards_path),
            "--chapters",
            str(chapters_path),
            "--output",
            str(output_path),
        ],
        check=True,
    )

    rows = read_jsonl(output_path)
    assert output_path.exists()
    assert rows[0]["output"] == "正文"
    assert "泄漏文本" not in rows[0]["input"]


def test_build_sft_dataset_cli_writes_llamafactory_dataset_info(tmp_path):
    cards_path = tmp_path / "cards.jsonl"
    chapters_path = tmp_path / "chapters.jsonl"
    output_path = tmp_path / "sft_chapter_v1.jsonl"
    dataset_info_path = tmp_path / "dataset_info.json"
    write_jsonl(
        cards_path,
        [
            {
                "id": "c1",
                "style_contract": "契约",
                "previous_summary": "前情",
                "chapter_goal": "目标",
                "target_word_count": "2000-2500中文汉字",
                "chapter_structure": [
                    {"step": 1, "name": "开场", "goal": "引出冲突", "estimated_chars": "300"}
                ],
                "character_states": [{"name": "林默", "state": "冷静", "speech_style": "短句"}],
                "must_include": ["加钱"],
                "must_not_include": ["提前揭露真相"],
                "ending_hook": "箱子响了一下",
            }
        ],
    )
    write_jsonl(chapters_path, [{"id": "c1", "text": "正文", "split": "train", "quality_tag": "A"}])

    subprocess.run(
        [
            sys.executable,
            "scripts/build_sft_dataset.py",
            "--cards",
            str(cards_path),
            "--chapters",
            str(chapters_path),
            "--output",
            str(output_path),
            "--dataset-info-output",
            str(dataset_info_path),
        ],
        check=True,
    )

    rows = read_jsonl(output_path)
    assert output_path.exists()
    assert rows[0]["output"] == "正文"

    info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
    assert info["sft_chapter_v1"]["file_name"] == "sft_chapter_v1.jsonl"
    assert info["sft_chapter_v1"]["formatting"] == "alpaca"
    assert info["sft_chapter_v1"]["columns"] == {
        "prompt": "instruction",
        "query": "input",
        "response": "output",
    }


def test_build_sft_dataset_cli_rejects_dataset_info_same_as_output(tmp_path):
    cards_path = tmp_path / "cards.jsonl"
    chapters_path = tmp_path / "chapters.jsonl"
    output_path = tmp_path / "sft_chapter_v1.jsonl"
    write_jsonl(
        cards_path,
        [
            {
                "id": "c1",
                "style_contract": "契约",
                "previous_summary": "前情",
                "chapter_goal": "目标",
                "target_word_count": "2000-2500中文汉字",
                "chapter_structure": [],
                "character_states": [],
                "must_include": [],
                "must_not_include": [],
                "ending_hook": "",
            }
        ],
    )
    write_jsonl(chapters_path, [{"id": "c1", "text": "正文", "split": "train", "quality_tag": "A"}])

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_sft_dataset.py",
            "--cards",
            str(cards_path),
            "--chapters",
            str(chapters_path),
            "--output",
            str(output_path),
            "--dataset-info-output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--dataset-info-output must not be the same path as --output" in result.stderr
    assert not output_path.exists()
