from __future__ import annotations

import subprocess
import sys

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
    chapters = [{"id": "c1", "text": "正文"}]
    rows = build_sft_rows(cards, chapters)
    assert rows[0]["instruction"].startswith("你是作者的正文执行器")
    assert rows[0]["output"] == "正文"


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
    write_jsonl(chapters_path, [{"id": "c1", "text": "正文"}])

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
