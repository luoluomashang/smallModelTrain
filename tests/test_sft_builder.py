from __future__ import annotations

import json
import subprocess
import sys

import pytest

from small_model_train.io_utils import read_jsonl, write_jsonl
from small_model_train.prompt_renderer import SYSTEM_PROMPT
from small_model_train.sft_builder import INSTRUCTION, build_sft_rows, render_sft_input

VALID_STYLE_HASH = "a" * 64


def _sft_card(card_id: str = "c1", **overrides) -> dict:
    card = {
        "id": card_id,
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
    card.update(overrides)
    return card


def _approved_sft_card(card_id: str = "c1", **overrides) -> dict:
    card = _sft_card(
        card_id,
        draft_only=False,
        approval_status="approved",
        style_contract_id="contract-v1",
        style_contract_sha256=VALID_STYLE_HASH,
    )
    card.update(overrides)
    return card


def _train_chapter(chapter_id: str = "c1") -> dict:
    return {"id": chapter_id, "text": "正文", "split": "train", "quality_tag": "A"}


def test_sft_instruction_reuses_system_prompt():
    assert INSTRUCTION == SYSTEM_PROMPT


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


def test_build_sft_rows_rejects_draft_cards_in_formal_mode():
    card = _approved_sft_card(draft_only=True, approval_status="draft")

    with pytest.raises(ValueError, match="draft card cannot enter formal SFT: c1"):
        build_sft_rows([card], [_train_chapter()], require_approved_cards=True)


@pytest.mark.parametrize(
    "card",
    [
        _approved_sft_card(draft_only=True, approval_status="draft"),
        _approved_sft_card(approval_status="pending"),
    ],
    ids=["draft", "unapproved"],
)
@pytest.mark.parametrize(
    "chapters",
    [
        [],
        [{**_train_chapter(), "split": "eval"}],
        [{**_train_chapter(), "quality_tag": "B"}],
    ],
    ids=["missing-chapter", "eval-split", "non-a-quality"],
)
def test_build_sft_rows_formal_mode_skips_non_sft_candidates_before_approval_gate(card, chapters):
    assert build_sft_rows([card], chapters, require_approved_cards=True) == []


@pytest.mark.parametrize("approval_status", [None, "draft", "pending"])
def test_build_sft_rows_rejects_missing_or_unapproved_status_in_formal_mode(approval_status):
    card = _approved_sft_card()
    if approval_status is None:
        del card["approval_status"]
    else:
        card["approval_status"] = approval_status

    with pytest.raises(ValueError, match="approval_status.*c1"):
        build_sft_rows([card], [_train_chapter()], require_approved_cards=True)


def test_build_sft_rows_rejects_missing_style_contract_id_in_formal_mode():
    card = _approved_sft_card()
    del card["style_contract_id"]

    with pytest.raises(ValueError, match="style_contract_id.*c1"):
        build_sft_rows([card], [_train_chapter()], require_approved_cards=True)


@pytest.mark.parametrize("style_hash", ["", "not-hex", "g" * 64, "a" * 63])
def test_build_sft_rows_rejects_invalid_style_contract_hash_in_formal_mode(style_hash):
    card = _approved_sft_card(style_contract_sha256=style_hash)

    with pytest.raises(ValueError, match="style_contract_sha256.*c1"):
        build_sft_rows([card], [_train_chapter()], require_approved_cards=True)


@pytest.mark.parametrize("approval_status", ["approved", "frozen"])
def test_build_sft_rows_accepts_approved_or_frozen_cards_in_formal_mode(approval_status):
    card = _approved_sft_card(approval_status=approval_status)

    rows = build_sft_rows([card], [_train_chapter()], require_approved_cards=True)

    assert rows[0]["output"] == "正文"


def test_build_sft_dataset_cli_rejects_draft_cards_by_default_and_allows_with_flag(tmp_path):
    cards_path = tmp_path / "cards.jsonl"
    chapters_path = tmp_path / "chapters.jsonl"
    output_path = tmp_path / "sft.jsonl"
    allowed_output_path = tmp_path / "sft_allowed.jsonl"
    write_jsonl(cards_path, [_approved_sft_card(draft_only=True, approval_status="draft")])
    write_jsonl(chapters_path, [_train_chapter()])

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
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "draft card cannot enter formal SFT: c1" in result.stderr
    assert not output_path.exists()

    subprocess.run(
        [
            sys.executable,
            "scripts/build_sft_dataset.py",
            "--cards",
            str(cards_path),
            "--chapters",
            str(chapters_path),
            "--output",
            str(allowed_output_path),
            "--allow-draft-cards",
        ],
        check=True,
    )

    rows = read_jsonl(allowed_output_path)
    assert len(rows) == 1
    assert rows[0]["output"] == "正文"


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
            "--allow-draft-cards",
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
            "--allow-draft-cards",
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
