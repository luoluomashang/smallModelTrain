from __future__ import annotations

import subprocess
import sys

import pytest

from small_model_train.io_utils import read_jsonl, write_jsonl


def _style_contract() -> dict:
    from small_model_train.style_contract import build_style_contract_asset

    return build_style_contract_asset(
        style_contract_id="contract-v1",
        approval_status="approved",
        source_corpus={
            "path": "chapters.jsonl",
            "sha256": "b" * 64,
            "quality_filter": "quality_tag=A",
            "row_count": 1,
            "selected_rows": 1,
            "split_summary": {"train": 1},
        },
        profile_metrics={
            "chapter_count": 1,
            "avg_dialogue_ratio": 0.1,
            "avg_paragraph_chars": 20,
            "ai_taste": {"phrase_hits": {}, "total_hits": 0, "hits_per_10k_chars": 0},
        },
    )


def _draft_card(**overrides) -> dict:
    card = {
        "id": "c1",
        "style_contract": "旧风格契约",
        "previous_summary": "上一章林默接下委托。",
        "chapter_goal": "林默进入旧仓库并完成谈判。",
        "chapter_structure": [
            {"step": 1, "name": "入场", "goal": "交代仓库压力", "estimated_chars": "300-500"}
        ],
        "character_states": [
            {"name": "林默", "state": "警惕", "speech_style": "短句"}
        ],
        "conflict_beat": "周衡压价。",
        "payoff_beat": "林默发现箱子异常。",
        "must_include": ["旧仓库", "加钱"],
        "must_not_include": ["作者说明"],
        "ending_hook": "箱子自己响了一下。",
        "target_word_count": "2000-2500中文汉字",
    }
    card.update(overrides)
    return card


def _chapter() -> dict:
    return {
        "id": "c1",
        "text": "这一章用于计算来源哈希。",
        "split": "train",
        "quality_tag": "A",
    }


def test_compile_draft_card_outputs_reviewed_formal_candidate():
    from small_model_train.cards.card_compiler import compile_chapter_execution_card

    card = compile_chapter_execution_card(
        draft_card=_draft_card(),
        chapter=_chapter(),
        style_contract=_style_contract(),
        group_id="group-c1",
        split="train",
    )

    assert card["card_id"] == "card-c1-v1"
    assert card["chapter_id"] == "c1"
    assert card["card_status"] == "reviewed"
    assert card["style_contract_id"] == "contract-v1"
    assert card["hard_constraints"]["must_include"] == ["旧仓库", "加钱"]
    assert card["execution_plan"]["conflict_beat"] == "周衡压价。"
    assert card["creative_space"]["allowed_scene_expansion"]


def test_compile_rejects_abstract_only_card():
    from small_model_train.cards.card_compiler import compile_chapter_execution_card

    with pytest.raises(ValueError, match="abstract-only"):
        compile_chapter_execution_card(
            draft_card=_draft_card(
                chapter_goal="节奏紧凑，写得爽一点，减少 AI 味。",
                conflict_beat="",
                payoff_beat="",
                must_include=[],
                ending_hook="",
            ),
            chapter=_chapter(),
            style_contract=_style_contract(),
            group_id="group-c1",
            split="train",
        )


def test_render_chapter_execution_input_uses_style_contract_and_formal_sections():
    from small_model_train.cards.card_compiler import compile_chapter_execution_card
    from small_model_train.cards.card_renderer import render_chapter_execution_input

    contract = _style_contract()
    formal_card = compile_chapter_execution_card(
        draft_card=_draft_card(),
        chapter=_chapter(),
        style_contract=contract,
        group_id="group-c1",
        split="train",
    )

    rendered = render_chapter_execution_input(formal_card, contract)

    assert "【风格契约】" in rendered
    assert contract["prompt_rules"]["style_contract_text"] in rendered
    assert "【本章目标】\n林默进入旧仓库并完成谈判。" in rendered
    assert "【创作自由】" in rendered
    assert "这一章用于计算来源哈希" not in rendered


def test_render_chapter_execution_input_includes_all_formal_negative_constraints():
    from small_model_train.cards.card_compiler import compile_chapter_execution_card
    from small_model_train.cards.card_renderer import render_chapter_execution_input
    from small_model_train.schemas.chapter_execution_card import canonical_card_sha256

    contract = _style_contract()
    formal_card = compile_chapter_execution_card(
        draft_card=_draft_card(),
        chapter=_chapter(),
        style_contract=contract,
        group_id="group-c1",
        split="train",
    )
    formal_card["hard_constraints"]["forbidden_future_facts"] = ["不要提前揭露幕后老板"]
    formal_card["hard_constraints"]["style_bans"] = ["不要使用总结式旁白"]
    formal_card["card_sha256"] = canonical_card_sha256(formal_card)

    rendered = render_chapter_execution_input(formal_card, contract)

    assert "- 作者说明" in rendered
    assert "- 不要提前揭露幕后老板" in rendered
    assert "- 不要使用总结式旁白" in rendered


def test_compile_chapter_execution_cards_cli_writes_reviewed_cards(tmp_path):
    from small_model_train.style_contract import write_style_contract_asset

    cards_path = tmp_path / "draft_cards.jsonl"
    chapters_path = tmp_path / "chapters.jsonl"
    contract_path = tmp_path / "style_contract.json"
    output_path = tmp_path / "formal_cards.jsonl"
    write_jsonl(cards_path, [_draft_card()])
    write_jsonl(chapters_path, [_chapter()])
    write_style_contract_asset(contract_path, _style_contract())

    result = subprocess.run(
        [
            sys.executable,
            "scripts/compile_chapter_execution_cards.py",
            "--cards",
            str(cards_path),
            "--chapters",
            str(chapters_path),
            "--style-contract-json",
            str(contract_path),
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    rows = read_jsonl(output_path)
    assert rows[0]["card_status"] == "reviewed"
    assert rows[0]["chapter_id"] == "c1"
