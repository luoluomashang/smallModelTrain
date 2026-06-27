from __future__ import annotations

import json
import subprocess
import sys

import pytest

from small_model_train.io_utils import read_jsonl, write_jsonl
from small_model_train.prompt_renderer import SYSTEM_PROMPT
from small_model_train.sft_builder import (
    INSTRUCTION,
    build_formal_sft_rows,
    build_sft_rows,
    render_sft_input,
)

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


def _formal_style_contract_for_stage5c(source_sha256: str = "b" * 64) -> dict:
    from small_model_train.style_contract import build_style_contract_asset

    return build_style_contract_asset(
        style_contract_id="contract-v1",
        approval_status="approved",
        source_corpus={
            "path": "chapters.jsonl",
            "sha256": source_sha256,
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


def _formal_card_for_stage5c(
    chapter_id: str = "c1",
    text: str = "这一章用于计算来源哈希。",
) -> dict:
    from small_model_train.schemas.chapter_execution_card import build_chapter_execution_card

    contract = _formal_style_contract_for_stage5c()
    return build_chapter_execution_card(
        card_id=f"card-{chapter_id}-v1",
        chapter_id=chapter_id,
        card_status="approved",
        style_contract_id=contract["style_contract_id"],
        style_contract_sha256=contract["contract_sha256"],
        source_chapter_text=text,
        target_platform="local",
        genre_tags=["都市"],
        hard_constraints={
            "must_include": ["合同证据"],
            "must_not_include": ["作者说明"],
            "continuity_facts": ["林默刚拿到关键证据。"],
            "forbidden_future_facts": ["不要提前揭露幕后买家。"],
            "style_bans": ["不要使用AI味套话。"],
        },
        execution_plan={
            "chapter_goal": "林默用证据稳住局面。",
            "conflict_beat": "岳家当众施压，逼他交出合同。",
            "payoff_beat": "林默亮出备份，让对方第一次失声。",
            "chapter_structure": [
                {"step": 1, "name": "压迫", "goal": "把林默逼到台前。", "estimated_chars": "300-400"}
            ],
            "character_states": [{"name": "林默", "state": "冷静", "speech_style": "短句"}],
            "ending_hook": "门外响起真正买家的声音。",
            "target_word_count": "2000-2500中文汉字",
        },
        creative_space={
            "optional_sensory_details": ["酒杯轻响"],
            "optional_dialogue_moves": ["短句反问"],
            "optional_micro_conflicts": ["旁观者低声议论"],
            "allowed_scene_expansion": ["补足宴会厅动线"],
        },
        provenance={
            "source_card_id": "",
            "compiler_version": "test",
            "created_at": "2026-01-01T00:00:00Z",
            "reviewer": "qa",
            "reviewed_at": "2026-01-01T00:00:00Z",
            "review_notes": "",
            "group_id": "g1",
            "split": "train",
        },
    )


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


def test_build_formal_sft_rows_requires_one_formal_card_per_train_chapter():
    chapter_text = "这一章用于计算来源哈希。"
    chapters = [{**_train_chapter(), "text": chapter_text}]
    contract = _formal_style_contract_for_stage5c()
    card = _formal_card_for_stage5c(text=chapter_text)

    rows = build_formal_sft_rows([card], chapters, contract)

    assert rows == [
        {
            "instruction": INSTRUCTION,
            "input": rows[0]["input"],
            "output": chapter_text,
        }
    ]
    assert "【创作自由】" in rows[0]["input"]


def test_build_formal_sft_rows_rejects_missing_card():
    chapter_text = "这一章用于计算来源哈希。"
    chapters = [{**_train_chapter(), "text": chapter_text}]
    contract = _formal_style_contract_for_stage5c()

    with pytest.raises(ValueError, match="missing formal card for train chapter: c1"):
        build_formal_sft_rows([], chapters, contract)


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
    contract = _style_contract_asset("approved")

    assert (
        build_sft_rows(
            [card],
            chapters,
            require_approved_cards=True,
            style_contract=contract,
        )
        == []
    )


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
    contract = _style_contract_asset("approved")
    card = _approved_sft_card(
        approval_status=approval_status,
        style_contract_id=contract["style_contract_id"],
        style_contract_sha256=contract["contract_sha256"],
    )

    rows = build_sft_rows(
        [card],
        [_train_chapter()],
        require_approved_cards=True,
        style_contract=contract,
    )

    assert rows[0]["output"] == "正文"


def test_build_sft_dataset_cli_rejects_draft_cards_by_default_and_allows_with_flag(tmp_path):
    from small_model_train.artifact_manifest import file_sha256
    from small_model_train.style_contract import write_style_contract_asset

    cards_path = tmp_path / "cards.jsonl"
    chapters_path = tmp_path / "chapters.jsonl"
    contract_path = tmp_path / "style_contract.json"
    output_path = tmp_path / "sft.jsonl"
    allowed_output_path = tmp_path / "sft_allowed.jsonl"
    write_jsonl(
        cards_path,
        [_approved_sft_card(draft_only=True, approval_status="draft", schema_version=1)],
    )
    write_jsonl(chapters_path, [_train_chapter()])
    contract = _style_contract_asset(
        "approved",
        source_sha256=file_sha256(chapters_path),
    )
    write_style_contract_asset(contract_path, contract)

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
            "--style-contract-json",
            str(contract_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "card_id is required" in result.stderr
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


def test_build_sft_dataset_cli_rejects_dataset_manifest_same_as_output(tmp_path):
    cards_path = tmp_path / "cards.jsonl"
    chapters_path = tmp_path / "chapters.jsonl"
    output_path = tmp_path / "sft_chapter_v1.jsonl"
    write_jsonl(cards_path, [_sft_card()])
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
            "--dataset-manifest-output",
            str(output_path),
            "--allow-draft-cards",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "output path collision" in result.stderr
    assert not output_path.exists()


def test_build_sft_dataset_cli_rejects_dataset_manifest_same_as_dataset_info(tmp_path):
    cards_path = tmp_path / "cards.jsonl"
    chapters_path = tmp_path / "chapters.jsonl"
    output_path = tmp_path / "sft_chapter_v1.jsonl"
    sidecar_path = tmp_path / "dataset_sidecar.json"
    write_jsonl(cards_path, [_sft_card()])
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
            "--dataset-info-output",
            str(sidecar_path),
            "--dataset-manifest-output",
            str(sidecar_path),
            "--allow-draft-cards",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "output path collision" in result.stderr
    assert not output_path.exists()
    assert not sidecar_path.exists()


def _style_contract_asset(
    status: str = "approved",
    source_sha256: str = "b" * 64,
) -> dict:
    from small_model_train.style_contract import build_style_contract_asset

    return build_style_contract_asset(
        style_contract_id="contract-v1",
        approval_status=status,
        source_corpus={
            "path": "chapters.jsonl",
            "sha256": source_sha256,
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


def test_build_sft_rows_rejects_pending_style_contract_in_formal_mode():
    card = _approved_sft_card()
    contract = _style_contract_asset("pending_review")

    with pytest.raises(ValueError, match="approved or frozen"):
        build_sft_rows(
            [card],
            [_train_chapter()],
            require_approved_cards=True,
            style_contract=contract,
        )


def test_build_sft_rows_rejects_pending_style_contract_without_trainable_candidates():
    card = _approved_sft_card()
    contract = _style_contract_asset("pending_review")

    with pytest.raises(ValueError, match="approved or frozen"):
        build_sft_rows(
            [card],
            [{**_train_chapter(), "split": "eval"}],
            require_approved_cards=True,
            style_contract=contract,
        )


def test_build_sft_rows_rejects_malformed_style_contract_without_trainable_candidates():
    card = _approved_sft_card()
    contract = _style_contract_asset("approved")
    del contract["contract_sha256"]

    with pytest.raises(ValueError, match="missing required fields: contract_sha256"):
        build_sft_rows(
            [card],
            [{**_train_chapter(), "split": "eval"}],
            require_approved_cards=True,
            style_contract=contract,
        )


def test_build_sft_rows_rejects_style_contract_hash_mismatch():
    contract = _style_contract_asset("approved")
    card = _approved_sft_card(style_contract_sha256="c" * 64)

    with pytest.raises(ValueError, match="style_contract_sha256 mismatch"):
        build_sft_rows(
            [card],
            [_train_chapter()],
            require_approved_cards=True,
            style_contract=contract,
        )


def test_build_sft_rows_accepts_matching_approved_style_contract():
    contract = _style_contract_asset("approved")
    card = _approved_sft_card(
        style_contract_id=contract["style_contract_id"],
        style_contract_sha256=contract["contract_sha256"],
    )

    rows = build_sft_rows(
        [card],
        [_train_chapter()],
        require_approved_cards=True,
        style_contract=contract,
    )

    assert rows[0]["output"] == "正文"


def test_build_sft_rows_accepts_matching_frozen_style_contract():
    contract = _style_contract_asset("frozen")
    card = _approved_sft_card(
        style_contract_id=contract["style_contract_id"],
        style_contract_sha256=contract["contract_sha256"],
    )

    rows = build_sft_rows(
        [card],
        [_train_chapter()],
        require_approved_cards=True,
        style_contract=contract,
    )

    assert rows[0]["output"] == "正文"


def test_build_sft_rows_requires_style_contract_even_without_trainable_candidates():
    card = _approved_sft_card()

    with pytest.raises(ValueError, match="style contract JSON is required for formal SFT"):
        build_sft_rows(
            [card],
            [{**_train_chapter(), "split": "eval"}],
            require_approved_cards=True,
        )


def test_build_sft_dataset_cli_requires_style_contract_json_for_formal(tmp_path):
    cards_path = tmp_path / "cards.jsonl"
    chapters_path = tmp_path / "chapters.jsonl"
    output_path = tmp_path / "sft.jsonl"
    card = _approved_sft_card()
    write_jsonl(cards_path, [card])
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
    assert "style contract JSON is required for formal SFT" in result.stderr
    assert not output_path.exists()


def test_build_sft_dataset_cli_rejects_missing_style_contract_json_path(tmp_path):
    cards_path = tmp_path / "cards.jsonl"
    chapters_path = tmp_path / "chapters.jsonl"
    contract_path = tmp_path / "missing_style_contract.json"
    output_path = tmp_path / "sft.jsonl"
    card = _approved_sft_card()
    write_jsonl(cards_path, [card])
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
            "--style-contract-json",
            str(contract_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "style contract JSON not found" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output_path.exists()


def test_build_sft_dataset_cli_formal_cards_write_manifest(tmp_path):
    from small_model_train.artifact_manifest import file_sha256
    from small_model_train.schemas.chapter_execution_card import canonical_card_sha256, text_sha256
    from small_model_train.style_contract import write_style_contract_asset

    cards_path = tmp_path / "cards.jsonl"
    chapters_path = tmp_path / "chapters.jsonl"
    contract_path = tmp_path / "style_contract.json"
    output_path = tmp_path / "sft.jsonl"
    manifest_path = tmp_path / "dataset_manifest.json"
    chapter_text = "这一章用于计算来源哈希。"
    chapters = [{**_train_chapter(), "text": chapter_text}]
    write_jsonl(chapters_path, chapters)
    contract = _formal_style_contract_for_stage5c(
        source_sha256=file_sha256(chapters_path),
    )
    card = _formal_card_for_stage5c(text=chapter_text)
    card["style_contract_sha256"] = contract["contract_sha256"]
    card["card_sha256"] = canonical_card_sha256(card)
    write_jsonl(cards_path, [card])
    write_style_contract_asset(contract_path, contract)

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
            "--style-contract-json",
            str(contract_path),
            "--dataset-manifest-output",
            str(manifest_path),
        ],
        check=True,
    )

    rows = read_jsonl(output_path)
    assert rows[0]["output"] == chapter_text
    assert "【创作自由】" in rows[0]["input"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["formal_dataset"] is True
    assert manifest["row_count"] == 1
    assert manifest["style_contract_sha256"] == contract["contract_sha256"]
    assert manifest["split_manifest"]["counts"] == {"train": 1}
    assert manifest["card_hashes"] == {card["card_id"]: card["card_sha256"]}
    assert manifest["chapter_hashes"] == {"c1": text_sha256(chapter_text)}


def test_build_sft_dataset_cli_accepts_matching_approved_contract(tmp_path):
    from small_model_train.artifact_manifest import file_sha256
    from small_model_train.schemas.chapter_execution_card import canonical_card_sha256
    from small_model_train.style_contract import write_style_contract_asset

    cards_path = tmp_path / "cards.jsonl"
    chapters_path = tmp_path / "chapters.jsonl"
    contract_path = tmp_path / "style_contract.json"
    output_path = tmp_path / "sft.jsonl"
    chapter = _train_chapter()
    write_jsonl(chapters_path, [chapter])
    contract = _style_contract_asset(
        "approved",
        source_sha256=file_sha256(chapters_path),
    )
    card = _formal_card_for_stage5c(text=chapter["text"])
    card["style_contract_sha256"] = contract["contract_sha256"]
    card["card_sha256"] = canonical_card_sha256(card)
    write_jsonl(cards_path, [card])
    write_style_contract_asset(contract_path, contract)

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
            "--style-contract-json",
            str(contract_path),
        ],
        check=True,
    )

    assert read_jsonl(output_path)[0]["output"] == "正文"


def test_build_sft_dataset_cli_rejects_chapters_hash_mismatch(tmp_path):
    from small_model_train.style_contract import write_style_contract_asset

    contract = _style_contract_asset("approved", source_sha256="0" * 64)
    cards_path = tmp_path / "cards.jsonl"
    chapters_path = tmp_path / "chapters.jsonl"
    contract_path = tmp_path / "style_contract.json"
    output_path = tmp_path / "sft.jsonl"
    card = _approved_sft_card(
        style_contract_id=contract["style_contract_id"],
        style_contract_sha256=contract["contract_sha256"],
    )
    write_jsonl(cards_path, [card])
    write_jsonl(chapters_path, [_train_chapter()])
    write_style_contract_asset(contract_path, contract)

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
            "--style-contract-json",
            str(contract_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "chapters sha256 does not match style contract source_corpus.sha256" in (
        result.stderr
    )
    assert "Traceback" not in result.stderr
    assert not output_path.exists()


def test_build_sft_dataset_cli_allow_draft_with_style_contract_uses_legacy_cards(tmp_path):
    from small_model_train.artifact_manifest import file_sha256
    from small_model_train.style_contract import write_style_contract_asset

    cards_path = tmp_path / "cards.jsonl"
    chapters_path = tmp_path / "chapters.jsonl"
    contract_path = tmp_path / "style_contract.json"
    output_path = tmp_path / "sft.jsonl"
    implicit_manifest_path = tmp_path / "dataset_manifest.json"
    write_jsonl(cards_path, [_approved_sft_card(draft_only=True, approval_status="draft")])
    write_jsonl(chapters_path, [_train_chapter()])
    contract = _style_contract_asset(
        "approved",
        source_sha256=file_sha256(chapters_path),
    )
    write_style_contract_asset(contract_path, contract)

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
            "--style-contract-json",
            str(contract_path),
        ],
        check=True,
    )

    rows = read_jsonl(output_path)
    assert len(rows) == 1
    assert rows[0]["output"] == "正文"
    assert not implicit_manifest_path.exists()


def test_build_sft_dataset_cli_rejects_manifest_in_draft_mode(tmp_path):
    from small_model_train.artifact_manifest import file_sha256
    from small_model_train.style_contract import write_style_contract_asset

    cards_path = tmp_path / "cards.jsonl"
    chapters_path = tmp_path / "chapters.jsonl"
    contract_path = tmp_path / "style_contract.json"
    output_path = tmp_path / "sft.jsonl"
    manifest_path = tmp_path / "dataset_manifest.json"
    write_jsonl(cards_path, [_approved_sft_card(draft_only=True, approval_status="draft")])
    write_jsonl(chapters_path, [_train_chapter()])
    contract = _style_contract_asset(
        "approved",
        source_sha256=file_sha256(chapters_path),
    )
    write_style_contract_asset(contract_path, contract)

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
            "--allow-draft-cards",
            "--style-contract-json",
            str(contract_path),
            "--dataset-manifest-output",
            str(manifest_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "dataset manifest requires formal SFT mode" in result.stderr
    assert not output_path.exists()
    assert not manifest_path.exists()


def test_build_formal_sft_rows_rejects_duplicate_train_chapter_id_before_rows():
    contract = _formal_style_contract_for_stage5c()
    chapters = [
        {"id": "c1", "text": "第一条重复章节正文。", "split": "train", "quality_tag": "A"},
        {"id": "c1", "text": "这一章用于计算来源哈希。", "split": "train", "quality_tag": "A"},
    ]
    card = _formal_card_for_stage5c(text="这一章用于计算来源哈希。")

    with pytest.raises(ValueError, match="duplicate trainable chapter id: c1"):
        build_formal_sft_rows([card], chapters, contract)
