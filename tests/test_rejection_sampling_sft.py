from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from small_model_train.io_utils import read_jsonl, write_jsonl


MODEL_OUTPUT = "林默把合同推过去，对方沉默。"
REVISED_OUTPUT = "林默没有解释，只把合同推到桌面。岳家的人第一次停住。"


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


def _formal_card(contract: dict | None = None, *, card_status: str = "approved") -> dict:
    from small_model_train.schemas.chapter_execution_card import build_chapter_execution_card

    contract = contract or _style_contract()
    return build_chapter_execution_card(
        card_id="card-c1-v1",
        chapter_id="c1",
        card_status=card_status,
        style_contract_id=contract["style_contract_id"],
        style_contract_sha256=contract["contract_sha256"],
        source_chapter_text="这一章用于计算来源哈希。",
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


def _revision(card: dict, contract: dict, **overrides) -> dict:
    from small_model_train.cards.card_renderer import render_chapter_execution_input
    from small_model_train.schemas.chapter_execution_card import text_sha256

    record = {
        "revision_id": "rev-c1-001",
        "schema_version": 1,
        "card_id": card["card_id"],
        "chapter_id": card["chapter_id"],
        "style_contract_id": contract["style_contract_id"],
        "style_contract_sha256": contract["contract_sha256"],
        "prompt_sha256": text_sha256(render_chapter_execution_input(card, contract)),
        "raw_output_sha256": text_sha256(MODEL_OUTPUT),
        "model_output": MODEL_OUTPUT,
        "revised_output": REVISED_OUTPUT,
        "revision_status": "accepted",
        "revision_author": "author",
        "revised_at": "2026-06-27T01:00:00Z",
        "edit_summary": "把解释改成动作和反应。",
        "defect_record_ids": ["review-c1-001"],
        "acceptance_reason": "同剧情更像作者正文。",
    }
    record.update(overrides)
    return record


def test_build_rejection_sampling_sft_rows_uses_formal_prompt_and_revised_output():
    from small_model_train.cards.card_renderer import render_chapter_execution_input
    from small_model_train.review.rejection_sampling import build_rejection_sampling_sft_rows
    from small_model_train.schemas.chapter_execution_card import text_sha256
    from small_model_train.sft_builder import INSTRUCTION

    contract = _style_contract()
    card = _formal_card(contract)

    rows = build_rejection_sampling_sft_rows([_revision(card, contract)], [card], contract)

    assert rows == [
        {
            "instruction": INSTRUCTION,
            "input": render_chapter_execution_input(card, contract),
            "output": REVISED_OUTPUT,
            "revision_id": "rev-c1-001",
            "card_id": "card-c1-v1",
            "chapter_id": "c1",
            "style_contract_sha256": contract["contract_sha256"],
            "raw_output_sha256": text_sha256(MODEL_OUTPUT),
        }
    ]


def test_build_rejection_sampling_sft_rows_rejects_unaccepted_revision():
    from small_model_train.review.rejection_sampling import build_rejection_sampling_sft_rows

    contract = _style_contract()
    card = _formal_card(contract)

    with pytest.raises(ValueError, match="revision_status must be accepted"):
        build_rejection_sampling_sft_rows(
            [_revision(card, contract, revision_status="rejected")],
            [card],
            contract,
        )


def test_build_rejection_sampling_sft_rows_rejects_reviewed_card_status():
    from small_model_train.review.rejection_sampling import build_rejection_sampling_sft_rows

    contract = _style_contract()
    reviewed_card = _formal_card(contract, card_status="reviewed")

    with pytest.raises(ValueError, match="formal card status must be approved or frozen"):
        build_rejection_sampling_sft_rows(
            [_revision(reviewed_card, contract)],
            [reviewed_card],
            contract,
        )


def test_build_rejection_sampling_sft_rows_rejects_pending_style_contract():
    from small_model_train.review.rejection_sampling import build_rejection_sampling_sft_rows
    from small_model_train.style_contract import canonical_style_contract_sha256

    pending_contract = _style_contract()
    pending_contract["approval_status"] = "pending_review"
    pending_contract["contract_sha256"] = canonical_style_contract_sha256(pending_contract)
    card = _formal_card(pending_contract)

    with pytest.raises(ValueError, match="style contract approval_status must be approved or frozen"):
        build_rejection_sampling_sft_rows(
            [_revision(card, pending_contract)],
            [card],
            pending_contract,
        )


def test_build_rejection_sampling_sft_rows_rejects_prompt_hash_mismatch():
    from small_model_train.review.rejection_sampling import build_rejection_sampling_sft_rows

    contract = _style_contract()
    card = _formal_card(contract)

    with pytest.raises(ValueError, match="revision prompt_sha256 mismatch"):
        build_rejection_sampling_sft_rows(
            [_revision(card, contract, prompt_sha256="c" * 64)],
            [card],
            contract,
        )


def test_build_rejection_sampling_sft_cli_writes_jsonl(tmp_path: Path):
    from small_model_train.style_contract import write_style_contract_asset

    contract = _style_contract()
    card = _formal_card(contract)
    revisions_path = tmp_path / "revisions.jsonl"
    cards_path = tmp_path / "cards.jsonl"
    contract_path = tmp_path / "style_contract.json"
    output_path = tmp_path / "rejection_sft.jsonl"
    write_jsonl(revisions_path, [_revision(card, contract)])
    write_jsonl(cards_path, [card])
    write_style_contract_asset(contract_path, contract)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_rejection_sampling_sft.py",
            "--revisions",
            str(revisions_path),
            "--cards",
            str(cards_path),
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
    assert f"wrote 1 rejection-sampling SFT rows to {output_path}" in result.stdout
    rows = read_jsonl(output_path)
    assert rows[0]["revision_id"] == "rev-c1-001"
    assert rows[0]["output"] == REVISED_OUTPUT


def test_build_rejection_sampling_sft_cli_fails_when_revisions_jsonl_missing(tmp_path: Path):
    from small_model_train.style_contract import write_style_contract_asset

    contract = _style_contract()
    card = _formal_card(contract)
    revisions_path = tmp_path / "missing_revisions.jsonl"
    cards_path = tmp_path / "cards.jsonl"
    contract_path = tmp_path / "style_contract.json"
    output_path = tmp_path / "rejection_sft.jsonl"
    write_jsonl(cards_path, [card])
    write_style_contract_asset(contract_path, contract)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_rejection_sampling_sft.py",
            "--revisions",
            str(revisions_path),
            "--cards",
            str(cards_path),
            "--style-contract-json",
            str(contract_path),
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "revisions JSONL not found" in result.stderr
    assert not output_path.exists()


def test_build_rejection_sampling_sft_cli_fails_when_cards_jsonl_missing(tmp_path: Path):
    from small_model_train.style_contract import write_style_contract_asset

    contract = _style_contract()
    card = _formal_card(contract)
    revisions_path = tmp_path / "revisions.jsonl"
    cards_path = tmp_path / "missing_cards.jsonl"
    contract_path = tmp_path / "style_contract.json"
    output_path = tmp_path / "rejection_sft.jsonl"
    write_jsonl(revisions_path, [_revision(card, contract)])
    write_style_contract_asset(contract_path, contract)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_rejection_sampling_sft.py",
            "--revisions",
            str(revisions_path),
            "--cards",
            str(cards_path),
            "--style-contract-json",
            str(contract_path),
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "cards JSONL not found" in result.stderr
    assert not output_path.exists()
