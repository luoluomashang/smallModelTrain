from __future__ import annotations

import json
from pathlib import Path

import pytest


STYLE_HASH = "a" * 64
SOURCE_HASH = "b" * 64


def _card(**overrides) -> dict:
    from small_model_train.schemas.chapter_execution_card import (
        build_chapter_execution_card,
    )

    card = build_chapter_execution_card(
        card_id="card-c1-v1",
        chapter_id="c1",
        card_status="approved",
        style_contract_id="contract-v1",
        style_contract_sha256=STYLE_HASH,
        source_chapter_text="这一章用于计算来源哈希。",
        target_platform="hybrid_fanqie_qidian",
        genre_tags=["xuanhuan", "system"],
        hard_constraints={
            "must_include": ["旧仓库", "加钱"],
            "must_not_include": ["作者说明"],
            "continuity_facts": ["林默上一章已经接下委托"],
            "forbidden_future_facts": ["最终幕后身份"],
            "style_bans": ["不要输出提纲"],
        },
        execution_plan={
            "chapter_goal": "林默进入旧仓库并完成谈判。",
            "conflict_beat": "周衡压价并试探林默底线。",
            "payoff_beat": "林默用箱子异常逼对方退让。",
            "chapter_structure": [
                {"step": 1, "name": "入场", "goal": "交代压力", "estimated_chars": "300-500"}
            ],
            "character_states": [
                {"name": "林默", "state": "警惕", "speech_style": "短句"}
            ],
            "ending_hook": "箱子自己响了一下。",
            "target_word_count": "2000-2500中文汉字",
        },
        creative_space={
            "optional_sensory_details": ["雨声", "铁锈味"],
            "optional_dialogue_moves": ["短暂停顿"],
            "optional_micro_conflicts": ["临时加价"],
            "allowed_scene_expansion": ["仓库外等待的人群"],
        },
        provenance={
            "source_card_id": "draft-c1",
            "compiler_version": "stage5c_v1",
            "created_at": "2026-06-25T00:00:00Z",
            "reviewer": "author",
            "reviewed_at": "2026-06-25T01:00:00Z",
            "review_notes": "批准。",
            "group_id": "group-c1",
            "split": "train",
        },
    )
    card.update(overrides)
    if overrides:
        from small_model_train.schemas.chapter_execution_card import (
            canonical_card_sha256,
        )

        card["card_sha256"] = canonical_card_sha256(card)
    return card


def test_validate_chapter_execution_card_accepts_valid_card():
    from small_model_train.schemas.chapter_execution_card import (
        is_card_approved_for_formal_sft,
        validate_chapter_execution_card,
    )

    card = validate_chapter_execution_card(_card())

    assert card["schema_version"] == 1
    assert len(card["card_sha256"]) == 64
    assert len(card["source_chapter_sha256"]) == 64
    assert is_card_approved_for_formal_sft(card) is True


def test_validate_chapter_execution_card_accepts_pending_review_metadata():
    from small_model_train.schemas.chapter_execution_card import (
        canonical_card_sha256,
        validate_chapter_execution_card,
    )

    card = _card()
    card["provenance"]["reviewer"] = ""
    card["provenance"]["reviewed_at"] = ""
    card["provenance"]["review_notes"] = ""
    card["card_sha256"] = canonical_card_sha256(card)

    assert validate_chapter_execution_card(card)["provenance"]["reviewer"] == ""


def test_utc_now_iso_uses_z_suffix():
    from small_model_train.schemas.chapter_execution_card import utc_now_iso

    assert utc_now_iso().endswith("Z")


def test_card_hash_excludes_card_sha256():
    from small_model_train.schemas.chapter_execution_card import canonical_card_sha256

    card = _card()
    original = card["card_sha256"]
    card["card_sha256"] = "0" * 64

    assert canonical_card_sha256(card) == original


@pytest.mark.parametrize("status", ["draft", "reviewed", "rejected"])
def test_non_formal_status_is_not_formal(status: str):
    from small_model_train.schemas.chapter_execution_card import (
        is_card_approved_for_formal_sft,
    )

    assert is_card_approved_for_formal_sft(_card(card_status=status)) is False


def test_invalid_status_is_rejected():
    from small_model_train.schemas.chapter_execution_card import (
        validate_chapter_execution_card,
    )

    with pytest.raises(ValueError, match="card_status"):
        validate_chapter_execution_card(_card(card_status="pending"))


def test_missing_nested_required_field_is_rejected():
    from small_model_train.schemas.chapter_execution_card import (
        canonical_card_sha256,
        validate_chapter_execution_card,
    )

    card = _card()
    del card["execution_plan"]["chapter_goal"]
    card["card_sha256"] = canonical_card_sha256(card)

    with pytest.raises(ValueError, match="execution_plan.chapter_goal"):
        validate_chapter_execution_card(card)


def test_hash_mismatch_is_rejected():
    from small_model_train.schemas.chapter_execution_card import (
        validate_chapter_execution_card,
    )

    card = _card()
    card["execution_plan"]["chapter_goal"] = "偷偷改目标。"

    with pytest.raises(ValueError, match="card_sha256 mismatch"):
        validate_chapter_execution_card(card)


def test_read_and_write_chapter_execution_cards_round_trip(tmp_path: Path):
    from small_model_train.schemas.chapter_execution_card import (
        read_chapter_execution_cards,
        write_chapter_execution_cards,
    )

    path = tmp_path / "cards.jsonl"
    write_chapter_execution_cards(path, [_card()])

    assert read_chapter_execution_cards(path)[0]["card_id"] == "card-c1-v1"
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["card_id"] == "card-c1-v1"
