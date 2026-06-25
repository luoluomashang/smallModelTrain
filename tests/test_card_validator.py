from __future__ import annotations


def _style_contract(approval_status: str = "approved") -> dict:
    from small_model_train.style_contract import build_style_contract_asset

    return build_style_contract_asset(
        style_contract_id="contract-v1",
        approval_status=approval_status,
        source_corpus={
            "path": "chapters.jsonl",
            "sha256": "b" * 64,
            "quality_filter": "quality_tag=A",
            "row_count": 2,
            "selected_rows": 2,
            "split_summary": {"train": 2},
        },
        profile_metrics={
            "chapter_count": 1,
            "avg_dialogue_ratio": 0.1,
            "avg_paragraph_chars": 20,
            "ai_taste": {"phrase_hits": {}, "total_hits": 0, "hits_per_10k_chars": 0},
        },
        created_at="2026-06-25T00:00:00",
    )


def _formal_card(
    chapter_id: str,
    text: str = "这一章用于计算来源哈希。",
    style_contract: dict | None = None,
    **overrides,
) -> dict:
    from small_model_train.schemas.chapter_execution_card import (
        build_chapter_execution_card,
    )

    contract = style_contract or _style_contract()
    card = build_chapter_execution_card(
        card_id=f"card-{chapter_id}-v1",
        chapter_id=chapter_id,
        card_status="approved",
        style_contract_id=contract["style_contract_id"],
        style_contract_sha256=contract["contract_sha256"],
        source_chapter_text=text,
        target_platform="hybrid_fanqie_qidian",
        genre_tags=["xuanhuan"],
        hard_constraints={
            "must_include": ["旧仓库"],
            "must_not_include": ["作者说明"],
            "continuity_facts": ["上一章压力还在"],
            "forbidden_future_facts": ["未来真相"],
            "style_bans": [],
        },
        execution_plan={
            "chapter_goal": "林默进入旧仓库。",
            "conflict_beat": "对手压价。",
            "payoff_beat": "林默反制。",
            "chapter_structure": [
                {"step": 1, "name": "入场", "goal": "交代压力", "estimated_chars": "300"}
            ],
            "character_states": [
                {"name": "林默", "state": "警惕", "speech_style": "短句"}
            ],
            "ending_hook": "箱子响了一下。",
            "target_word_count": "2000-2500中文汉字",
        },
        creative_space={
            "optional_sensory_details": [],
            "optional_dialogue_moves": [],
            "optional_micro_conflicts": [],
            "allowed_scene_expansion": [],
        },
        provenance={
            "source_card_id": chapter_id,
            "compiler_version": "stage5c_v1",
            "created_at": "2026-06-25T00:00:00Z",
            "reviewer": "",
            "reviewed_at": "",
            "review_notes": "",
            "group_id": f"group-{chapter_id}",
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


def test_validate_formal_card_batch_accepts_one_card_per_train_chapter():
    from small_model_train.cards.card_validator import validate_formal_card_batch

    chapters = [{"id": "c1", "text": "这一章用于计算来源哈希。", "split": "train", "quality_tag": "A"}]
    result = validate_formal_card_batch([_formal_card("c1")], chapters, _style_contract())

    assert result["passed"] is True
    assert result["errors"] == []
    assert result["card_by_chapter_id"]["c1"]["card_id"] == "card-c1-v1"


def test_validate_formal_card_batch_rejects_missing_train_chapter_card():
    from small_model_train.cards.card_validator import validate_formal_card_batch

    chapters = [{"id": "c1", "text": "正文", "split": "train", "quality_tag": "A"}]
    result = validate_formal_card_batch([], chapters, _style_contract())

    assert result["passed"] is False
    assert "missing formal card for train chapter: c1" in result["errors"]


def test_validate_formal_card_batch_rejects_duplicate_cards():
    from small_model_train.cards.card_validator import validate_formal_card_batch

    chapters = [{"id": "c1", "text": "这一章用于计算来源哈希。", "split": "train", "quality_tag": "A"}]
    duplicate = _formal_card("c1", card_id="card-c1-v2")
    result = validate_formal_card_batch([_formal_card("c1"), duplicate], chapters, _style_contract())

    assert result["passed"] is False
    assert "duplicate formal cards for chapter c1" in "\n".join(result["errors"])


def test_validate_formal_card_batch_rejects_source_hash_mismatch():
    from small_model_train.cards.card_validator import validate_formal_card_batch

    chapters = [{"id": "c1", "text": "不同正文", "split": "train", "quality_tag": "A"}]
    result = validate_formal_card_batch([_formal_card("c1")], chapters, _style_contract())

    assert result["passed"] is False
    assert "source_chapter_sha256 mismatch: c1" in "\n".join(result["errors"])


def test_validate_formal_card_batch_rejects_future_context_leakage():
    from small_model_train.cards.card_validator import validate_formal_card_batch
    from small_model_train.schemas.chapter_execution_card import canonical_card_sha256

    leaked_fragment = "密室墙后藏着真正账本和旧印章"
    chapters = [
        {"id": "c1", "text": "这一章用于计算来源哈希。", "split": "train", "quality_tag": "A"},
        {
            "id": "c2",
            "text": f"封存章节里才揭示：{leaked_fragment}。",
            "split": "sealed",
            "quality_tag": "A",
        },
    ]
    card = _formal_card("c1")
    card["hard_constraints"]["must_include"].append(leaked_fragment)
    card["card_sha256"] = canonical_card_sha256(card)

    result = validate_formal_card_batch([card], chapters, _style_contract())

    assert result["passed"] is False
    assert "future-context leakage" in "\n".join(result["errors"])


def test_validate_formal_card_batch_rejects_later_train_chapter_leakage():
    from small_model_train.cards.card_validator import validate_formal_card_batch
    from small_model_train.schemas.chapter_execution_card import canonical_card_sha256

    leaked_fragment = "第二章才揭开的暗门机关编号"
    chapters = [
        {"id": "c1", "text": "这一章用于计算来源哈希。", "split": "train", "quality_tag": "A"},
        {
            "id": "c2",
            "text": f"后续章节才会写到：{leaked_fragment}。",
            "split": "train",
            "quality_tag": "A",
        },
    ]
    card = _formal_card("c1")
    card["hard_constraints"]["must_include"].append(leaked_fragment)
    card["card_sha256"] = canonical_card_sha256(card)
    c2_card = _formal_card("c2", text=chapters[1]["text"])

    result = validate_formal_card_batch([card, c2_card], chapters, _style_contract())

    assert result["passed"] is False
    assert "future-context leakage" in "\n".join(result["errors"])


def test_validate_formal_card_batch_rejects_pending_review_style_contract():
    from small_model_train.cards.card_validator import validate_formal_card_batch

    contract = _style_contract("pending_review")
    chapters = [{"id": "c1", "text": "这一章用于计算来源哈希。", "split": "train", "quality_tag": "A"}]
    card = _formal_card("c1", style_contract=contract)

    result = validate_formal_card_batch([card], chapters, contract)

    assert result["passed"] is False
    assert "style contract" in "\n".join(result["errors"])
    assert "approved or frozen" in "\n".join(result["errors"])


def test_validate_formal_card_batch_rejects_duplicate_card_id_across_chapters():
    from small_model_train.cards.card_validator import validate_formal_card_batch

    shared_card_id = "card-shared-v1"
    c1_text = "这一章用于计算来源哈希。"
    c2_text = "第二章用于计算另一段来源哈希。"
    chapters = [
        {"id": "c1", "text": c1_text, "split": "train", "quality_tag": "A"},
        {"id": "c2", "text": c2_text, "split": "train", "quality_tag": "A"},
    ]
    cards = [
        _formal_card("c1", text=c1_text, card_id=shared_card_id),
        _formal_card("c2", text=c2_text, card_id=shared_card_id),
    ]

    result = validate_formal_card_batch(cards, chapters, _style_contract())

    assert result["passed"] is False
    assert "duplicate formal card id" in "\n".join(result["errors"])
