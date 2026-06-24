from __future__ import annotations

import json
from pathlib import Path

import pytest

from small_model_train.style_contract import (
    APPROVED_FORMAL_STATUSES,
    build_style_contract_asset,
    canonical_style_contract_sha256,
    is_contract_approved_for_formal_sft,
    read_style_contract_asset,
    render_style_contract_markdown,
    validate_style_contract_asset,
    write_style_contract_asset,
)


def _profile() -> dict:
    return {
        "chapter_count": 2,
        "avg_chinese_chars": 1200,
        "avg_paragraph_chars": 80,
        "avg_dialogue_ratio": 0.42,
        "chinese_chars": {"min": 1000, "max": 1400, "avg": 1200, "p50": 1000, "p90": 1400},
        "paragraph_chars": {"min": 20, "max": 120, "avg": 80, "p50": 60, "p90": 120},
        "dialogue_ratio": {"min": 0.2, "max": 0.6, "avg": 0.42, "p50": 0.4, "p90": 0.6},
        "sentence_chars": {"min": 5, "max": 30, "avg": 12, "p50": 10, "p90": 30},
        "punctuation_density": {"。": 0.02},
        "ai_taste": {"phrase_hits": {"空气仿佛凝固了": 0}, "total_hits": 0, "hits_per_10k_chars": 0},
        "source_filter": {"total_rows": 2, "selected_rows": 2, "skipped_rows": 0, "quality_filter": "quality_tag=A"},
    }


def _asset(status: str = "pending_review") -> dict:
    return build_style_contract_asset(
        style_contract_id="author_main_v1",
        approval_status=status,
        source_corpus={
            "path": "data_clean/chapters_split.jsonl",
            "sha256": "a" * 64,
            "quality_filter": "quality_tag=A",
            "row_count": 2,
            "selected_rows": 2,
            "split_summary": {"train": 2},
        },
        profile_metrics=_profile(),
        author_notes="",
    )


def test_build_style_contract_asset_defaults_to_hash_bound_pending_review():
    asset = _asset()

    assert asset["schema_version"] == 1
    assert asset["style_contract_id"] == "author_main_v1"
    assert asset["approval_status"] == "pending_review"
    assert len(asset["contract_sha256"]) == 64
    assert canonical_style_contract_sha256(asset) == asset["contract_sha256"]
    assert validate_style_contract_asset(asset) == asset
    assert is_contract_approved_for_formal_sft(asset) is False


@pytest.mark.parametrize("status", sorted(APPROVED_FORMAL_STATUSES))
def test_approved_and_frozen_contracts_are_formal(status: str):
    assert is_contract_approved_for_formal_sft(_asset(status)) is True


@pytest.mark.parametrize("status", ["", "pending", "approved_by_author"])
def test_invalid_approval_status_is_rejected(status: str):
    asset = _asset()
    asset["approval_status"] = status
    asset["contract_sha256"] = canonical_style_contract_sha256(asset)

    with pytest.raises(ValueError, match="approval_status"):
        validate_style_contract_asset(asset)


def test_contract_hash_mismatch_is_rejected():
    asset = _asset()
    asset["prompt_rules"]["output"] = "tampered"

    with pytest.raises(ValueError, match="contract_sha256 mismatch"):
        validate_style_contract_asset(asset)


def test_style_contract_read_write_roundtrip(tmp_path: Path):
    path = tmp_path / "style_contract.json"
    asset = _asset("approved")

    write_style_contract_asset(path, asset)
    loaded = read_style_contract_asset(path)

    assert loaded == asset


def test_render_style_contract_markdown_is_human_reviewable():
    markdown = render_style_contract_markdown(_asset())

    assert "# Style Contract author_main_v1" in markdown
    assert "approval_status: pending_review" in markdown
    assert "只输出正文" in markdown
