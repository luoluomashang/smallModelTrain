from __future__ import annotations

from pathlib import Path

import pytest


MODEL_OUTPUT = "林默把合同推过去，对方沉默。"
REVISED_OUTPUT = "林默没有解释，只把合同推到桌面。岳家的人第一次停住。"


def _revision(**overrides) -> dict:
    from small_model_train.schemas.chapter_execution_card import text_sha256

    record = {
        "revision_id": "rev-c1-001",
        "schema_version": 1,
        "card_id": "card-c1-v1",
        "chapter_id": "c1",
        "style_contract_id": "contract-v1",
        "style_contract_sha256": "a" * 64,
        "prompt_sha256": "b" * 64,
        "raw_output_sha256": text_sha256(MODEL_OUTPUT),
        "model_output": MODEL_OUTPUT,
        "revised_output": REVISED_OUTPUT,
        "revision_status": "accepted_with_minor_edits",
        "revision_author": "author",
        "revised_at": "2026-06-27T01:00:00Z",
        "edit_summary": "把解释改成动作和反应。",
        "defect_record_ids": ["review-c1-001"],
        "acceptance_reason": "同剧情更像作者正文。",
    }
    record.update(overrides)
    return record


def test_validate_revision_record_accepts_same_plot_revision():
    from small_model_train.review.revision_records import (
        is_revision_accepted_for_rejection_sampling,
        validate_revision_record,
    )

    record = validate_revision_record(_revision())

    assert record["revision_id"] == "rev-c1-001"
    assert is_revision_accepted_for_rejection_sampling(record) is True


@pytest.mark.parametrize("status", ["rejected", "needs_rewrite"])
def test_rejected_revision_is_not_sft_candidate(status):
    from small_model_train.review.revision_records import (
        is_revision_accepted_for_rejection_sampling,
        validate_revision_record,
    )

    record = validate_revision_record(_revision(revision_status=status))

    assert is_revision_accepted_for_rejection_sampling(record) is False


def test_validate_revision_record_rejects_raw_hash_mismatch():
    from small_model_train.review.revision_records import validate_revision_record

    with pytest.raises(ValueError, match="raw_output_sha256 mismatch"):
        validate_revision_record(_revision(raw_output_sha256="c" * 64))


def test_validate_revision_record_rejects_empty_revised_output():
    from small_model_train.review.revision_records import validate_revision_record

    with pytest.raises(ValueError, match="revised_output"):
        validate_revision_record(_revision(revised_output=""))


def test_validate_revision_record_provenance_accepts_matching_context():
    from small_model_train.review.revision_records import validate_revision_record_provenance

    card = {
        "card_id": "card-c1-v1",
        "chapter_id": "c1",
        "style_contract_id": "contract-v1",
        "style_contract_sha256": "a" * 64,
    }

    record = validate_revision_record_provenance(
        _revision(),
        card=card,
        style_contract_id="contract-v1",
        style_contract_sha256="a" * 64,
        prompt_sha256="b" * 64,
    )

    assert record["revision_id"] == "rev-c1-001"


def test_validate_revision_record_provenance_rejects_card_id_mismatch():
    from small_model_train.review.revision_records import validate_revision_record_provenance

    card = {
        "card_id": "other-card",
        "chapter_id": "c1",
        "style_contract_id": "contract-v1",
        "style_contract_sha256": "a" * 64,
    }

    with pytest.raises(ValueError, match="revision card_id mismatch"):
        validate_revision_record_provenance(
            _revision(),
            card=card,
            style_contract_id="contract-v1",
            style_contract_sha256="a" * 64,
            prompt_sha256="b" * 64,
        )


def test_validate_revision_record_provenance_rejects_prompt_sha256_mismatch():
    from small_model_train.review.revision_records import validate_revision_record_provenance

    card = {
        "card_id": "card-c1-v1",
        "chapter_id": "c1",
        "style_contract_id": "contract-v1",
        "style_contract_sha256": "a" * 64,
    }

    with pytest.raises(ValueError, match="revision prompt_sha256 mismatch"):
        validate_revision_record_provenance(
            _revision(),
            card=card,
            style_contract_id="contract-v1",
            style_contract_sha256="a" * 64,
            prompt_sha256="c" * 64,
        )


def test_read_write_revision_records_round_trip(tmp_path: Path):
    from small_model_train.review.revision_records import read_revision_records, write_revision_records

    path = tmp_path / "revisions.jsonl"
    write_revision_records(path, [_revision()])

    assert read_revision_records(path)[0]["revision_id"] == "rev-c1-001"
