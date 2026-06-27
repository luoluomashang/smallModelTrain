from __future__ import annotations

from pathlib import Path

import pytest


RAW_OUTPUT = "林默没有解释，只把合同推到桌面。岳家的人第一次停住。"


def _record(**overrides) -> dict:
    from small_model_train.schemas.chapter_execution_card import text_sha256

    record = {
        "record_id": "review-c1-001",
        "schema_version": 1,
        "card_id": "card-c1-v1",
        "chapter_id": "c1",
        "style_contract_id": "contract-v1",
        "style_contract_sha256": "a" * 64,
        "source_output_id": "gen-c1",
        "raw_output_sha256": text_sha256(RAW_OUTPUT),
        "reviewer": "author",
        "reviewed_at": "2026-06-27T00:00:00Z",
        "defects": [
            {
                "label": "dialogue_flatness",
                "severity": "minor",
                "evidence_text": "岳家的人第一次停住",
                "evidence_start": RAW_OUTPUT.index("岳家的人第一次停住"),
                "evidence_end": RAW_OUTPUT.index("岳家的人第一次停住") + len("岳家的人第一次停住"),
                "suggested_fix": "补一处更具体的压迫反应。",
            }
        ],
        "overall_acceptance": "accepted_with_minor_edits",
        "notes": "可用。",
    }
    record.update(overrides)
    return record


def test_validate_review_record_accepts_matching_span():
    from small_model_train.review.evidence import validate_review_record

    record = validate_review_record(_record(), raw_output=RAW_OUTPUT)

    assert record["defects"][0]["evidence_text"] == "岳家的人第一次停住"


def test_validate_review_record_rejects_sanitized_only_without_raw_text():
    from small_model_train.review.evidence import validate_review_record

    with pytest.raises(ValueError, match="raw_output is required"):
        validate_review_record(_record(), raw_output="")


def test_validate_review_record_rejects_mismatched_span():
    from small_model_train.review.evidence import validate_review_record

    record = _record()
    record["defects"][0]["evidence_start"] = 0
    record["defects"][0]["evidence_end"] = 2

    with pytest.raises(ValueError, match="evidence span does not match evidence_text"):
        validate_review_record(record, raw_output=RAW_OUTPUT)


def test_resolve_evidence_text_fills_offsets():
    from small_model_train.review.evidence import resolve_evidence_text

    defect = resolve_evidence_text(
        {
            "label": "generic_phrase",
            "severity": "major",
            "evidence_text": "合同推到桌面",
            "suggested_fix": "换成动作链。",
        },
        raw_output=RAW_OUTPUT,
        index=0,
    )

    assert RAW_OUTPUT[defect["evidence_start"] : defect["evidence_end"]] == "合同推到桌面"


def test_read_write_review_records_round_trip(tmp_path: Path):
    from small_model_train.review.evidence import read_review_records, write_review_records

    path = tmp_path / "review_records.jsonl"
    write_review_records(path, [_record()], raw_outputs={"gen-c1": RAW_OUTPUT})

    assert read_review_records(path, raw_outputs={"gen-c1": RAW_OUTPUT})[0]["record_id"] == "review-c1-001"


def test_validate_review_records_reports_first_record_as_one_based():
    from small_model_train.review.evidence import validate_review_records

    record = _record(source_output_id="missing-output")

    with pytest.raises(ValueError, match=r"review record 1:"):
        validate_review_records([record], raw_outputs={"gen-c1": RAW_OUTPUT})
