from __future__ import annotations

import json
from pathlib import Path

import pytest

from small_model_train.artifact_manifest import file_sha256
from small_model_train.io_utils import write_jsonl
from small_model_train.style_contract import build_style_contract_asset, write_style_contract_asset


def test_build_dataset_manifest_records_hashes_and_formal_flag(tmp_path: Path):
    from small_model_train.data.dataset_manifest import (
        build_dataset_manifest,
        write_dataset_manifest,
    )

    sft_path = tmp_path / "sft.jsonl"
    chapters_path = tmp_path / "chapters.jsonl"
    cards_path = tmp_path / "cards.jsonl"
    style_path = tmp_path / "style.json"
    manifest_path = tmp_path / "manifest.json"

    write_jsonl(sft_path, [{"instruction": "i", "input": "x", "output": "y"}])
    write_jsonl(chapters_path, [{"id": "c1", "text": "正文", "split": "train"}])
    write_jsonl(
        cards_path,
        [{"card_id": "card-c1-v1", "chapter_id": "c1", "card_sha256": "a" * 64}],
    )
    style_contract = _style_contract_asset(chapters_path)
    write_style_contract_asset(style_path, style_contract)

    manifest = build_dataset_manifest(
        sft_dataset_path=sft_path,
        chapters_path=chapters_path,
        cards_path=cards_path,
        style_contract_path=style_path,
        style_contract=style_contract,
        split_manifest={"counts": {"train": 1}},
        card_hashes={"card-c1-v1": "a" * 64},
        chapter_hashes={"c1": "c" * 64},
        leakage_report={"passed": True, "errors": []},
        near_duplicate_report=[],
        formal_dataset=True,
    )
    write_dataset_manifest(manifest_path, manifest)

    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded["row_count"] == 1
    assert loaded["formal_dataset"] is True
    assert loaded["style_contract_id"] == "contract-v1"
    assert loaded["sft_dataset_sha256"] == file_sha256(sft_path)
    assert loaded["chapters_sha256"] == file_sha256(chapters_path)
    assert loaded["cards_sha256"] == file_sha256(cards_path)
    assert loaded["style_contract_file_sha256"] == file_sha256(style_path)
    assert loaded["style_contract_sha256"] == style_contract["contract_sha256"]
    assert loaded["sft_dataset_schema"]["valid"] is True
    assert loaded["card_hashes"]["card-c1-v1"] == "a" * 64


def test_build_dataset_manifest_rejects_invalid_sft_schema(tmp_path: Path):
    from small_model_train.data.dataset_manifest import build_dataset_manifest

    sft_path = tmp_path / "sft.jsonl"
    chapters_path = tmp_path / "chapters.jsonl"
    cards_path = tmp_path / "cards.jsonl"
    style_path = tmp_path / "style.json"

    write_jsonl(sft_path, [{}])
    write_jsonl(chapters_path, [{"id": "c1", "text": "正文", "split": "train"}])
    write_jsonl(
        cards_path,
        [{"card_id": "card-c1-v1", "chapter_id": "c1", "card_sha256": "a" * 64}],
    )
    style_contract = _style_contract_asset(chapters_path)
    write_style_contract_asset(style_path, style_contract)

    with pytest.raises(ValueError, match="SFT dataset schema invalid"):
        build_dataset_manifest(
            sft_dataset_path=sft_path,
            chapters_path=chapters_path,
            cards_path=cards_path,
            style_contract_path=style_path,
            style_contract=style_contract,
            split_manifest={"counts": {"train": 1}},
            card_hashes={"card-c1-v1": "a" * 64},
            chapter_hashes={"c1": "c" * 64},
            leakage_report={"passed": True, "errors": []},
            near_duplicate_report=[],
            formal_dataset=True,
        )


def test_build_dataset_manifest_rejects_invalid_style_contract_file(tmp_path: Path):
    from small_model_train.data.dataset_manifest import build_dataset_manifest

    sft_path = tmp_path / "sft.jsonl"
    chapters_path = tmp_path / "chapters.jsonl"
    cards_path = tmp_path / "cards.jsonl"
    style_path = tmp_path / "style.json"

    write_jsonl(sft_path, [{"instruction": "i", "input": "x", "output": "y"}])
    write_jsonl(chapters_path, [{"id": "c1", "text": "正文", "split": "train"}])
    write_jsonl(
        cards_path,
        [{"card_id": "card-c1-v1", "chapter_id": "c1", "card_sha256": "a" * 64}],
    )
    style_contract = _style_contract_asset(chapters_path)
    style_path.write_text(
        json.dumps({"style_contract_id": "contract-v1"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="StyleContract"):
        build_dataset_manifest(
            sft_dataset_path=sft_path,
            chapters_path=chapters_path,
            cards_path=cards_path,
            style_contract_path=style_path,
            style_contract=style_contract,
            split_manifest={"counts": {"train": 1}},
            card_hashes={"card-c1-v1": "a" * 64},
            chapter_hashes={"c1": "c" * 64},
            leakage_report={"passed": True, "errors": []},
            near_duplicate_report=[],
            formal_dataset=True,
        )


def test_build_dataset_manifest_rejects_style_contract_provenance_mismatch(tmp_path: Path):
    from small_model_train.data.dataset_manifest import build_dataset_manifest

    sft_path = tmp_path / "sft.jsonl"
    chapters_path = tmp_path / "chapters.jsonl"
    cards_path = tmp_path / "cards.jsonl"
    style_path = tmp_path / "style.json"

    write_jsonl(sft_path, [{"instruction": "i", "input": "x", "output": "y"}])
    write_jsonl(chapters_path, [{"id": "c1", "text": "正文", "split": "train"}])
    write_jsonl(
        cards_path,
        [{"card_id": "card-c1-v1", "chapter_id": "c1", "card_sha256": "a" * 64}],
    )
    file_style_contract = _style_contract_asset(chapters_path)
    provided_style_contract = _style_contract_asset(
        chapters_path,
        style_contract_id="contract-v2",
    )
    write_style_contract_asset(style_path, file_style_contract)

    with pytest.raises(ValueError, match="StyleContract.*mismatch"):
        build_dataset_manifest(
            sft_dataset_path=sft_path,
            chapters_path=chapters_path,
            cards_path=cards_path,
            style_contract_path=style_path,
            style_contract=provided_style_contract,
            split_manifest={"counts": {"train": 1}},
            card_hashes={"card-c1-v1": "a" * 64},
            chapter_hashes={"c1": "c" * 64},
            leakage_report={"passed": True, "errors": []},
            near_duplicate_report=[],
            formal_dataset=True,
        )


def _style_contract_asset(
    chapters_path: Path,
    *,
    style_contract_id: str = "contract-v1",
) -> dict:
    return build_style_contract_asset(
        style_contract_id=style_contract_id,
        approval_status="approved",
        source_corpus={
            "path": str(chapters_path),
            "sha256": file_sha256(chapters_path),
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
