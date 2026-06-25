from __future__ import annotations

import json
from pathlib import Path

from small_model_train.io_utils import write_jsonl


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
    style_path.write_text(
        json.dumps({"style_contract_id": "contract-v1"}) + "\n",
        encoding="utf-8",
    )

    manifest = build_dataset_manifest(
        sft_dataset_path=sft_path,
        chapters_path=chapters_path,
        cards_path=cards_path,
        style_contract_path=style_path,
        style_contract={"style_contract_id": "contract-v1", "contract_sha256": "b" * 64},
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
    assert loaded["card_hashes"]["card-c1-v1"] == "a" * 64
