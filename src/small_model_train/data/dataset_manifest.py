from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from small_model_train.artifact_manifest import file_sha256, summarize_jsonl_artifact


SCHEMA_VERSION = 1


def build_dataset_manifest(
    *,
    sft_dataset_path: str | Path,
    chapters_path: str | Path,
    cards_path: str | Path,
    style_contract_path: str | Path,
    style_contract: dict[str, Any],
    split_manifest: dict[str, Any],
    card_hashes: dict[str, str],
    chapter_hashes: dict[str, str],
    leakage_report: dict[str, Any],
    near_duplicate_report: list[dict[str, Any]],
    formal_dataset: bool,
) -> dict[str, Any]:
    sft_summary = summarize_jsonl_artifact(
        sft_dataset_path,
        label="sft_dataset",
        validate_sft_dataset_schema=True,
    )
    sft_schema = sft_summary.get("schema", {})
    if not sft_schema.get("valid"):
        errors = sft_schema.get("errors", [])
        error_text = "; ".join(str(error) for error in errors) or "unknown error"
        raise ValueError(f"SFT dataset schema invalid: {error_text}")

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        ),
        "sft_dataset_path": str(sft_dataset_path),
        "sft_dataset_sha256": sft_summary["sha256"],
        "sft_dataset_schema": sft_schema,
        "row_count": sft_summary["row_count"],
        "chapters_path": str(chapters_path),
        "chapters_sha256": file_sha256(chapters_path),
        "cards_path": str(cards_path),
        "cards_sha256": file_sha256(cards_path),
        "style_contract_path": str(style_contract_path),
        "style_contract_file_sha256": file_sha256(style_contract_path),
        "style_contract_id": style_contract["style_contract_id"],
        "style_contract_sha256": style_contract["contract_sha256"],
        "split_manifest": split_manifest,
        "card_hashes": card_hashes,
        "chapter_hashes": chapter_hashes,
        "leakage_report": leakage_report,
        "near_duplicate_report": near_duplicate_report,
        "formal_dataset": bool(formal_dataset),
    }


def write_dataset_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
