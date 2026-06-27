from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from small_model_train.schemas.chapter_execution_card import text_sha256


SCHEMA_VERSION = 1
REVISION_STATUSES = {"accepted", "accepted_with_minor_edits", "rejected", "needs_rewrite"}
ACCEPTED_REVISION_STATUSES = {"accepted", "accepted_with_minor_edits"}
REQUIRED_FIELDS = (
    "revision_id",
    "schema_version",
    "card_id",
    "chapter_id",
    "style_contract_id",
    "style_contract_sha256",
    "prompt_sha256",
    "raw_output_sha256",
    "model_output",
    "revised_output",
    "revision_status",
    "revision_author",
    "revised_at",
    "edit_summary",
    "defect_record_ids",
    "acceptance_reason",
)
STRING_FIELDS = (
    "revision_id",
    "card_id",
    "chapter_id",
    "style_contract_id",
    "model_output",
    "revised_output",
    "revision_author",
    "revised_at",
    "edit_summary",
    "acceptance_reason",
)
SHA256_FIELDS = ("style_contract_sha256", "prompt_sha256", "raw_output_sha256")
LOWER_HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def validate_revision_record(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("revision record must be a JSON object")

    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"{missing[0]} is required")

    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")

    for field in STRING_FIELDS:
        if not isinstance(record[field], str) or not record[field].strip():
            raise ValueError(f"{field} must be a non-empty string")

    for field in SHA256_FIELDS:
        if not _is_lower_hex_sha256(record[field]):
            raise ValueError(f"{field} must be a 64-character lowercase hex string")

    if record["revision_status"] not in REVISION_STATUSES:
        raise ValueError(
            "revision_status must be one of: " + ", ".join(sorted(REVISION_STATUSES))
        )

    if record["raw_output_sha256"] != text_sha256(record["model_output"]):
        raise ValueError("raw_output_sha256 mismatch")

    defect_record_ids = record["defect_record_ids"]
    if not isinstance(defect_record_ids, list):
        raise ValueError("defect_record_ids must be a list")
    if not all(isinstance(record_id, str) and record_id.strip() for record_id in defect_record_ids):
        raise ValueError("defect_record_ids must contain non-empty strings")

    return record


def validate_revision_record_provenance(
    record: dict[str, Any],
    *,
    card: dict[str, Any],
    style_contract_id: str,
    style_contract_sha256: str,
    prompt_sha256: str,
) -> dict[str, Any]:
    validated = validate_revision_record(record)
    if validated["card_id"] != card.get("card_id"):
        raise ValueError("revision card_id mismatch")
    if validated["chapter_id"] != card.get("chapter_id"):
        raise ValueError("revision chapter_id mismatch")
    if validated["style_contract_id"] != style_contract_id:
        raise ValueError("revision style_contract_id mismatch")
    if validated["style_contract_sha256"] != style_contract_sha256:
        raise ValueError("revision style_contract_sha256 mismatch")
    if validated["prompt_sha256"] != prompt_sha256:
        raise ValueError("revision prompt_sha256 mismatch")
    return validated


def is_revision_accepted_for_rejection_sampling(record: dict[str, Any]) -> bool:
    validated = validate_revision_record(record)
    return validated["revision_status"] in ACCEPTED_REVISION_STATUSES


def validate_revision_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        raise ValueError("revision records must be a list")

    validated: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        try:
            validated.append(validate_revision_record(record))
        except ValueError as exc:
            raise ValueError(f"revision record {index}: {exc}") from exc
    return validated


def write_revision_records(path: str | Path, records: list[dict[str, Any]]) -> None:
    validated_records = validate_revision_records(records)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in validated_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_revision_records(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        raise ValueError(f"revision records JSONL not found: {input_path}")

    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{input_path}:{line_number} is not valid JSON") from exc

    return validate_revision_records(records)


def _is_lower_hex_sha256(value: object) -> bool:
    return isinstance(value, str) and LOWER_HEX_SHA256_RE.fullmatch(value) is not None
