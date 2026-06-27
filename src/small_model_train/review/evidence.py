from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from small_model_train.review.style_defects import validate_style_defect, validate_style_defects
from small_model_train.schemas.chapter_execution_card import text_sha256


SCHEMA_VERSION = 1
ACCEPTANCE_STATUSES = {"accepted", "accepted_with_minor_edits", "rejected", "needs_rewrite"}
REQUIRED_FIELDS = (
    "record_id",
    "schema_version",
    "card_id",
    "chapter_id",
    "style_contract_id",
    "style_contract_sha256",
    "source_output_id",
    "raw_output_sha256",
    "reviewer",
    "reviewed_at",
    "defects",
    "overall_acceptance",
    "notes",
)
STRING_FIELDS = (
    "record_id",
    "card_id",
    "chapter_id",
    "style_contract_id",
    "style_contract_sha256",
    "source_output_id",
    "raw_output_sha256",
    "reviewer",
    "reviewed_at",
    "notes",
)


def resolve_evidence_text(defect: dict[str, Any], *, raw_output: str, index: int) -> dict[str, Any]:
    if not isinstance(raw_output, str) or not raw_output:
        raise ValueError("raw_output is required")
    if not isinstance(defect, dict):
        raise ValueError(f"defects[{index}] must be a JSON object")

    evidence_text = defect.get("evidence_text")
    if not isinstance(evidence_text, str) or not evidence_text.strip():
        raise ValueError(f"defects[{index}].evidence_text must be a non-empty string")

    start = raw_output.find(evidence_text)
    if start < 0:
        raise ValueError(f"defects[{index}].evidence_text was not found in raw_output")

    resolved = dict(defect)
    resolved["evidence_start"] = start
    resolved["evidence_end"] = start + len(evidence_text)
    return validate_style_defect(resolved, index=index)


def validate_review_record(record: dict[str, Any], *, raw_output: str) -> dict[str, Any]:
    if not isinstance(raw_output, str) or not raw_output:
        raise ValueError("raw_output is required")
    if not isinstance(record, dict):
        raise ValueError("review record must be a JSON object")

    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"{missing[0]} is required")

    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")

    for field in STRING_FIELDS:
        if not isinstance(record[field], str):
            raise ValueError(f"{field} must be a string")

    if record["overall_acceptance"] not in ACCEPTANCE_STATUSES:
        raise ValueError(
            "overall_acceptance must be one of: " + ", ".join(sorted(ACCEPTANCE_STATUSES))
        )

    if record["raw_output_sha256"] != text_sha256(raw_output):
        raise ValueError("raw_output_sha256 mismatch")

    validate_style_defects(record["defects"])
    for index, defect in enumerate(record["defects"]):
        evidence_text = defect["evidence_text"]
        evidence_start = defect["evidence_start"]
        evidence_end = defect["evidence_end"]
        if raw_output[evidence_start:evidence_end] != evidence_text:
            raise ValueError(f"defects[{index}].evidence span does not match evidence_text")

    return record


def validate_review_records(
    records: list[dict[str, Any]], *, raw_outputs: dict[str, str]
) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        raise ValueError("review records must be a list")
    if not isinstance(raw_outputs, dict):
        raise ValueError("raw_outputs must be a dict")

    validated: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        try:
            if not isinstance(record, dict):
                raise ValueError("review record must be a JSON object")
            source_output_id = record.get("source_output_id")
            if source_output_id not in raw_outputs:
                raise ValueError(f"raw output not found for source_output_id: {source_output_id}")
            validated.append(validate_review_record(record, raw_output=raw_outputs[source_output_id]))
        except ValueError as exc:
            raise ValueError(f"review record {index}: {exc}") from exc
    return validated


def write_review_records(
    path: str | Path, records: list[dict[str, Any]], *, raw_outputs: dict[str, str]
) -> None:
    validated_records = validate_review_records(records, raw_outputs=raw_outputs)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in validated_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_review_records(path: str | Path, *, raw_outputs: dict[str, str]) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        raise ValueError(f"review records JSONL not found: {input_path}")

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

    return validate_review_records(records, raw_outputs=raw_outputs)
