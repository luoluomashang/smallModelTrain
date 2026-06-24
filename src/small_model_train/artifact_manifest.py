from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from small_model_train.execution_cards import validate_execution_cards


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_jsonl_artifact(
    path: str | Path,
    *,
    label: str,
    validate_execution_card_schema: bool = False,
    validate_sft_dataset_schema: bool = False,
) -> dict[str, Any]:
    artifact_path = Path(path)
    if validate_execution_card_schema:
        schema_name = "execution_cards"
    elif validate_sft_dataset_schema:
        schema_name = "sft_dataset"
    else:
        schema_name = "jsonl"
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    row_count = 0
    exists = artifact_path.exists()
    sha256 = ""

    if not exists:
        errors.append(f"{label} is missing: {artifact_path}")
    else:
        sha256 = file_sha256(artifact_path)
        if artifact_path.stat().st_size == 0:
            errors.append(f"{label} is empty: {artifact_path}")
        else:
            try:
                if validate_execution_card_schema:
                    rows = _read_jsonl_objects(artifact_path)
                    row_count = len(rows)
                elif validate_sft_dataset_schema:
                    row_count, errors = _count_and_validate_sft_rows(artifact_path)
                else:
                    row_count = _count_jsonl_objects(artifact_path)
            except ValueError as exc:
                errors.append(str(exc))

    if exists and not errors and row_count == 0:
        errors.append(f"{label} has no JSONL rows: {artifact_path}")

    if validate_execution_card_schema and not errors:
        try:
            validate_execution_cards(rows)
        except ValueError as exc:
            errors.append(str(exc))

    return {
        "label": label,
        "path": str(artifact_path),
        "exists": exists,
        "sha256": sha256,
        "row_count": row_count,
        "schema": {
            "name": schema_name,
            "valid": not errors,
            "errors": errors,
        },
    }


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _count_jsonl_objects(path: Path) -> int:
    row_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            row_count += 1
    return row_count


def _count_and_validate_sft_rows(path: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    row_count = 0
    required_fields = ("instruction", "input", "output")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            row_count += 1
            invalid_fields = [
                field
                for field in required_fields
                if not isinstance(row.get(field), str) or not row[field].strip()
            ]
            if invalid_fields:
                errors.append(
                    f"{path}:{line_number} missing/invalid SFT fields: "
                    + ", ".join(invalid_fields)
                )
    return row_count, errors
