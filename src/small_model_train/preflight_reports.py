from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def build_preflight_report(
    kind: str,
    passed: bool,
    payload: dict[str, Any],
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(passed, bool):
        raise ValueError("preflight report passed must be a bool")
    if not isinstance(payload, dict):
        raise ValueError("preflight report payload must be a JSON object")

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "passed": passed,
        "checked_at": _utc_now_iso(),
        "errors": _list_or_empty(errors, "errors"),
        "warnings": _list_or_empty(warnings, "warnings"),
        "payload": payload,
    }


def write_preflight_report(path: str | Path, report: dict[str, Any]) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_preflight_report(path: str | Path, expected_kind: str) -> dict[str, Any]:
    report_path = Path(path)
    if not report_path.is_file():
        raise ValueError(f"preflight report is missing: {report_path}")

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"preflight report invalid JSON: {report_path}: {exc}"
        ) from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"preflight report could not be read: {report_path}: {exc}"
        ) from exc

    if not isinstance(report, dict):
        raise ValueError(f"preflight report must be a JSON object: {report_path}")
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "preflight report schema mismatch: "
            f"expected {SCHEMA_VERSION}, got {report.get('schema_version')!r}"
        )
    if report.get("kind") != expected_kind:
        raise ValueError(
            "preflight report kind mismatch: "
            f"expected {expected_kind!r}, got {report.get('kind')!r}"
        )
    if not isinstance(report.get("passed"), bool):
        raise ValueError("preflight report passed must be a bool")
    _validate_checked_at(report.get("checked_at"))
    if not isinstance(report.get("errors"), list):
        raise ValueError("preflight report errors must be a list")
    _validate_string_list(report["errors"], "errors")
    if not isinstance(report.get("warnings"), list):
        raise ValueError("preflight report warnings must be a list")
    _validate_string_list(report["warnings"], "warnings")
    if not isinstance(report.get("payload"), dict):
        raise ValueError("preflight report payload must be a JSON object")

    return report


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _list_or_empty(items: list[str] | None, field: str) -> list[str]:
    if items is None:
        return []
    if not isinstance(items, list):
        raise ValueError(f"preflight report {field} must be a list")
    _validate_string_list(items, field)
    return list(items)


def _validate_string_list(items: list[Any], field: str) -> None:
    for index, item in enumerate(items):
        if not isinstance(item, str):
            raise ValueError(
                f"preflight report {field} must be a list of strings; "
                f"item {index} is {type(item).__name__}"
            )


def _validate_checked_at(value: Any) -> None:
    if not isinstance(value, str):
        raise ValueError("preflight report checked_at must be an ISO UTC string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            "preflight report checked_at must be an ISO UTC string"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(None):
        raise ValueError("preflight report checked_at must be an ISO UTC string")
