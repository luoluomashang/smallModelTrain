from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "training_config_snapshot.yaml",
)


def check_adapter_dir(adapter_dir: str | Path) -> dict[str, Any]:
    root = Path(adapter_dir)
    missing_files = [
        name for name in REQUIRED_ADAPTER_FILES if not (root / name).is_file()
    ]
    zero_size_files = [
        name
        for name in REQUIRED_ADAPTER_FILES
        if (root / name).is_file() and (root / name).stat().st_size == 0
    ]
    errors: list[str] = []
    _check_adapter_config(root / "adapter_config.json", errors)
    _check_safetensors_header(root / "adapter_model.safetensors", errors)

    return {
        "adapter_dir": str(root),
        "passed": not missing_files and not zero_size_files and not errors,
        "missing_files": missing_files,
        "zero_size_files": zero_size_files,
        "errors": errors,
    }


def render_adapter_report(title: str, result: dict[str, Any]) -> str:
    decision = "允许进入下一步" if result.get("passed") else "不允许进入下一步"
    lines = [
        f"# {title}",
        "",
        f"- adapter_dir: {result.get('adapter_dir', '')}",
        f"- decision: {decision}",
        "",
        "## Missing Files",
    ]
    lines.extend(_render_list(result.get("missing_files", [])))
    lines.extend(["", "## Zero Size Files"])
    lines.extend(_render_list(result.get("zero_size_files", [])))
    lines.extend(["", "## Errors"])
    lines.extend(_render_list(result.get("errors", [])))
    lines.append("")
    return "\n".join(lines)


def _check_adapter_config(path: Path, errors: list[str]) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        return

    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        errors.append(f"adapter_config.json invalid JSON: {_format_exception(exc)}")
        return

    if not isinstance(config, dict):
        errors.append("adapter_config.json must be a JSON object")


def _check_safetensors_header(path: Path, errors: list[str]) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        return

    try:
        payload = path.read_bytes()
    except OSError as exc:
        errors.append(
            f"adapter_model.safetensors read failed: {_format_exception(exc)}"
        )
        return

    if len(payload) < 8:
        errors.append(
            "adapter_model.safetensors invalid safetensors header: "
            "file is smaller than the 8-byte header length"
        )
        return

    header_len = int.from_bytes(payload[:8], "little")
    header_end = 8 + header_len
    if len(payload) < header_end:
        errors.append(
            "adapter_model.safetensors invalid safetensors header: "
            f"file has {len(payload)} bytes but header declares {header_len} bytes"
        )
        return

    try:
        header = json.loads(payload[8:header_end].decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        errors.append(
            "adapter_model.safetensors invalid safetensors header: "
            f"{_format_exception(exc)}"
        )
        return

    if not isinstance(header, dict):
        errors.append("adapter_model.safetensors header must be a JSON object")


def _format_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _render_list(items: list[str]) -> list[str]:
    if not items:
        return ["- 无"]
    return [f"- {item}" for item in items]
