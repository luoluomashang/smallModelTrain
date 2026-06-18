from __future__ import annotations

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

    return {
        "adapter_dir": str(root),
        "passed": not missing_files and not zero_size_files,
        "missing_files": missing_files,
        "zero_size_files": zero_size_files,
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
    lines.append("")
    return "\n".join(lines)


def _render_list(items: list[str]) -> list[str]:
    if not items:
        return ["- 无"]
    return [f"- {item}" for item in items]
