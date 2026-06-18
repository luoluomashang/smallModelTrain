from __future__ import annotations

import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable

REQUIRED_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
)


def check_model_files(model_dir: str | Path) -> dict[str, Any]:
    model_path = Path(model_dir)
    missing_files = [
        name for name in REQUIRED_FILES if not (model_path / name).is_file()
    ]
    errors: list[str] = []

    shards = sorted(
        shard for shard in model_path.glob("model-*.safetensors") if shard.is_file()
    )
    if not shards:
        missing_files.append("model-*.safetensors")

    index_path = model_path / "model.safetensors.index.json"
    indexed_shards: list[str] = []
    if index_path.is_file():
        indexed_shards = _read_indexed_shards(index_path, errors)
        for shard_name in indexed_shards:
            shard_path = model_path / shard_name
            if not shard_path.is_file() and shard_name not in missing_files:
                missing_files.append(shard_name)

    zero_size_files = [
        shard.name for shard in shards if shard.stat().st_size == 0
    ]
    for shard_name in indexed_shards:
        shard_path = model_path / shard_name
        if (
            shard_path.is_file()
            and shard_path.stat().st_size == 0
            and shard_name not in zero_size_files
        ):
            zero_size_files.append(shard_name)
    errors.extend([
        f"missing required file: {name}" for name in missing_files
    ])
    errors.extend([
        f"zero-size model shard: {name}" for name in zero_size_files
    ])

    return {
        "model_dir": str(model_path),
        "passed": not missing_files and not zero_size_files and not errors,
        "missing_files": missing_files,
        "zero_size_files": zero_size_files,
        "shard_count": len(shards),
        "load_checks": {"config": "not_run", "tokenizer": "not_run"},
        "errors": errors,
    }


def _read_indexed_shards(index_path: Path, errors: list[str]) -> list[str]:
    try:
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        errors.append(f"invalid index: {_format_exception(exc)}")
        return []

    weight_map = index_data.get("weight_map") if isinstance(index_data, dict) else None
    if not isinstance(weight_map, dict):
        errors.append("invalid index: weight_map must be a dict")
        return []

    shard_names: list[str] = []
    for layer_name, shard_name in weight_map.items():
        if not isinstance(shard_name, str):
            errors.append(
                "invalid index: "
                f"weight_map entry {layer_name!r} must reference a shard filename"
            )
            continue
        if not _is_plain_shard_filename(shard_name):
            errors.append(f"invalid shard path in index: {shard_name}")
            continue
        if shard_name not in shard_names:
            shard_names.append(shard_name)
    return shard_names


def _is_plain_shard_filename(shard_name: str) -> bool:
    windows_path = PureWindowsPath(shard_name)
    posix_path = PurePosixPath(shard_name)
    return (
        bool(shard_name)
        and shard_name not in {".", ".."}
        and "/" not in shard_name
        and "\\" not in shard_name
        and ".." not in windows_path.parts
        and ".." not in posix_path.parts
        and not windows_path.drive
        and not windows_path.is_absolute()
        and not posix_path.is_absolute()
        and windows_path.name == shard_name
        and posix_path.name == shard_name
    )


def run_transformers_load_checks(
    result: dict[str, Any],
    config_loader: Callable[[str], Any] | None = None,
    tokenizer_loader: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    load_checks = result.setdefault(
        "load_checks", {"config": "not_run", "tokenizer": "not_run"}
    )
    load_checks.setdefault("config", "not_run")
    load_checks.setdefault("tokenizer", "not_run")

    if config_loader is None or tokenizer_loader is None:
        try:
            from transformers import AutoConfig, AutoTokenizer
        except Exception as exc:  # pragma: no cover - depends on local environment
            load_checks["config"] = "failed"
            load_checks["tokenizer"] = "failed"
            result.setdefault("errors", []).append(
                f"transformers load check setup failed: {_format_exception(exc)}"
            )
            _recompute_passed(result)
            return result

        config_loader = config_loader or AutoConfig.from_pretrained
        tokenizer_loader = tokenizer_loader or AutoTokenizer.from_pretrained

    model_dir = result["model_dir"]
    _run_loader(result, "config", config_loader, model_dir)
    _run_loader(result, "tokenizer", tokenizer_loader, model_dir)
    _recompute_passed(result)
    return result


def render_model_check_report(result: dict[str, Any]) -> str:
    decision = "允许进入训练" if result["passed"] else "不进入训练"
    lines = [
        "# Local Model Check Report",
        "",
        f"- Model dir: `{result['model_dir']}`",
        f"- Decision: **{decision}**",
        f"- Shard count: {result['shard_count']}",
        "",
        "## Missing Files",
    ]

    lines.extend(_render_list(result.get("missing_files", [])))
    lines.extend(["", "## Zero-size Files"])
    lines.extend(_render_list(result.get("zero_size_files", [])))
    lines.extend(["", "## Load Checks"])
    for name, status in result.get("load_checks", {}).items():
        lines.append(f"- {name}: {status}")

    if not result.get("load_checks"):
        lines.append("- none")

    lines.extend(["", "## Errors"])
    lines.extend(_render_list(result.get("errors", [])))
    lines.append("")
    return "\n".join(lines)


def _run_loader(
    result: dict[str, Any],
    name: str,
    loader: Callable[[str], Any],
    model_dir: str,
) -> None:
    try:
        loader(model_dir)
    except Exception as exc:
        result["load_checks"][name] = "failed"
        result.setdefault("errors", []).append(
            f"{name} load failed: {_format_exception(exc)}"
        )
    else:
        result["load_checks"][name] = "passed"


def _format_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _recompute_passed(result: dict[str, Any]) -> None:
    load_checks = result.get("load_checks", {})
    result["passed"] = (
        not result.get("missing_files")
        and not result.get("zero_size_files")
        and not result.get("errors")
        and all(status != "failed" for status in load_checks.values())
    )


def _render_list(items: list[str]) -> list[str]:
    if not items:
        return ["- none"]
    return [f"- {item}" for item in items]
