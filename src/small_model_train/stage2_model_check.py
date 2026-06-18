from __future__ import annotations

from pathlib import Path
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

    shards = sorted(
        shard for shard in model_path.glob("model-*.safetensors") if shard.is_file()
    )
    if not shards:
        missing_files.append("model-*.safetensors")

    zero_size_files = [
        shard.name for shard in shards if shard.stat().st_size == 0
    ]
    errors = [
        f"missing required file: {name}" for name in missing_files
    ] + [
        f"zero-size model shard: {name}" for name in zero_size_files
    ]

    return {
        "model_dir": str(model_path),
        "passed": not missing_files and not zero_size_files,
        "missing_files": missing_files,
        "zero_size_files": zero_size_files,
        "shard_count": len(shards),
        "load_checks": {"config": "not_run", "tokenizer": "not_run"},
        "errors": errors,
    }


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
                f"transformers load check setup failed: {exc}"
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
        result.setdefault("errors", []).append(f"{name} load failed: {exc}")
    else:
        result["load_checks"][name] = "passed"


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
