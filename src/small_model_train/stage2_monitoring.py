from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

ERROR_RULES = (
    {
        "error_type": "cuda_oom",
        "markers": ("cuda out of memory", "cublas_status_alloc_failed"),
        "suggestion": "降低 cutoff_len，必要时降低 lora_rank",
    },
    {
        "error_type": "process_killed",
        "markers": ("killed", "terminated"),
        "suggestion": "查看 GPU 采样和系统稳定性，确认是否被系统结束",
    },
    {
        "error_type": "driver_reset",
        "markers": ("device lost", "cuda driver", "unknown error"),
        "suggestion": "重启训练环境并检查驱动状态",
    },
    {
        "error_type": "bnb_load_error",
        "markers": ("bitsandbytes", "4-bit", "4bit"),
        "suggestion": "检查 bitsandbytes、CUDA 和 PyTorch 版本匹配",
    },
    {
        "error_type": "tokenizer_error",
        "markers": ("tokenizer", "autotokenizer", "autoconfig"),
        "suggestion": "检查本地模型目录和 tokenizer 文件",
    },
    {
        "error_type": "dataset_error",
        "markers": ("dataset_info", "filenotfounderror", "jsondecodeerror", "dataset"),
        "suggestion": "检查数据路径和 LLaMA-Factory dataset_info",
    },
    {
        "error_type": "llamafactory_error",
        "markers": ("llamafactory", "unrecognized arguments", "invalid choice"),
        "suggestion": "检查 LLaMA-Factory 命令和参数",
    },
    {
        "error_type": "adapter_save_error",
        "markers": ("adapter_model", "save_pretrained", "permission denied"),
        "suggestion": "检查 output_dir 权限和磁盘空间",
    },
)

DATASET_CONTEXT_MARKERS = (
    "dataset_info",
    "filenotfounderror",
    "jsondecodeerror",
    "dataset",
    "data_sft",
    "data/",
    "data\\",
    ".jsonl",
    "input/",
    "input\\",
)

ADAPTER_SAVE_CONTEXT_MARKERS = (
    "adapter",
    "output_dir",
    "save_pretrained",
    "adapter_model",
)

HIGH_SPECIFICITY_ERROR_TYPES = (
    "cuda_oom",
    "bnb_load_error",
    "tokenizer_error",
    "llamafactory_error",
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def append_event(
    path: str | Path,
    phase: str,
    status: str,
    detail: dict[str, Any] | None = None,
) -> None:
    event_path = Path(path)
    event_path.parent.mkdir(parents=True, exist_ok=True)

    row: dict[str, Any] = {
        "time": now_iso(),
        "phase": phase,
        "status": status,
    }
    if detail is not None:
        row["detail"] = detail

    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def classify_training_error(stderr: str, exit_code: int | None) -> dict[str, str]:
    lowered_stderr = stderr.lower()

    specific_error = _match_error_rule(lowered_stderr, HIGH_SPECIFICITY_ERROR_TYPES)
    if specific_error is not None:
        return _error_result(specific_error)

    if _contains_any(lowered_stderr, DATASET_CONTEXT_MARKERS):
        return _error_result("dataset_error")

    if "permission denied" in lowered_stderr:
        if _contains_any(lowered_stderr, ADAPTER_SAVE_CONTEXT_MARKERS):
            return _error_result("adapter_save_error")
        if exit_code not in (0, None):
            return _error_result("process_killed")
        return {"error_type": "none", "suggestion": "无"}

    remaining_error = _match_error_rule(
        lowered_stderr,
        exclude_error_types=HIGH_SPECIFICITY_ERROR_TYPES + ("dataset_error",),
    )
    if remaining_error is not None:
        return _error_result(remaining_error)

    if exit_code not in (0, None):
        return _error_result("process_killed")

    return {"error_type": "none", "suggestion": "无"}


def parse_gpu_process_samples(text: str) -> list[dict[str, int | str]]:
    samples: list[dict[str, int | str]] = []
    for row in csv.reader(text.splitlines()):
        if len(row) != 3:
            continue

        try:
            pid = int(row[0].strip())
            used_mb = int(row[2].strip())
        except ValueError:
            continue

        samples.append(
            {
                "pid": pid,
                "name": row[1].strip(),
                "used_mb": used_mb,
            }
        )
    return samples


def collect_gpu_processes() -> list[dict[str, int | str]]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return []

    return parse_gpu_process_samples(result.stdout)


def render_failure_summary(
    error: dict[str, str],
    last_events: list[dict[str, Any]],
    last_gpu_samples: list[dict[str, Any]],
    exit_code: int | None,
    stderr_tail: str,
) -> str:
    lines = [
        "# Training Failure Summary",
        "",
        f"- Exit code: {exit_code}",
        f"- Error type: {error.get('error_type', 'none')}",
        f"- Suggestion: {error.get('suggestion', '')}",
        "",
        "## Last Events",
    ]
    lines.extend(_render_dict_rows(last_events))
    lines.extend(["", "## Last GPU Samples"])
    lines.extend(_render_dict_rows(last_gpu_samples))
    lines.extend(["", "## stderr tail", "```text", stderr_tail, "```", ""])
    return "\n".join(lines)


def _error_result(error_type: str) -> dict[str, str]:
    for rule in ERROR_RULES:
        if rule["error_type"] == error_type:
            return {
                "error_type": str(rule["error_type"]),
                "suggestion": str(rule["suggestion"]),
            }
    return {"error_type": error_type, "suggestion": ""}


def _match_error_rule(
    text: str,
    include_error_types: tuple[str, ...] | None = None,
    exclude_error_types: tuple[str, ...] = (),
) -> str | None:
    for rule in ERROR_RULES:
        error_type = str(rule["error_type"])
        if include_error_types is not None and error_type not in include_error_types:
            continue
        if error_type in exclude_error_types:
            continue
        if any(marker in text for marker in rule["markers"]):
            return error_type
    return None


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _render_dict_rows(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- none"]
    return [f"- {json.dumps(row, ensure_ascii=False)}" for row in rows]
