from __future__ import annotations

import csv
import os
import platform
import shutil
import subprocess
from importlib import metadata
from typing import Any


def import_version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "missing"


def parse_nvidia_smi_memory(text: str) -> dict[str, int | str]:
    line = next((candidate for candidate in text.splitlines() if candidate.strip()), "")
    if not line:
        return _empty_gpu()

    try:
        row = next(csv.reader([line]))
    except csv.Error:
        return _empty_gpu()

    if len(row) != 3:
        return _empty_gpu()

    gpu_name = row[0].strip()
    try:
        total_mb = int(row[1].strip())
        free_mb = int(row[2].strip())
    except ValueError:
        return _empty_gpu()

    return {"gpu_name": gpu_name, "total_mb": total_mb, "free_mb": free_mb}


def query_nvidia_smi_memory() -> dict[str, int | str]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free",
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
        return _empty_gpu()

    return parse_nvidia_smi_memory(result.stdout)


def vram_recommendation(free_mb: int) -> dict[str, Any]:
    if free_mb >= 13 * 1024:
        return {
            "allow_training": True,
            "cutoff_len": 8192,
            "message": "允许 8192 cutoff_len 冒烟训练",
        }
    if free_mb >= 11 * 1024:
        return {
            "allow_training": True,
            "cutoff_len": 6144,
            "message": "建议从 6144 cutoff_len 开始",
        }
    return {
        "allow_training": False,
        "cutoff_len": 0,
        "message": "不建议启动训练，先关闭占用 GPU 的程序",
    }


def collect_training_env() -> dict[str, Any]:
    gpu = query_nvidia_smi_memory()
    return {
        "python": platform.python_version(),
        "imports": {
            "torch": import_version("torch"),
            "transformers": import_version("transformers"),
            "bitsandbytes": import_version("bitsandbytes"),
            "peft": import_version("peft"),
        },
        "cuda_available": _cuda_available(),
        "gpu": gpu,
        "llamafactory": "available" if shutil.which("llamafactory-cli") else "missing",
        "env": {
            "HF_HOME": os.environ.get("HF_HOME", ""),
            "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE", ""),
            "HF_ENDPOINT": os.environ.get("HF_ENDPOINT", ""),
        },
        "recommendation": vram_recommendation(int(gpu["free_mb"])),
    }


def render_env_report(snapshot: dict[str, Any]) -> str:
    gpu = snapshot.get("gpu", {})
    recommendation = snapshot.get("recommendation", {})
    lines = [
        "# Training Environment Report",
        "",
        f"- Python: {snapshot.get('python', '')}",
        f"- CUDA available: {snapshot.get('cuda_available', False)}",
        f"- LLaMA-Factory: {snapshot.get('llamafactory', 'missing')}",
        "",
        "## GPU",
        f"- Name: {gpu.get('gpu_name', '')}",
        f"- Total memory: {gpu.get('total_mb', 0)} MB",
        f"- Free memory: {gpu.get('free_mb', 0)} MB",
        "",
        "## Python Packages",
    ]
    for package, version in snapshot.get("imports", {}).items():
        lines.append(f"- {package}: {version}")

    lines.extend(["", "## Environment Variables"])
    for name, value in snapshot.get("env", {}).items():
        lines.append(f"- {name}: {value}")

    lines.extend(
        [
            "",
            "## Recommendation",
            f"- Allow training: {recommendation.get('allow_training', False)}",
            f"- Cutoff length: {recommendation.get('cutoff_len', 0)}",
            f"- Message: {recommendation.get('message', '')}",
            "",
        ]
    )
    return "\n".join(lines)


def _cuda_available() -> bool:
    try:
        import torch
    except Exception:  # pragma: no cover - depends on local environment
        return False

    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - depends on local environment
        return False


def _empty_gpu() -> dict[str, int | str]:
    return {"gpu_name": "", "total_mb": 0, "free_mb": 0}
