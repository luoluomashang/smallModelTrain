# Qwen3 QLoRA Stage 2 Training Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Stage 2 training execution loop for local Qwen3-4B QLoRA SFT: model checks, environment checks, monitored smoke/full training, adapter checks, OOM diagnosis, and fixed eval inference.

**Architecture:** Keep the existing project shape: testable logic in `src/small_model_train/`, thin CLI wrappers in `scripts/`, and Markdown/JSONL/YAML artifacts under `reports/`, `logs/`, and `outputs/`. High-risk GPU work is launched through subprocess supervisors that preserve stderr, event logs, GPU samples, and failure classification when the worker process crashes.

**Tech Stack:** Python 3.11 stdlib (`argparse`, `dataclasses`, `json`, `pathlib`, `subprocess`, `threading`, `time`, `importlib.metadata`), existing JSONL helpers, `pytest`, optional runtime imports (`torch`, `transformers`, `bitsandbytes`, `peft`) only inside environment/model/probe checks and GPU workers.

---

## File Structure

Create these files:

```text
src/small_model_train/stage2_model_check.py
src/small_model_train/stage2_env_check.py
src/small_model_train/stage2_config.py
src/small_model_train/stage2_monitoring.py
src/small_model_train/stage2_training.py
src/small_model_train/stage2_adapter.py
src/small_model_train/stage2_inference.py
scripts/check_local_model.py
scripts/check_training_env.py
scripts/run_sft_smoke.py
scripts/run_sft_train.py
scripts/check_adapter.py
scripts/run_oom_probe.py
scripts/run_eval_inference.py
scripts/stage2_eval_worker.py
tests/test_stage2_model_check.py
tests/test_stage2_env_check.py
tests/test_stage2_config.py
tests/test_stage2_monitoring.py
tests/test_stage2_training.py
tests/test_stage2_adapter.py
tests/test_stage2_inference.py
```

Modify these files:

```text
README.md
```

Responsibilities:

- `stage2_model_check.py`: local model file validation and optional `transformers` config/tokenizer load checks.
- `stage2_env_check.py`: import/version checks, CUDA/GPU snapshot parsing, environment report rendering.
- `stage2_config.py`: flat YAML read/write, training config snapshots, LLaMA-Factory command construction.
- `stage2_monitoring.py`: event JSONL writing, GPU sample parsing, stderr error classification, failure summary rendering.
- `stage2_training.py`: training input validation, smoke/full run specs, subprocess execution with monitoring hooks.
- `stage2_adapter.py`: adapter directory validation and report rendering.
- `stage2_inference.py`: eval prompt construction and generated JSONL row formatting.
- `scripts/*.py`: command-line wrappers with repo-root import setup.

## Task 1: Local Model Check

**Files:**
- Create: `src/small_model_train/stage2_model_check.py`
- Create: `scripts/check_local_model.py`
- Test: `tests/test_stage2_model_check.py`

- [ ] **Step 1: Write failing tests for file validation and report rendering**

Create `tests/test_stage2_model_check.py`:

```python
from pathlib import Path

from small_model_train.stage2_model_check import check_model_files, render_model_check_report


def write_file(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_check_model_files_passes_for_required_files(tmp_path: Path):
    model_dir = tmp_path / "model"
    for name in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "model-00001-of-00001.safetensors",
    ]:
        write_file(model_dir / name)

    result = check_model_files(model_dir)

    assert result["passed"] is True
    assert result["missing_files"] == []
    assert result["zero_size_files"] == []
    assert result["shard_count"] == 1


def test_check_model_files_reports_missing_and_zero_size_files(tmp_path: Path):
    model_dir = tmp_path / "model"
    write_file(model_dir / "config.json")
    write_file(model_dir / "tokenizer.json")
    write_file(model_dir / "tokenizer_config.json")
    write_file(model_dir / "model.safetensors.index.json")
    shard = model_dir / "model-00001-of-00001.safetensors"
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_bytes(b"")

    result = check_model_files(model_dir)

    assert result["passed"] is False
    assert result["zero_size_files"] == ["model-00001-of-00001.safetensors"]


def test_render_model_check_report_contains_decision(tmp_path: Path):
    result = {
        "model_dir": str(tmp_path / "model"),
        "passed": False,
        "missing_files": ["config.json"],
        "zero_size_files": [],
        "shard_count": 0,
        "load_checks": {"config": "skipped", "tokenizer": "skipped"},
        "errors": ["missing required file: config.json"],
    }

    report = render_model_check_report(result)

    assert "# Local Model Check Report" in report
    assert "missing required file: config.json" in report
    assert "不进入训练" in report
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_stage2_model_check.py -v
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `small_model_train.stage2_model_check`.

- [ ] **Step 3: Implement local model check module**

Create `src/small_model_train/stage2_model_check.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Callable


REQUIRED_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
]


def check_model_files(model_dir: str | Path) -> dict:
    root = Path(model_dir)
    missing_files = [name for name in REQUIRED_FILES if not (root / name).exists()]
    shard_paths = sorted(root.glob("model-*.safetensors"))
    if not shard_paths:
        missing_files.append("model-*.safetensors")
    zero_size_files = [path.name for path in shard_paths if path.stat().st_size == 0]
    errors = [f"missing required file: {name}" for name in missing_files]
    errors.extend(f"empty weight shard: {name}" for name in zero_size_files)
    return {
        "model_dir": str(root),
        "passed": not errors,
        "missing_files": missing_files,
        "zero_size_files": zero_size_files,
        "shard_count": len(shard_paths),
        "load_checks": {"config": "not_run", "tokenizer": "not_run"},
        "errors": errors,
    }


def run_transformers_load_checks(
    result: dict,
    config_loader: Callable[[str], object] | None = None,
    tokenizer_loader: Callable[[str], object] | None = None,
) -> dict:
    updated = dict(result)
    load_checks = dict(updated.get("load_checks", {}))
    model_dir = updated["model_dir"]
    try:
        if config_loader is None or tokenizer_loader is None:
            from transformers import AutoConfig, AutoTokenizer

            config_loader = AutoConfig.from_pretrained
            tokenizer_loader = AutoTokenizer.from_pretrained
        config_loader(model_dir)
        load_checks["config"] = "ok"
    except Exception as exc:
        load_checks["config"] = f"failed: {type(exc).__name__}: {exc}"
        updated.setdefault("errors", []).append(load_checks["config"])
    try:
        tokenizer_loader(model_dir)
        load_checks["tokenizer"] = "ok"
    except Exception as exc:
        load_checks["tokenizer"] = f"failed: {type(exc).__name__}: {exc}"
        updated.setdefault("errors", []).append(load_checks["tokenizer"])
    updated["load_checks"] = load_checks
    updated["passed"] = updated.get("passed", False) and not updated.get("errors", [])
    return updated


def render_model_check_report(result: dict) -> str:
    decision = "允许进入训练" if result.get("passed") else "不进入训练"
    lines = [
        "# Local Model Check Report",
        "",
        f"- 模型目录：{result.get('model_dir', '')}",
        f"- 检查结论：{decision}",
        f"- safetensors 分片数：{result.get('shard_count', 0)}",
        "",
        "## Load Checks",
    ]
    for name, status in result.get("load_checks", {}).items():
        lines.append(f"- {name}: {status}")
    lines.extend(["", "## Errors"])
    errors = result.get("errors", [])
    if errors:
        for error in errors:
            lines.append(f"- {error}")
    else:
        lines.append("- 无")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Implement local model check CLI**

Create `scripts/check_local_model.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.stage2_model_check import (
    check_model_files,
    render_model_check_report,
    run_transformers_load_checks,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=r"E:\models\Qwen3-4B-Instruct-2507")
    parser.add_argument("--report", default="reports/model_check_report.md")
    parser.add_argument("--skip-transformers-load", action="store_true")
    args = parser.parse_args()

    result = check_model_files(args.model_dir)
    if not args.skip_transformers_load and result["passed"]:
        result = run_transformers_load_checks(result)
    report = render_model_check_report(result)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"wrote model check report to {report_path}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_stage2_model_check.py -v
```

Expected: PASS for all tests in `tests/test_stage2_model_check.py`.

- [ ] **Step 6: Manually verify against the downloaded model**

Run:

```powershell
python scripts/check_local_model.py --model-dir E:\models\Qwen3-4B-Instruct-2507 --report reports/model_check_report.md
```

Expected: exits 0 and writes `reports/model_check_report.md` with `检查结论：允许进入训练`.

- [ ] **Step 7: Commit**

Run:

```powershell
git add src/small_model_train/stage2_model_check.py scripts/check_local_model.py tests/test_stage2_model_check.py reports/model_check_report.md
git commit -m "feat: add stage two local model check"
```

Expected: commit succeeds.

## Task 2: Training Environment Check

**Files:**
- Create: `src/small_model_train/stage2_env_check.py`
- Create: `scripts/check_training_env.py`
- Test: `tests/test_stage2_env_check.py`

- [ ] **Step 1: Write failing tests for GPU CSV parsing and report rendering**

Create `tests/test_stage2_env_check.py`:

```python
from small_model_train.stage2_env_check import parse_nvidia_smi_memory, render_env_report, vram_recommendation


def test_parse_nvidia_smi_memory_reads_free_and_total_mb():
    text = "NVIDIA RTX, 16384, 14000\n"

    result = parse_nvidia_smi_memory(text)

    assert result == {"gpu_name": "NVIDIA RTX", "total_mb": 16384, "free_mb": 14000}


def test_vram_recommendation_uses_stage_two_thresholds():
    assert vram_recommendation(14000)["cutoff_len"] == 8192
    assert vram_recommendation(12000)["cutoff_len"] == 6144
    assert vram_recommendation(8000)["allow_training"] is False


def test_render_env_report_contains_dependency_status():
    snapshot = {
        "python": "3.11.8",
        "imports": {"torch": "2.5.0", "bitsandbytes": "missing"},
        "cuda_available": True,
        "gpu": {"gpu_name": "NVIDIA RTX", "total_mb": 16384, "free_mb": 14000},
        "llamafactory": "available",
        "env": {"HF_HOME": "", "TRANSFORMERS_CACHE": "", "HF_ENDPOINT": ""},
        "recommendation": {"allow_training": True, "cutoff_len": 8192, "message": "允许 8192 cutoff_len 冒烟训练"},
    }

    report = render_env_report(snapshot)

    assert "# Training Environment Report" in report
    assert "- bitsandbytes: missing" in report
    assert "允许 8192 cutoff_len 冒烟训练" in report
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_stage2_env_check.py -v
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `small_model_train.stage2_env_check`.

- [ ] **Step 3: Implement environment check module**

Create `src/small_model_train/stage2_env_check.py`:

```python
from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import subprocess


def import_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def parse_nvidia_smi_memory(text: str) -> dict:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if not first_line:
        return {"gpu_name": "", "total_mb": 0, "free_mb": 0}
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) < 3:
        return {"gpu_name": first_line, "total_mb": 0, "free_mb": 0}
    return {
        "gpu_name": parts[0],
        "total_mb": int(float(parts[1])),
        "free_mb": int(float(parts[2])),
    }


def query_nvidia_smi_memory() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=False, text=True, timeout=10)
    except FileNotFoundError:
        return {"gpu_name": "", "total_mb": 0, "free_mb": 0}
    if result.returncode != 0:
        return {"gpu_name": "", "total_mb": 0, "free_mb": 0}
    return parse_nvidia_smi_memory(result.stdout)


def vram_recommendation(free_mb: int) -> dict:
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


def collect_training_env() -> dict:
    imports = {
        "torch": import_version("torch"),
        "transformers": import_version("transformers"),
        "bitsandbytes": import_version("bitsandbytes"),
        "peft": import_version("peft"),
    }
    cuda_available = False
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False
    gpu = query_nvidia_smi_memory()
    return {
        "python": platform.python_version(),
        "imports": imports,
        "cuda_available": cuda_available,
        "gpu": gpu,
        "llamafactory": "available" if shutil.which("llamafactory-cli") else "missing",
        "env": {
            "HF_HOME": os.environ.get("HF_HOME", ""),
            "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE", ""),
            "HF_ENDPOINT": os.environ.get("HF_ENDPOINT", ""),
        },
        "recommendation": vram_recommendation(int(gpu.get("free_mb", 0))),
    }


def render_env_report(snapshot: dict) -> str:
    lines = [
        "# Training Environment Report",
        "",
        f"- Python: {snapshot.get('python', '')}",
        f"- CUDA available: {snapshot.get('cuda_available', False)}",
        f"- LLaMA-Factory: {snapshot.get('llamafactory', '')}",
        "",
        "## GPU",
    ]
    gpu = snapshot.get("gpu", {})
    lines.extend(
        [
            f"- Name: {gpu.get('gpu_name', '')}",
            f"- Total MB: {gpu.get('total_mb', 0)}",
            f"- Free MB: {gpu.get('free_mb', 0)}",
            "",
            "## Python Packages",
        ]
    )
    for name, status in snapshot.get("imports", {}).items():
        lines.append(f"- {name}: {status}")
    lines.extend(["", "## Environment Variables"])
    for name, value in snapshot.get("env", {}).items():
        lines.append(f"- {name}: {value}")
    recommendation = snapshot.get("recommendation", {})
    lines.extend(
        [
            "",
            "## Recommendation",
            f"- allow_training: {recommendation.get('allow_training', False)}",
            f"- cutoff_len: {recommendation.get('cutoff_len', 0)}",
            f"- message: {recommendation.get('message', '')}",
        ]
    )
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Implement environment check CLI**

Create `scripts/check_training_env.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.stage2_env_check import collect_training_env, render_env_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="reports/training_env_report.md")
    args = parser.parse_args()

    snapshot = collect_training_env()
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_env_report(snapshot), encoding="utf-8")
    print(f"wrote training environment report to {report_path}")
    if not snapshot["recommendation"]["allow_training"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_stage2_env_check.py -v
```

Expected: PASS for all tests in `tests/test_stage2_env_check.py`.

- [ ] **Step 6: Manually collect the real environment report**

Run:

```powershell
python scripts/check_training_env.py --report reports/training_env_report.md
```

Expected: writes `reports/training_env_report.md`. Exit code is 0 when free VRAM is at least 11GB; exit code is 1 when the report recommends closing other GPU programs.

- [ ] **Step 7: Commit**

Run:

```powershell
git add src/small_model_train/stage2_env_check.py scripts/check_training_env.py tests/test_stage2_env_check.py reports/training_env_report.md
git commit -m "feat: add stage two environment check"
```

Expected: commit succeeds.

## Task 3: Config Snapshots And Training Input Validation

**Files:**
- Create: `src/small_model_train/stage2_config.py`
- Test: `tests/test_stage2_config.py`

- [ ] **Step 1: Write failing tests for flat YAML parsing and smoke snapshot overrides**

Create `tests/test_stage2_config.py`:

```python
from pathlib import Path

from small_model_train.stage2_config import (
    build_llamafactory_command,
    make_training_snapshot,
    read_flat_yaml,
    write_flat_yaml,
)


def test_read_and_write_flat_yaml_round_trip(tmp_path: Path):
    path = tmp_path / "config.yaml"
    write_flat_yaml(path, {"bf16": True, "cutoff_len": 8192, "learning_rate": 3.0e-5, "template": "qwen3"})

    result = read_flat_yaml(path)

    assert result["bf16"] is True
    assert result["cutoff_len"] == 8192
    assert result["learning_rate"] == 3.0e-5
    assert result["template"] == "qwen3"


def test_make_training_snapshot_overrides_model_output_and_smoke_values(tmp_path: Path):
    source = tmp_path / "source.yaml"
    output = tmp_path / "outputs" / "training_config_snapshot.yaml"
    write_flat_yaml(source, {"model_name_or_path": "remote", "output_dir": "old", "cutoff_len": 8192})

    snapshot = make_training_snapshot(
        source_config=source,
        output_config=output,
        model_dir=r"E:\models\Qwen3-4B-Instruct-2507",
        output_dir="outputs/sft_smoke",
        overrides={"max_samples": 100, "num_train_epochs": 1},
    )

    assert snapshot["model_name_or_path"] == r"E:\models\Qwen3-4B-Instruct-2507"
    assert snapshot["output_dir"] == "outputs/sft_smoke"
    assert snapshot["max_samples"] == 100
    assert output.exists()


def test_build_llamafactory_command_uses_config_path():
    command = build_llamafactory_command("outputs/sft_smoke/training_config_snapshot.yaml")

    assert command == ["llamafactory-cli", "train", "outputs/sft_smoke/training_config_snapshot.yaml"]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_stage2_config.py -v
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `small_model_train.stage2_config`.

- [ ] **Step 3: Implement flat YAML config helpers**

Create `src/small_model_train/stage2_config.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any


def parse_scalar(value: str) -> Any:
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        if any(marker in stripped for marker in [".", "e", "E"]):
            return float(stripped)
        return int(stripped)
    except ValueError:
        return stripped.strip('"').strip("'")


def format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def read_flat_yaml(path: str | Path) -> dict:
    rows: dict[str, Any] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        rows[key.strip()] = parse_scalar(value)
    return rows


def write_flat_yaml(path: str | Path, values: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}: {format_scalar(value)}" for key, value in values.items()]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_training_snapshot(
    source_config: str | Path,
    output_config: str | Path,
    model_dir: str,
    output_dir: str,
    overrides: dict | None = None,
) -> dict:
    snapshot = read_flat_yaml(source_config)
    snapshot["model_name_or_path"] = model_dir
    snapshot["output_dir"] = output_dir
    if overrides:
        snapshot.update(overrides)
    write_flat_yaml(output_config, snapshot)
    return snapshot


def build_llamafactory_command(config_path: str | Path) -> list[str]:
    return ["llamafactory-cli", "train", str(config_path)]
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_stage2_config.py -v
```

Expected: PASS for all tests in `tests/test_stage2_config.py`.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/small_model_train/stage2_config.py tests/test_stage2_config.py
git commit -m "feat: add stage two config snapshots"
```

Expected: commit succeeds.

## Task 4: Monitoring, GPU Sampling, And Error Classification

**Files:**
- Create: `src/small_model_train/stage2_monitoring.py`
- Test: `tests/test_stage2_monitoring.py`

- [ ] **Step 1: Write failing tests for event logs, GPU parsing, and error classification**

Create `tests/test_stage2_monitoring.py`:

```python
import json
from pathlib import Path

from small_model_train.stage2_monitoring import (
    append_event,
    classify_training_error,
    parse_gpu_process_samples,
    render_failure_summary,
)


def test_append_event_writes_jsonl(tmp_path: Path):
    path = tmp_path / "logs" / "events.jsonl"

    append_event(path, "prepare_config", "start", {"run": "smoke"})

    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["phase"] == "prepare_config"
    assert row["status"] == "start"
    assert row["detail"] == {"run": "smoke"}


def test_classify_training_error_detects_cuda_oom():
    result = classify_training_error("RuntimeError: CUDA out of memory", 1)

    assert result["error_type"] == "cuda_oom"
    assert "降低 cutoff_len" in result["suggestion"]


def test_classify_training_error_detects_dataset_error():
    result = classify_training_error("FileNotFoundError: dataset_info.json", 1)

    assert result["error_type"] == "dataset_error"


def test_parse_gpu_process_samples_reads_nvidia_smi_csv():
    text = "1234, python.exe, 11800\n5678, chrome.exe, 600\n"

    samples = parse_gpu_process_samples(text)

    assert samples == [
        {"pid": 1234, "name": "python.exe", "used_mb": 11800},
        {"pid": 5678, "name": "chrome.exe", "used_mb": 600},
    ]


def test_render_failure_summary_contains_last_phase_and_gpu_sample():
    summary = render_failure_summary(
        error={"error_type": "cuda_oom", "suggestion": "降低 cutoff_len"},
        last_events=[{"phase": "first_backward", "status": "start"}],
        last_gpu_samples=[{"free_mb": 512, "used_mb": 15872}],
        exit_code=1,
        stderr_tail="CUDA out of memory",
    )

    assert "first_backward" in summary
    assert "cuda_oom" in summary
    assert "CUDA out of memory" in summary
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_stage2_monitoring.py -v
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `small_model_train.stage2_monitoring`.

- [ ] **Step 3: Implement monitoring module**

Create `src/small_model_train/stage2_monitoring.py`:

```python
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path


ERROR_RULES = [
    ("cuda_oom", ["cuda out of memory", "cublas_status_alloc_failed"], "降低 cutoff_len，必要时降低 lora_rank"),
    ("process_killed", ["killed", "terminated"], "查看 GPU 采样和系统稳定性，确认是否被系统结束"),
    ("driver_reset", ["device lost", "cuda driver", "unknown error"], "重启训练环境并检查驱动状态"),
    ("bnb_load_error", ["bitsandbytes", "4-bit", "4bit"], "检查 bitsandbytes、CUDA 和 PyTorch 版本匹配"),
    ("tokenizer_error", ["tokenizer", "autotokenizer", "autoconfig"], "检查本地模型目录和 tokenizer 文件"),
    ("dataset_error", ["dataset_info", "filenotfounderror", "jsondecodeerror", "dataset"], "检查数据路径和 LLaMA-Factory dataset_info"),
    ("llamafactory_error", ["llamafactory", "unrecognized arguments", "invalid choice"], "检查 LLaMA-Factory 命令和参数"),
    ("adapter_save_error", ["adapter_model", "save_pretrained", "permission denied"], "检查 output_dir 权限和磁盘空间"),
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def append_event(path: str | Path, phase: str, status: str, detail: dict | None = None) -> None:
    event_path = Path(path)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"time": now_iso(), "phase": phase, "status": status}
    if detail is not None:
        row["detail"] = detail
    with event_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def classify_training_error(stderr: str, exit_code: int | None) -> dict:
    lowered = stderr.lower()
    for error_type, markers, suggestion in ERROR_RULES:
        if any(marker in lowered for marker in markers):
            return {"error_type": error_type, "suggestion": suggestion}
    if exit_code not in (0, None):
        return {"error_type": "process_killed", "suggestion": "查看 stderr、事件日志和 GPU 采样"}
    return {"error_type": "none", "suggestion": "无"}


def parse_gpu_process_samples(text: str) -> list[dict]:
    samples: list[dict] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = [part.strip() for part in stripped.split(",")]
        if len(parts) < 3:
            continue
        samples.append({"pid": int(parts[0]), "name": parts[1], "used_mb": int(float(parts[2]))})
    return samples


def collect_gpu_processes() -> list[dict]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=False, text=True, timeout=10)
    except FileNotFoundError:
        return []
    if result.returncode != 0:
        return []
    return parse_gpu_process_samples(result.stdout)


def render_failure_summary(
    error: dict,
    last_events: list[dict],
    last_gpu_samples: list[dict],
    exit_code: int | None,
    stderr_tail: str,
) -> str:
    lines = [
        "# Training Failure Summary",
        "",
        f"- exit_code: {exit_code}",
        f"- error_type: {error.get('error_type', '')}",
        f"- suggestion: {error.get('suggestion', '')}",
        "",
        "## Last Events",
    ]
    if last_events:
        for event in last_events:
            lines.append(f"- {event.get('phase')}: {event.get('status')}")
    else:
        lines.append("- 无")
    lines.extend(["", "## Last GPU Samples"])
    if last_gpu_samples:
        for sample in last_gpu_samples:
            lines.append(json.dumps(sample, ensure_ascii=False))
    else:
        lines.append("- 无")
    lines.extend(["", "## stderr Tail", "```text", stderr_tail, "```"])
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_stage2_monitoring.py -v
```

Expected: PASS for all tests in `tests/test_stage2_monitoring.py`.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/small_model_train/stage2_monitoring.py tests/test_stage2_monitoring.py
git commit -m "feat: add stage two training monitoring"
```

Expected: commit succeeds.

## Task 5: Smoke And Full Training Launchers

**Files:**
- Create: `src/small_model_train/stage2_training.py`
- Create: `scripts/run_sft_smoke.py`
- Create: `scripts/run_sft_train.py`
- Test: `tests/test_stage2_training.py`

- [ ] **Step 1: Write failing tests for input validation, run specs, and dry-run command output**

Create `tests/test_stage2_training.py`:

```python
from pathlib import Path

from small_model_train.stage2_training import (
    build_train_run,
    run_training_dry,
    validate_training_inputs,
)


def test_validate_training_inputs_reports_missing_files(tmp_path: Path):
    result = validate_training_inputs(
        sft_dataset=tmp_path / "missing_sft.jsonl",
        eval_cards=tmp_path / "missing_eval.jsonl",
    )

    assert result["passed"] is False
    assert "missing_sft.jsonl" in "\n".join(result["errors"])


def test_build_train_run_creates_smoke_snapshot(tmp_path: Path):
    config = tmp_path / "sft.yaml"
    config.write_text("model_name_or_path: remote\noutput_dir: old\ncutoff_len: 8192\n", encoding="utf-8")

    run = build_train_run(
        name="sft_smoke",
        source_config=config,
        model_dir=r"E:\models\Qwen3-4B-Instruct-2507",
        output_dir=tmp_path / "outputs" / "sft_smoke",
        log_dir=tmp_path / "logs" / "training",
        smoke=True,
    )

    assert run["snapshot"]["max_samples"] == 100
    assert run["command"][0:2] == ["llamafactory-cli", "train"]
    assert Path(run["config_path"]).exists()


def test_run_training_dry_writes_event_and_command(tmp_path: Path):
    run = {
        "name": "sft_smoke",
        "command": ["llamafactory-cli", "train", "config.yaml"],
        "event_log": str(tmp_path / "events.jsonl"),
        "stderr_log": str(tmp_path / "stderr.log"),
    }

    result = run_training_dry(run)

    assert result["exit_code"] == 0
    assert "llamafactory-cli train config.yaml" in result["command_text"]
    assert Path(run["event_log"]).exists()
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_stage2_training.py -v
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `small_model_train.stage2_training`.

- [ ] **Step 3: Implement training run builder and dry-run helper**

Create `src/small_model_train/stage2_training.py`:

```python
from __future__ import annotations

import subprocess
from pathlib import Path

from small_model_train.stage2_config import build_llamafactory_command, make_training_snapshot
from small_model_train.stage2_monitoring import append_event, classify_training_error


SMOKE_OVERRIDES = {
    "max_samples": 100,
    "save_steps": 50,
    "logging_steps": 5,
    "num_train_epochs": 1,
}


def validate_training_inputs(sft_dataset: str | Path, eval_cards: str | Path) -> dict:
    errors: list[str] = []
    for path in [Path(sft_dataset), Path(eval_cards)]:
        if not path.exists():
            errors.append(f"missing input file: {path}")
        elif path.stat().st_size == 0:
            errors.append(f"empty input file: {path}")
    return {"passed": not errors, "errors": errors}


def build_train_run(
    name: str,
    source_config: str | Path,
    model_dir: str,
    output_dir: str | Path,
    log_dir: str | Path,
    smoke: bool,
) -> dict:
    output_path = Path(output_dir)
    config_path = output_path / "training_config_snapshot.yaml"
    overrides = dict(SMOKE_OVERRIDES) if smoke else {}
    snapshot = make_training_snapshot(
        source_config=source_config,
        output_config=config_path,
        model_dir=model_dir,
        output_dir=str(output_path),
        overrides=overrides,
    )
    log_path = Path(log_dir)
    return {
        "name": name,
        "snapshot": snapshot,
        "config_path": str(config_path),
        "command": build_llamafactory_command(config_path),
        "event_log": str(log_path / f"{name}_events.jsonl"),
        "gpu_log": str(log_path / f"{name}_gpu.jsonl"),
        "stderr_log": str(log_path / f"{name}_stderr.log"),
    }


def run_training_dry(run: dict) -> dict:
    append_event(run["event_log"], "prepare_config", "ok", {"run": run["name"]})
    append_event(run["event_log"], "launch_train_subprocess", "dry_run", {"command": run["command"]})
    return {"exit_code": 0, "command_text": " ".join(run["command"]), "error": {"error_type": "none", "suggestion": "无"}}


def run_training_subprocess(run: dict) -> dict:
    append_event(run["event_log"], "launch_train_subprocess", "start", {"command": run["command"]})
    completed = subprocess.run(run["command"], capture_output=True, check=False, text=True)
    stderr_path = Path(run["stderr_log"])
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    status = "ok" if completed.returncode == 0 else "failed"
    append_event(run["event_log"], "launch_train_subprocess", status, {"exit_code": completed.returncode})
    error = classify_training_error(completed.stderr, completed.returncode)
    return {"exit_code": completed.returncode, "command_text": " ".join(run["command"]), "error": error}
```

- [ ] **Step 4: Implement smoke training CLI**

Create `scripts/run_sft_smoke.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.stage2_training import build_train_run, run_training_dry, run_training_subprocess, validate_training_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft_qlora_qwen3_4b.yaml")
    parser.add_argument("--model-dir", default=r"E:\models\Qwen3-4B-Instruct-2507")
    parser.add_argument("--output-dir", default="outputs/sft_smoke")
    parser.add_argument("--log-dir", default="logs/training")
    parser.add_argument("--sft-dataset", default="data_sft/sft_chapter_v1.jsonl")
    parser.add_argument("--eval-cards", default="data_cards/eval_cards_50.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    validation = validate_training_inputs(args.sft_dataset, args.eval_cards)
    if not validation["passed"]:
        for error in validation["errors"]:
            print(error, file=sys.stderr)
        raise SystemExit(1)
    run = build_train_run(
        name="sft_smoke",
        source_config=args.config,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        smoke=True,
    )
    result = run_training_dry(run) if args.dry_run else run_training_subprocess(run)
    print(result["command_text"])
    if result["exit_code"] != 0:
        raise SystemExit(result["exit_code"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Implement full SFT v1 training CLI**

Create `scripts/run_sft_train.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.stage2_training import build_train_run, run_training_dry, run_training_subprocess, validate_training_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft_qlora_qwen3_4b.yaml")
    parser.add_argument("--model-dir", default=r"E:\models\Qwen3-4B-Instruct-2507")
    parser.add_argument("--output-dir", default="outputs/sft_v1")
    parser.add_argument("--log-dir", default="logs/training")
    parser.add_argument("--sft-dataset", default="data_sft/sft_chapter_v1.jsonl")
    parser.add_argument("--eval-cards", default="data_cards/eval_cards_50.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    validation = validate_training_inputs(args.sft_dataset, args.eval_cards)
    if not validation["passed"]:
        for error in validation["errors"]:
            print(error, file=sys.stderr)
        raise SystemExit(1)
    run = build_train_run(
        name="sft_v1",
        source_config=args.config,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        smoke=False,
    )
    result = run_training_dry(run) if args.dry_run else run_training_subprocess(run)
    print(result["command_text"])
    if result["exit_code"] != 0:
        raise SystemExit(result["exit_code"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_stage2_training.py -v
```

Expected: PASS for all tests in `tests/test_stage2_training.py`.

- [ ] **Step 7: Dry-run the smoke command after data files exist**

Run:

```powershell
python scripts/run_sft_smoke.py --dry-run
```

Expected: prints `llamafactory-cli train outputs\sft_smoke\training_config_snapshot.yaml` or the same path with forward slashes.

- [ ] **Step 8: Commit**

Run:

```powershell
git add src/small_model_train/stage2_training.py scripts/run_sft_smoke.py scripts/run_sft_train.py tests/test_stage2_training.py
git commit -m "feat: add stage two training launchers"
```

Expected: commit succeeds.

## Task 6: Adapter Check And OOM Probe Report

**Files:**
- Create: `src/small_model_train/stage2_adapter.py`
- Create: `scripts/check_adapter.py`
- Create: `scripts/run_oom_probe.py`
- Test: `tests/test_stage2_adapter.py`

- [ ] **Step 1: Write failing tests for adapter validation and report rendering**

Create `tests/test_stage2_adapter.py`:

```python
from pathlib import Path

from small_model_train.stage2_adapter import check_adapter_dir, render_adapter_report


def test_check_adapter_dir_passes_when_required_files_exist(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "training_config_snapshot.yaml").write_text("output_dir: adapter\n", encoding="utf-8")

    result = check_adapter_dir(adapter)

    assert result["passed"] is True
    assert result["missing_files"] == []


def test_check_adapter_dir_reports_missing_files(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()

    result = check_adapter_dir(adapter)

    assert result["passed"] is False
    assert "adapter_config.json" in result["missing_files"]


def test_render_adapter_report_contains_decision(tmp_path: Path):
    result = {
        "adapter_dir": str(tmp_path / "adapter"),
        "passed": False,
        "missing_files": ["adapter_model.safetensors"],
        "zero_size_files": [],
    }

    report = render_adapter_report("SFT v1 Adapter", result)

    assert "# SFT v1 Adapter" in report
    assert "adapter_model.safetensors" in report
    assert "不允许进入下一步" in report
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_stage2_adapter.py -v
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `small_model_train.stage2_adapter`.

- [ ] **Step 3: Implement adapter check module**

Create `src/small_model_train/stage2_adapter.py`:

```python
from __future__ import annotations

from pathlib import Path


REQUIRED_ADAPTER_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "training_config_snapshot.yaml",
]


def check_adapter_dir(adapter_dir: str | Path) -> dict:
    root = Path(adapter_dir)
    missing_files = [name for name in REQUIRED_ADAPTER_FILES if not (root / name).exists()]
    zero_size_files = [
        name
        for name in REQUIRED_ADAPTER_FILES
        if (root / name).exists() and (root / name).is_file() and (root / name).stat().st_size == 0
    ]
    return {
        "adapter_dir": str(root),
        "passed": not missing_files and not zero_size_files,
        "missing_files": missing_files,
        "zero_size_files": zero_size_files,
    }


def render_adapter_report(title: str, result: dict) -> str:
    decision = "允许进入下一步" if result.get("passed") else "不允许进入下一步"
    lines = [
        f"# {title}",
        "",
        f"- adapter_dir: {result.get('adapter_dir', '')}",
        f"- decision: {decision}",
        "",
        "## Missing Files",
    ]
    missing = result.get("missing_files", [])
    if missing:
        for name in missing:
            lines.append(f"- {name}")
    else:
        lines.append("- 无")
    lines.extend(["", "## Zero Size Files"])
    zero_size = result.get("zero_size_files", [])
    if zero_size:
        for name in zero_size:
            lines.append(f"- {name}")
    else:
        lines.append("- 无")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Implement adapter check CLI**

Create `scripts/check_adapter.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.stage2_adapter import check_adapter_dir, render_adapter_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--title", default="Adapter Check Report")
    args = parser.parse_args()

    result = check_adapter_dir(args.adapter_dir)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_adapter_report(args.title, result), encoding="utf-8")
    print(f"wrote adapter report to {report_path}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Implement OOM probe CLI as a dry-run first probe plan**

Create `scripts/run_oom_probe.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path


PROBES = [
    "Probe 1: load tokenizer and config",
    "Probe 2: load 4-bit base model",
    "Probe 3: inject LoRA",
    "Probe 4: tokenize one sample",
    "Probe 5: cutoff_len=8192, max_steps=1",
    "Probe 6: cutoff_len=6144, max_steps=1",
    "Probe 7: cutoff_len=6144, lora_rank=8, max_steps=1",
]


def render_probe_report() -> str:
    lines = ["# OOM Probe Report", "", "## Probe Plan"]
    for probe in PROBES:
        lines.append(f"- {probe}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "- Probe 2 失败：基座 4-bit 加载不稳，优先查 bitsandbytes / CUDA / 显存占用。",
            "- Probe 3 失败：LoRA target 或 PEFT 配置问题。",
            "- Probe 5 失败但 Probe 6 成功：8192 cutoff_len 过高。",
            "- Probe 6 失败但 Probe 7 成功：LoRA rank 或反向传播显存压力过高。",
            "- 所有 probe 通过但正式训练失败：检查数据长度分布、保存、日志或长时间运行稳定性。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="reports/oom_probe_report.md")
    args = parser.parse_args()
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_probe_report(), encoding="utf-8")
    print(f"wrote OOM probe report to {report_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_stage2_adapter.py -v
```

Expected: PASS for all tests in `tests/test_stage2_adapter.py`.

- [ ] **Step 7: Generate the OOM probe report**

Run:

```powershell
python scripts/run_oom_probe.py --report reports/oom_probe_report.md
```

Expected: writes `reports/oom_probe_report.md` with all seven probe steps and their interpretation.

- [ ] **Step 8: Commit**

Run:

```powershell
git add src/small_model_train/stage2_adapter.py scripts/check_adapter.py scripts/run_oom_probe.py tests/test_stage2_adapter.py reports/oom_probe_report.md
git commit -m "feat: add adapter checks and oom probe report"
```

Expected: commit succeeds.

## Task 7: Fixed Eval Inference Rows And README Stage 2 Usage

**Files:**
- Create: `src/small_model_train/stage2_inference.py`
- Create: `scripts/run_eval_inference.py`
- Modify: `README.md`
- Test: `tests/test_stage2_inference.py`

- [ ] **Step 1: Write failing tests for eval row formatting**

Create `tests/test_stage2_inference.py`:

```python
from small_model_train.stage2_inference import build_generation_row, default_inference_params, render_eval_prompt


def test_render_eval_prompt_uses_existing_card_fields():
    card = {
        "id": "case1",
        "style_contract": "只输出正文。",
        "previous_summary": "上一章结束。",
        "chapter_goal": "完成交易。",
        "chapter_structure": [],
        "character_states": [],
        "must_include": ["加钱"],
        "must_not_include": ["真相"],
        "ending_hook": "箱子响了一下。",
        "target_word_count": "2000-2500中文汉字",
    }

    prompt = render_eval_prompt(card)

    assert "完成交易" in prompt
    assert "只输出正文" in prompt
    assert "加钱" in prompt


def test_build_generation_row_keeps_fixed_schema():
    row = build_generation_row("case1", "正文", "sft_v1", {"temperature": 0.7})

    assert row == {
        "id": "case1",
        "output": "正文",
        "model": "sft_v1",
        "params": {"temperature": 0.7},
    }


def test_default_inference_params_match_stage_two_config():
    params = default_inference_params()

    assert params["temperature"] == 0.7
    assert params["top_p"] == 0.8
    assert params["top_k"] == 20
    assert params["repetition_penalty"] == 1.05
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_stage2_inference.py -v
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `small_model_train.stage2_inference`.

- [ ] **Step 3: Implement inference prompt and row helpers**

Create `src/small_model_train/stage2_inference.py`:

```python
from __future__ import annotations

from small_model_train.sft_builder import render_sft_input


def default_inference_params() -> dict:
    return {
        "max_new_tokens": 5120,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.05,
    }


def render_eval_prompt(card: dict) -> str:
    return render_sft_input(card)


def build_generation_row(sample_id: str, output: str, model: str, params: dict) -> dict:
    return {"id": sample_id, "output": output, "model": model, "params": params}
```

- [ ] **Step 4: Implement eval inference worker for real GPU generation**

Create `scripts/stage2_eval_worker.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.io_utils import read_jsonl, write_jsonl
from small_model_train.stage2_inference import build_generation_row, default_inference_params, render_eval_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="sft_v1")
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    params = default_inference_params()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    rows = []
    for card in read_jsonl(args.cards):
        prompt = render_eval_prompt(card)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        generated = model.generate(
            **inputs,
            max_new_tokens=params["max_new_tokens"],
            temperature=params["temperature"],
            top_p=params["top_p"],
            top_k=params["top_k"],
            repetition_penalty=params["repetition_penalty"],
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tokens = generated[0][inputs["input_ids"].shape[-1] :]
        output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        rows.append(build_generation_row(card["id"], output, args.model_name, params))
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} generation rows to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Implement eval inference supervisor CLI with dry-run support**

Create `scripts/run_eval_inference.py`:

```python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.io_utils import read_jsonl, write_jsonl
from small_model_train.stage2_monitoring import append_event, classify_training_error
from small_model_train.stage2_inference import build_generation_row, default_inference_params, render_eval_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", default="data_cards/eval_cards_50.jsonl")
    parser.add_argument("--model-dir", default=r"E:\models\Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter-dir", default="outputs/sft_v1")
    parser.add_argument("--output", default="outputs/sft_v1/generated.jsonl")
    parser.add_argument("--model-name", default="sft_v1")
    parser.add_argument("--event-log", default="logs/training/sft_v1_eval_events.jsonl")
    parser.add_argument("--stderr-log", default="logs/training/sft_v1_eval_stderr.log")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cards = read_jsonl(args.cards)
    if args.dry_run:
        rows = [
            build_generation_row(card["id"], f"[DRY RUN] {render_eval_prompt(card)[:80]}", args.model_name, default_inference_params())
            for card in cards
        ]
        write_jsonl(args.output, rows)
        print(f"wrote {len(rows)} dry-run generation rows to {args.output}")
        return
    command = [
        sys.executable,
        "scripts/stage2_eval_worker.py",
        "--cards",
        args.cards,
        "--model-dir",
        args.model_dir,
        "--adapter-dir",
        args.adapter_dir,
        "--output",
        args.output,
        "--model-name",
        args.model_name,
    ]
    append_event(args.event_log, "eval_first_generation", "start", {"command": command})
    completed = subprocess.run(command, capture_output=True, check=False, text=True)
    stderr_path = Path(args.stderr_log)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    status = "ok" if completed.returncode == 0 else "failed"
    append_event(args.event_log, "eval_first_generation", status, {"exit_code": completed.returncode})
    if completed.stdout:
        print(completed.stdout.strip())
    if completed.returncode != 0:
        error = classify_training_error(completed.stderr, completed.returncode)
        print(f"{error['error_type']}: {error['suggestion']}", file=sys.stderr)
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Update README with Stage 2 command sequence**

Modify `README.md` by appending:

```markdown

## Stage 2 Training Execution

The local model path is:

```powershell
E:\models\Qwen3-4B-Instruct-2507
```

Run checks before training:

```powershell
python scripts/check_local_model.py --model-dir E:\models\Qwen3-4B-Instruct-2507 --report reports/model_check_report.md
python scripts/check_training_env.py --report reports/training_env_report.md
```

Create and inspect smoke-run config before launching real GPU training:

```powershell
python scripts/run_sft_smoke.py --dry-run
```

When the dry run is correct and GPU memory is available, launch smoke training:

```powershell
python scripts/run_sft_smoke.py
python scripts/check_adapter.py --adapter-dir outputs/sft_smoke --report reports/sft_smoke_report.md --title "SFT Smoke Adapter"
```

After smoke training passes, launch SFT v1:

```powershell
python scripts/run_sft_train.py
python scripts/check_adapter.py --adapter-dir outputs/sft_v1 --report reports/sft_v1_training_report.md --title "SFT v1 Adapter"
```

Generate fixed eval outputs and reuse Stage 1 scoring:

```powershell
python scripts/run_eval_inference.py
python scripts/score_outputs.py --cards data_cards/eval_cards_50.jsonl --outputs outputs/sft_v1/generated.jsonl --output outputs/sft_v1/metrics.jsonl
python scripts/evaluate_outputs.py --scores outputs/sft_v1/metrics.jsonl --report reports/sft_v1_report.md --title "SFT v1 Report"
```

If training OOMs or crashes, create the probe report:

```powershell
python scripts/run_oom_probe.py --report reports/oom_probe_report.md
```
```

- [ ] **Step 7: Run tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_stage2_inference.py -v
```

Expected: PASS for all tests in `tests/test_stage2_inference.py`.

- [ ] **Step 8: Run the full test suite**

Run:

```powershell
python -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 9: Commit**

Run:

```powershell
git add src/small_model_train/stage2_inference.py scripts/run_eval_inference.py scripts/stage2_eval_worker.py tests/test_stage2_inference.py README.md
git commit -m "feat: add stage two eval inference"
```

Expected: commit succeeds.

## Manual Stage 2 Execution Checklist

After all code tasks pass, run the Stage 2 flow manually:

```powershell
python scripts/check_local_model.py --model-dir E:\models\Qwen3-4B-Instruct-2507 --report reports/model_check_report.md
python scripts/check_training_env.py --report reports/training_env_report.md
python scripts/run_sft_smoke.py --dry-run
python scripts/run_sft_smoke.py
python scripts/check_adapter.py --adapter-dir outputs/sft_smoke --report reports/sft_smoke_report.md --title "SFT Smoke Adapter"
python scripts/run_sft_train.py
python scripts/check_adapter.py --adapter-dir outputs/sft_v1 --report reports/sft_v1_training_report.md --title "SFT v1 Adapter"
python scripts/run_eval_inference.py
python scripts/score_outputs.py --cards data_cards/eval_cards_50.jsonl --outputs outputs/sft_v1/generated.jsonl --output outputs/sft_v1/metrics.jsonl
python scripts/evaluate_outputs.py --scores outputs/sft_v1/metrics.jsonl --report reports/sft_v1_report.md --title "SFT v1 Report"
```

Expected final artifacts:

```text
reports/model_check_report.md
reports/training_env_report.md
reports/sft_smoke_report.md
reports/sft_v1_training_report.md
outputs/sft_v1/generated.jsonl
outputs/sft_v1/metrics.jsonl
reports/sft_v1_report.md
```

If smoke or full training fails, run:

```powershell
python scripts/run_oom_probe.py --report reports/oom_probe_report.md
```

Expected failure artifacts:

```text
logs/training/sft_smoke_events.jsonl
logs/training/sft_smoke_stderr.log
logs/training/sft_v1_events.jsonl
logs/training/sft_v1_stderr.log
reports/oom_probe_report.md
```

## Self-Review

Spec coverage:

- Local model checks are implemented by Task 1.
- Training environment checks are implemented by Task 2.
- Runtime config snapshots are implemented by Task 3.
- Event logs, stderr classification, and failure summaries are implemented by Task 4.
- Smoke and full SFT launchers are implemented by Task 5.
- Adapter checks and the OOM probe report are implemented by Task 6.
- Fixed eval output schema and Stage 1 scoring handoff are implemented by Task 7.
- Stage 3 remains a design-only forward plan in the approved spec and is not implemented here.

Type consistency:

- Training run dictionaries consistently use `name`, `snapshot`, `config_path`, `command`, `event_log`, `gpu_log`, and `stderr_log`.
- Generation rows consistently use `id`, `output`, `model`, and `params`.
- Check result dictionaries consistently expose `passed` plus file-specific lists or `errors`.

Execution boundary:

- Automated tests do not require real GPU training.
- Real QLoRA training remains a manual Stage 2 execution step because it depends on CUDA, bitsandbytes, LLaMA-Factory, and local data readiness.
- The eval inference supervisor launches `scripts/stage2_eval_worker.py` as a subprocess so GPU crashes preserve event and stderr logs.
