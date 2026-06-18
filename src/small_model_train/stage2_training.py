from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from small_model_train.stage2_config import (
    build_llamafactory_command,
    make_training_snapshot,
)
from small_model_train.stage2_monitoring import (
    append_event,
    classify_training_error,
)

SMOKE_OVERRIDES = {
    "max_samples": 100,
    "save_steps": 50,
    "logging_steps": 5,
    "num_train_epochs": 1,
}


def validate_training_inputs(
    sft_dataset: str | Path,
    eval_cards: str | Path,
) -> dict[str, Any]:
    errors = []
    for label, raw_path in (
        ("SFT dataset", sft_dataset),
        ("eval cards", eval_cards),
    ):
        path = Path(raw_path)
        if not path.exists():
            errors.append(f"{label} is missing: {path}")
        elif path.stat().st_size == 0:
            errors.append(f"{label} is empty: {path}")

    return {"passed": not errors, "errors": errors}


def build_train_run(
    name: str,
    source_config: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    log_dir: str | Path,
    smoke: bool,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    config_path = output_path / "training_config_snapshot.yaml"
    log_path = Path(log_dir)
    overrides = dict(SMOKE_OVERRIDES) if smoke else {}
    snapshot = make_training_snapshot(
        source_config=source_config,
        output_config=config_path,
        model_dir=model_dir,
        output_dir=output_path,
        overrides=overrides,
    )

    return {
        "name": name,
        "snapshot": snapshot,
        "config_path": str(config_path),
        "command": build_llamafactory_command(config_path),
        "event_log": str(log_path / f"{name}_events.jsonl"),
        "gpu_log": str(log_path / f"{name}_gpu.jsonl"),
        "stderr_log": str(log_path / f"{name}_stderr.log"),
    }


def run_training_dry(run: dict[str, Any]) -> dict[str, Any]:
    command_text = _command_text(run["command"])
    append_event(
        run["event_log"],
        "prepare_config",
        "ok",
        {"run": run["name"], "config_path": run.get("config_path")},
    )
    append_event(
        run["event_log"],
        "launch_train_subprocess",
        "dry_run",
        {"run": run["name"], "command": command_text},
    )

    return {
        "exit_code": 0,
        "command_text": command_text,
        "error": {"error_type": "none", "suggestion": "无"},
    }


def run_training_subprocess(run: dict[str, Any]) -> dict[str, Any]:
    command_text = _command_text(run["command"])
    append_event(
        run["event_log"],
        "launch_train_subprocess",
        "start",
        {"run": run["name"], "command": command_text},
    )
    result = subprocess.run(
        run["command"],
        capture_output=True,
        check=False,
        text=True,
    )
    stderr = result.stderr or ""
    stderr_log = Path(run["stderr_log"])
    stderr_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.write_text(stderr, encoding="utf-8")

    error = classify_training_error(stderr, result.returncode)
    status = "ok" if result.returncode == 0 else "failed"
    append_event(
        run["event_log"],
        "launch_train_subprocess",
        status,
        {
            "run": run["name"],
            "exit_code": result.returncode,
            "error": error,
        },
    )

    return {
        "exit_code": result.returncode,
        "command_text": command_text,
        "error": error,
    }


def _command_text(command: list[str]) -> str:
    return " ".join(str(part) for part in command)
