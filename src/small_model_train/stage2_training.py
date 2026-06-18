from __future__ import annotations

import json
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from small_model_train.stage2_config import (
    build_llamafactory_command,
    make_training_snapshot,
)
from small_model_train.stage2_monitoring import (
    append_event,
    classify_training_error,
    collect_gpu_processes,
    now_iso,
    render_failure_summary,
)

SMOKE_OVERRIDES = {
    "max_samples": 100,
    "save_steps": 50,
    "logging_steps": 5,
    "num_train_epochs": 1,
}

LOG_PHASE_MARKERS = (
    ("load_tokenizer", ("loading tokenizer", "load tokenizer", "autotokenizer")),
    (
        "load_base_model_4bit",
        ("load 4-bit base model", "load 4bit base model", "4-bit base model"),
    ),
    ("prepare_lora", ("prepare lora", "preparing lora", "inject lora", "lora")),
    ("tokenize_dataset", ("tokenize dataset", "tokenizing dataset")),
    ("first_forward", ("first forward", "first_forward")),
    ("first_backward", ("first backward", "first_backward")),
    ("first_optimizer_step", ("first optimizer step", "first_optimizer_step")),
    ("save_adapter", ("save adapter", "saving adapter", "save_pretrained")),
)


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
        "stdout_log": str(log_path / f"{name}_stdout.log"),
        "failure_report": str(log_path / f"{name}_failure_report.md"),
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
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    seen_phases: set[str] = set()
    append_event(
        run["event_log"],
        "launch_train_subprocess",
        "start",
        {"run": run["name"], "command": command_text},
    )
    _append_gpu_sample(run, "before_subprocess")
    try:
        process = subprocess.Popen(
            run["command"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        _append_gpu_sample(run, "during_subprocess")
        stdout_thread = _start_stream_thread(
            process.stdout,
            run.get("stdout_log"),
            stdout_chunks,
            run,
            seen_phases,
        )
        stderr_thread = _start_stream_thread(
            process.stderr,
            run.get("stderr_log"),
            stderr_chunks,
            run,
            seen_phases,
        )
        interval = float(run.get("gpu_sample_interval_seconds", 2.0))
        while process.poll() is None:
            time.sleep(max(interval, 0.001))
            _append_gpu_sample(run, "during_subprocess")
        exit_code = process.wait()
        stdout_thread.join()
        stderr_thread.join()
    except (FileNotFoundError, OSError) as exc:
        exit_code = 127
        stderr_chunks.append(f"{type(exc).__name__}: {exc}")
        _write_text_log(run.get("stdout_log"), "")
        _write_text_log(run["stderr_log"], stderr_chunks[0])
    finally:
        _append_gpu_sample(run, "after_subprocess")

    stdout = "".join(stdout_chunks)
    stderr = "".join(stderr_chunks)
    combined_output = stderr + "\n" + stdout

    error = classify_training_error(combined_output, exit_code)
    status = "ok" if exit_code == 0 else "failed"
    append_event(
        run["event_log"],
        "launch_train_subprocess",
        status,
        {
            "run": run["name"],
            "exit_code": exit_code,
            "error": error,
        },
    )
    if exit_code != 0:
        _write_failure_report(run, error, exit_code, stdout, stderr, combined_output)

    return {
        "exit_code": exit_code,
        "command_text": command_text,
        "error": error,
    }


def _command_text(command: list[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in command])


def _write_text_log(path: str | Path | None, text: str) -> None:
    if path is None:
        return

    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(text, encoding="utf-8")


def _start_stream_thread(
    stream: Any,
    log_path: str | Path | None,
    chunks: list[str],
    run: dict[str, Any],
    seen_phases: set[str],
) -> threading.Thread:
    _write_text_log(log_path, "")
    thread = threading.Thread(
        target=_stream_output,
        args=(stream, log_path, chunks, run, seen_phases),
        daemon=True,
    )
    thread.start()
    return thread


def _stream_output(
    stream: Any,
    log_path: str | Path | None,
    chunks: list[str],
    run: dict[str, Any],
    seen_phases: set[str],
) -> None:
    if stream is None:
        return

    for line in stream:
        chunks.append(line)
        _append_text_log(log_path, line)
        _record_log_markers(run, seen_phases, line)


def _append_text_log(path: str | Path | None, text: str) -> None:
    if path is None:
        return

    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def _record_log_markers(
    run: dict[str, Any],
    seen_phases: set[str],
    text: str,
) -> None:
    lowered = text.lower()
    for phase, markers in LOG_PHASE_MARKERS:
        if phase in seen_phases:
            continue
        if any(marker in lowered for marker in markers):
            seen_phases.add(phase)
            append_event(
                run["event_log"],
                phase,
                "seen_in_log",
                {"run": run["name"], "line": text.strip()},
            )


def _write_failure_report(
    run: dict[str, Any],
    error: dict[str, str],
    exit_code: int | None,
    stdout: str,
    stderr: str,
    combined_output: str,
) -> None:
    report_path = run.get("failure_report")
    if not report_path:
        return

    summary = render_failure_summary(
        error=error,
        last_events=_read_jsonl_tail(run.get("event_log"), 20),
        last_gpu_samples=_read_jsonl_tail(run.get("gpu_log"), 10),
        exit_code=exit_code,
        stderr_tail=_tail_text(stderr),
        stdout_tail=_tail_text(stdout),
        combined_tail=_tail_text(combined_output),
    )
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary, encoding="utf-8")


def _read_jsonl_tail(path: str | Path | None, limit: int) -> list[dict[str, Any]]:
    if path is None:
        return []
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return []

    rows: deque[dict[str, Any]] = deque(maxlen=limit)
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return list(rows)


def _tail_text(text: str, max_lines: int = 80) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def _append_gpu_sample(run: dict[str, Any], phase: str) -> None:
    gpu_log = run.get("gpu_log")
    if not gpu_log:
        return

    try:
        processes = collect_gpu_processes()
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        processes = []

    row = {
        "time": now_iso(),
        "phase": phase,
        "processes": processes,
    }
    gpu_path = Path(gpu_log)
    gpu_path.parent.mkdir(parents=True, exist_ok=True)
    with gpu_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
