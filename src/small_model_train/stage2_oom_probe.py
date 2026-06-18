"""OOM and crash probe orchestration for Stage 2.

Each probe runs in a separate worker process. That isolation is deliberate: if
CUDA, bitsandbytes, or the driver kills a worker, the parent can still record
which probe failed and where the logs live.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from small_model_train.stage2_monitoring import (
    append_event,
    classify_training_error,
    collect_gpu_processes,
    now_iso,
)

PROBES = [
    "Probe 1: load tokenizer and config",
    "Probe 2: load 4-bit base model",
    "Probe 3: inject LoRA",
    "Probe 4: tokenize one sample",
    "Probe 5: cutoff_len=8192, max_steps=1",
    "Probe 6: cutoff_len=6144, max_steps=1",
    "Probe 7: cutoff_len=6144, lora_rank=8, max_steps=1",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_PATH = REPO_ROOT / "scripts" / "stage2_oom_probe_worker.py"


def run_oom_probes(
    model_dir: str | Path,
    cards: str | Path,
    sft_dataset: str | Path,
    config: str | Path,
    log_dir: str | Path,
) -> list[dict[str, Any]]:
    results = []
    for index, probe in enumerate(PROBES, start=1):
        results.append(
            run_one_probe(
                probe=probe,
                index=index,
                model_dir=model_dir,
                cards=cards,
                sft_dataset=sft_dataset,
                config=config,
                log_dir=log_dir,
            )
        )
    return results


def run_one_probe(
    probe: str,
    index: int,
    model_dir: str | Path,
    cards: str | Path,
    sft_dataset: str | Path,
    config: str | Path,
    log_dir: str | Path,
) -> dict[str, Any]:
    root = Path(log_dir)
    root.mkdir(parents=True, exist_ok=True)
    slug = _probe_slug(index, probe)
    # Every probe gets separate stdout, stderr, event, and GPU logs so the last successful phase is recoverable after a crash.
    stdout_log = root / f"{slug}_stdout.log"
    stderr_log = root / f"{slug}_stderr.log"
    gpu_log = root / f"{slug}_gpu.jsonl"
    event_log = root / f"{slug}_events.jsonl"
    command = build_probe_command(
        index=index,
        model_dir=model_dir,
        cards=cards,
        sft_dataset=sft_dataset,
        config=config,
        log_dir=log_dir,
    )

    append_event(event_log, "launch_probe", "start", {"probe": probe, "command": command})
    _append_gpu_sample(gpu_log, "before_probe")
    stdout = ""
    stderr = ""
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        stdout, stderr = _communicate_streams(process, stdout_log, stderr_log)
        exit_code = process.wait()
    except (FileNotFoundError, OSError) as exc:
        exit_code = 127
        stderr = f"{type(exc).__name__}: {exc}"
        _write_text(stdout_log, "")
        _write_text(stderr_log, stderr)
    finally:
        _append_gpu_sample(gpu_log, "after_probe")

    error = classify_training_error(stderr + "\n" + stdout, exit_code)
    status = "passed" if exit_code == 0 else "failed"
    append_event(
        event_log,
        "launch_probe",
        status,
        {"probe": probe, "exit_code": exit_code, "error": error},
    )
    return {
        "probe": probe,
        "status": status,
        "exit_code": exit_code,
        "error_type": error["error_type"],
        "suggestion": error["suggestion"],
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "event_log": str(event_log),
        "gpu_log": str(gpu_log),
        "command": command,
    }


def build_probe_command(
    index: int,
    model_dir: str | Path,
    cards: str | Path,
    sft_dataset: str | Path,
    config: str | Path,
    log_dir: str | Path,
) -> list[str]:
    return [
        sys.executable,
        str(WORKER_PATH),
        "--probe",
        str(index),
        "--model-dir",
        str(model_dir),
        "--cards",
        str(cards),
        "--sft-dataset",
        str(sft_dataset),
        "--config",
        str(config),
        "--log-dir",
        str(log_dir),
    ]


def render_probe_report(results: list[dict[str, Any]] | None = None) -> str:
    lines = ["# OOM Probe Report", "", "## Probe Plan"]
    lines.extend(f"- {probe}" for probe in PROBES)
    if results is not None:
        lines.extend(["", "## Probe Results"])
        for result in results:
            lines.append(
                "- "
                f"{result.get('probe', '')}: {result.get('status', '')} "
                f"(exit={result.get('exit_code')}, "
                f"error={result.get('error_type', 'none')})"
            )
            suggestion = result.get("suggestion")
            if suggestion:
                lines.append(f"  suggestion: {suggestion}")
            stdout_log = result.get("stdout_log")
            stderr_log = result.get("stderr_log")
            if stdout_log or stderr_log:
                lines.append(f"  logs: stdout={stdout_log}, stderr={stderr_log}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "- Probe 2 失败：基座 4-bit 加载不稳，优先查 bitsandbytes / CUDA / 显存占用。",
            "- Probe 3 失败：LoRA target 或 PEFT 配置问题。",
            "- Probe 5 失败但 Probe 6 成功：8192 cutoff_len 过高。",
            "- Probe 6 失败但 Probe 7 成功：LoRA rank 或反向传播显存压力过高。",
            "- 所有 probe 通过但正式训练失败：检查数据长度分布、保存、日志或长时间运行稳定性。",
            "",
        ]
    )
    return "\n".join(lines)


def _communicate_streams(
    process: subprocess.Popen[str],
    stdout_log: Path,
    stderr_log: Path,
) -> tuple[str, str]:
    # The parent streams both pipes while the child runs; this avoids losing buffered output when the child exits abruptly.
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    _write_text(stdout_log, "")
    _write_text(stderr_log, "")
    stdout_thread = threading.Thread(
        target=_stream_to_log,
        args=(process.stdout, stdout_log, stdout_chunks),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_stream_to_log,
        args=(process.stderr, stderr_log, stderr_chunks),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    stdout_thread.join()
    stderr_thread.join()
    return "".join(stdout_chunks), "".join(stderr_chunks)


def _stream_to_log(stream: Any, path: Path, chunks: list[str]) -> None:
    if stream is None:
        return

    for line in stream:
        chunks.append(line)
        _append_text(path, line)


def _append_gpu_sample(path: str | Path, phase: str) -> None:
    try:
        processes = collect_gpu_processes()
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        processes = []
    row = {"time": now_iso(), "phase": phase, "processes": processes}
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_text(path: str | Path, text: str) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(text, encoding="utf-8")


def _append_text(path: str | Path, text: str) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def _probe_slug(index: int, probe: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", probe.lower()).strip("_")
    return f"{index:02d}_{slug}"
