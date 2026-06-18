import json
from pathlib import Path

import pytest

from small_model_train import stage2_training
from small_model_train.stage2_training import (
    build_train_run,
    run_training_dry,
    run_training_subprocess,
    validate_training_inputs,
)


def test_validate_training_inputs_reports_missing_files(tmp_path: Path):
    result = validate_training_inputs(
        sft_dataset=tmp_path / "missing_sft.jsonl",
        eval_cards=tmp_path / "missing_eval.jsonl",
    )

    assert result["passed"] is False
    assert "missing_sft.jsonl" in "\n".join(result["errors"])


def test_validate_training_inputs_reports_empty_files(tmp_path: Path):
    sft_dataset = tmp_path / "empty_sft.jsonl"
    eval_cards = tmp_path / "empty_eval.jsonl"
    sft_dataset.touch()
    eval_cards.touch()

    result = validate_training_inputs(
        sft_dataset=sft_dataset,
        eval_cards=eval_cards,
    )

    errors = "\n".join(result["errors"])
    assert result["passed"] is False
    assert "empty_sft.jsonl" in errors
    assert "empty_eval.jsonl" in errors


def test_build_train_run_creates_smoke_snapshot(tmp_path: Path):
    config = tmp_path / "sft.yaml"
    config.write_text(
        "model_name_or_path: remote\noutput_dir: old\ncutoff_len: 8192\n",
        encoding="utf-8",
    )

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
    assert run["stdout_log"] == str(
        tmp_path / "logs" / "training" / "sft_smoke_stdout.log"
    )


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


def test_run_training_subprocess_writes_stderr_and_failed_event(
    tmp_path: Path,
    monkeypatch,
):
    run = {
        "name": "sft_smoke",
        "command": ["llamafactory-cli", "train", "config.yaml"],
        "event_log": str(tmp_path / "events.jsonl"),
        "gpu_log": str(tmp_path / "gpu.jsonl"),
        "stderr_log": str(tmp_path / "stderr.log"),
        "stdout_log": str(tmp_path / "stdout.log"),
    }
    calls = []

    class CompletedProcess:
        returncode = 1
        stderr = "RuntimeError: CUDA out of memory"
        stdout = ""

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return CompletedProcess()

    monkeypatch.setattr(stage2_training.subprocess, "run", fake_run)
    monkeypatch.setattr(
        stage2_training,
        "collect_gpu_processes",
        lambda: [],
        raising=False,
    )

    result = run_training_subprocess(run)

    assert calls == [
        (
            ["llamafactory-cli", "train", "config.yaml"],
            {"capture_output": True, "check": False, "text": True},
        )
    ]
    assert Path(run["stderr_log"]).read_text(encoding="utf-8") == (
        "RuntimeError: CUDA out of memory"
    )
    events = [
        json.loads(line)
        for line in Path(run["event_log"]).read_text(encoding="utf-8").splitlines()
    ]
    assert [event["status"] for event in events] == ["start", "failed"]
    assert events[-1]["detail"]["exit_code"] == 1
    assert result["exit_code"] == 1
    assert result["error"]["error_type"] == "cuda_oom"


def test_run_training_subprocess_preserves_stdout_and_gpu_samples(
    tmp_path: Path,
    monkeypatch,
):
    run = {
        "name": "sft_smoke",
        "command": ["llamafactory-cli", "train", "config.yaml"],
        "event_log": str(tmp_path / "events.jsonl"),
        "gpu_log": str(tmp_path / "gpu.jsonl"),
        "stderr_log": str(tmp_path / "stderr.log"),
        "stdout_log": str(tmp_path / "stdout.log"),
    }
    gpu_samples = [
        [{"pid": 1234, "name": "python.exe", "used_mb": 12000}],
        [{"pid": 1234, "name": "python.exe", "used_mb": 12600}],
    ]

    class CompletedProcess:
        returncode = 0
        stderr = ""
        stdout = "loss: 1.23\n"

    def fake_collect_gpu_processes():
        return gpu_samples.pop(0)

    monkeypatch.setattr(
        stage2_training.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(),
    )
    monkeypatch.setattr(
        stage2_training,
        "collect_gpu_processes",
        fake_collect_gpu_processes,
        raising=False,
    )

    result = run_training_subprocess(run)

    assert result["exit_code"] == 0
    assert Path(run["stdout_log"]).read_text(encoding="utf-8") == "loss: 1.23\n"
    assert Path(run["stderr_log"]).read_text(encoding="utf-8") == ""
    gpu_rows = [
        json.loads(line)
        for line in Path(run["gpu_log"]).read_text(encoding="utf-8").splitlines()
    ]
    assert [row["phase"] for row in gpu_rows] == [
        "before_subprocess",
        "after_subprocess",
    ]
    assert gpu_rows[0]["processes"] == [
        {"pid": 1234, "name": "python.exe", "used_mb": 12000}
    ]
    assert gpu_rows[1]["processes"] == [
        {"pid": 1234, "name": "python.exe", "used_mb": 12600}
    ]


def test_run_training_subprocess_handles_launcher_exceptions(
    tmp_path: Path,
    monkeypatch,
):
    run = {
        "name": "sft_smoke",
        "command": ["llamafactory-cli", "train", "config.yaml"],
        "event_log": str(tmp_path / "events.jsonl"),
        "gpu_log": str(tmp_path / "gpu.jsonl"),
        "stderr_log": str(tmp_path / "stderr.log"),
        "stdout_log": str(tmp_path / "stdout.log"),
    }

    def fake_run(*args, **kwargs):
        raise FileNotFoundError("llamafactory-cli")

    monkeypatch.setattr(stage2_training.subprocess, "run", fake_run)
    monkeypatch.setattr(
        stage2_training,
        "collect_gpu_processes",
        lambda: [],
        raising=False,
    )

    try:
        result = run_training_subprocess(run)
    except FileNotFoundError as exc:
        pytest.fail(f"run_training_subprocess raised {exc!r}")

    stderr_text = Path(run["stderr_log"]).read_text(encoding="utf-8")
    events = [
        json.loads(line)
        for line in Path(run["event_log"]).read_text(encoding="utf-8").splitlines()
    ]

    assert result["exit_code"] != 0
    assert result["error"]["error_type"] in {"process_killed", "llamafactory_error"}
    assert "FileNotFoundError" in stderr_text
    assert "llamafactory-cli" in stderr_text
    assert Path(run["stdout_log"]).read_text(encoding="utf-8") == ""
    assert [event["status"] for event in events] == ["start", "failed"]
    assert events[-1]["detail"]["exit_code"] != 0


def test_command_text_quotes_windows_paths_with_spaces(tmp_path: Path):
    run = {
        "name": "sft_smoke",
        "command": [
            "llamafactory-cli",
            "train",
            r"E:\train configs\snapshot.yaml",
        ],
        "event_log": str(tmp_path / "events.jsonl"),
        "stderr_log": str(tmp_path / "stderr.log"),
    }

    result = run_training_dry(run)

    assert result["command_text"] == (
        'llamafactory-cli train "E:\\train configs\\snapshot.yaml"'
    )
