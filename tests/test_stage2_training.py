import json
import io
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
    assert run["failure_report"] == str(
        tmp_path / "logs" / "training" / "sft_smoke_failure_report.md"
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

    class FakePopen:
        def __init__(self, command, **kwargs):
            calls.append((command, kwargs))
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO("RuntimeError: CUDA out of memory")
            self.returncode = 1

        def poll(self):
            return self.returncode

        def wait(self):
            return self.returncode

    monkeypatch.setattr(stage2_training.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(
        stage2_training,
        "collect_gpu_processes",
        lambda: [],
        raising=False,
    )

    result = run_training_subprocess(run)

    assert calls[0][0] == ["llamafactory-cli", "train", "config.yaml"]
    assert calls[0][1]["stdout"] == stage2_training.subprocess.PIPE
    assert calls[0][1]["stderr"] == stage2_training.subprocess.PIPE
    assert calls[0][1]["text"] is True
    assert calls[0][1]["bufsize"] == 1
    assert calls[0][1]["env"]["WANDB_DISABLED"] == "true"
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
        "gpu_sample_interval_seconds": 0.01,
    }
    gpu_samples = [
        [{"pid": 1234, "name": "python.exe", "used_mb": 12000}],
        [{"pid": 1234, "name": "python.exe", "used_mb": 12600}],
        [{"pid": 1234, "name": "python.exe", "used_mb": 12900}],
    ]

    class FakePopen:
        def __init__(self, *_args, **_kwargs):
            self.stdout = io.StringIO("loss: 1.23\n")
            self.stderr = io.StringIO("")
            self.returncode = 0

        def poll(self):
            return self.returncode

        def wait(self):
            return self.returncode

    def fake_collect_gpu_processes():
        return gpu_samples.pop(0)

    monkeypatch.setattr(
        stage2_training.subprocess,
        "Popen",
        FakePopen,
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
        "during_subprocess",
        "after_subprocess",
    ]
    assert gpu_rows[0]["processes"] == [
        {"pid": 1234, "name": "python.exe", "used_mb": 12000}
    ]
    assert gpu_rows[1]["processes"] == [
        {"pid": 1234, "name": "python.exe", "used_mb": 12600}
    ]
    assert gpu_rows[2]["processes"] == [
        {"pid": 1234, "name": "python.exe", "used_mb": 12900}
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

    def fake_popen(*args, **kwargs):
        raise FileNotFoundError("llamafactory-cli")

    monkeypatch.setattr(stage2_training.subprocess, "Popen", fake_popen)
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


def test_run_training_subprocess_records_phase_markers_from_streamed_logs(
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

    class FakePopen:
        def __init__(self, *_args, **_kwargs):
            self.stdout = io.StringIO(
                "Loading tokenizer\n"
                "load 4-bit base model\n"
                "Preparing LoRA adapter\n"
                "Tokenize dataset\n"
                "first forward\n"
                "save adapter\n"
            )
            self.stderr = io.StringIO("first backward\nfirst optimizer step\n")
            self.returncode = 0

        def poll(self):
            return self.returncode

        def wait(self):
            return self.returncode

    monkeypatch.setattr(stage2_training.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(stage2_training, "collect_gpu_processes", lambda: [])

    result = run_training_subprocess(run)

    events = [
        json.loads(line)
        for line in Path(run["event_log"]).read_text(encoding="utf-8").splitlines()
    ]
    seen_phases = {
        event["phase"] for event in events if event["status"] == "seen_in_log"
    }
    assert result["exit_code"] == 0
    assert {
        "load_tokenizer",
        "load_base_model_4bit",
        "prepare_lora",
        "tokenize_dataset",
        "first_forward",
        "first_backward",
        "first_optimizer_step",
        "save_adapter",
    } <= seen_phases


def test_run_training_subprocess_does_not_mark_prepare_lora_for_lora_rank_config(
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

    class FakePopen:
        def __init__(self, *_args, **_kwargs):
            self.stdout = io.StringIO("training_config: lora_rank: 8\n")
            self.stderr = io.StringIO("")
            self.returncode = 0

        def poll(self):
            return self.returncode

        def wait(self):
            return self.returncode

    monkeypatch.setattr(stage2_training.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(stage2_training, "collect_gpu_processes", lambda: [])

    result = run_training_subprocess(run)

    events = [
        json.loads(line)
        for line in Path(run["event_log"]).read_text(encoding="utf-8").splitlines()
    ]
    assert result["exit_code"] == 0
    assert not any(
        event["phase"] == "prepare_lora" and event["status"] == "seen_in_log"
        for event in events
    )


def test_run_training_subprocess_classifies_stdout_oom_and_writes_failure_report(
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
        "failure_report": str(tmp_path / "failure.md"),
    }

    class FakePopen:
        def __init__(self, *_args, **_kwargs):
            self.stdout = io.StringIO(
                "Loading tokenizer\nRuntimeError: CUDA out of memory\n"
            )
            self.stderr = io.StringIO("")
            self.returncode = 1

        def poll(self):
            return self.returncode

        def wait(self):
            return self.returncode

    monkeypatch.setattr(stage2_training.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(
        stage2_training,
        "collect_gpu_processes",
        lambda: [{"pid": 7, "name": "python.exe", "used_mb": 16000}],
    )

    result = run_training_subprocess(run)

    report = Path(run["failure_report"]).read_text(encoding="utf-8")
    assert result["error"]["error_type"] == "cuda_oom"
    assert "cuda_oom" in report
    assert "load_tokenizer" in report
    assert "RuntimeError: CUDA out of memory" in report
    assert "16000" in report


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


def test_validate_full_training_prerequisites_requires_reports_and_smoke_adapter(
    tmp_path: Path,
):
    from scripts.run_sft_train import validate_full_training_prerequisites

    model_report = tmp_path / "reports" / "model.md"
    env_report = tmp_path / "reports" / "env.md"
    smoke_adapter = tmp_path / "outputs" / "sft_smoke"

    result = validate_full_training_prerequisites(
        model_report=model_report,
        env_report=env_report,
        smoke_adapter_dir=smoke_adapter,
    )

    assert result["passed"] is False
    assert "model.md" in "\n".join(result["errors"])
    assert "env.md" in "\n".join(result["errors"])
    assert "adapter_config.json" in "\n".join(result["errors"])

    model_report.parent.mkdir(parents=True)
    model_report.write_text("ok\n", encoding="utf-8")
    env_report.write_text("ok\n", encoding="utf-8")
    smoke_adapter.mkdir(parents=True)
    (smoke_adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    header = b"{}"
    (smoke_adapter / "adapter_model.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header
    )
    (smoke_adapter / "training_config_snapshot.yaml").write_text(
        "output_dir: smoke\n",
        encoding="utf-8",
    )

    result = validate_full_training_prerequisites(
        model_report=model_report,
        env_report=env_report,
        smoke_adapter_dir=smoke_adapter,
    )

    assert result == {"passed": True, "errors": []}
