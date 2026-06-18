import json
from pathlib import Path

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
        "stderr_log": str(tmp_path / "stderr.log"),
    }
    calls = []

    class CompletedProcess:
        returncode = 1
        stderr = "RuntimeError: CUDA out of memory"

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return CompletedProcess()

    monkeypatch.setattr(stage2_training.subprocess, "run", fake_run)

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
