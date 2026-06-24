import json
import io
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from small_model_train import stage2_training
from small_model_train.io_utils import write_jsonl
from small_model_train.stage2_training import (
    build_train_run,
    run_training_dry,
    run_training_subprocess,
    validate_training_inputs,
)


def _execution_card(sample_id: str = "case1") -> dict[str, Any]:
    return {
        "id": sample_id,
        "target_platform": "hybrid_fanqie_qidian",
        "genre_tags": ["xuanhuan", "system"],
        "style_contract": "短句推进，强冲突。",
        "chapter_goal": "主角发现任务并反击。",
        "chapter_structure": [
            {"step": 1, "name": "压迫", "goal": "建立冲突", "estimated_chars": "800"}
        ],
        "conflict_beat": "旧势力当众羞辱主角。",
        "payoff_beat": "主角用证据完成反击。",
        "must_include": ["系统面板"],
        "must_not_include": ["女频误会流"],
        "ending_hook": "新的任务出现。",
        "target_word_count": "2000-2500中文汉字",
    }


def test_summarize_jsonl_artifact_records_sha_rows_and_schema(tmp_path: Path):
    from small_model_train.artifact_manifest import summarize_jsonl_artifact

    cards_path = tmp_path / "eval_cards.jsonl"
    write_jsonl(cards_path, [_execution_card("case1")])

    summary = summarize_jsonl_artifact(
        cards_path,
        label="eval_cards",
        validate_execution_card_schema=True,
    )

    assert summary["path"] == str(cards_path)
    assert summary["sha256"]
    assert summary["row_count"] == 1
    assert summary["schema"]["name"] == "execution_cards"
    assert summary["schema"]["valid"] is True
    assert summary["schema"]["errors"] == []


def test_summarize_jsonl_artifact_rejects_raw_eval_cards_when_schema_required(
    tmp_path: Path,
):
    from small_model_train.artifact_manifest import summarize_jsonl_artifact

    raw_cards = tmp_path / "eval_cards_50.jsonl"
    write_jsonl(
        raw_cards,
        [{"id": "case1", "text": "原文", "quality_tag": "A", "split": "eval"}],
    )

    summary = summarize_jsonl_artifact(
        raw_cards,
        label="eval_cards",
        validate_execution_card_schema=True,
    )

    assert summary["schema"]["valid"] is False
    assert "missing execution-card fields" in "\n".join(summary["schema"]["errors"])


def test_summarize_jsonl_artifact_streams_non_schema_summary(
    monkeypatch,
    tmp_path: Path,
):
    from small_model_train import artifact_manifest

    dataset_path = tmp_path / "sft.jsonl"
    write_jsonl(dataset_path, [{"id": "row1"}, {"id": "row2"}])

    def fail_materializing_read(_path: Path):
        raise AssertionError("non-schema summary must not materialize rows")

    monkeypatch.setattr(
        artifact_manifest,
        "_read_jsonl_objects",
        fail_materializing_read,
    )

    summary = artifact_manifest.summarize_jsonl_artifact(
        dataset_path,
        label="sft_dataset",
        validate_execution_card_schema=False,
    )

    assert summary["row_count"] == 2
    assert summary["schema"] == {"name": "jsonl", "valid": True, "errors": []}


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


def test_append_event_serializes_concurrent_writes(monkeypatch, tmp_path: Path):
    from small_model_train import stage2_monitoring

    event_log = tmp_path / "events.jsonl"
    start_barrier = threading.Barrier(2)
    active_writes = 0
    active_writes_lock = threading.Lock()
    saw_concurrent_write = False
    written_lines: list[str] = []

    class SlowEventLogHandle:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def write(self, text: str) -> int:
            nonlocal active_writes, saw_concurrent_write
            with active_writes_lock:
                active_writes += 1
                if active_writes > 1:
                    saw_concurrent_write = True
            time.sleep(0.01)
            written_lines.append(text)
            with active_writes_lock:
                active_writes -= 1
            return len(text)

    def fake_open(self, *_args, **_kwargs):
        if self == event_log:
            return SlowEventLogHandle()
        return original_open(self, *_args, **_kwargs)

    original_open = Path.open
    monkeypatch.setattr(Path, "open", fake_open)

    def append_from_thread(index: int) -> None:
        start_barrier.wait()
        stage2_monitoring.append_event(
            event_log,
            phase=f"phase_{index}",
            status="ok",
            detail={"index": index},
        )

    threads = [
        threading.Thread(target=append_from_thread, args=(index,))
        for index in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert saw_concurrent_write is False
    assert len(written_lines) == 2
    assert [json.loads(line)["status"] for line in written_lines] == ["ok", "ok"]


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


def write_json_preflight(
    path: Path,
    *,
    kind: str,
    passed: bool,
    payload: dict[str, Any] | None = None,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
    schema_version: int = 1,
    checked_at: str = "2026-06-23T00:00:00Z",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "kind": kind,
                "passed": passed,
                "checked_at": checked_at,
                "errors": errors or [],
                "warnings": warnings or [],
                "payload": payload or {},
            }
        ),
        encoding="utf-8",
    )


def write_valid_adapter(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    header = b"{}"
    (path / "adapter_model.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header
    )
    (path / "training_config_snapshot.yaml").write_text(
        "output_dir: adapter\n",
        encoding="utf-8",
    )


def _style_contract_asset(status: str = "approved") -> dict[str, Any]:
    from small_model_train.style_contract import build_style_contract_asset

    return build_style_contract_asset(
        style_contract_id="author_main_v1",
        approval_status=status,
        source_corpus={
            "path": "data_clean/chapters_split.jsonl",
            "sha256": "a" * 64,
            "quality_filter": "quality_tag=A",
            "row_count": 1,
            "selected_rows": 1,
            "split_summary": {"train": 1},
        },
        profile_metrics={
            "chapter_count": 1,
            "avg_chinese_chars": 1200,
            "avg_paragraph_chars": 80,
            "avg_dialogue_ratio": 0.3,
            "chinese_chars": {
                "min": 1200,
                "max": 1200,
                "avg": 1200,
                "p50": 1200,
                "p90": 1200,
            },
            "paragraph_chars": {"min": 80, "max": 80, "avg": 80, "p50": 80, "p90": 80},
            "dialogue_ratio": {"min": 0.3, "max": 0.3, "avg": 0.3, "p50": 0.3, "p90": 0.3},
            "sentence_chars": {"min": 12, "max": 12, "avg": 12, "p50": 12, "p90": 12},
            "punctuation_density": {"。": 0.02},
            "ai_taste": {
                "phrase_hits": {"空气仿佛凝固了": 0},
                "total_hits": 0,
                "hits_per_10k_chars": 0,
            },
            "source_filter": {
                "total_rows": 1,
                "selected_rows": 1,
                "skipped_rows": 0,
                "quality_filter": "quality_tag=A",
            },
        },
    )


def test_read_preflight_report_rejects_missing_and_invalid_json(tmp_path: Path):
    from small_model_train.preflight_reports import read_preflight_report

    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="missing"):
        read_preflight_report(missing, expected_kind="model")

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        read_preflight_report(invalid, expected_kind="model")

    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        read_preflight_report(non_object, expected_kind="model")


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"schema_version": 2}, "schema"),
        ({"kind": "environment"}, "kind"),
        ({"passed": "true"}, "passed"),
        ({"checked_at": "not-a-date"}, "checked_at"),
        ({"errors": ["ok", 123]}, "errors"),
        ({"warnings": ["ok", {"bad": True}]}, "warnings"),
    ],
)
def test_read_preflight_report_rejects_invalid_schema_fields(
    tmp_path: Path,
    updates: dict[str, Any],
    match: str,
):
    from small_model_train.preflight_reports import read_preflight_report

    report = {
        "schema_version": 1,
        "kind": "model",
        "passed": True,
        "checked_at": "2026-06-23T00:00:00Z",
        "errors": [],
        "warnings": [],
        "payload": {},
    }
    report.update(updates)
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        read_preflight_report(path, expected_kind="model")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"errors": ["ok", 123]},
        {"warnings": ["ok", object()]},
    ],
)
def test_build_preflight_report_rejects_non_string_list_items(kwargs: dict[str, Any]):
    from small_model_train.preflight_reports import build_preflight_report

    with pytest.raises(ValueError, match=next(iter(kwargs))):
        build_preflight_report(
            kind="model",
            passed=True,
            payload={},
            **kwargs,
        )


def test_validate_full_training_prerequisites_requires_json_reports_and_smoke_adapter(
    tmp_path: Path,
):
    from scripts.run_sft_train import validate_full_training_prerequisites

    model_report = tmp_path / "reports" / "model.json"
    env_report = tmp_path / "reports" / "env.json"
    smoke_adapter = tmp_path / "outputs" / "sft_smoke"

    result = validate_full_training_prerequisites(
        model_report=model_report,
        env_report=env_report,
        smoke_adapter_dir=smoke_adapter,
    )

    assert result["passed"] is False
    assert "model.json" in "\n".join(result["errors"])
    assert "env.json" in "\n".join(result["errors"])
    assert "adapter_config.json" in "\n".join(result["errors"])

    write_json_preflight(model_report, kind="model", passed=True)
    write_json_preflight(env_report, kind="environment", passed=True)
    write_valid_adapter(smoke_adapter)

    result = validate_full_training_prerequisites(
        model_report=model_report,
        env_report=env_report,
        smoke_adapter_dir=smoke_adapter,
    )

    assert result == {"passed": True, "errors": []}


def test_failed_json_preflight_blocks_training_prerequisites(tmp_path: Path):
    from scripts.run_sft_train import validate_full_training_prerequisites

    model_report = tmp_path / "reports" / "model.json"
    env_report = tmp_path / "reports" / "env.json"
    smoke_adapter = tmp_path / "outputs" / "sft_smoke"
    write_json_preflight(
        model_report,
        kind="model",
        passed=False,
        errors=["missing required file: config.json"],
    )
    write_json_preflight(env_report, kind="environment", passed=True)
    write_valid_adapter(smoke_adapter)

    result = validate_full_training_prerequisites(
        model_report=model_report,
        env_report=env_report,
        smoke_adapter_dir=smoke_adapter,
    )

    assert result["passed"] is False
    errors = "\n".join(result["errors"])
    assert "model preflight report did not pass" in errors
    assert "missing required file: config.json" in errors


def test_build_run_manifest_records_training_evidence(monkeypatch, tmp_path: Path):
    from small_model_train import run_manifest

    monkeypatch.setattr(run_manifest, "current_git_commit", lambda _cwd=None: "abc123")

    manifest = run_manifest.build_run_manifest(
        run_name="sft_v1",
        command=["llamafactory-cli", "train", "snapshot.yaml"],
        training_exit_code=0,
        model_dir=tmp_path / "model",
        output_dir=tmp_path / "output",
        config_path=tmp_path / "output" / "training_config_snapshot.yaml",
        preflight_reports={
            "model": {"path": "reports/model.json", "passed": True},
            "environment": {"path": "reports/env.json", "passed": True},
        },
        adapter_check={"passed": True, "errors": []},
        passed=True,
    )

    assert manifest["schema_version"] == 1
    assert manifest["git_commit"] == "abc123"
    assert manifest["run_name"] == "sft_v1"
    assert manifest["command"] == ["llamafactory-cli", "train", "snapshot.yaml"]
    assert manifest["training_exit_code"] == 0
    assert manifest["model_dir"] == str(tmp_path / "model")
    assert manifest["output_dir"] == str(tmp_path / "output")
    assert manifest["config_path"] == str(
        tmp_path / "output" / "training_config_snapshot.yaml"
    )
    assert manifest["preflight_reports"]["model"]["passed"] is True
    assert manifest["adapter_check"] == {"passed": True, "errors": []}
    assert manifest["passed"] is True
    assert manifest["created_at"].endswith("Z")


def test_current_git_commit_returns_empty_string_on_subprocess_error(monkeypatch):
    from small_model_train import run_manifest

    def raise_subprocess_error(*_args, **_kwargs):
        raise subprocess.SubprocessError("git failed")

    monkeypatch.setattr(run_manifest.subprocess, "run", raise_subprocess_error)

    assert run_manifest.current_git_commit() == ""


def test_current_git_commit_passes_cwd_to_git(monkeypatch, tmp_path: Path):
    from small_model_train import run_manifest

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=" abc123 \n")

    monkeypatch.setattr(run_manifest.subprocess, "run", fake_run)

    assert run_manifest.current_git_commit(cwd=tmp_path) == "abc123"
    assert calls == [
        (
            ["git", "rev-parse", "HEAD"],
            {
                "capture_output": True,
                "check": True,
                "text": True,
                "cwd": tmp_path,
            },
        )
    ]


def test_run_sft_train_rejects_legacy_markdown_report_args(monkeypatch):
    from scripts import run_sft_train

    monkeypatch.setattr(
        run_sft_train.sys,
        "argv",
        [
            "run_sft_train.py",
            "--model-report",
            "reports/model_check_report.md",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        run_sft_train.main()

    assert excinfo.value.code == 2


def test_run_sft_train_rejects_legacy_env_report_arg(monkeypatch):
    from scripts import run_sft_train

    monkeypatch.setattr(
        run_sft_train.sys,
        "argv",
        [
            "run_sft_train.py",
            "--env-report",
            "reports/training_env_report.md",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        run_sft_train.main()

    assert excinfo.value.code == 2


def test_run_sft_train_dry_run_rejects_raw_eval_cards_before_manifest(
    monkeypatch,
    tmp_path: Path,
):
    from scripts import run_sft_train

    sft_dataset = tmp_path / "data" / "sft.jsonl"
    raw_eval_cards = tmp_path / "data" / "eval_cards_50.jsonl"
    sft_dataset.parent.mkdir(parents=True)
    sft_dataset.write_text("{}\n", encoding="utf-8")
    write_jsonl(
        raw_eval_cards,
        [{"id": "case1", "text": "原文", "quality_tag": "A", "split": "eval"}],
    )

    model_report = tmp_path / "reports" / "model.json"
    env_report = tmp_path / "reports" / "env.json"
    write_json_preflight(model_report, kind="model", passed=True)
    write_json_preflight(env_report, kind="environment", passed=True)
    write_valid_adapter(tmp_path / "outputs" / "sft_smoke")

    output_dir = tmp_path / "outputs" / "sft_v1"

    def fail_build_train_run(**_kwargs):
        raise AssertionError("schema gate must fail before command construction")

    monkeypatch.setattr(run_sft_train, "build_train_run", fail_build_train_run)
    monkeypatch.setattr(
        run_sft_train.sys,
        "argv",
        [
            "run_sft_train.py",
            "--dry-run",
            "--sft-dataset",
            str(sft_dataset),
            "--eval-cards",
            str(raw_eval_cards),
            "--model-report-json",
            str(model_report),
            "--env-report-json",
            str(env_report),
            "--smoke-adapter-dir",
            str(tmp_path / "outputs" / "sft_smoke"),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert run_sft_train.main() == 1
    assert not (output_dir / "run_manifest.json").exists()


def test_run_sft_smoke_rejects_raw_eval_cards_before_command(
    monkeypatch,
    tmp_path: Path,
):
    from scripts import run_sft_smoke

    sft_dataset = tmp_path / "data" / "sft.jsonl"
    raw_eval_cards = tmp_path / "data" / "eval_cards_50.jsonl"
    sft_dataset.parent.mkdir(parents=True)
    sft_dataset.write_text("{}\n", encoding="utf-8")
    write_jsonl(
        raw_eval_cards,
        [{"id": "case1", "text": "原文", "quality_tag": "A", "split": "eval"}],
    )

    def fail_build_train_run(**_kwargs):
        raise AssertionError("schema gate must fail before smoke command construction")

    monkeypatch.setattr(run_sft_smoke, "build_train_run", fail_build_train_run)
    monkeypatch.setattr(
        run_sft_smoke.sys,
        "argv",
        [
            "run_sft_smoke.py",
            "--dry-run",
            "--sft-dataset",
            str(sft_dataset),
            "--eval-cards",
            str(raw_eval_cards),
            "--output-dir",
            str(tmp_path / "outputs" / "sft_smoke"),
        ],
    )

    assert run_sft_smoke.main() == 1


def test_run_sft_train_dry_run_writes_manifest_without_output_adapter(
    monkeypatch,
    tmp_path: Path,
):
    from scripts import run_sft_train

    sft_dataset = tmp_path / "data" / "sft.jsonl"
    eval_cards = tmp_path / "data" / "eval.jsonl"
    sft_dataset.parent.mkdir(parents=True)
    sft_dataset.write_text("{}\n", encoding="utf-8")
    write_jsonl(eval_cards, [_execution_card("case1")])

    model_report = tmp_path / "reports" / "model.json"
    env_report = tmp_path / "reports" / "env.json"
    write_json_preflight(model_report, kind="model", passed=True)
    write_json_preflight(env_report, kind="environment", passed=True)
    write_valid_adapter(tmp_path / "outputs" / "sft_smoke")

    output_dir = tmp_path / "outputs" / "sft_v1"
    config_path = output_dir / "training_config_snapshot.yaml"

    def fake_build_train_run(**kwargs):
        config_path.parent.mkdir(parents=True)
        config_path.write_text("output_dir: sft_v1\n", encoding="utf-8")
        return {
            "name": kwargs["name"],
            "config_path": str(config_path),
            "command": ["llamafactory-cli", "train", str(config_path)],
        }

    monkeypatch.setattr(run_sft_train, "build_train_run", fake_build_train_run)
    monkeypatch.setattr(
        run_sft_train,
        "run_training_dry",
        lambda _run: {
            "exit_code": 0,
            "command_text": f"llamafactory-cli train {config_path}",
            "error": {"error_type": "none", "suggestion": "dry-run"},
        },
    )
    monkeypatch.setattr(
        run_sft_train.sys,
        "argv",
        [
            "run_sft_train.py",
            "--dry-run",
            "--config",
            str(tmp_path / "config.yaml"),
            "--model-dir",
            str(tmp_path / "model"),
            "--output-dir",
            str(output_dir),
            "--log-dir",
            str(tmp_path / "logs"),
            "--sft-dataset",
            str(sft_dataset),
            "--eval-cards",
            str(eval_cards),
            "--model-report-json",
            str(model_report),
            "--env-report-json",
            str(env_report),
            "--smoke-adapter-dir",
            str(tmp_path / "outputs" / "sft_smoke"),
        ],
    )

    exit_code = run_sft_train.main()

    manifest = json.loads((output_dir / "run_manifest.json").read_text("utf-8"))
    assert exit_code == 0
    assert manifest["training_exit_code"] == 0
    assert manifest["adapter_check"]["status"] == "skipped"
    assert manifest["adapter_check"]["passed"] is False
    assert manifest["adapter_check"]["reason"] == "dry-run does not produce an adapter"
    assert manifest["passed"] is True
    assert manifest["sft_dataset"]["row_count"] == 1
    assert manifest["eval_cards"]["schema"]["valid"] is True
    assert manifest["formal_evidence"] is False


def test_run_sft_train_dry_run_records_style_contract_provenance(
    monkeypatch,
    tmp_path: Path,
):
    from scripts import run_sft_train
    from small_model_train.style_contract import write_style_contract_asset

    sft_dataset = tmp_path / "data" / "sft.jsonl"
    eval_cards = tmp_path / "data" / "eval.jsonl"
    contract_path = tmp_path / "data_style" / "style_contract.json"
    sft_dataset.parent.mkdir(parents=True)
    sft_dataset.write_text("{}\n", encoding="utf-8")
    write_jsonl(eval_cards, [_execution_card("case1")])
    contract = _style_contract_asset("approved")
    write_style_contract_asset(contract_path, contract)

    model_report = tmp_path / "reports" / "model.json"
    env_report = tmp_path / "reports" / "env.json"
    write_json_preflight(model_report, kind="model", passed=True)
    write_json_preflight(env_report, kind="environment", passed=True)
    write_valid_adapter(tmp_path / "outputs" / "sft_smoke")

    output_dir = tmp_path / "outputs" / "sft_v1"
    config_path = output_dir / "training_config_snapshot.yaml"

    def fake_build_train_run(**kwargs):
        config_path.parent.mkdir(parents=True)
        config_path.write_text("output_dir: sft_v1\n", encoding="utf-8")
        return {
            "name": kwargs["name"],
            "config_path": str(config_path),
            "command": ["llamafactory-cli", "train", str(config_path)],
        }

    monkeypatch.setattr(run_sft_train, "build_train_run", fake_build_train_run)
    monkeypatch.setattr(
        run_sft_train,
        "run_training_dry",
        lambda _run: {
            "exit_code": 0,
            "command_text": f"llamafactory-cli train {config_path}",
            "error": {"error_type": "none", "suggestion": "dry-run"},
        },
    )
    monkeypatch.setattr(
        run_sft_train.sys,
        "argv",
        [
            "run_sft_train.py",
            "--dry-run",
            "--sft-dataset",
            str(sft_dataset),
            "--eval-cards",
            str(eval_cards),
            "--model-report-json",
            str(model_report),
            "--env-report-json",
            str(env_report),
            "--smoke-adapter-dir",
            str(tmp_path / "outputs" / "sft_smoke"),
            "--style-contract-json",
            str(contract_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert run_sft_train.main() == 0
    manifest = json.loads((output_dir / "run_manifest.json").read_text("utf-8"))
    assert manifest["style_contract"]["style_contract_id"] == contract["style_contract_id"]
    assert manifest["style_contract"]["contract_sha256"] == contract["contract_sha256"]
    assert manifest["style_contract"]["approval_status"] == "approved"
    assert manifest["style_contract"]["schema"]["name"] == "style_contract"
    assert manifest["style_contract"]["schema"]["valid"] is True
    assert manifest["formal_evidence"] is False


@pytest.mark.parametrize(
    ("approval_status", "expected_formal_evidence"),
    [
        ("approved", True),
        ("frozen", True),
        ("pending_review", False),
        ("rejected", False),
        ("draft", False),
    ],
)
def test_run_sft_train_formal_evidence_requires_approved_or_frozen_style_contract(
    monkeypatch,
    tmp_path: Path,
    approval_status: str,
    expected_formal_evidence: bool,
):
    from scripts import run_sft_train
    from small_model_train.style_contract import write_style_contract_asset

    sft_dataset = tmp_path / "data" / "sft.jsonl"
    eval_cards = tmp_path / "data" / "eval.jsonl"
    contract_path = tmp_path / "data_style" / "style_contract.json"
    sft_dataset.parent.mkdir(parents=True)
    sft_dataset.write_text("{}\n", encoding="utf-8")
    write_jsonl(eval_cards, [_execution_card("case1")])
    contract = _style_contract_asset(approval_status)
    write_style_contract_asset(contract_path, contract)

    model_report = tmp_path / "reports" / "model.json"
    env_report = tmp_path / "reports" / "env.json"
    write_json_preflight(model_report, kind="model", passed=True)
    write_json_preflight(env_report, kind="environment", passed=True)
    write_valid_adapter(tmp_path / "outputs" / "sft_smoke")

    output_dir = tmp_path / "outputs" / "sft_v1"
    write_valid_adapter(output_dir)
    config_path = output_dir / "training_config_snapshot.yaml"

    def fake_build_train_run(**kwargs):
        return {
            "name": kwargs["name"],
            "config_path": str(config_path),
            "command": ["llamafactory-cli", "train", str(config_path)],
        }

    monkeypatch.setattr(run_sft_train, "build_train_run", fake_build_train_run)
    monkeypatch.setattr(
        run_sft_train,
        "run_training_subprocess",
        lambda _run: {
            "exit_code": 0,
            "command_text": f"llamafactory-cli train {config_path}",
            "error": {"error_type": "none", "suggestion": "无"},
        },
    )
    monkeypatch.setattr(
        run_sft_train.sys,
        "argv",
        [
            "run_sft_train.py",
            "--sft-dataset",
            str(sft_dataset),
            "--eval-cards",
            str(eval_cards),
            "--model-report-json",
            str(model_report),
            "--env-report-json",
            str(env_report),
            "--smoke-adapter-dir",
            str(tmp_path / "outputs" / "sft_smoke"),
            "--style-contract-json",
            str(contract_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert run_sft_train.main() == 0
    manifest = json.loads((output_dir / "run_manifest.json").read_text("utf-8"))
    assert manifest["adapter_check"]["passed"] is True
    assert manifest["sft_dataset"]["schema"]["valid"] is True
    assert manifest["eval_cards"]["schema"]["valid"] is True
    assert manifest["style_contract"]["schema"]["name"] == "style_contract"
    assert manifest["style_contract"]["schema"]["valid"] is True
    assert manifest["style_contract"]["approval_status"] == approval_status
    assert manifest["formal_evidence"] is expected_formal_evidence


def test_run_sft_train_formal_evidence_requires_passed_preflight_reports(
    monkeypatch,
    tmp_path: Path,
):
    from scripts import run_sft_train
    from small_model_train.style_contract import write_style_contract_asset

    sft_dataset = tmp_path / "data" / "sft.jsonl"
    eval_cards = tmp_path / "data" / "eval.jsonl"
    contract_path = tmp_path / "data_style" / "style_contract.json"
    sft_dataset.parent.mkdir(parents=True)
    sft_dataset.write_text("{}\n", encoding="utf-8")
    write_jsonl(eval_cards, [_execution_card("case1")])
    contract = _style_contract_asset("approved")
    write_style_contract_asset(contract_path, contract)

    write_valid_adapter(tmp_path / "outputs" / "sft_smoke")

    output_dir = tmp_path / "outputs" / "sft_v1"
    write_valid_adapter(output_dir)
    config_path = output_dir / "training_config_snapshot.yaml"

    def fake_build_train_run(**kwargs):
        return {
            "name": kwargs["name"],
            "config_path": str(config_path),
            "command": ["llamafactory-cli", "train", str(config_path)],
        }

    monkeypatch.setattr(run_sft_train, "build_train_run", fake_build_train_run)
    monkeypatch.setattr(
        run_sft_train,
        "run_training_subprocess",
        lambda _run: {
            "exit_code": 0,
            "command_text": f"llamafactory-cli train {config_path}",
            "error": {"error_type": "none", "suggestion": "无"},
        },
    )
    monkeypatch.setattr(
        run_sft_train.sys,
        "argv",
        [
            "run_sft_train.py",
            "--skip-prereq-checks",
            "--sft-dataset",
            str(sft_dataset),
            "--eval-cards",
            str(eval_cards),
            "--model-report-json",
            str(tmp_path / "missing" / "model.json"),
            "--env-report-json",
            str(tmp_path / "missing" / "env.json"),
            "--smoke-adapter-dir",
            str(tmp_path / "outputs" / "sft_smoke"),
            "--style-contract-json",
            str(contract_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert run_sft_train.main() == 0
    manifest = json.loads((output_dir / "run_manifest.json").read_text("utf-8"))
    assert manifest["adapter_check"]["passed"] is True
    assert manifest["sft_dataset"]["schema"]["valid"] is True
    assert manifest["eval_cards"]["schema"]["valid"] is True
    assert manifest["style_contract"]["approval_status"] == "approved"
    assert manifest["preflight_reports"]["model"]["passed"] is False
    assert manifest["preflight_reports"]["environment"]["passed"] is False
    assert manifest["formal_evidence"] is False


def test_run_sft_train_rejects_missing_style_contract_before_manifest(
    monkeypatch,
    tmp_path: Path,
    capsys,
):
    from scripts import run_sft_train

    sft_dataset = tmp_path / "data" / "sft.jsonl"
    eval_cards = tmp_path / "data" / "eval.jsonl"
    missing_contract = tmp_path / "data_style" / "missing_style_contract.json"
    sft_dataset.parent.mkdir(parents=True)
    sft_dataset.write_text("{}\n", encoding="utf-8")
    write_jsonl(eval_cards, [_execution_card("case1")])

    model_report = tmp_path / "reports" / "model.json"
    env_report = tmp_path / "reports" / "env.json"
    write_json_preflight(model_report, kind="model", passed=True)
    write_json_preflight(env_report, kind="environment", passed=True)
    write_valid_adapter(tmp_path / "outputs" / "sft_smoke")

    output_dir = tmp_path / "outputs" / "sft_v1"

    def fail_build_train_run(**_kwargs):
        raise AssertionError("style contract gate must fail before command construction")

    monkeypatch.setattr(run_sft_train, "build_train_run", fail_build_train_run)
    monkeypatch.setattr(
        run_sft_train.sys,
        "argv",
        [
            "run_sft_train.py",
            "--dry-run",
            "--sft-dataset",
            str(sft_dataset),
            "--eval-cards",
            str(eval_cards),
            "--model-report-json",
            str(model_report),
            "--env-report-json",
            str(env_report),
            "--smoke-adapter-dir",
            str(tmp_path / "outputs" / "sft_smoke"),
            "--style-contract-json",
            str(missing_contract),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert run_sft_train.main() == 1
    assert "style contract JSON not found" in capsys.readouterr().err
    assert not (output_dir / "run_manifest.json").exists()


def test_run_sft_train_rejects_invalid_style_contract_before_manifest(
    monkeypatch,
    tmp_path: Path,
    capsys,
):
    from scripts import run_sft_train

    sft_dataset = tmp_path / "data" / "sft.jsonl"
    eval_cards = tmp_path / "data" / "eval.jsonl"
    invalid_contract = tmp_path / "data_style" / "invalid_style_contract.json"
    sft_dataset.parent.mkdir(parents=True)
    invalid_contract.parent.mkdir(parents=True)
    sft_dataset.write_text("{}\n", encoding="utf-8")
    write_jsonl(eval_cards, [_execution_card("case1")])
    invalid_contract.write_text("{not json\n", encoding="utf-8")

    model_report = tmp_path / "reports" / "model.json"
    env_report = tmp_path / "reports" / "env.json"
    write_json_preflight(model_report, kind="model", passed=True)
    write_json_preflight(env_report, kind="environment", passed=True)
    write_valid_adapter(tmp_path / "outputs" / "sft_smoke")

    output_dir = tmp_path / "outputs" / "sft_v1"

    def fail_build_train_run(**_kwargs):
        raise AssertionError("style contract gate must fail before command construction")

    monkeypatch.setattr(run_sft_train, "build_train_run", fail_build_train_run)
    monkeypatch.setattr(
        run_sft_train.sys,
        "argv",
        [
            "run_sft_train.py",
            "--dry-run",
            "--sft-dataset",
            str(sft_dataset),
            "--eval-cards",
            str(eval_cards),
            "--model-report-json",
            str(model_report),
            "--env-report-json",
            str(env_report),
            "--smoke-adapter-dir",
            str(tmp_path / "outputs" / "sft_smoke"),
            "--style-contract-json",
            str(invalid_contract),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert run_sft_train.main() == 1
    assert "is not valid JSON" in capsys.readouterr().err
    assert not (output_dir / "run_manifest.json").exists()


def test_style_contract_manifest_summary_records_invalid_schema_name(tmp_path: Path):
    from scripts.run_sft_train import _style_contract_for_manifest

    invalid_contract = tmp_path / "style_contract.json"
    invalid_contract.write_text("{not json\n", encoding="utf-8")

    summary = _style_contract_for_manifest(invalid_contract)

    assert summary is not None
    assert summary["schema"]["name"] == "style_contract"
    assert summary["schema"]["valid"] is False
    assert "is not valid JSON" in "\n".join(summary["schema"]["errors"])


def test_run_sft_train_dry_run_records_pending_style_contract_without_formal_evidence(
    monkeypatch,
    tmp_path: Path,
):
    from scripts import run_sft_train
    from small_model_train.style_contract import write_style_contract_asset

    sft_dataset = tmp_path / "data" / "sft.jsonl"
    eval_cards = tmp_path / "data" / "eval.jsonl"
    contract_path = tmp_path / "data_style" / "style_contract.json"
    sft_dataset.parent.mkdir(parents=True)
    sft_dataset.write_text("{}\n", encoding="utf-8")
    write_jsonl(eval_cards, [_execution_card("case1")])
    contract = _style_contract_asset("pending_review")
    write_style_contract_asset(contract_path, contract)

    model_report = tmp_path / "reports" / "model.json"
    env_report = tmp_path / "reports" / "env.json"
    write_json_preflight(model_report, kind="model", passed=True)
    write_json_preflight(env_report, kind="environment", passed=True)
    write_valid_adapter(tmp_path / "outputs" / "sft_smoke")

    output_dir = tmp_path / "outputs" / "sft_v1"
    config_path = output_dir / "training_config_snapshot.yaml"

    def fake_build_train_run(**kwargs):
        config_path.parent.mkdir(parents=True)
        config_path.write_text("output_dir: sft_v1\n", encoding="utf-8")
        return {
            "name": kwargs["name"],
            "config_path": str(config_path),
            "command": ["llamafactory-cli", "train", str(config_path)],
        }

    monkeypatch.setattr(run_sft_train, "build_train_run", fake_build_train_run)
    monkeypatch.setattr(
        run_sft_train,
        "run_training_dry",
        lambda _run: {
            "exit_code": 0,
            "command_text": f"llamafactory-cli train {config_path}",
            "error": {"error_type": "none", "suggestion": "dry-run"},
        },
    )
    monkeypatch.setattr(
        run_sft_train.sys,
        "argv",
        [
            "run_sft_train.py",
            "--dry-run",
            "--sft-dataset",
            str(sft_dataset),
            "--eval-cards",
            str(eval_cards),
            "--model-report-json",
            str(model_report),
            "--env-report-json",
            str(env_report),
            "--smoke-adapter-dir",
            str(tmp_path / "outputs" / "sft_smoke"),
            "--style-contract-json",
            str(contract_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert run_sft_train.main() == 0
    manifest = json.loads((output_dir / "run_manifest.json").read_text("utf-8"))
    assert manifest["style_contract"]["approval_status"] == "pending_review"
    assert manifest["style_contract"]["schema"]["valid"] is True
    assert manifest["formal_evidence"] is False


def test_run_sft_train_skip_prereqs_marks_preflights_skipped(
    monkeypatch,
    tmp_path: Path,
):
    from scripts import run_sft_train

    sft_dataset = tmp_path / "data" / "sft.jsonl"
    eval_cards = tmp_path / "data" / "eval.jsonl"
    sft_dataset.parent.mkdir(parents=True)
    sft_dataset.write_text("{}\n", encoding="utf-8")
    write_jsonl(eval_cards, [_execution_card("case1")])

    output_dir = tmp_path / "outputs" / "sft_v1"
    config_path = output_dir / "training_config_snapshot.yaml"

    def fake_build_train_run(**kwargs):
        config_path.parent.mkdir(parents=True)
        config_path.write_text("output_dir: sft_v1\n", encoding="utf-8")
        return {
            "name": kwargs["name"],
            "config_path": str(config_path),
            "command": ["llamafactory-cli", "train", str(config_path)],
        }

    monkeypatch.setattr(run_sft_train, "build_train_run", fake_build_train_run)
    monkeypatch.setattr(
        run_sft_train,
        "run_training_dry",
        lambda _run: {
            "exit_code": 0,
            "command_text": f"llamafactory-cli train {config_path}",
            "error": {"error_type": "none", "suggestion": "dry-run"},
        },
    )
    monkeypatch.setattr(
        run_sft_train.sys,
        "argv",
        [
            "run_sft_train.py",
            "--dry-run",
            "--skip-prereq-checks",
            "--config",
            str(tmp_path / "config.yaml"),
            "--model-dir",
            str(tmp_path / "model"),
            "--output-dir",
            str(output_dir),
            "--log-dir",
            str(tmp_path / "logs"),
            "--sft-dataset",
            str(sft_dataset),
            "--eval-cards",
            str(eval_cards),
            "--model-report-json",
            str(tmp_path / "missing" / "model.json"),
            "--env-report-json",
            str(tmp_path / "missing" / "env.json"),
        ],
    )

    exit_code = run_sft_train.main()

    manifest = json.loads((output_dir / "run_manifest.json").read_text("utf-8"))
    assert exit_code == 0
    assert manifest["preflight_reports"]["model"]["status"] == "skipped"
    assert manifest["preflight_reports"]["model"]["passed"] is False
    assert manifest["preflight_reports"]["environment"]["status"] == "skipped"
    assert manifest["preflight_reports"]["environment"]["passed"] is False


def test_run_sft_train_writes_manifest_when_training_exits_nonzero(
    monkeypatch,
    tmp_path: Path,
):
    from scripts import run_sft_train

    sft_dataset = tmp_path / "data" / "sft.jsonl"
    eval_cards = tmp_path / "data" / "eval.jsonl"
    sft_dataset.parent.mkdir(parents=True)
    sft_dataset.write_text("{}\n", encoding="utf-8")
    write_jsonl(eval_cards, [_execution_card("case1")])

    model_report = tmp_path / "reports" / "model.json"
    env_report = tmp_path / "reports" / "env.json"
    write_json_preflight(model_report, kind="model", passed=True)
    write_json_preflight(env_report, kind="environment", passed=True)
    write_valid_adapter(tmp_path / "outputs" / "sft_smoke")

    output_dir = tmp_path / "outputs" / "sft_v1"
    config_path = output_dir / "training_config_snapshot.yaml"

    def fake_build_train_run(**kwargs):
        config_path.parent.mkdir(parents=True)
        config_path.write_text("output_dir: sft_v1\n", encoding="utf-8")
        return {
            "name": kwargs["name"],
            "config_path": str(config_path),
            "command": ["llamafactory-cli", "train", str(config_path)],
        }

    monkeypatch.setattr(run_sft_train, "build_train_run", fake_build_train_run)
    monkeypatch.setattr(
        run_sft_train,
        "run_training_subprocess",
        lambda _run: {
            "exit_code": 42,
            "command_text": f"llamafactory-cli train {config_path}",
            "error": {"error_type": "llamafactory_error", "suggestion": "inspect logs"},
        },
    )
    monkeypatch.setattr(
        run_sft_train.sys,
        "argv",
        [
            "run_sft_train.py",
            "--config",
            str(tmp_path / "config.yaml"),
            "--model-dir",
            str(tmp_path / "model"),
            "--output-dir",
            str(output_dir),
            "--log-dir",
            str(tmp_path / "logs"),
            "--sft-dataset",
            str(sft_dataset),
            "--eval-cards",
            str(eval_cards),
            "--model-report-json",
            str(model_report),
            "--env-report-json",
            str(env_report),
            "--smoke-adapter-dir",
            str(tmp_path / "outputs" / "sft_smoke"),
        ],
    )

    exit_code = run_sft_train.main()

    manifest = json.loads((output_dir / "run_manifest.json").read_text("utf-8"))
    assert exit_code == 42
    assert manifest["training_exit_code"] == 42
    assert manifest["adapter_check"]["status"] == "not_run"
    assert manifest["adapter_check"]["passed"] is False
    assert manifest["passed"] is False


def test_run_sft_train_formal_evidence_requires_valid_sft_schema(
    monkeypatch,
    tmp_path: Path,
):
    from scripts import run_sft_train

    sft_dataset = tmp_path / "data" / "sft.jsonl"
    eval_cards = tmp_path / "data" / "eval.jsonl"
    sft_dataset.parent.mkdir(parents=True)
    sft_dataset.write_text("{not json\n", encoding="utf-8")
    write_jsonl(eval_cards, [_execution_card("case1")])

    model_report = tmp_path / "reports" / "model.json"
    env_report = tmp_path / "reports" / "env.json"
    write_json_preflight(model_report, kind="model", passed=True)
    write_json_preflight(env_report, kind="environment", passed=True)
    write_valid_adapter(tmp_path / "outputs" / "sft_smoke")

    output_dir = tmp_path / "outputs" / "sft_v1"
    write_valid_adapter(output_dir)
    config_path = output_dir / "training_config_snapshot.yaml"

    def fake_build_train_run(**kwargs):
        return {
            "name": kwargs["name"],
            "config_path": str(config_path),
            "command": ["llamafactory-cli", "train", str(config_path)],
        }

    monkeypatch.setattr(run_sft_train, "build_train_run", fake_build_train_run)
    monkeypatch.setattr(
        run_sft_train,
        "run_training_subprocess",
        lambda _run: {
            "exit_code": 0,
            "command_text": f"llamafactory-cli train {config_path}",
            "error": {"error_type": "none", "suggestion": "无"},
        },
    )
    monkeypatch.setattr(
        run_sft_train.sys,
        "argv",
        [
            "run_sft_train.py",
            "--config",
            str(tmp_path / "config.yaml"),
            "--model-dir",
            str(tmp_path / "model"),
            "--output-dir",
            str(output_dir),
            "--log-dir",
            str(tmp_path / "logs"),
            "--sft-dataset",
            str(sft_dataset),
            "--eval-cards",
            str(eval_cards),
            "--model-report-json",
            str(model_report),
            "--env-report-json",
            str(env_report),
            "--smoke-adapter-dir",
            str(tmp_path / "outputs" / "sft_smoke"),
        ],
    )

    exit_code = run_sft_train.main()

    manifest = json.loads((output_dir / "run_manifest.json").read_text("utf-8"))
    assert exit_code == 0
    assert manifest["adapter_check"]["passed"] is True
    assert manifest["eval_cards"]["schema"]["valid"] is True
    assert manifest["sft_dataset"]["schema"]["valid"] is False
    assert "not valid JSON" in "\n".join(manifest["sft_dataset"]["schema"]["errors"])
    assert manifest["formal_evidence"] is False


def test_run_sft_train_writes_failed_manifest_when_adapter_invalid(
    monkeypatch,
    tmp_path: Path,
):
    from scripts import run_sft_train

    sft_dataset = tmp_path / "data" / "sft.jsonl"
    eval_cards = tmp_path / "data" / "eval.jsonl"
    sft_dataset.parent.mkdir(parents=True)
    sft_dataset.write_text("{}\n", encoding="utf-8")
    write_jsonl(eval_cards, [_execution_card("case1")])

    model_report = tmp_path / "reports" / "model.json"
    env_report = tmp_path / "reports" / "env.json"
    write_json_preflight(model_report, kind="model", passed=True)
    write_json_preflight(env_report, kind="environment", passed=True)
    write_valid_adapter(tmp_path / "outputs" / "sft_smoke")

    output_dir = tmp_path / "outputs" / "sft_v1"
    config_path = output_dir / "training_config_snapshot.yaml"

    def fake_build_train_run(**kwargs):
        config_path.parent.mkdir(parents=True)
        config_path.write_text("output_dir: sft_v1\n", encoding="utf-8")
        return {
            "name": kwargs["name"],
            "config_path": str(config_path),
            "command": ["llamafactory-cli", "train", str(config_path)],
        }

    monkeypatch.setattr(run_sft_train, "build_train_run", fake_build_train_run)
    monkeypatch.setattr(
        run_sft_train,
        "run_training_subprocess",
        lambda _run: {
            "exit_code": 0,
            "command_text": f"llamafactory-cli train {config_path}",
            "error": {"error_type": "none", "suggestion": "无"},
        },
    )
    monkeypatch.setattr(
        run_sft_train.sys,
        "argv",
        [
            "run_sft_train.py",
            "--config",
            str(tmp_path / "config.yaml"),
            "--model-dir",
            str(tmp_path / "model"),
            "--output-dir",
            str(output_dir),
            "--log-dir",
            str(tmp_path / "logs"),
            "--sft-dataset",
            str(sft_dataset),
            "--eval-cards",
            str(eval_cards),
            "--model-report-json",
            str(model_report),
            "--env-report-json",
            str(env_report),
            "--smoke-adapter-dir",
            str(tmp_path / "outputs" / "sft_smoke"),
        ],
    )

    exit_code = run_sft_train.main()

    manifest_path = output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert exit_code != 0
    assert manifest["training_exit_code"] == 0
    assert manifest["adapter_check"]["passed"] is False
    assert "adapter_config.json" in manifest["adapter_check"]["missing_files"]
    assert manifest["passed"] is False
