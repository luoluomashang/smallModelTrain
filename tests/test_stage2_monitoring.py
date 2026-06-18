import json
import re
from pathlib import Path

from small_model_train import stage2_monitoring
from small_model_train.stage2_monitoring import (
    append_event,
    classify_training_error,
    parse_gpu_process_samples,
    render_failure_summary,
)


EXPECTED_SUGGESTIONS = {
    "cuda_oom": "降低 cutoff_len，必要时降低 lora_rank",
    "process_killed": "查看 GPU 采样和系统稳定性，确认是否被系统结束",
    "driver_reset": "重启训练环境并检查驱动状态",
    "bnb_load_error": "检查 bitsandbytes、CUDA 和 PyTorch 版本匹配",
    "tokenizer_error": "检查本地模型目录和 tokenizer 文件",
    "dataset_error": "检查数据路径和 LLaMA-Factory dataset_info",
    "llamafactory_error": "检查 LLaMA-Factory 命令和参数",
    "adapter_save_error": "检查 output_dir 权限和磁盘空间",
}


def test_now_iso_uses_local_timezone_offset(monkeypatch):
    class FakeLocalDateTime:
        def isoformat(self, timespec="auto"):
            assert timespec == "seconds"
            return "2026-01-02T11:04:05+08:00"

    class FakeNowDateTime:
        def __init__(self, tz):
            self.tz = tz

        def astimezone(self):
            assert self.tz is None
            return FakeLocalDateTime()

        def isoformat(self, timespec="auto"):
            assert timespec == "seconds"
            return "2026-01-02T03:04:05+00:00"

    class FakeDateTime:
        @classmethod
        def now(cls, tz=None):
            return FakeNowDateTime(tz)

    monkeypatch.setattr(stage2_monitoring, "datetime", FakeDateTime)

    timestamp = stage2_monitoring.now_iso()

    iso_with_offset = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}"
    assert re.fullmatch(iso_with_offset, timestamp)
    assert timestamp.endswith("+08:00")
    assert not timestamp.endswith("+00:00")


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


def test_classify_training_error_prefers_cuda_oom_over_dataset_context():
    result = classify_training_error(
        "RuntimeError: CUDA out of memory while iterating dataset",
        1,
    )

    assert result == {
        "error_type": "cuda_oom",
        "suggestion": EXPECTED_SUGGESTIONS["cuda_oom"],
    }


def test_classify_training_error_detects_dataset_error():
    result = classify_training_error("FileNotFoundError: dataset_info.json", 1)

    assert result["error_type"] == "dataset_error"


def test_classify_training_error_prefers_dataset_context_over_unknown_error():
    result = classify_training_error(
        "RuntimeError: unknown error while loading dataset_info.json",
        1,
    )

    assert result["error_type"] == "dataset_error"


def test_classify_training_error_treats_data_permission_denied_as_dataset_error():
    result = classify_training_error(
        "Permission denied: data_sft/sft_chapter_v1.jsonl",
        1,
    )

    assert result["error_type"] == "dataset_error"


def test_classify_training_error_prefers_llamafactory_error_over_dataset_argument():
    result = classify_training_error(
        "llamafactory-cli: unrecognized arguments: --dataset foo",
        1,
    )

    assert result == {
        "error_type": "llamafactory_error",
        "suggestion": EXPECTED_SUGGESTIONS["llamafactory_error"],
    }


def test_classify_training_error_does_not_treat_bare_permission_denied_as_adapter_save_error():
    result = classify_training_error("Permission denied", 1)

    assert result == {
        "error_type": "process_killed",
        "suggestion": EXPECTED_SUGGESTIONS["process_killed"],
    }


def test_classify_training_error_detects_empty_nonzero_exit_as_process_killed():
    result = classify_training_error("", 1)

    assert result["error_type"] == "process_killed"


def test_classify_training_error_returns_none_for_clean_exit_without_stderr():
    assert classify_training_error("", 0) == {
        "error_type": "none",
        "suggestion": "无",
    }
    assert classify_training_error("", None) == {
        "error_type": "none",
        "suggestion": "无",
    }


def test_classify_training_error_ignores_successful_stderr_markers():
    cases = [
        "Loading dataset train split",
        "Loading tokenizer files",
        "bitsandbytes warning",
    ]

    for stderr in cases:
        assert classify_training_error(stderr, 0) == {
            "error_type": "none",
            "suggestion": "无",
        }


def test_classify_training_error_returns_plan_suggestions_exactly():
    cases = [
        ("RuntimeError: CUDA out of memory", "cuda_oom"),
        ("RuntimeError: killed", "process_killed"),
        ("RuntimeError: device lost", "driver_reset"),
        ("ImportError: bitsandbytes 4-bit", "bnb_load_error"),
        ("AutoTokenizer failed", "tokenizer_error"),
        ("FileNotFoundError: dataset_info.json", "dataset_error"),
        ("llamafactory-cli: unrecognized arguments", "llamafactory_error"),
        ("Permission denied during save_pretrained adapter_model", "adapter_save_error"),
    ]

    for stderr, error_type in cases:
        result = classify_training_error(stderr, 1)

        assert result == {
            "error_type": error_type,
            "suggestion": EXPECTED_SUGGESTIONS[error_type],
        }


def test_parse_gpu_process_samples_reads_nvidia_smi_csv():
    text = "1234, python.exe, 11800\n5678, chrome.exe, 600\n"

    samples = parse_gpu_process_samples(text)

    assert samples == [
        {"pid": 1234, "name": "python.exe", "used_mb": 11800},
        {"pid": 5678, "name": "chrome.exe", "used_mb": 600},
    ]


def test_collect_gpu_processes_runs_nvidia_smi_and_parses_stdout(monkeypatch):
    calls = []

    class CompletedProcess:
        stdout = "1234, python.exe, 11800\n"

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return CompletedProcess()

    monkeypatch.setattr(stage2_monitoring.subprocess, "run", fake_run)

    samples = stage2_monitoring.collect_gpu_processes()

    assert samples == [{"pid": 1234, "name": "python.exe", "used_mb": 11800}]
    command, kwargs = calls[0]
    assert "--query-compute-apps=pid,process_name,used_memory" in command
    assert "--format=csv,noheader,nounits" in command
    assert kwargs["capture_output"] is True
    assert kwargs["check"] is True
    assert kwargs["text"] is True
    assert kwargs["timeout"] == 10


def test_collect_gpu_processes_returns_empty_when_nvidia_smi_is_missing(monkeypatch):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(stage2_monitoring.subprocess, "run", fake_run)

    assert stage2_monitoring.collect_gpu_processes() == []


def test_render_failure_summary_contains_last_phase_and_gpu_sample():
    summary = render_failure_summary(
        error={"error_type": "cuda_oom", "suggestion": "降低 cutoff_len"},
        last_events=[{"phase": "first_backward", "status": "start"}],
        last_gpu_samples=[{"free_mb": 512, "used_mb": 15872}],
        exit_code=1,
        stderr_tail="CUDA out of memory",
    )

    assert "# Training Failure Summary" in summary
    assert "- Exit code: 1" in summary
    assert "first_backward" in summary
    assert "cuda_oom" in summary
    assert "降低 cutoff_len" in summary
    assert "## Last Events" in summary
    assert "## Last GPU Samples" in summary
    assert "15872" in summary
    assert "```text\nCUDA out of memory\n```" in summary
    assert "CUDA out of memory" in summary
