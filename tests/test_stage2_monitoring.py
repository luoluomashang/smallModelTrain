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


def test_classify_training_error_detects_empty_nonzero_exit_as_process_killed():
    result = classify_training_error("", 1)

    assert result["error_type"] == "process_killed"


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
