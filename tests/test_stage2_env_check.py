from types import SimpleNamespace

from small_model_train import stage2_env_check
from small_model_train.stage2_env_check import (
    collect_training_env,
    parse_nvidia_smi_memory,
    query_nvidia_smi_memory,
    render_env_report,
    vram_recommendation,
)


def test_parse_nvidia_smi_memory_reads_free_and_total_mb():
    text = "NVIDIA RTX, 16384, 14000\n"

    result = parse_nvidia_smi_memory(text)

    assert result == {"gpu_name": "NVIDIA RTX", "total_mb": 16384, "free_mb": 14000}


def test_parse_nvidia_smi_memory_returns_zeroes_for_malformed_input():
    assert parse_nvidia_smi_memory("") == {
        "gpu_name": "",
        "total_mb": 0,
        "free_mb": 0,
    }
    assert parse_nvidia_smi_memory("NVIDIA RTX, nope, 14000") == {
        "gpu_name": "",
        "total_mb": 0,
        "free_mb": 0,
    }
    assert parse_nvidia_smi_memory("NVIDIA RTX, 16384") == {
        "gpu_name": "",
        "total_mb": 0,
        "free_mb": 0,
    }


def test_vram_recommendation_uses_stage_two_thresholds():
    assert vram_recommendation(14000)["cutoff_len"] == 8192
    assert vram_recommendation(12000)["cutoff_len"] == 6144
    assert vram_recommendation(8000)["allow_training"] is False


def test_collect_training_env_blocks_high_vram_when_environment_is_not_ready(
    monkeypatch,
):
    monkeypatch.setattr(
        stage2_env_check,
        "query_nvidia_smi_memory",
        lambda: {"gpu_name": "NVIDIA RTX", "total_mb": 16384, "free_mb": 14000},
    )
    monkeypatch.setattr(stage2_env_check, "_cuda_available", lambda: False)
    monkeypatch.setattr(stage2_env_check.shutil, "which", lambda _name: None)
    monkeypatch.setattr(stage2_env_check, "import_version", lambda _package: "missing")

    snapshot = collect_training_env()

    recommendation = snapshot["recommendation"]
    assert recommendation["cutoff_len"] == 8192
    assert recommendation["allow_training"] is False
    blocking_reasons = recommendation["blocking_reasons"]
    assert "CUDA is not available" in blocking_reasons
    assert "torch is missing" in blocking_reasons
    assert "transformers is missing" in blocking_reasons
    assert "bitsandbytes is missing" in blocking_reasons
    assert "peft is missing" in blocking_reasons
    assert "LLaMA-Factory CLI is missing" in blocking_reasons

    report = render_env_report(snapshot)

    assert "## Blocking Reasons" in report
    assert "- CUDA is not available" in report


def test_query_nvidia_smi_memory_chooses_gpu_with_highest_free_memory(monkeypatch):
    def fake_run(_command, **_kwargs):
        return SimpleNamespace(
            stdout="NVIDIA RTX 3090, 24576, 8000\nNVIDIA RTX 4090, 24576, 14000\n"
        )

    monkeypatch.setattr(stage2_env_check.subprocess, "run", fake_run)

    result = query_nvidia_smi_memory()

    assert result == {
        "gpu_name": "NVIDIA RTX 4090",
        "total_mb": 24576,
        "free_mb": 14000,
    }


def test_render_env_report_contains_dependency_status():
    snapshot = {
        "python": "3.11.8",
        "imports": {"torch": "2.5.0", "bitsandbytes": "missing"},
        "cuda_available": True,
        "gpu": {"gpu_name": "NVIDIA RTX", "total_mb": 16384, "free_mb": 14000},
        "llamafactory": "available",
        "env": {"HF_HOME": "", "TRANSFORMERS_CACHE": "", "HF_ENDPOINT": ""},
        "recommendation": {
            "allow_training": True,
            "cutoff_len": 8192,
            "message": "允许 8192 cutoff_len 冒烟训练",
        },
    }

    report = render_env_report(snapshot)

    assert "# Training Environment Report" in report
    assert "- bitsandbytes: missing" in report
    assert "允许 8192 cutoff_len 冒烟训练" in report
