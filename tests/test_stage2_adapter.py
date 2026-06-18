from pathlib import Path

from small_model_train.stage2_adapter import check_adapter_dir, render_adapter_report
from scripts.run_oom_probe import PROBES, render_probe_report


def test_check_adapter_dir_passes_when_required_files_exist(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "training_config_snapshot.yaml").write_text(
        "output_dir: adapter\n", encoding="utf-8"
    )

    result = check_adapter_dir(adapter)

    assert result["passed"] is True
    assert result["missing_files"] == []


def test_check_adapter_dir_reports_missing_files(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()

    result = check_adapter_dir(adapter)

    assert result["passed"] is False
    assert "adapter_config.json" in result["missing_files"]


def test_check_adapter_dir_reports_zero_size_files(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"")
    (adapter / "training_config_snapshot.yaml").write_text(
        "output_dir: adapter\n", encoding="utf-8"
    )

    result = check_adapter_dir(adapter)

    assert result["passed"] is False
    assert result["missing_files"] == []
    assert result["zero_size_files"] == ["adapter_model.safetensors"]


def test_render_adapter_report_contains_decision(tmp_path: Path):
    result = {
        "adapter_dir": str(tmp_path / "adapter"),
        "passed": False,
        "missing_files": ["adapter_model.safetensors"],
        "zero_size_files": [],
    }

    report = render_adapter_report("SFT v1 Adapter", result)

    assert "# SFT v1 Adapter" in report
    assert "adapter_model.safetensors" in report
    assert "不允许进入下一步" in report


def test_render_probe_report_contains_probe_plan_and_interpretation():
    report = render_probe_report()

    assert "# OOM Probe Report" in report
    for probe in PROBES:
        assert probe in report
    assert "Probe 2 失败：基座 4-bit 加载不稳" in report
    assert "Probe 5 失败但 Probe 6 成功：8192 cutoff_len 过高" in report
    assert "所有 probe 通过但正式训练失败" in report
