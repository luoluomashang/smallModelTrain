from pathlib import Path

from small_model_train.stage2_adapter import check_adapter_dir, render_adapter_report
from scripts.run_oom_probe import PROBES, render_probe_report


def write_minimal_safetensors(path: Path) -> None:
    header = b"{}"
    path.write_bytes(len(header).to_bytes(8, "little") + header)


def test_check_adapter_dir_passes_when_required_files_exist(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    write_minimal_safetensors(adapter / "adapter_model.safetensors")
    (adapter / "training_config_snapshot.yaml").write_text(
        "output_dir: adapter\n", encoding="utf-8"
    )

    result = check_adapter_dir(adapter)

    assert result["passed"] is True
    assert result["missing_files"] == []
    assert result["zero_size_files"] == []
    assert result["errors"] == []


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


def test_check_adapter_dir_reports_invalid_adapter_config_json(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("not json", encoding="utf-8")
    write_minimal_safetensors(adapter / "adapter_model.safetensors")
    (adapter / "training_config_snapshot.yaml").write_text(
        "output_dir: adapter\n", encoding="utf-8"
    )

    result = check_adapter_dir(adapter)

    assert result["passed"] is False
    assert any("adapter_config.json" in error for error in result["errors"])


def test_check_adapter_dir_requires_adapter_config_json_object(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("[]", encoding="utf-8")
    write_minimal_safetensors(adapter / "adapter_model.safetensors")
    (adapter / "training_config_snapshot.yaml").write_text(
        "output_dir: adapter\n", encoding="utf-8"
    )

    result = check_adapter_dir(adapter)

    assert result["passed"] is False
    assert any("adapter_config.json" in error for error in result["errors"])


def test_check_adapter_dir_reports_invalid_safetensors_header(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"x")
    (adapter / "training_config_snapshot.yaml").write_text(
        "output_dir: adapter\n", encoding="utf-8"
    )

    result = check_adapter_dir(adapter)

    assert result["passed"] is False
    assert any("adapter_model.safetensors" in error for error in result["errors"])


def test_check_adapter_dir_does_not_read_entire_safetensors_file(
    tmp_path: Path, monkeypatch
):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    write_minimal_safetensors(adapter / "adapter_model.safetensors")
    (adapter / "training_config_snapshot.yaml").write_text(
        "output_dir: adapter\n", encoding="utf-8"
    )

    def fail_full_file_read(_path: Path) -> bytes:
        raise OSError("full adapter weight read is not allowed")

    monkeypatch.setattr(Path, "read_bytes", fail_full_file_read)

    result = check_adapter_dir(adapter)

    assert result["passed"] is True
    assert result["errors"] == []


def test_render_adapter_report_contains_decision(tmp_path: Path):
    result = {
        "adapter_dir": str(tmp_path / "adapter"),
        "passed": False,
        "missing_files": ["adapter_model.safetensors"],
        "zero_size_files": [],
        "errors": ["adapter_model.safetensors has invalid safetensors header"],
    }

    report = render_adapter_report("SFT v1 Adapter", result)

    assert "# SFT v1 Adapter" in report
    assert "adapter_model.safetensors" in report
    assert "不允许进入下一步" in report
    assert "## Errors" in report
    assert "invalid safetensors header" in report


def test_render_adapter_report_uses_empty_marker_for_errors(tmp_path: Path):
    result = {
        "adapter_dir": str(tmp_path / "adapter"),
        "passed": True,
        "missing_files": [],
        "zero_size_files": [],
        "errors": [],
    }

    report = render_adapter_report("SFT v1 Adapter", result)

    assert "## Errors\n- 无" in report


def test_render_probe_report_contains_probe_plan_and_interpretation():
    report = render_probe_report()

    assert "# OOM Probe Report" in report
    for probe in PROBES:
        assert probe in report
    assert "Probe 2 失败：基座 4-bit 加载不稳" in report
    assert "Probe 5 失败但 Probe 6 成功：8192 cutoff_len 过高" in report
    assert "所有 probe 通过但正式训练失败" in report
