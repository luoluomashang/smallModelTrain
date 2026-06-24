import json
from pathlib import Path

from small_model_train.stage2_model_check import (
    check_model_files,
    render_model_check_report,
    run_transformers_load_checks,
)


def write_file(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_check_model_files_passes_for_required_files(tmp_path: Path):
    model_dir = tmp_path / "model"
    for name in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model-00001-of-00001.safetensors",
    ]:
        write_file(model_dir / name)
    write_file(
        model_dir / "model.safetensors.index.json",
        json.dumps({"weight_map": {"layer.a": "model-00001-of-00001.safetensors"}}),
    )

    result = check_model_files(model_dir)

    assert result["passed"] is True
    assert result["missing_files"] == []
    assert result["zero_size_files"] == []
    assert result["shard_count"] == 1


def test_check_model_files_reports_missing_and_zero_size_files(tmp_path: Path):
    model_dir = tmp_path / "model"
    write_file(model_dir / "config.json")
    write_file(model_dir / "tokenizer.json")
    write_file(model_dir / "tokenizer_config.json")
    write_file(model_dir / "model.safetensors.index.json")
    shard = model_dir / "model-00001-of-00001.safetensors"
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_bytes(b"")

    result = check_model_files(model_dir)

    assert result["passed"] is False
    assert result["zero_size_files"] == ["model-00001-of-00001.safetensors"]


def test_check_model_files_reports_missing_shards(tmp_path: Path):
    model_dir = tmp_path / "model"
    for name in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    ]:
        write_file(model_dir / name)

    result = check_model_files(model_dir)

    assert result["passed"] is False
    assert "model-*.safetensors" in result["missing_files"]
    assert any("model-*.safetensors" in error for error in result["errors"])


def test_check_model_files_reports_missing_indexed_shards(tmp_path: Path):
    model_dir = tmp_path / "model"
    write_file(model_dir / "config.json")
    write_file(model_dir / "tokenizer.json")
    write_file(model_dir / "tokenizer_config.json")
    write_file(
        model_dir / "model.safetensors.index.json",
        json.dumps(
            {
                "weight_map": {
                    "layer.a": "model-00001-of-00002.safetensors",
                    "layer.b": "model-00002-of-00002.safetensors",
                }
            }
        ),
    )
    write_file(model_dir / "model-00001-of-00002.safetensors")

    result = check_model_files(model_dir)

    assert result["passed"] is False
    assert "model-00002-of-00002.safetensors" in result["missing_files"]
    assert any(
        "model-00002-of-00002.safetensors" in error for error in result["errors"]
    )


def test_check_model_files_rejects_unsafe_indexed_shard_paths(tmp_path: Path):
    model_dir = tmp_path / "model"
    write_file(model_dir / "config.json")
    write_file(model_dir / "tokenizer.json")
    write_file(model_dir / "tokenizer_config.json")
    write_file(
        model_dir / "model.safetensors.index.json",
        json.dumps(
            {
                "weight_map": {
                    "layer.a": "model-00001-of-00002.safetensors",
                    "layer.b": "../model-00002-of-00002.safetensors",
                }
            }
        ),
    )
    write_file(model_dir / "model-00001-of-00002.safetensors")
    write_file(tmp_path / "model-00002-of-00002.safetensors")

    result = check_model_files(model_dir)

    assert result["passed"] is False
    assert any(
        "invalid shard path" in error
        and "../model-00002-of-00002.safetensors" in error
        for error in result["errors"]
    )


def test_check_model_files_reports_corrupt_index_bytes(tmp_path: Path):
    model_dir = tmp_path / "model"
    write_file(model_dir / "config.json")
    write_file(model_dir / "tokenizer.json")
    write_file(model_dir / "tokenizer_config.json")
    write_file(model_dir / "model-00001-of-00001.safetensors")
    index_path = model_dir / "model.safetensors.index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_bytes(b"\xff\xfe\xfa")

    try:
        result = check_model_files(model_dir)
    except UnicodeDecodeError as exc:
        raise AssertionError("check_model_files raised UnicodeDecodeError") from exc

    assert result["passed"] is False
    assert any("invalid index" in error for error in result["errors"])


def test_run_transformers_load_checks_reports_exception_types(tmp_path: Path):
    result = {
        "model_dir": str(tmp_path / "model"),
        "passed": True,
        "missing_files": [],
        "zero_size_files": [],
        "shard_count": 1,
        "load_checks": {"config": "not_run", "tokenizer": "not_run"},
        "errors": [],
    }

    def raise_boom(_model_dir: str) -> None:
        raise RuntimeError("boom")

    run_transformers_load_checks(
        result,
        config_loader=raise_boom,
        tokenizer_loader=raise_boom,
    )

    assert result["passed"] is False
    assert any("RuntimeError: boom" in error for error in result["errors"])


def test_render_model_check_report_contains_decision(tmp_path: Path):
    result = {
        "model_dir": str(tmp_path / "model"),
        "passed": False,
        "missing_files": ["config.json"],
        "zero_size_files": [],
        "shard_count": 0,
        "load_checks": {"config": "skipped", "tokenizer": "skipped"},
        "errors": ["missing required file: config.json"],
    }

    report = render_model_check_report(result)

    assert "# Local Model Check Report" in report
    assert "missing required file: config.json" in report
    assert "不进入训练" in report


def test_check_local_model_writes_json_preflight_report(
    tmp_path: Path,
    monkeypatch,
):
    from scripts import check_local_model

    result = {
        "model_dir": str(tmp_path / "model"),
        "passed": False,
        "missing_files": ["config.json"],
        "zero_size_files": ["model-00001-of-00001.safetensors"],
        "shard_count": 1,
        "load_checks": {"config": "skipped", "tokenizer": "skipped"},
        "errors": ["custom model error"],
    }
    report_path = tmp_path / "reports" / "model.md"
    json_path = tmp_path / "reports" / "model.json"

    monkeypatch.setattr(check_local_model, "check_model_files", lambda _path: result)
    monkeypatch.setattr(
        check_local_model.sys,
        "argv",
        [
            "check_local_model.py",
            "--model-dir",
            str(tmp_path / "model"),
            "--report",
            str(report_path),
            "--json-output",
            str(json_path),
            "--skip-transformers-load",
        ],
    )

    exit_code = check_local_model.main()

    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert report["schema_version"] == 1
    assert report["kind"] == "model"
    assert report["passed"] is False
    assert report["payload"] == result
    assert "custom model error" in report["errors"]
    assert "missing required file: config.json" in report["errors"]
    assert (
        "zero-size model shard: model-00001-of-00001.safetensors"
        in report["errors"]
    )
    assert report_path.exists()


def test_check_local_model_writes_default_json_preflight_report(
    tmp_path: Path,
    monkeypatch,
):
    from scripts import check_local_model

    result = {
        "model_dir": str(tmp_path / "model"),
        "passed": True,
        "missing_files": [],
        "zero_size_files": [],
        "shard_count": 1,
        "load_checks": {"config": "skipped", "tokenizer": "skipped"},
        "errors": [],
    }

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(check_local_model, "check_model_files", lambda _path: result)
    monkeypatch.setattr(
        check_local_model.sys,
        "argv",
        [
            "check_local_model.py",
            "--skip-transformers-load",
        ],
    )

    exit_code = check_local_model.main()

    json_path = tmp_path / "reports" / "model_check_report.json"
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["kind"] == "model"
    assert report["passed"] is True
    assert report["payload"] == result
