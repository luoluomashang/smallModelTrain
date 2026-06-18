from pathlib import Path

from small_model_train.stage2_model_check import check_model_files, render_model_check_report


def write_file(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_check_model_files_passes_for_required_files(tmp_path: Path):
    model_dir = tmp_path / "model"
    for name in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "model-00001-of-00001.safetensors",
    ]:
        write_file(model_dir / name)

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
