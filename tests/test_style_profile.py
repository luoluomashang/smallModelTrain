import json
import subprocess
import sys
from pathlib import Path

from small_model_train.io_utils import write_jsonl
from small_model_train import style_profile
from small_model_train.style_profile import build_style_profile, render_style_contract


def test_build_style_profile_counts_core_metrics():
    rows = [
        {"id": "c1", "text": "林默看了一眼。\n\n“加钱。”"},
        {"id": "c2", "text": "苏小满愣住。\n\n“多少？”"},
    ]
    profile = build_style_profile(rows)
    assert profile["chapter_count"] == 2
    assert profile["avg_chinese_chars"] > 0
    assert profile["avg_dialogue_ratio"] == 0.5


def test_build_style_profile_reports_distributions_and_ai_trace_metrics():
    rows = [
        {"id": "a", "quality_tag": "A", "text": "林默点头。空气仿佛凝固了。\n\n“加钱。”"},
        {"id": "b", "quality_tag": "B", "text": "这行也参与函数级统计。"},
    ]

    profile = build_style_profile(rows)

    assert profile["chapter_count"] == 2
    assert profile["chinese_chars"]["min"] > 0
    assert profile["chinese_chars"]["p50"] > 0
    assert profile["paragraph_chars"]["avg"] > 0
    assert profile["dialogue_ratio"]["avg"] >= 0
    assert profile["sentence_chars"]["p90"] > 0
    assert profile["punctuation_density"]["。"] > 0
    assert profile["ai_taste"]["phrase_hits"]["空气仿佛凝固了"] == 1
    assert profile["source_filter"]["selected_rows"] == 2


def test_build_style_profile_handles_empty_input():
    profile = build_style_profile([])

    assert profile["chapter_count"] == 0
    assert profile["chinese_chars"]["avg"] == 0
    assert profile["paragraph_chars"]["p90"] == 0
    assert profile["ai_taste"]["total_hits"] == 0


def test_style_profile_distribution_uses_linear_interpolated_percentiles():
    two_values = style_profile._distribution([10, 20])
    four_values = style_profile._distribution([10, 20, 30, 40])

    assert two_values["p50"] == 15
    assert two_values["p90"] == 19
    assert four_values["p50"] == 25
    assert four_values["p90"] == 37


def test_render_style_contract_accepts_legacy_nested_scalar_and_none_metrics():
    profiles = [
        {"avg_dialogue_ratio": 0.5, "avg_paragraph_chars": 8},
        {"dialogue_ratio": {"avg": 0.25}, "paragraph_chars": {"avg": 12}},
        {"dialogue_ratio": 0.25, "paragraph_chars": 12},
        {"dialogue_ratio": None, "paragraph_chars": None},
    ]

    for profile in profiles:
        contract = render_style_contract(profile)

        assert "对话比例参考" in contract
        assert "段落长度参考" in contract


def test_build_style_profile_skips_invalid_ai_trace_phrases(monkeypatch):
    monkeypatch.setattr(
        style_profile,
        "AI_TRACE_PHRASES",
        ["空气仿佛凝固了", "", None, 123],
    )

    profile = build_style_profile(
        [{"id": "a", "text": "空气仿佛凝固了。空气仿佛凝固了。"}]
    )

    assert profile["ai_taste"]["phrase_hits"] == {"空气仿佛凝固了": 2}
    assert profile["ai_taste"]["total_hits"] == 2


def test_render_style_contract_contains_project_rules():
    contract = render_style_contract({"avg_dialogue_ratio": 0.5, "avg_paragraph_chars": 8})
    assert "只输出正文" in contract
    assert "不要输出提纲" in contract
    assert "对话比例参考" in contract


def test_build_style_contract_script_writes_profile_and_contract_for_a_rows(tmp_path):
    chapters_path = tmp_path / "chapters.jsonl"
    contract_path = tmp_path / "style_contract.txt"
    profile_path = tmp_path / "style_profile.json"
    write_jsonl(
        chapters_path,
        [
            {"id": "a", "quality_tag": "A", "text": "林默点头。\n\n“成交。”"},
            {"id": "b", "quality_tag": "B", "text": "这行不应该参与统计。"},
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_style_contract.py",
            "--chapters",
            str(chapters_path),
            "--contract-output",
            str(contract_path),
            "--contract-json-output",
            str(tmp_path / "style_contract.json"),
            "--metrics-output",
            str(tmp_path / "style_metrics.json"),
            "--profile-output",
            str(profile_path),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert contract_path.exists()
    assert profile_path.exists()
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile["chapter_count"] == 1


def test_build_style_contract_script_writes_json_markdown_and_metrics(tmp_path):
    chapters_path = tmp_path / "chapters.jsonl"
    contract_json_path = tmp_path / "data_style" / "style_contract_author_main_v1.json"
    contract_md_path = tmp_path / "style_contract.md"
    metrics_path = tmp_path / "data_style" / "style_metrics_author_main_v1.json"
    write_jsonl(
        chapters_path,
        [
            {"id": "a", "quality_tag": "A", "split": "train", "text": "林默点头。\n\n“成交。”"},
            {"id": "b", "quality_tag": "B", "split": "train", "text": "这行不应该参与统计。"},
            {"id": "c", "quality_tag": "B", "split": "eval", "text": "这行也不应该参与统计。"},
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_style_contract.py",
            "--chapters",
            str(chapters_path),
            "--contract-json-output",
            str(contract_json_path),
            "--contract-output",
            str(contract_md_path),
            "--metrics-output",
            str(metrics_path),
            "--style-contract-id",
            "author_main_v1",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    asset = json.loads(contract_json_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert asset["approval_status"] == "pending_review"
    assert asset["style_contract_id"] == "author_main_v1"
    assert asset["contract_sha256"]
    assert asset["source_corpus"]["row_count"] == 3
    assert asset["source_corpus"]["selected_rows"] == 1
    assert asset["source_corpus"]["split_summary"] == {"train": 1}
    assert metrics["chapter_count"] == 1
    assert metrics["source_filter"]["total_rows"] == 3
    assert metrics["source_filter"]["selected_rows"] == 1
    assert metrics["source_filter"]["skipped_rows"] == 2
    assert "# Style Contract author_main_v1" in contract_md_path.read_text(encoding="utf-8")


def test_build_style_contract_rejects_duplicate_outputs(tmp_path):
    chapters_path = tmp_path / "chapters.jsonl"
    output_path = tmp_path / "same.json"
    write_jsonl(chapters_path, [{"id": "a", "quality_tag": "A", "text": "正文"}])

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_style_contract.py",
            "--chapters",
            str(chapters_path),
            "--contract-json-output",
            str(output_path),
            "--contract-output",
            str(output_path),
            "--metrics-output",
            str(tmp_path / "metrics.json"),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    assert "output paths must be distinct" in result.stderr


def test_build_style_contract_rejects_empty_selected_corpus_without_outputs(tmp_path):
    chapters_path = tmp_path / "chapters.jsonl"
    contract_json_path = tmp_path / "style_contract.json"
    contract_md_path = tmp_path / "style_contract.md"
    metrics_path = tmp_path / "style_metrics.json"
    write_jsonl(chapters_path, [{"id": "b", "quality_tag": "B", "text": "正文"}])

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_style_contract.py",
            "--chapters",
            str(chapters_path),
            "--contract-json-output",
            str(contract_json_path),
            "--contract-output",
            str(contract_md_path),
            "--metrics-output",
            str(metrics_path),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    assert "no rows matched quality_tag=A" in result.stderr
    assert not contract_json_path.exists()
    assert not contract_md_path.exists()
    assert not metrics_path.exists()


def test_build_style_contract_uses_default_outputs(tmp_path):
    chapters_path = tmp_path / "chapters.jsonl"
    script_path = Path("scripts/build_style_contract.py").resolve()
    write_jsonl(chapters_path, [{"id": "a", "quality_tag": "A", "text": "正文"}])

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--chapters",
            str(chapters_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "style_contract.md").exists()
    assert (tmp_path / "data_style" / "style_contract_author_main_v1.json").exists()
    assert (tmp_path / "data_style" / "style_metrics_author_main_v1.json").exists()


def test_build_style_contract_rejects_blank_style_contract_id(tmp_path):
    chapters_path = tmp_path / "chapters.jsonl"
    contract_json_path = tmp_path / "style_contract.json"
    contract_md_path = tmp_path / "style_contract.md"
    metrics_path = tmp_path / "style_metrics.json"
    write_jsonl(chapters_path, [{"id": "a", "quality_tag": "A", "text": "正文"}])

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_style_contract.py",
            "--chapters",
            str(chapters_path),
            "--contract-json-output",
            str(contract_json_path),
            "--contract-output",
            str(contract_md_path),
            "--metrics-output",
            str(metrics_path),
            "--style-contract-id",
            " ",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    assert "style-contract-id must be non-empty" in result.stderr
    assert not contract_json_path.exists()
    assert not contract_md_path.exists()
    assert not metrics_path.exists()
