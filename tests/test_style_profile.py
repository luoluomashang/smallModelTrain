import json
import subprocess
import sys

from small_model_train.io_utils import write_jsonl
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
