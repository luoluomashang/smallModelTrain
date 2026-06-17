from __future__ import annotations

import subprocess
import sys

from small_model_train.io_utils import read_jsonl, write_jsonl
from small_model_train.scoring import detect_ai_trace, score_output


def test_detect_ai_trace_counts_known_phrases():
    result = detect_ai_trace("空气仿佛凝固了，他心中涌起一股复杂的情绪。")
    assert result["count"] == 2
    assert "空气仿佛凝固了" in result["matches"]


def test_score_output_applies_hard_gates_and_failure_types():
    card = {
        "must_include": ["加钱"],
        "must_not_include": ["真相是他父亲"],
    }
    output = "林默说加钱。真相是他父亲。"
    score = score_output("case1", card, output)
    assert score["hard_gate_pass"] is False
    assert "forbidden_violation" in score["failure_types"]
    assert score["must_include_coverage"] == 1.0


def test_score_output_marks_length_short():
    score = score_output("case2", {"must_include": [], "must_not_include": []}, "太短。")
    assert "length_short" in score["failure_types"]


def test_score_output_marks_outline_leak():
    output = "以下是正文：" + "林默加钱。" * 500
    score = score_output("case3", {"must_include": [], "must_not_include": []}, output)
    assert "outline_leak" in score["failure_types"]
    assert score["hard_gate_pass"] is False


def test_score_output_marks_missing_must_include():
    card = {"must_include": ["加钱", "雨夜"], "must_not_include": []}
    score = score_output("case4", card, "林默走进房间。" * 400)
    assert "must_include_missing" in score["failure_types"]
    assert score["must_include_coverage"] == 0.0


def test_score_outputs_cli_reads_cards_and_outputs_jsonl(tmp_path):
    cards_path = tmp_path / "cards.jsonl"
    outputs_path = tmp_path / "outputs.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    write_jsonl(
        cards_path,
        [
            {
                "id": "case1",
                "must_include": ["加钱"],
                "must_not_include": ["真相是他父亲"],
            }
        ],
    )
    write_jsonl(outputs_path, [{"id": "case1", "output": "林默说加钱。真相是他父亲。"}])

    result = subprocess.run(
        [
            sys.executable,
            "scripts/score_outputs.py",
            "--cards",
            str(cards_path),
            "--outputs",
            str(outputs_path),
            "--output",
            str(scores_path),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    rows = read_jsonl(scores_path)
    assert len(rows) == 1
    assert rows[0]["id"] == "case1"
    assert rows[0]["hard_gate_pass"] is False
    assert "forbidden_violation" in rows[0]["failure_types"]
