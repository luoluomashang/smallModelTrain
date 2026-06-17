from __future__ import annotations

import subprocess
import sys

from small_model_train.io_utils import write_jsonl
from small_model_train.reporting import build_markdown_report, summarize_scores


def test_summarize_scores_counts_hard_gate_and_failures():
    scores = [
        {"id": "a", "hard_gate_pass": True, "failure_types": [], "char_count_zh": 2200},
        {"id": "b", "hard_gate_pass": False, "failure_types": ["length_short"], "char_count_zh": 900},
    ]
    summary = summarize_scores(scores)
    assert summary["sample_count"] == 2
    assert summary["hard_gate_pass_rate"] == 0.5
    assert summary["failure_counts"]["length_short"] == 1


def test_summarize_scores_empty_returns_zeroes():
    summary = summarize_scores([])
    assert summary["sample_count"] == 0
    assert summary["hard_gate_pass_rate"] == 0
    assert summary["avg_chinese_chars"] == 0


def test_build_markdown_report_contains_decision():
    report = build_markdown_report(
        title="SFT v1",
        scores=[{"id": "a", "hard_gate_pass": True, "failure_types": [], "char_count_zh": 2200}],
        config_snapshot={"model": "qwen3"},
    )
    assert "# SFT v1" in report
    assert "是否进入下一阶段" in report


def test_evaluate_outputs_cli_reads_scores_and_writes_report(tmp_path):
    scores_path = tmp_path / "scores.jsonl"
    report_path = tmp_path / "reports" / "evaluation.md"
    write_jsonl(
        scores_path,
        [
            {
                "id": "case1",
                "hard_gate_pass": True,
                "failure_types": [],
                "char_count_zh": 2200,
            }
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_outputs.py",
            "--scores",
            str(scores_path),
            "--report",
            str(report_path),
            "--title",
            "SFT v1",
            "--config-json",
            '{"model":"qwen3"}',
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = report_path.read_text(encoding="utf-8")
    assert "# SFT v1" in report
    assert '"model": "qwen3"' in report
