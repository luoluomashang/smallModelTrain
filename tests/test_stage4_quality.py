from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from small_model_train.io_utils import read_jsonl, write_jsonl
from small_model_train.stage4_quality import (
    detect_outline_markers,
    render_quality_budget_report,
    select_quality_subset,
    summarize_quality_budget,
)


def _card(sample_id: str) -> dict:
    return {"id": sample_id, "chapter_goal": f"goal {sample_id}"}


def test_select_quality_subset_prioritizes_outline_leak_metrics():
    cards = [_card("a"), _card("b"), _card("c")]
    metrics = [
        {
            "id": "a",
            "failure_types": ["length_short"],
            "char_count_zh": 300,
        },
        {
            "id": "b",
            "failure_types": ["length_short", "outline_leak"],
            "char_count_zh": 280,
        },
        {
            "id": "c",
            "failure_types": [],
            "char_count_zh": 2200,
        },
    ]

    subset = select_quality_subset(cards, metrics, count=2)

    assert [row["id"] for row in subset] == ["b", "a"]


def test_select_quality_subset_falls_back_to_eval_order_without_metrics():
    subset = select_quality_subset([_card("a"), _card("b")], [], count=1)

    assert [row["id"] for row in subset] == ["a"]


def test_detect_outline_markers_reports_known_markers():
    assert detect_outline_markers("以下是正文：【章节结构】") == [
        "【",
        "】",
        "章节结构",
        "以下是正文",
    ]


def test_summarize_quality_budget_counts_rows_tokens_and_failures():
    cards = [_card("a"), _card("b")]
    generated = [
        {"id": "a", "output": "短文", "params": {"max_new_tokens": 1024}},
        {
            "id": "b",
            "output": "【章节结构】",
            "params": {"max_new_tokens": 1024},
        },
    ]
    metrics = [
        {
            "id": "a",
            "hard_gate_pass": False,
            "char_count_zh": 300,
            "failure_types": ["length_short"],
        },
        {
            "id": "b",
            "hard_gate_pass": False,
            "char_count_zh": 320,
            "failure_types": ["length_short", "outline_leak"],
        },
    ]

    summary = summarize_quality_budget(cards, generated, metrics)

    assert summary["expected_rows"] == 2
    assert summary["generated_rows"] == 2
    assert summary["metrics_rows"] == 2
    assert summary["max_new_tokens"] == [1024]
    assert summary["failure_counts"] == {"length_short": 2, "outline_leak": 1}
    assert summary["decision"] == "blocked_length_short"


def test_render_quality_budget_report_does_not_include_generated_text():
    cards = [_card("a")]
    generated = [
        {
            "id": "a",
            "output": "以下是正文：秘密文本",
            "params": {"max_new_tokens": 1024},
        }
    ]
    metrics = [
        {
            "id": "a",
            "hard_gate_pass": False,
            "char_count_zh": 300,
            "failure_types": ["outline_leak"],
        }
    ]
    summary = summarize_quality_budget(cards, generated, metrics)

    report = render_quality_budget_report("Stage 4.1", summary)

    assert "# Stage 4.1" in report
    assert "outline_leak: 1" in report
    assert "以下是正文：秘密文本" not in report


def test_build_eval_quality_subset_cli_writes_prioritized_subset(tmp_path: Path):
    cards_path = tmp_path / "cards.jsonl"
    metrics_path = tmp_path / "metrics.jsonl"
    output_path = tmp_path / "subset.jsonl"
    write_jsonl(cards_path, [_card("a"), _card("b")])
    write_jsonl(
        metrics_path,
        [
            {"id": "a", "failure_types": ["length_short"], "char_count_zh": 300},
            {
                "id": "b",
                "failure_types": ["length_short", "outline_leak"],
                "char_count_zh": 280,
            },
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_eval_quality_subset.py",
            "--cards",
            str(cards_path),
            "--metrics",
            str(metrics_path),
            "--output",
            str(output_path),
            "--count",
            "1",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert [row["id"] for row in read_jsonl(output_path)] == ["b"]
    assert "wrote 1 quality eval cards" in result.stdout


def test_build_stage4_quality_report_cli_writes_report(tmp_path: Path):
    cards_path = tmp_path / "cards.jsonl"
    generated_path = tmp_path / "generated.jsonl"
    metrics_path = tmp_path / "metrics.jsonl"
    report_path = tmp_path / "report.md"
    write_jsonl(cards_path, [_card("a")])
    write_jsonl(
        generated_path,
        [{"id": "a", "output": "短文", "params": {"max_new_tokens": 1024}}],
    )
    write_jsonl(
        metrics_path,
        [
            {
                "id": "a",
                "hard_gate_pass": False,
                "char_count_zh": 300,
                "failure_types": ["length_short"],
            }
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_stage4_quality_report.py",
            "--cards",
            str(cards_path),
            "--generated",
            str(generated_path),
            "--metrics",
            str(metrics_path),
            "--report",
            str(report_path),
            "--title",
            "Stage 4.1",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = report_path.read_text(encoding="utf-8")
    assert "# Stage 4.1" in report
    assert "blocked_length_short" in report
    assert "wrote Stage 4.1 quality report" in result.stdout
