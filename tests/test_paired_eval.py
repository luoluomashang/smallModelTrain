from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from small_model_train.evaluation.paired_eval import (
    render_paired_eval_report,
    summarize_paired_eval,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_summarize_paired_eval_uses_review_judgment_over_metric_tie():
    summary = summarize_paired_eval(
        baseline_metrics=[
            {"id": "card-1", "hard_gate_pass": True, "failure_types": []},
        ],
        candidate_metrics=[
            {"id": "card-1", "hard_gate_pass": True, "failure_types": []},
        ],
        judgments=[
            {"id": "card-1", "winner": "candidate", "source": "author"},
        ],
    )

    assert summary["paired_rows"] == 1
    assert summary["wins"] == 1
    assert summary["losses"] == 0
    assert summary["ties"] == 0
    assert summary["comparisons"][0]["winner"] == "candidate"
    assert summary["comparisons"][0]["winner_source"] == "review"


def test_summarize_paired_eval_flags_candidate_regression_without_review():
    summary = summarize_paired_eval(
        baseline_metrics=[
            {"id": "card-1", "hard_gate_pass": True, "failure_types": []},
        ],
        candidate_metrics=[
            {"id": "card-1", "hard_gate_pass": False, "failure_types": ["outline_leak"]},
        ],
        judgments=[],
    )

    assert summary["wins"] == 0
    assert summary["losses"] == 1
    assert summary["regression_ids"] == ["card-1"]
    assert summary["comparisons"][0]["winner"] == "baseline"


def test_summarize_paired_eval_rejects_judgment_id_outside_paired_rows():
    with pytest.raises(
        ValueError,
        match="judgment id not found in paired rows: card-typo",
    ):
        summarize_paired_eval(
            baseline_metrics=[
                {"id": "card-1", "hard_gate_pass": True, "failure_types": []},
            ],
            candidate_metrics=[
                {"id": "card-1", "hard_gate_pass": True, "failure_types": []},
            ],
            judgments=[
                {"id": "card-typo", "winner": "candidate"},
            ],
        )


def test_summarize_paired_eval_rejects_missing_candidate_ids():
    with pytest.raises(ValueError, match=r"missing_candidate_ids.*card-2"):
        summarize_paired_eval(
            baseline_metrics=[
                {"id": "card-1", "hard_gate_pass": True, "failure_types": []},
                {"id": "card-2", "hard_gate_pass": True, "failure_types": []},
            ],
            candidate_metrics=[
                {"id": "card-1", "hard_gate_pass": True, "failure_types": []},
            ],
            judgments=[],
        )


def test_summarize_paired_eval_rejects_missing_baseline_ids():
    with pytest.raises(ValueError, match=r"missing_baseline_ids.*card-3"):
        summarize_paired_eval(
            baseline_metrics=[
                {"id": "card-1", "hard_gate_pass": True, "failure_types": []},
            ],
            candidate_metrics=[
                {"id": "card-1", "hard_gate_pass": True, "failure_types": []},
                {"id": "card-3", "hard_gate_pass": True, "failure_types": []},
            ],
            judgments=[],
        )


def test_summarize_paired_eval_records_regression_when_review_picks_candidate():
    summary = summarize_paired_eval(
        baseline_metrics=[
            {"id": "card-1", "hard_gate_pass": True, "failure_types": []},
        ],
        candidate_metrics=[
            {"id": "card-1", "hard_gate_pass": False, "failure_types": ["outline_leak"]},
        ],
        judgments=[
            {"id": "card-1", "winner": "candidate"},
        ],
    )

    assert summary["wins"] == 1
    assert summary["losses"] == 0
    assert summary["regression_ids"] == ["card-1"]
    assert summary["comparisons"][0]["winner"] == "candidate"
    assert summary["comparisons"][0]["winner_source"] == "review"


def test_summarize_paired_eval_rejects_no_paired_rows():
    with pytest.raises(ValueError, match="paired eval requires at least one paired row"):
        summarize_paired_eval(
            baseline_metrics=[],
            candidate_metrics=[],
            judgments=[],
        )


def test_render_paired_eval_report_keeps_counts_and_boundary():
    summary = summarize_paired_eval(
        baseline_metrics=[
            {"id": "card-1", "hard_gate_pass": True, "failure_types": []},
        ],
        candidate_metrics=[
            {"id": "card-1", "hard_gate_pass": True, "failure_types": []},
        ],
        judgments=[
            {"id": "card-1", "winner": "candidate"},
        ],
    )

    report = render_paired_eval_report(summary)

    assert "# Stage 5E Paired Eval Report" in report
    assert "- Candidate wins: 1" in report
    assert "paired_eval_no_training" in report


def test_build_paired_eval_report_cli_writes_summary_and_report(tmp_path):
    baseline_metrics = tmp_path / "baseline.jsonl"
    candidate_metrics = tmp_path / "candidate.jsonl"
    judgments = tmp_path / "judgments.jsonl"
    summary_output = tmp_path / "summary.json"
    report_output = tmp_path / "report.md"
    _write_jsonl(
        baseline_metrics,
        [{"id": "card-1", "hard_gate_pass": True, "failure_types": []}],
    )
    _write_jsonl(
        candidate_metrics,
        [{"id": "card-1", "hard_gate_pass": True, "failure_types": []}],
    )
    _write_jsonl(judgments, [{"id": "card-1", "winner": "candidate"}])

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_paired_eval_report.py"),
            "--baseline-metrics",
            str(baseline_metrics),
            "--candidate-metrics",
            str(candidate_metrics),
            "--judgments",
            str(judgments),
            "--summary-output",
            str(summary_output),
            "--report-output",
            str(report_output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == f"wrote Stage 5E paired eval report to {report_output}"
    assert json.loads(summary_output.read_text(encoding="utf-8"))["wins"] == 1
    assert "# Stage 5E Paired Eval Report" in report_output.read_text(encoding="utf-8")


def test_build_paired_eval_report_cli_reports_validation_errors(tmp_path):
    baseline_metrics = tmp_path / "baseline.jsonl"
    candidate_metrics = tmp_path / "candidate.jsonl"
    judgments = tmp_path / "judgments.jsonl"
    _write_jsonl(
        baseline_metrics,
        [
            {"id": "card-1", "hard_gate_pass": True, "failure_types": []},
            {"id": "card-1", "hard_gate_pass": True, "failure_types": []},
        ],
    )
    _write_jsonl(candidate_metrics, [])
    _write_jsonl(judgments, [])

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_paired_eval_report.py"),
            "--baseline-metrics",
            str(baseline_metrics),
            "--candidate-metrics",
            str(candidate_metrics),
            "--judgments",
            str(judgments),
            "--summary-output",
            str(tmp_path / "summary.json"),
            "--report-output",
            str(tmp_path / "report.md"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "error: duplicate baseline metric id: card-1" in result.stderr


def test_build_paired_eval_report_cli_requires_existing_input_files(tmp_path):
    baseline_metrics = tmp_path / "missing-baseline.jsonl"
    candidate_metrics = tmp_path / "candidate.jsonl"
    judgments = tmp_path / "judgments.jsonl"
    summary_output = tmp_path / "summary.json"
    report_output = tmp_path / "report.md"
    _write_jsonl(
        candidate_metrics,
        [{"id": "card-1", "hard_gate_pass": True, "failure_types": []}],
    )
    _write_jsonl(judgments, [])

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_paired_eval_report.py"),
            "--baseline-metrics",
            str(baseline_metrics),
            "--candidate-metrics",
            str(candidate_metrics),
            "--judgments",
            str(judgments),
            "--summary-output",
            str(summary_output),
            "--report-output",
            str(report_output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert f"error: required input path does not exist: {baseline_metrics}" in result.stderr
    assert not summary_output.exists()
    assert not report_output.exists()


def test_build_paired_eval_report_cli_removes_stale_outputs_on_failure(tmp_path):
    baseline_metrics = tmp_path / "baseline.jsonl"
    candidate_metrics = tmp_path / "candidate.jsonl"
    judgments = tmp_path / "judgments.jsonl"
    summary_output = tmp_path / "summary.json"
    report_output = tmp_path / "report.md"
    _write_jsonl(
        baseline_metrics,
        [{"id": "card-1", "hard_gate_pass": True, "failure_types": []}],
    )
    _write_jsonl(
        candidate_metrics,
        [{"id": "card-1", "hard_gate_pass": True, "failure_types": []}],
    )
    _write_jsonl(judgments, [{"id": "card-stale", "winner": "candidate"}])
    summary_output.write_text('{"stale": true}\n', encoding="utf-8")
    report_output.write_text("# Stale\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_paired_eval_report.py"),
            "--baseline-metrics",
            str(baseline_metrics),
            "--candidate-metrics",
            str(candidate_metrics),
            "--judgments",
            str(judgments),
            "--summary-output",
            str(summary_output),
            "--report-output",
            str(report_output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "judgment id not found in paired rows: card-stale" in result.stderr
    assert not summary_output.exists()
    assert not report_output.exists()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
