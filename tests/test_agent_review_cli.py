from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from small_model_train.agent_review import REVIEWERS
from small_model_train.execution_cards import DEFAULT_TARGET_PLATFORM, RUBRIC_VERSION
from small_model_train.io_utils import read_jsonl, write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]


def _card(sample_id: str = "case1") -> dict:
    return {
        "id": sample_id,
        "target_platform": DEFAULT_TARGET_PLATFORM,
        "genre_tags": ["xuanhuan", "system"],
        "style_contract": "男频爽文，节奏紧，强钩子。",
        "chapter_goal": "主角发现系统任务并反击压迫者。",
        "chapter_structure": [
            {
                "step": 1,
                "name": "开局压迫",
                "goal": "建立困境和目标",
                "estimated_chars": "800",
            }
        ],
        "conflict_beat": "旧势力当众羞辱主角。",
        "payoff_beat": "主角用系统奖励完成反杀。",
        "must_include": ["系统面板", "当众反击"],
        "must_not_include": ["女频误会流"],
        "ending_hook": "新的悬赏任务出现。",
        "target_word_count": 1800,
    }


def _review(sample_id: str, reviewer: str, passed: bool = True) -> dict:
    return {
        "id": sample_id,
        "target_platform": DEFAULT_TARGET_PLATFORM,
        "genre_tags": ["xuanhuan", "system"],
        "rubric_version": RUBRIC_VERSION,
        "reviewer": reviewer,
        "pass": passed,
        "severity": "none" if passed else "major",
        "issues": [] if passed else ["semantic_repetition"],
        "evidence": ["concise review evidence"],
        "recommendation": "accept" if passed else "revise",
        "confidence": 0.9,
    }


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/run_agent_review.py", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_run_agent_review_import_mode_writes_votes_and_report(tmp_path: Path):
    cards_path = tmp_path / "cards.jsonl"
    outputs_path = tmp_path / "outputs.jsonl"
    metrics_path = tmp_path / "metrics.jsonl"
    imported_reviews_path = tmp_path / "reviews_in.jsonl"
    output_reviews_path = tmp_path / "reviews_out.jsonl"
    votes_path = tmp_path / "votes.jsonl"
    summary_path = tmp_path / "summary.jsonl"
    report_path = tmp_path / "report.md"
    write_jsonl(cards_path, [_card()])
    write_jsonl(outputs_path, [{"id": "case1", "output": "正文"}])
    write_jsonl(metrics_path, [{"id": "case1", "hard_gate_pass": True, "failure_types": []}])
    write_jsonl(
        imported_reviews_path,
        [_review("case1", reviewer, True) for reviewer in sorted(REVIEWERS)],
    )

    result = _run_cli(
        "--cards",
        str(cards_path),
        "--outputs",
        str(outputs_path),
        "--metrics",
        str(metrics_path),
        "--target-platform",
        DEFAULT_TARGET_PLATFORM,
        "--reviews-import",
        str(imported_reviews_path),
        "--output",
        str(output_reviews_path),
        "--votes-output",
        str(votes_path),
        "--summary-output",
        str(summary_path),
        "--report",
        str(report_path),
    )

    assert result.returncode == 0, result.stderr
    assert len(read_jsonl(output_reviews_path)) == 3
    assert read_jsonl(votes_path)[0]["agent_gate_pass"] is True
    assert read_jsonl(summary_path)[0]["decision"] == "ready_for_next_expansion"
    assert "ready_for_next_expansion" in report_path.read_text(encoding="utf-8")


def test_run_agent_review_import_mode_exits_nonzero_when_outputs_missing(
    tmp_path: Path,
):
    cards_path = tmp_path / "cards.jsonl"
    outputs_path = tmp_path / "missing_outputs.jsonl"
    metrics_path = tmp_path / "metrics.jsonl"
    imported_reviews_path = tmp_path / "reviews_in.jsonl"
    output_reviews_path = tmp_path / "reviews_out.jsonl"
    votes_path = tmp_path / "votes.jsonl"
    report_path = tmp_path / "report.md"
    write_jsonl(cards_path, [_card()])
    write_jsonl(metrics_path, [{"id": "case1", "hard_gate_pass": True, "failure_types": []}])
    write_jsonl(
        imported_reviews_path,
        [_review("case1", reviewer, True) for reviewer in sorted(REVIEWERS)],
    )

    result = _run_cli(
        "--cards",
        str(cards_path),
        "--outputs",
        str(outputs_path),
        "--metrics",
        str(metrics_path),
        "--target-platform",
        DEFAULT_TARGET_PLATFORM,
        "--reviews-import",
        str(imported_reviews_path),
        "--output",
        str(output_reviews_path),
        "--votes-output",
        str(votes_path),
        "--report",
        str(report_path),
    )

    assert result.returncode != 0
    assert "outputs file is missing or empty" in result.stderr


def test_run_agent_review_import_mode_exits_nonzero_when_metrics_missing(
    tmp_path: Path,
):
    cards_path = tmp_path / "cards.jsonl"
    outputs_path = tmp_path / "outputs.jsonl"
    metrics_path = tmp_path / "missing_metrics.jsonl"
    imported_reviews_path = tmp_path / "reviews_in.jsonl"
    output_reviews_path = tmp_path / "reviews_out.jsonl"
    votes_path = tmp_path / "votes.jsonl"
    report_path = tmp_path / "report.md"
    write_jsonl(cards_path, [_card()])
    write_jsonl(outputs_path, [{"id": "case1", "output": "正文"}])
    write_jsonl(
        imported_reviews_path,
        [_review("case1", reviewer, True) for reviewer in sorted(REVIEWERS)],
    )

    result = _run_cli(
        "--cards",
        str(cards_path),
        "--outputs",
        str(outputs_path),
        "--metrics",
        str(metrics_path),
        "--target-platform",
        DEFAULT_TARGET_PLATFORM,
        "--reviews-import",
        str(imported_reviews_path),
        "--output",
        str(output_reviews_path),
        "--votes-output",
        str(votes_path),
        "--report",
        str(report_path),
    )

    assert result.returncode != 0
    assert "metrics file is missing or empty" in result.stderr


def test_run_agent_review_import_mode_exits_nonzero_when_metrics_empty(
    tmp_path: Path,
):
    cards_path = tmp_path / "cards.jsonl"
    outputs_path = tmp_path / "outputs.jsonl"
    metrics_path = tmp_path / "metrics.jsonl"
    imported_reviews_path = tmp_path / "reviews_in.jsonl"
    output_reviews_path = tmp_path / "reviews_out.jsonl"
    votes_path = tmp_path / "votes.jsonl"
    report_path = tmp_path / "report.md"
    write_jsonl(cards_path, [_card()])
    write_jsonl(outputs_path, [{"id": "case1", "output": "正文"}])
    write_jsonl(metrics_path, [])
    write_jsonl(
        imported_reviews_path,
        [_review("case1", reviewer, True) for reviewer in sorted(REVIEWERS)],
    )

    result = _run_cli(
        "--cards",
        str(cards_path),
        "--outputs",
        str(outputs_path),
        "--metrics",
        str(metrics_path),
        "--target-platform",
        DEFAULT_TARGET_PLATFORM,
        "--reviews-import",
        str(imported_reviews_path),
        "--output",
        str(output_reviews_path),
        "--votes-output",
        str(votes_path),
        "--report",
        str(report_path),
    )

    assert result.returncode != 0
    assert "metrics file is missing or empty" in result.stderr


def test_run_agent_review_import_mode_exits_nonzero_when_outputs_missing_card_id(
    tmp_path: Path,
):
    cards_path = tmp_path / "cards.jsonl"
    outputs_path = tmp_path / "outputs.jsonl"
    metrics_path = tmp_path / "metrics.jsonl"
    imported_reviews_path = tmp_path / "reviews_in.jsonl"
    output_reviews_path = tmp_path / "reviews_out.jsonl"
    votes_path = tmp_path / "votes.jsonl"
    report_path = tmp_path / "report.md"
    write_jsonl(cards_path, [_card("case1"), _card("case2")])
    write_jsonl(outputs_path, [{"id": "case1", "output": "正文"}])
    write_jsonl(
        metrics_path,
        [
            {"id": "case1", "hard_gate_pass": True, "failure_types": []},
            {"id": "case2", "hard_gate_pass": True, "failure_types": []},
        ],
    )
    write_jsonl(
        imported_reviews_path,
        [
            _review(sample_id, reviewer, True)
            for sample_id in ("case1", "case2")
            for reviewer in sorted(REVIEWERS)
        ],
    )

    result = _run_cli(
        "--cards",
        str(cards_path),
        "--outputs",
        str(outputs_path),
        "--metrics",
        str(metrics_path),
        "--target-platform",
        DEFAULT_TARGET_PLATFORM,
        "--reviews-import",
        str(imported_reviews_path),
        "--output",
        str(output_reviews_path),
        "--votes-output",
        str(votes_path),
        "--report",
        str(report_path),
    )

    assert result.returncode != 0
    assert "outputs missing rows for card ids: case2" in result.stderr


def test_run_agent_review_import_mode_exits_nonzero_when_metrics_missing_card_id(
    tmp_path: Path,
):
    cards_path = tmp_path / "cards.jsonl"
    outputs_path = tmp_path / "outputs.jsonl"
    metrics_path = tmp_path / "metrics.jsonl"
    imported_reviews_path = tmp_path / "reviews_in.jsonl"
    output_reviews_path = tmp_path / "reviews_out.jsonl"
    votes_path = tmp_path / "votes.jsonl"
    report_path = tmp_path / "report.md"
    write_jsonl(cards_path, [_card("case1"), _card("case2")])
    write_jsonl(
        outputs_path,
        [{"id": "case1", "output": "正文"}, {"id": "case2", "output": "正文"}],
    )
    write_jsonl(metrics_path, [{"id": "case1", "hard_gate_pass": True, "failure_types": []}])
    write_jsonl(
        imported_reviews_path,
        [
            _review(sample_id, reviewer, True)
            for sample_id in ("case1", "case2")
            for reviewer in sorted(REVIEWERS)
        ],
    )

    result = _run_cli(
        "--cards",
        str(cards_path),
        "--outputs",
        str(outputs_path),
        "--metrics",
        str(metrics_path),
        "--target-platform",
        DEFAULT_TARGET_PLATFORM,
        "--reviews-import",
        str(imported_reviews_path),
        "--output",
        str(output_reviews_path),
        "--votes-output",
        str(votes_path),
        "--report",
        str(report_path),
    )

    assert result.returncode != 0
    assert "metrics missing rows for card ids: case2" in result.stderr

def test_run_agent_review_mock_mode_exits_nonzero_on_failed_metrics(tmp_path: Path):
    cards_path = tmp_path / "cards.jsonl"
    outputs_path = tmp_path / "outputs.jsonl"
    metrics_path = tmp_path / "metrics.jsonl"
    output_reviews_path = tmp_path / "reviews_out.jsonl"
    votes_path = tmp_path / "votes.jsonl"
    report_path = tmp_path / "report.md"
    write_jsonl(cards_path, [_card()])
    write_jsonl(outputs_path, [{"id": "case1", "output": "正文"}])
    write_jsonl(
        metrics_path,
        [
            {
                "id": "case1",
                "hard_gate_pass": False,
                "failure_types": ["semantic_repetition"],
            }
        ],
    )

    result = _run_cli(
        "--cards",
        str(cards_path),
        "--outputs",
        str(outputs_path),
        "--metrics",
        str(metrics_path),
        "--target-platform",
        DEFAULT_TARGET_PLATFORM,
        "--backend",
        "mock",
        "--output",
        str(output_reviews_path),
        "--votes-output",
        str(votes_path),
        "--report",
        str(report_path),
    )

    assert result.returncode == 1
    assert read_jsonl(votes_path)[0]["agent_gate_pass"] is False
    assert "blocked_by_agent_review" in report_path.read_text(encoding="utf-8")
