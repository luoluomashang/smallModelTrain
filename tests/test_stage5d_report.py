from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_stage5d_review_report.py"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_build_stage5d_summary_counts_review_and_revision_metrics():
    from small_model_train.review.stage5d_report import build_stage5d_summary

    review_records = [
        {
            "id": "out-1",
            "defects": [
                {"label": "generic_phrase", "severity": "minor"},
                {"label": "plan_execution_regression", "severity": "major"},
            ],
        },
        {
            "id": "out-2",
            "defects": [
                {"label": "plan_execution_regression", "severity": "blocker"},
            ],
        },
    ]
    revision_records = [
        {"revision_status": "accepted", "model_output": "你好世界", "revised_output": "你好"},
        {
            "revision_status": "accepted_with_minor_edits",
            "model_output": "春风又绿江南岸",
            "revised_output": "春风绿岸",
        },
        {"revision_status": "rejected", "model_output": "甲乙丙", "revised_output": "甲乙丙丁"},
    ]

    summary = build_stage5d_summary(
        review_records,
        revision_records,
        rejection_sampling_rows=[{"id": "rs-1"}, {"id": "rs-2"}],
        preference_rows=[{"id": "pref-1"}],
    )

    assert summary["reviewed_outputs"] == 2
    assert summary["defects"] == {
        "total_defects": 3,
        "by_label": {"generic_phrase": 1, "plan_execution_regression": 2},
        "by_severity": {"blocker": 1, "major": 1, "minor": 1},
    }
    assert summary["revision_records"] == 3
    assert summary["accepted_revisions"] == 2
    assert summary["author_acceptance_rate"] == 0.6667
    assert summary["changed_char_delta"] == 4
    assert summary["rejection_sampling_sft_rows"] == 2
    assert summary["preference_candidate_rows"] == 1
    assert summary["plan_execution_regressions"] == 2
    assert summary["boundary"] == "candidate_data_only_no_preference_training"


def test_build_stage5d_summary_empty_input_returns_zero_boundary_summary():
    from small_model_train.review.stage5d_report import build_stage5d_summary

    summary = build_stage5d_summary([], [], [], [])

    assert summary == {
        "reviewed_outputs": 0,
        "defects": {"total_defects": 0, "by_label": {}, "by_severity": {}},
        "revision_records": 0,
        "accepted_revisions": 0,
        "author_acceptance_rate": 0.0,
        "changed_char_delta": 0,
        "rejection_sampling_sft_rows": 0,
        "preference_candidate_rows": 0,
        "plan_execution_regressions": 0,
        "boundary": "candidate_data_only_no_preference_training",
    }


def test_render_stage5d_report_includes_title_and_preference_training_boundary():
    from small_model_train.review.stage5d_report import render_stage5d_report

    report = render_stage5d_report(
        {
            "reviewed_outputs": 1,
            "defects": {
                "total_defects": 1,
                "by_label": {"generic_phrase": 1},
                "by_severity": {"minor": 1},
            },
            "revision_records": 1,
            "accepted_revisions": 1,
            "author_acceptance_rate": 1.0,
            "changed_char_delta": 0,
            "rejection_sampling_sft_rows": 1,
            "preference_candidate_rows": 1,
            "plan_execution_regressions": 0,
            "boundary": "candidate_data_only_no_preference_training",
        }
    )

    assert "Stage 5D Review Report" in report
    assert "这些 preference rows 只是候选数据，不代表已经运行 DPO/SimPO/ORPO/KTO。" in report


def test_cli_writes_summary_json_and_markdown_report(tmp_path: Path):
    review_records = tmp_path / "review.jsonl"
    revisions = tmp_path / "revisions.jsonl"
    rejection_sampling_rows = tmp_path / "rs.jsonl"
    preference_rows = tmp_path / "pref.jsonl"
    summary_output = tmp_path / "out" / "summary.json"
    report_output = tmp_path / "out" / "report.md"
    _write_jsonl(
        review_records,
        [{"id": "out-1", "defects": [{"label": "generic_phrase", "severity": "minor"}]}],
    )
    _write_jsonl(
        revisions,
        [{"revision_status": "accepted", "model_output": "山河", "revised_output": "山河故人"}],
    )
    _write_jsonl(rejection_sampling_rows, [{"id": "rs-1"}])
    _write_jsonl(preference_rows, [{"id": "pref-1"}, {"id": "pref-2"}])

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--review-records",
            str(review_records),
            "--revisions",
            str(revisions),
            "--rejection-sampling-rows",
            str(rejection_sampling_rows),
            "--preference-rows",
            str(preference_rows),
            "--summary-output",
            str(summary_output),
            "--report-output",
            str(report_output),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"wrote Stage 5D summary to {summary_output}" in result.stdout
    summary = json.loads(summary_output.read_text(encoding="utf-8"))
    assert summary["reviewed_outputs"] == 1
    assert summary["preference_candidate_rows"] == 2
    assert summary["changed_char_delta"] == 2
    assert "Stage 5D Review Report" in report_output.read_text(encoding="utf-8")


def test_cli_fails_nonzero_without_outputs_when_required_input_path_is_missing(tmp_path: Path):
    missing_review_records = tmp_path / "missing-review.jsonl"
    revisions = tmp_path / "revisions.jsonl"
    rejection_sampling_rows = tmp_path / "rs.jsonl"
    preference_rows = tmp_path / "pref.jsonl"
    summary_output = tmp_path / "summary.json"
    report_output = tmp_path / "report.md"
    _write_jsonl(revisions, [])
    _write_jsonl(rejection_sampling_rows, [])
    _write_jsonl(preference_rows, [])

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--review-records",
            str(missing_review_records),
            "--revisions",
            str(revisions),
            "--rejection-sampling-rows",
            str(rejection_sampling_rows),
            "--preference-rows",
            str(preference_rows),
            "--summary-output",
            str(summary_output),
            "--report-output",
            str(report_output),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "required input path does not exist" in result.stderr
    assert str(missing_review_records) in result.stderr
    assert not summary_output.exists()
    assert not report_output.exists()
