from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_stage5d_review_report.py"
RUNBOOK = REPO_ROOT / "docs" / "stage5d-author-feedback-ai-taste-reduction.zh.md"


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
        raw_outputs={"out-1": "你好世界", "out-2": "春风又绿江南岸"},
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
    assert summary["changed_char_delta"] == 6
    assert summary["rejection_sampling_sft_rows"] == 2
    assert summary["preference_candidate_rows"] == 1
    assert summary["plan_execution_regressions"] == 2
    assert summary["boundary"] == "candidate_data_only_no_preference_training"


def test_build_stage5d_summary_tracks_density_edit_burden_and_split_risk():
    from small_model_train.review.stage5d_report import build_stage5d_summary

    review_records = [
        {
            "source_output_id": "gen-1",
            "review_source": "author",
            "defects": [{"label": "generic_phrase", "severity": "minor"}],
        },
        {
            "source_output_id": "gen-2",
            "review_source": "deterministic",
            "defects": [{"label": "plan_execution_regression", "severity": "major"}],
        },
    ]
    raw_outputs = {
        "gen-1": "林默没有解释，只把合同推到桌面。岳家的人第一次停住。",
        "gen-2": "门外响起脚步声。",
    }
    revision_records = [
        {
            "revision_id": "rev-1",
            "revision_status": "accepted",
            "model_output": "甲乙丙丁",
            "revised_output": "甲乙丙丁戊己庚辛",
        },
        {
            "revision_id": "rev-2",
            "revision_status": "accepted",
            "model_output": "山河故人",
            "revised_output": "山河故人春风又绿江",
        },
    ]
    rejection_sampling_rows = [
        {"revision_id": "rev-1", "source_split": "train"},
        {"revision_id": "rev-2", "source_split": "sealed"},
    ]

    summary = build_stage5d_summary(
        review_records,
        revision_records,
        rejection_sampling_rows=rejection_sampling_rows,
        preference_rows=[],
        raw_outputs=raw_outputs,
    )

    assert summary["reviewed_output_chars"] == 30
    assert summary["defect_density_per_10k_chars"] == round(2 / 30 * 10000, 4)
    assert summary["edit_burden"] == {"mean_changed_chars": 4.5, "median_changed_chars": 4.5}
    assert summary["candidate_split_counts"] == {"sealed": 1, "train": 1}
    assert summary["non_train_rejection_sampling_rows"] == ["rev-2"]
    assert summary["review_source_counts"] == {"author": 1, "deterministic": 1}


def test_build_stage5d_summary_rejects_missing_raw_output_ids():
    import pytest

    from small_model_train.review.stage5d_report import build_stage5d_summary

    with pytest.raises(ValueError, match="missing raw output for reviewed output id: gen-missing"):
        build_stage5d_summary(
            [{"source_output_id": "gen-missing", "defects": []}],
            [],
            rejection_sampling_rows=[],
            preference_rows=[],
            raw_outputs={},
        )


def test_build_stage5d_summary_counts_reviewer_when_review_source_is_absent():
    from small_model_train.review.stage5d_report import build_stage5d_summary

    summary = build_stage5d_summary(
        [{"id": "out-1", "reviewer": "author", "defects": []}],
        [],
        rejection_sampling_rows=[],
        preference_rows=[],
        raw_outputs={"out-1": "山河"},
    )

    assert summary["review_source_counts"] == {"author": 1}


def test_build_stage5d_summary_empty_input_returns_zero_boundary_summary():
    from small_model_train.review.stage5d_report import build_stage5d_summary

    summary = build_stage5d_summary([], [], [], [], raw_outputs={})

    assert summary == {
        "reviewed_outputs": 0,
        "reviewed_output_chars": 0,
        "defects": {"total_defects": 0, "by_label": {}, "by_severity": {}},
        "defect_density_per_10k_chars": 0.0,
        "revision_records": 0,
        "accepted_revisions": 0,
        "author_acceptance_rate": 0.0,
        "changed_char_delta": 0,
        "edit_burden": {"mean_changed_chars": 0.0, "median_changed_chars": 0.0},
        "rejection_sampling_sft_rows": 0,
        "preference_candidate_rows": 0,
        "candidate_split_counts": {},
        "non_train_rejection_sampling_rows": [],
        "review_source_counts": {},
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
            "reviewed_output_chars": 2,
            "defect_density_per_10k_chars": 5000.0,
            "edit_burden": {"mean_changed_chars": 0.0, "median_changed_chars": 0.0},
            "rejection_sampling_sft_rows": 1,
            "preference_candidate_rows": 1,
            "candidate_split_counts": {"train": 1},
            "non_train_rejection_sampling_rows": [],
            "review_source_counts": {"author": 1},
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
    raw_outputs = tmp_path / "raw_outputs.jsonl"
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
    _write_jsonl(raw_outputs, [{"id": "out-1", "output": "山河"}])

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
            "--raw-outputs",
            str(raw_outputs),
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
    assert summary["reviewed_output_chars"] == 2
    assert "Stage 5D Review Report" in report_output.read_text(encoding="utf-8")


def test_cli_requires_raw_outputs_for_density_metrics(tmp_path: Path):
    review_records = tmp_path / "review.jsonl"
    revisions = tmp_path / "revisions.jsonl"
    rejection_sampling_rows = tmp_path / "rs.jsonl"
    preference_rows = tmp_path / "pref.jsonl"
    summary_output = tmp_path / "out" / "summary.json"
    report_output = tmp_path / "out" / "report.md"
    _write_jsonl(review_records, [])
    _write_jsonl(revisions, [])
    _write_jsonl(rejection_sampling_rows, [])
    _write_jsonl(preference_rows, [])

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

    assert result.returncode != 0
    assert "--raw-outputs is required" in result.stderr
    assert not summary_output.exists()
    assert not report_output.exists()


def test_cli_raw_outputs_prefers_raw_output_before_output_and_text(tmp_path: Path):
    review_records = tmp_path / "review.jsonl"
    revisions = tmp_path / "revisions.jsonl"
    rejection_sampling_rows = tmp_path / "rs.jsonl"
    preference_rows = tmp_path / "pref.jsonl"
    raw_outputs = tmp_path / "raw_outputs.jsonl"
    summary_output = tmp_path / "out" / "summary.json"
    report_output = tmp_path / "out" / "report.md"
    _write_jsonl(review_records, [{"id": "out-1", "defects": []}])
    _write_jsonl(revisions, [])
    _write_jsonl(rejection_sampling_rows, [])
    _write_jsonl(preference_rows, [])
    _write_jsonl(raw_outputs, [{"id": "out-1", "output": "", "raw_output": "山河"}])

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
            "--raw-outputs",
            str(raw_outputs),
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
    summary = json.loads(summary_output.read_text(encoding="utf-8"))
    assert summary["reviewed_output_chars"] == 2


def test_cli_rejects_empty_raw_output_text(tmp_path: Path):
    review_records = tmp_path / "review.jsonl"
    revisions = tmp_path / "revisions.jsonl"
    rejection_sampling_rows = tmp_path / "rs.jsonl"
    preference_rows = tmp_path / "pref.jsonl"
    raw_outputs = tmp_path / "raw_outputs.jsonl"
    summary_output = tmp_path / "out" / "summary.json"
    report_output = tmp_path / "out" / "report.md"
    _write_jsonl(review_records, [{"id": "out-1", "defects": []}])
    _write_jsonl(revisions, [])
    _write_jsonl(rejection_sampling_rows, [])
    _write_jsonl(preference_rows, [])
    _write_jsonl(raw_outputs, [{"id": "out-1", "raw_output": ""}])

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
            "--raw-outputs",
            str(raw_outputs),
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
    assert "raw output row 1 has empty text" in result.stderr
    assert not summary_output.exists()
    assert not report_output.exists()


def test_cli_fails_nonzero_without_outputs_when_required_input_path_is_missing(tmp_path: Path):
    missing_review_records = tmp_path / "missing-review.jsonl"
    revisions = tmp_path / "revisions.jsonl"
    rejection_sampling_rows = tmp_path / "rs.jsonl"
    preference_rows = tmp_path / "pref.jsonl"
    raw_outputs = tmp_path / "raw_outputs.jsonl"
    summary_output = tmp_path / "summary.json"
    report_output = tmp_path / "report.md"
    _write_jsonl(revisions, [])
    _write_jsonl(rejection_sampling_rows, [])
    _write_jsonl(preference_rows, [])
    _write_jsonl(raw_outputs, [])

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
            "--raw-outputs",
            str(raw_outputs),
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


def test_stage5d_runbook_report_command_includes_required_raw_outputs():
    assert (
        "--raw-outputs outputs/stage5d_generation_records.jsonl"
        in RUNBOOK.read_text(encoding="utf-8")
    )
