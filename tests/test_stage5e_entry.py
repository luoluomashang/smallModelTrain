from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_stage5e_entry.py"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _summary(**overrides: object) -> dict:
    summary = {
        "reviewed_outputs": 1,
        "reviewed_output_chars": 40,
        "defects": {
            "total_defects": 3,
            "by_label": {"generic_phrase": 2, "plan_execution_regression": 1},
            "by_severity": {"major": 1, "minor": 2},
        },
        "defect_density_per_10k_chars": 750.0,
        "revision_records": 1,
        "accepted_revisions": 1,
        "author_acceptance_rate": 1.0,
        "edit_burden": {"mean_changed_chars": 4.0, "median_changed_chars": 4.0},
        "rejection_sampling_sft_rows": 1,
        "preference_candidate_rows": 1,
        "non_train_rejection_sampling_rows": [],
        "plan_execution_regressions": 0,
        "boundary": "candidate_data_only_no_preference_training",
    }
    summary.update(overrides)
    return summary


def _review(**overrides: object) -> dict:
    record = {
        "record_id": "review-1",
        "review_source": "author",
        "reviewer": "author",
        "overall_acceptance": "accepted",
    }
    record.update(overrides)
    return record


def _revision(**overrides: object) -> dict:
    revision = {
        "revision_id": "rev-1",
        "revision_status": "accepted",
        "card_id": "card-1",
        "chapter_id": "chapter-1",
        "style_contract_sha256": "a" * 64,
        "prompt_sha256": "b" * 64,
        "raw_output_sha256": "c" * 64,
    }
    revision.update(overrides)
    return revision


def _generation(**overrides: object) -> dict:
    generation = {
        "id": "gen-1",
        "card_id": "card-1",
        "style_contract_sha256": "a" * 64,
        "prompt_sha256": "b" * 64,
        "raw_output_sha256": "c" * 64,
        "seed": 314159,
        "model_role": "author_candidate",
        "generation_params_sha256": "d" * 64,
    }
    generation.update(overrides)
    return generation


def _rs_row(**overrides: object) -> dict:
    row = {"revision_id": "rev-1", "source_split": "train"}
    row.update(overrides)
    return row


def _preference_row(**overrides: object) -> dict:
    row = {"id": "pref-1", "defect_labels": ["generic_phrase"]}
    row.update(overrides)
    return row


def test_stage5e_entry_gate_passes_complete_stage5d_evidence():
    from small_model_train.review.stage5e_entry import check_stage5e_entry

    result = check_stage5e_entry(
        summary=_summary(),
        review_records=[_review()],
        revision_records=[_revision()],
        rejection_sampling_rows=[_rs_row()],
        preference_rows=[_preference_row()],
        generation_records=[_generation()],
    )

    assert result == {
        "passed": True,
        "errors": [],
        "entry": "stage5e_controlled_experimentation",
    }


def test_stage5e_entry_gate_rejects_non_train_candidate():
    from small_model_train.review.stage5e_entry import check_stage5e_entry

    result = check_stage5e_entry(
        summary=_summary(non_train_rejection_sampling_rows=["rev-2"]),
        review_records=[_review()],
        revision_records=[_revision()],
        rejection_sampling_rows=[_rs_row(source_split="sealed")],
        preference_rows=[_preference_row()],
        generation_records=[_generation()],
    )

    assert result["passed"] is False
    assert "non-train rejection-sampling rows block Stage 5E: rev-2" in result["errors"]


def test_stage5e_entry_gate_rejects_missing_seeded_generation_link():
    from small_model_train.review.stage5e_entry import check_stage5e_entry

    result = check_stage5e_entry(
        summary=_summary(),
        review_records=[_review()],
        revision_records=[_revision()],
        rejection_sampling_rows=[_rs_row()],
        preference_rows=[_preference_row()],
        generation_records=[],
    )

    assert result["passed"] is False
    assert (
        "accepted revision lacks same-card same-style same-seed generation record: rev-1"
        in result["errors"]
    )


def test_stage5e_entry_gate_rejects_accepted_revision_missing_link_field():
    from small_model_train.review.stage5e_entry import check_stage5e_entry

    result = check_stage5e_entry(
        summary=_summary(),
        review_records=[_review()],
        revision_records=[_revision(prompt_sha256="")],
        rejection_sampling_rows=[_rs_row()],
        preference_rows=[_preference_row()],
        generation_records=[_generation()],
    )

    assert result["passed"] is False
    assert (
        "accepted revision missing generation link field: rev-1 prompt_sha256"
        in result["errors"]
    )


def test_stage5e_entry_gate_rejects_generation_record_with_only_seed():
    from small_model_train.review.stage5e_entry import check_stage5e_entry

    result = check_stage5e_entry(
        summary=_summary(),
        review_records=[_review()],
        revision_records=[_revision()],
        rejection_sampling_rows=[_rs_row()],
        preference_rows=[_preference_row()],
        generation_records=[{"id": "gen-seed-only", "seed": 7}],
    )

    assert result["passed"] is False
    assert "generation record missing link field: gen-seed-only card_id" in result["errors"]
    assert (
        "generation record missing link field: gen-seed-only style_contract_sha256"
        in result["errors"]
    )


def test_stage5e_entry_gate_reports_malformed_generation_row_direct_errors():
    from small_model_train.review.stage5e_entry import check_stage5e_entry

    result = check_stage5e_entry(
        summary=_summary(),
        review_records=[_review()],
        revision_records=[_revision()],
        rejection_sampling_rows=[_rs_row()],
        preference_rows=[_preference_row()],
        generation_records=[
            "not-an-object",
            _generation(id="gen-bool-seed", seed=True),
        ],
    )

    assert result["passed"] is False
    assert "generation record must be a JSON object: row-1" in result["errors"]
    assert "generation record missing integer seed: gen-bool-seed" in result["errors"]


def test_stage5e_entry_gate_rejects_preference_rows_without_defect_labels():
    from small_model_train.review.stage5e_entry import check_stage5e_entry

    result = check_stage5e_entry(
        summary=_summary(),
        review_records=[_review()],
        revision_records=[_revision()],
        rejection_sampling_rows=[_rs_row()],
        preference_rows=[_preference_row(defect_labels=[])],
        generation_records=[_generation()],
    )

    assert result["passed"] is False
    assert "preference row requires non-empty defect_labels: pref-1" in result["errors"]


def test_stage5e_entry_gate_rejects_summary_preference_count_mismatch():
    from small_model_train.review.stage5e_entry import check_stage5e_entry

    result = check_stage5e_entry(
        summary=_summary(preference_candidate_rows=2),
        review_records=[_review()],
        revision_records=[_revision()],
        rejection_sampling_rows=[_rs_row()],
        preference_rows=[_preference_row()],
        generation_records=[_generation()],
    )

    assert result["passed"] is False
    assert (
        "preference_candidate_rows does not match preference rows: summary=2 actual=1"
        in result["errors"]
    )


def test_stage5e_entry_gate_rejects_empty_preference_candidates():
    from small_model_train.review.stage5e_entry import check_stage5e_entry

    result = check_stage5e_entry(
        summary=_summary(preference_candidate_rows=0),
        review_records=[_review()],
        revision_records=[_revision()],
        rejection_sampling_rows=[_rs_row()],
        preference_rows=[],
        generation_records=[_generation()],
    )

    assert result["passed"] is False
    assert "preference candidate rows are required before Stage 5E" in result["errors"]


def test_stage5e_entry_gate_requires_author_human_or_blind_review():
    from small_model_train.review.stage5e_entry import check_stage5e_entry

    result = check_stage5e_entry(
        summary=_summary(),
        review_records=[_review(review_source="deterministic", reviewer="rules")],
        revision_records=[_revision()],
        rejection_sampling_rows=[_rs_row()],
        preference_rows=[_preference_row()],
        generation_records=[_generation()],
    )

    assert result["passed"] is False
    assert (
        "accepted author, human, or blind-review evidence is required before Stage 5E"
        in result["errors"]
    )


def test_stage5e_entry_gate_rejects_author_review_without_acceptance():
    from small_model_train.review.stage5e_entry import check_stage5e_entry

    result = check_stage5e_entry(
        summary=_summary(),
        review_records=[_review(overall_acceptance="rejected")],
        revision_records=[_revision()],
        rejection_sampling_rows=[_rs_row()],
        preference_rows=[_preference_row()],
        generation_records=[_generation()],
    )

    assert result["passed"] is False
    assert (
        "accepted author, human, or blind-review evidence is required before Stage 5E"
        in result["errors"]
    )


def test_stage5e_entry_cli_writes_json_report(tmp_path: Path):
    summary = tmp_path / "summary.json"
    review_records = tmp_path / "reviews.jsonl"
    revisions = tmp_path / "revisions.jsonl"
    rejection_sampling_rows = tmp_path / "rs.jsonl"
    preference_rows = tmp_path / "pref.jsonl"
    generation_records = tmp_path / "generations.jsonl"
    output = tmp_path / "out" / "stage5e_entry.json"
    _write_json(summary, _summary())
    _write_jsonl(review_records, [_review()])
    _write_jsonl(revisions, [_revision()])
    _write_jsonl(rejection_sampling_rows, [_rs_row()])
    _write_jsonl(preference_rows, [_preference_row()])
    _write_jsonl(generation_records, [_generation()])

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--summary",
            str(summary),
            "--review-records",
            str(review_records),
            "--revisions",
            str(revisions),
            "--rejection-sampling-rows",
            str(rejection_sampling_rows),
            "--preference-rows",
            str(preference_rows),
            "--generation-records",
            str(generation_records),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["errors"] == []


def test_stage5e_entry_cli_writes_failed_gate_json_report(tmp_path: Path):
    summary = tmp_path / "summary.json"
    review_records = tmp_path / "reviews.jsonl"
    revisions = tmp_path / "revisions.jsonl"
    rejection_sampling_rows = tmp_path / "rs.jsonl"
    preference_rows = tmp_path / "pref.jsonl"
    generation_records = tmp_path / "generations.jsonl"
    output = tmp_path / "out" / "stage5e_entry.json"
    _write_json(summary, _summary())
    _write_jsonl(review_records, [_review(overall_acceptance="rejected")])
    _write_jsonl(revisions, [_revision()])
    _write_jsonl(rejection_sampling_rows, [_rs_row()])
    _write_jsonl(preference_rows, [_preference_row()])
    _write_jsonl(generation_records, [_generation()])

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--summary",
            str(summary),
            "--review-records",
            str(review_records),
            "--revisions",
            str(revisions),
            "--rejection-sampling-rows",
            str(rejection_sampling_rows),
            "--preference-rows",
            str(preference_rows),
            "--generation-records",
            str(generation_records),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert (
        "accepted author, human, or blind-review evidence is required before Stage 5E"
        in payload["errors"]
    )
    assert (
        "accepted author, human, or blind-review evidence is required before Stage 5E"
        in result.stderr
    )
