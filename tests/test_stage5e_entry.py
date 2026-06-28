from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_stage5e_entry.py"
MODEL_OUTPUT = "林默把合同推过去，对方沉默。"
REVISED_OUTPUT = "林默没有解释，只把合同推到桌面。岳家的人第一次停住。"
PROMPT_SHA256 = "b" * 64
STYLE_CONTRACT_SHA256 = "a" * 64


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
        "reviewed_output_chars": 12,
        "defects": {
            "total_defects": 1,
            "by_label": {"generic_phrase": 1},
            "by_severity": {"minor": 1},
        },
        "defect_density_per_10k_chars": 833.3333,
        "revision_records": 1,
        "accepted_revisions": 1,
        "author_acceptance_rate": 1.0,
        "changed_char_delta": 11,
        "edit_burden": {"mean_changed_chars": 11.0, "median_changed_chars": 11.0},
        "rejection_sampling_sft_rows": 1,
        "preference_candidate_rows": 1,
        "candidate_split_counts": {"train": 1},
        "non_train_rejection_sampling_rows": [],
        "review_source_counts": {"author": 1},
        "plan_execution_regressions": 0,
        "boundary": "candidate_data_only_no_preference_training",
    }
    summary.update(overrides)
    return summary


def _review(**overrides: object) -> dict:
    from small_model_train.schemas.chapter_execution_card import text_sha256

    evidence_text = "对方沉默"
    evidence_start = MODEL_OUTPUT.find(evidence_text)
    record = {
        "record_id": "review-1",
        "schema_version": 1,
        "card_id": "card-c1-v1",
        "chapter_id": "c1",
        "style_contract_id": "contract-v1",
        "style_contract_sha256": STYLE_CONTRACT_SHA256,
        "source_output_id": "card-c1-v1",
        "raw_output_sha256": text_sha256(MODEL_OUTPUT),
        "review_source": "author",
        "reviewer": "author",
        "reviewed_at": "2026-06-27T01:00:00Z",
        "defects": [
            {
                "label": "generic_phrase",
                "severity": "minor",
                "evidence_text": evidence_text,
                "evidence_start": evidence_start,
                "evidence_end": evidence_start + len(evidence_text),
                "suggested_fix": "换成具体动作反应。",
            }
        ],
        "overall_acceptance": "accepted",
        "notes": "同剧情人工复核通过。",
    }
    record.update(overrides)
    return record


def _revision(**overrides: object) -> dict:
    from small_model_train.schemas.chapter_execution_card import text_sha256

    revision = {
        "revision_id": "rev-c1-001",
        "schema_version": 1,
        "revision_status": "accepted",
        "card_id": "card-c1-v1",
        "chapter_id": "c1",
        "style_contract_id": "contract-v1",
        "style_contract_sha256": STYLE_CONTRACT_SHA256,
        "prompt_sha256": PROMPT_SHA256,
        "raw_output_sha256": text_sha256(MODEL_OUTPUT),
        "model_output": MODEL_OUTPUT,
        "revised_output": REVISED_OUTPUT,
        "revision_author": "author",
        "revised_at": "2026-06-27T01:00:00Z",
        "edit_summary": "把解释改成动作和反应。",
        "defect_record_ids": ["review-1"],
        "acceptance_reason": "同剧情更像作者正文。",
    }
    revision.update(overrides)
    return revision


def _generation(**overrides: object) -> dict:
    from small_model_train.stage2_inference import build_generation_row

    generation = build_generation_row(
        "card-c1-v1",
        MODEL_OUTPUT,
        "sft_v1",
        {"seed": 7},
        prompt_sha256=PROMPT_SHA256,
    )
    generation.update(overrides)
    return generation


def _rs_row(**overrides: object) -> dict:
    row = {
        "instruction": "续写下一章。",
        "input": "章节执行卡内容",
        "output": REVISED_OUTPUT,
        "revision_id": "rev-c1-001",
        "card_id": "card-c1-v1",
        "chapter_id": "c1",
        "style_contract_sha256": STYLE_CONTRACT_SHA256,
        "raw_output_sha256": _revision()["raw_output_sha256"],
        "source_split": "train",
    }
    row.update(overrides)
    return row


def _preference_row(**overrides: object) -> dict:
    row = {
        "id": "rev-c1-001",
        "prompt_sha256": PROMPT_SHA256,
        "card_id": "card-c1-v1",
        "chapter_id": "c1",
        "style_contract_sha256": STYLE_CONTRACT_SHA256,
        "chosen": REVISED_OUTPUT,
        "rejected": MODEL_OUTPUT,
        "defect_record_ids": ["review-1"],
        "defect_labels": ["generic_phrase"],
        "reject_type": "generic_phrase",
        "source": "stage5d_same_plot_revision",
    }
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
        "accepted revision lacks same-card, same-style when available, same-prompt, "
        "same-raw-output generation record with integer seed provenance: rev-c1-001"
        in result["errors"]
    )


def test_stage5e_entry_gate_rejects_forged_summary_metrics():
    from small_model_train.review.stage5e_entry import check_stage5e_entry

    result = check_stage5e_entry(
        summary=_summary(
            reviewed_output_chars=999,
            defect_density_per_10k_chars=1.23,
            plan_execution_regressions=99,
            candidate_split_counts={"sealed": 1},
            review_source_counts={"robot": 1},
        ),
        review_records=[_review()],
        revision_records=[_revision()],
        rejection_sampling_rows=[_rs_row()],
        preference_rows=[_preference_row()],
        generation_records=[_generation()],
    )

    assert result["passed"] is False
    assert (
        "reviewed_output_chars does not match recomputed Stage 5D summary: summary=999 actual=12"
        in result["errors"]
    )
    assert (
        "defect_density_per_10k_chars does not match recomputed Stage 5D summary: "
        "summary=1.23 actual=833.3333"
        in result["errors"]
    )
    assert (
        "plan_execution_regressions does not match recomputed Stage 5D summary: "
        "summary=99 actual=0"
        in result["errors"]
    )
    assert (
        "candidate_split_counts does not match recomputed Stage 5D summary: "
        "summary={'sealed': 1} actual={'train': 1}"
        in result["errors"]
    )
    assert (
        "review_source_counts does not match recomputed Stage 5D summary: "
        "summary={'robot': 1} actual={'author': 1}"
        in result["errors"]
    )


def test_stage5e_entry_gate_requires_human_review_for_each_accepted_revision():
    from small_model_train.review.stage5d_report import build_stage5d_summary
    from small_model_train.review.stage5e_entry import check_stage5e_entry
    from small_model_train.schemas.chapter_execution_card import text_sha256

    second_model_output = "许昭把名单压住，窗外雨声停了一瞬。"
    second_revised_output = "许昭没有抬头，只把名单压在灯下。窗外雨声停了一瞬。"
    second_evidence_text = "雨声停了一瞬"
    second_evidence_start = second_model_output.find(second_evidence_text)
    second_raw_output_sha256 = text_sha256(second_model_output)
    second_review = _review(
        record_id="review-2",
        card_id="card-c2-v1",
        chapter_id="c2",
        source_output_id="card-c2-v1",
        raw_output_sha256=second_raw_output_sha256,
        review_source="deterministic",
        reviewer="rules",
        defects=[
            {
                "label": "payoff_blur",
                "severity": "minor",
                "evidence_text": second_evidence_text,
                "evidence_start": second_evidence_start,
                "evidence_end": second_evidence_start + len(second_evidence_text),
                "suggested_fix": "把情绪落到动作上。",
            }
        ],
        overall_acceptance="accepted",
    )
    second_revision = _revision(
        revision_id="rev-c2-001",
        card_id="card-c2-v1",
        chapter_id="c2",
        raw_output_sha256=second_raw_output_sha256,
        model_output=second_model_output,
        revised_output=second_revised_output,
        defect_record_ids=["review-2"],
    )
    second_rs_row = _rs_row(
        output=second_revised_output,
        revision_id="rev-c2-001",
        card_id="card-c2-v1",
        chapter_id="c2",
        raw_output_sha256=second_raw_output_sha256,
    )
    second_preference_row = _preference_row(
        id="rev-c2-001",
        card_id="card-c2-v1",
        chapter_id="c2",
        chosen=second_revised_output,
        rejected=second_model_output,
        defect_record_ids=["review-2"],
        defect_labels=["payoff_blur"],
        reject_type="payoff_blur",
    )
    second_generation = _generation(
        id="card-c2-v1",
        output=second_model_output,
        raw_output=second_model_output,
    )
    review_records = [_review(), second_review]
    revision_records = [_revision(), second_revision]
    rejection_sampling_rows = [_rs_row(), second_rs_row]
    preference_rows = [_preference_row(), second_preference_row]
    raw_outputs = {
        "card-c1-v1": MODEL_OUTPUT,
        "card-c2-v1": second_model_output,
    }

    result = check_stage5e_entry(
        summary=build_stage5d_summary(
            review_records,
            revision_records,
            rejection_sampling_rows,
            preference_rows,
            raw_outputs=raw_outputs,
        ),
        review_records=review_records,
        revision_records=revision_records,
        rejection_sampling_rows=rejection_sampling_rows,
        preference_rows=preference_rows,
        generation_records=[_generation(), second_generation],
    )

    assert result["passed"] is False
    assert (
        "accepted revision lacks author, human, or blind-review evidence: rev-c2-001"
        in result["errors"]
    )
    assert (
        "accepted author, human, or blind-review evidence is required before Stage 5E"
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
        "accepted revision missing generation link field: rev-c1-001 prompt_sha256"
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
    assert "generation record missing prompt_sha256: gen-seed-only" in result["errors"]
    assert "generation record missing raw output: gen-seed-only" in result["errors"]


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
    assert "preference row requires non-empty defect_labels: rev-c1-001" in result["errors"]


def test_stage5e_entry_gate_rejects_minimal_rejection_sampling_row():
    from small_model_train.review.stage5e_entry import check_stage5e_entry

    result = check_stage5e_entry(
        summary=_summary(),
        review_records=[_review()],
        revision_records=[_revision()],
        rejection_sampling_rows=[{"source_split": "train"}],
        preference_rows=[_preference_row()],
        generation_records=[_generation()],
    )

    assert result["passed"] is False
    assert "rejection-sampling row missing required field: row-1 revision_id" in result["errors"]
    assert "rejection-sampling row not linked to accepted revision: row-1" in result["errors"]


def test_stage5e_entry_gate_rejects_minimal_preference_row():
    from small_model_train.review.stage5e_entry import check_stage5e_entry

    result = check_stage5e_entry(
        summary=_summary(),
        review_records=[_review()],
        revision_records=[_revision()],
        rejection_sampling_rows=[_rs_row()],
        preference_rows=[{"defect_labels": ["generic_phrase"]}],
        generation_records=[_generation()],
    )

    assert result["passed"] is False
    assert "preference row missing required field: row-1 id" in result["errors"]
    assert "preference row not linked to accepted revision: row-1" in result["errors"]


def test_stage5e_entry_gate_rejects_generation_raw_hash_mismatch():
    from small_model_train.review.stage5e_entry import check_stage5e_entry

    result = check_stage5e_entry(
        summary=_summary(),
        review_records=[_review()],
        revision_records=[_revision()],
        rejection_sampling_rows=[_rs_row()],
        preference_rows=[_preference_row()],
        generation_records=[_generation(raw_output_sha256="c" * 64)],
    )

    assert result["passed"] is False
    assert "generation record raw_output_sha256 mismatch: card-c1-v1" in result["errors"]


def test_stage5e_entry_gate_rejects_invalid_revision_schema():
    from small_model_train.review.stage5e_entry import check_stage5e_entry

    result = check_stage5e_entry(
        summary=_summary(accepted_revisions=0),
        review_records=[_review()],
        revision_records=[_revision(raw_output_sha256="not-a-sha")],
        rejection_sampling_rows=[_rs_row()],
        preference_rows=[_preference_row()],
        generation_records=[_generation()],
    )

    assert result["passed"] is False
    assert (
        "revision record invalid: rev-c1-001 raw_output_sha256 must be a 64-character lowercase hex string"
        in result["errors"]
    )


def test_stage5e_entry_gate_rejects_rejection_sampling_row_not_linked_to_revision():
    from small_model_train.review.stage5e_entry import check_stage5e_entry

    result = check_stage5e_entry(
        summary=_summary(),
        review_records=[_review()],
        revision_records=[_revision()],
        rejection_sampling_rows=[_rs_row(revision_id="rev-other")],
        preference_rows=[_preference_row()],
        generation_records=[_generation()],
    )

    assert result["passed"] is False
    assert "rejection-sampling row not linked to accepted revision: rev-other" in result["errors"]


def test_stage5e_entry_gate_rejects_rejection_sampling_field_mismatch():
    from small_model_train.review.stage5e_entry import check_stage5e_entry

    result = check_stage5e_entry(
        summary=_summary(),
        review_records=[_review()],
        revision_records=[_revision()],
        rejection_sampling_rows=[_rs_row(output="不是修订正文")],
        preference_rows=[_preference_row()],
        generation_records=[_generation()],
    )

    assert result["passed"] is False
    assert (
        "rejection-sampling row mismatch accepted revision: rev-c1-001 output"
        in result["errors"]
    )


def test_stage5e_entry_gate_rejects_preference_row_not_linked_to_revision():
    from small_model_train.review.stage5e_entry import check_stage5e_entry

    result = check_stage5e_entry(
        summary=_summary(),
        review_records=[_review()],
        revision_records=[_revision()],
        rejection_sampling_rows=[_rs_row()],
        preference_rows=[_preference_row(id="rev-other")],
        generation_records=[_generation()],
    )

    assert result["passed"] is False
    assert "preference row not linked to accepted revision: rev-other" in result["errors"]


def test_stage5e_entry_gate_rejects_preference_field_mismatch():
    from small_model_train.review.stage5e_entry import check_stage5e_entry

    result = check_stage5e_entry(
        summary=_summary(),
        review_records=[_review()],
        revision_records=[_revision()],
        rejection_sampling_rows=[_rs_row()],
        preference_rows=[_preference_row(chosen="不是修订正文")],
        generation_records=[_generation()],
    )

    assert result["passed"] is False
    assert "preference row mismatch accepted revision: rev-c1-001 chosen" in result["errors"]


def test_stage5e_entry_gate_rejects_forged_preference_defect_labels():
    from small_model_train.review.stage5e_entry import check_stage5e_entry

    result = check_stage5e_entry(
        summary=_summary(),
        review_records=[_review()],
        revision_records=[_revision()],
        rejection_sampling_rows=[_rs_row()],
        preference_rows=[
            _preference_row(defect_labels=["payoff_blur"], reject_type="payoff_blur")
        ],
        generation_records=[_generation()],
    )

    assert result["passed"] is False
    assert (
        "preference row defect_labels do not match referenced review defects: rev-c1-001"
        in result["errors"]
    )


def test_stage5e_entry_gate_rejects_unrelated_accepted_review_evidence():
    from small_model_train.review.stage5e_entry import check_stage5e_entry

    result = check_stage5e_entry(
        summary=_summary(),
        review_records=[_review(record_id="review-other")],
        revision_records=[_revision(defect_record_ids=["review-1"])],
        rejection_sampling_rows=[_rs_row()],
        preference_rows=[_preference_row()],
        generation_records=[_generation()],
    )

    assert result["passed"] is False
    assert "accepted revision referenced review missing: rev-c1-001 review-1" in result["errors"]
    assert (
        "accepted author, human, or blind-review evidence is required before Stage 5E"
        in result["errors"]
    )


def test_stage5e_entry_gate_accepts_generation_output_id_separate_from_card_id():
    from small_model_train.review.stage5e_entry import check_stage5e_entry

    result = check_stage5e_entry(
        summary=_summary(),
        review_records=[_review(source_output_id="gen-1")],
        revision_records=[_revision()],
        rejection_sampling_rows=[_rs_row()],
        preference_rows=[_preference_row()],
        generation_records=[_generation(id="gen-1", card_id="card-c1-v1")],
    )

    assert result["passed"] is True
    assert result["errors"] == []


def test_stage5e_entry_gate_rejects_duplicate_generation_output_id():
    from small_model_train.review.stage5e_entry import check_stage5e_entry

    result = check_stage5e_entry(
        summary=_summary(),
        review_records=[_review(source_output_id="gen-1")],
        revision_records=[_revision()],
        rejection_sampling_rows=[_rs_row()],
        preference_rows=[_preference_row()],
        generation_records=[
            _generation(id="gen-1", card_id="card-c1-v1"),
            _generation(id="gen-1", card_id="card-c1-v1"),
        ],
    )

    assert result["passed"] is False
    assert "duplicate generation output id: gen-1" in result["errors"]


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
