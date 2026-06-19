from __future__ import annotations

import subprocess
import sys

from small_model_train.io_utils import read_jsonl, write_jsonl
from small_model_train.stage3_data_readiness import (
    build_stage3_summary,
    decide_stage3_status,
    render_stage3_readiness_report,
)


def _card(card_id: str = "train_1", **overrides: object) -> dict:
    card = {
        "id": card_id,
        "style_contract": "契约",
        "previous_summary": "前情摘要",
        "chapter_goal": "推进冲突",
        "chapter_structure": [{"step": 1, "name": "开场", "goal": "抬高风险", "estimated_chars": "300"}],
        "character_states": [{"name": "林默", "state": "警惕", "speech_style": "短句"}],
        "must_include": ["旧钥匙"],
        "must_not_include": ["提前揭露真相"],
        "ending_hook": "门外传来脚步声",
        "target_word_count": "2000-2500中文汉字",
        "source_text": "原文只用于输出，不进入提示词。",
    }
    card.update(overrides)
    return card


def _write_ready_artifacts(tmp_path):
    raw_dir = tmp_path / "data_raw" / "novels"
    raw_dir.mkdir(parents=True)
    (raw_dir / "book.txt").write_text("第一章\n正文", encoding="utf-8")

    chapters_raw = tmp_path / "chapters_raw.jsonl"
    chapters = tmp_path / "chapters.jsonl"
    chapters_split = tmp_path / "chapters_split.jsonl"
    chapter_cards = tmp_path / "chapter_cards.jsonl"
    eval_cards = tmp_path / "eval_cards.jsonl"
    sft_dataset = tmp_path / "sft.jsonl"

    chapter_rows = [
        {"id": "train_1", "text": "这是训练章节正文。", "quality_tag": "A"},
        {"id": "eval_1", "text": "这是评估章节正文。", "quality_tag": "B"},
    ]
    write_jsonl(chapters_raw, chapter_rows)
    write_jsonl(chapters, chapter_rows)
    write_jsonl(
        chapters_split,
        [
            {"id": "train_1", "text": "这是训练章节正文。", "quality_tag": "A", "split": "train"},
            {"id": "eval_1", "text": "这是评估章节正文。", "quality_tag": "B", "split": "eval"},
        ],
    )
    write_jsonl(chapter_cards, [_card("train_1"), _card("eval_1")])
    write_jsonl(eval_cards, [_card("eval_1")])
    write_jsonl(
        sft_dataset,
        [
            {"instruction": "写正文", "input": "卡片1", "output": "正文1"},
            {"instruction": "写正文", "input": "卡片2", "output": "正文2"},
        ],
    )
    return raw_dir, chapters_raw, chapters, chapters_split, chapter_cards, eval_cards, sft_dataset


def _summary(tmp_path, **overrides):
    paths = _write_ready_artifacts(tmp_path)
    options = {
        "smoke_dry_run": {"exit_code": 0, "command": "python scripts/run_sft_smoke.py --dry-run", "stderr": ""},
        "min_trainable_sft": 2,
        "min_eval_cards": 1,
        "preferred_eval_cards": 1,
    }
    options.update(overrides)
    return build_stage3_summary(*paths, **options)


def test_ready_summary_with_synthetic_data(tmp_path):
    summary = _summary(tmp_path)

    assert summary["decision"] == "ready_for_stage4_smoke_training"
    assert summary["raw_text_file_count"] == 1
    assert summary["chapters_raw_count"] == 2
    assert summary["chapter_count"] == 2
    assert summary["split_count"] == 2
    assert summary["train_count"] == 1
    assert summary["eval_split_count"] == 1
    assert summary["quality_tag_counts"] == {"A": 1, "B": 1}
    assert summary["chapter_length"] == {"min": 8, "max": 8, "avg": 8}
    assert summary["chapter_card_count"] == 2
    assert summary["eval_card_count"] == 1
    assert summary["sft_row_count"] == 2
    assert summary["blockers"] == []
    assert summary["warnings"] == ["SFT dataset has 2 rows, below preferred 100"]


def test_missing_raw_text_blocks_summary(tmp_path):
    raw_dir, *paths = _write_ready_artifacts(tmp_path)
    (raw_dir / "book.txt").unlink()

    summary = build_stage3_summary(
        raw_dir,
        *paths,
        smoke_dry_run={"exit_code": 0, "command": "dry-run", "stderr": ""},
        min_trainable_sft=2,
        min_eval_cards=1,
        preferred_eval_cards=1,
    )

    assert summary["decision"] == "blocked_missing_raw_text"
    assert any("data_raw/novels has no .txt or .md files" in blocker for blocker in summary["blockers"])


def test_missing_required_card_field_blocks_with_exact_issue(tmp_path):
    raw_dir, chapters_raw, chapters, chapters_split, chapter_cards, eval_cards, sft_dataset = _write_ready_artifacts(tmp_path)
    broken_card = _card("train_1")
    del broken_card["chapter_goal"]
    write_jsonl(chapter_cards, [broken_card, _card("eval_1")])

    summary = build_stage3_summary(
        raw_dir,
        chapters_raw,
        chapters,
        chapters_split,
        chapter_cards,
        eval_cards,
        sft_dataset,
        smoke_dry_run={"exit_code": 0, "command": "dry-run", "stderr": ""},
        min_trainable_sft=2,
        min_eval_cards=1,
        preferred_eval_cards=1,
    )

    assert summary["decision"] == "blocked_missing_chapter_cards"
    assert summary["card_issues"]["missing_required_fields"] == [
        {"id": "train_1", "missing_fields": ["chapter_goal"]}
    ]


def test_source_text_leakage_via_previous_summary_blocks(tmp_path):
    raw_dir, chapters_raw, chapters, chapters_split, chapter_cards, eval_cards, sft_dataset = _write_ready_artifacts(tmp_path)
    write_jsonl(
        chapter_cards,
        [
            _card(
                "train_1",
                previous_summary="上一章直接复述：这是一段非常独特的原文句子，必须被抓住。",
                source_text="这是一段非常独特的原文句子，不能进入提示词。",
            ),
            _card("eval_1"),
        ],
    )

    summary = build_stage3_summary(
        raw_dir,
        chapters_raw,
        chapters,
        chapters_split,
        chapter_cards,
        eval_cards,
        sft_dataset,
        smoke_dry_run={"exit_code": 0, "command": "dry-run", "stderr": ""},
        min_trainable_sft=2,
        min_eval_cards=1,
        preferred_eval_cards=1,
    )

    assert summary["decision"] == "blocked_source_leakage"
    assert any("train_1:" in error for error in summary["card_issues"]["source_leakage_errors"])


def test_malformed_card_field_type_blocks_without_raising(tmp_path):
    raw_dir, chapters_raw, chapters, chapters_split, chapter_cards, eval_cards, sft_dataset = _write_ready_artifacts(tmp_path)
    write_jsonl(chapter_cards, [_card("train_1", chapter_structure="not-a-list"), _card("eval_1")])

    summary = build_stage3_summary(
        raw_dir,
        chapters_raw,
        chapters,
        chapters_split,
        chapter_cards,
        eval_cards,
        sft_dataset,
        smoke_dry_run={"exit_code": 0, "command": "dry-run", "stderr": ""},
        min_trainable_sft=2,
        min_eval_cards=1,
        preferred_eval_cards=1,
    )
    report = render_stage3_readiness_report(summary)

    assert summary["decision"] == "blocked_missing_chapter_cards"
    assert summary["card_issues"]["source_leakage_errors"] == []
    assert len(summary["card_issues"]["render_errors"]) == 1
    assert "train_1:" in summary["card_issues"]["render_errors"][0]
    assert "render_errors" in report


def test_scalar_prompt_list_field_blocks_as_schema_error(tmp_path):
    raw_dir, chapters_raw, chapters, chapters_split, chapter_cards, eval_cards, sft_dataset = _write_ready_artifacts(tmp_path)
    write_jsonl(chapter_cards, [_card("train_1", must_include="旧仓库"), _card("eval_1")])

    summary = build_stage3_summary(
        raw_dir,
        chapters_raw,
        chapters,
        chapters_split,
        chapter_cards,
        eval_cards,
        sft_dataset,
        smoke_dry_run={"exit_code": 0, "command": "dry-run", "stderr": ""},
        min_trainable_sft=2,
        min_eval_cards=1,
        preferred_eval_cards=1,
    )
    report = render_stage3_readiness_report(summary)

    assert summary["decision"] == "blocked_missing_chapter_cards"
    assert summary["card_issues"]["source_leakage_errors"] == []
    assert summary["card_issues"]["schema_errors"] == [
        "train_1: must_include must be a list of strings"
    ]
    assert "schema_errors" in report


def test_non_string_prompt_list_item_blocks_as_schema_error(tmp_path):
    raw_dir, chapters_raw, chapters, chapters_split, chapter_cards, eval_cards, sft_dataset = _write_ready_artifacts(tmp_path)
    write_jsonl(chapter_cards, [_card("train_1", must_not_include=[123]), _card("eval_1")])

    summary = build_stage3_summary(
        raw_dir,
        chapters_raw,
        chapters,
        chapters_split,
        chapter_cards,
        eval_cards,
        sft_dataset,
        smoke_dry_run={"exit_code": 0, "command": "dry-run", "stderr": ""},
        min_trainable_sft=2,
        min_eval_cards=1,
        preferred_eval_cards=1,
    )

    assert summary["decision"] == "blocked_missing_chapter_cards"
    assert summary["card_issues"]["source_leakage_errors"] == []
    assert summary["card_issues"]["schema_errors"] == [
        "train_1: must_not_include must be a list of strings"
    ]


def test_readiness_blocks_malformed_chapter_structure(tmp_path):
    raw_dir, chapters_raw, chapters, chapters_split, chapter_cards, eval_cards, sft_dataset = _write_ready_artifacts(tmp_path)
    rows = read_jsonl(chapter_cards)
    rows[0]["chapter_structure"] = [{"beat": "承接", "goal": "推进", "estimated_chars": "300"}]
    write_jsonl(chapter_cards, rows)

    summary = build_stage3_summary(
        raw_dir=raw_dir,
        chapters_raw=chapters_raw,
        chapters=chapters,
        chapters_split=chapters_split,
        chapter_cards=chapter_cards,
        eval_cards=eval_cards,
        sft_dataset=sft_dataset,
        smoke_dry_run={"exit_code": 0, "command": "dry-run", "stderr": ""},
        min_trainable_sft=2,
        min_eval_cards=1,
        preferred_eval_cards=1,
    )

    assert summary["decision"] == "blocked_missing_chapter_cards"
    assert summary["card_issues"]["schema_errors"]
    assert any("chapter_structure[0].step" in error for error in summary["card_issues"]["schema_errors"])
    assert summary["card_issues"]["source_leakage_errors"] == []


def test_readiness_classifies_structure_render_value_errors_as_render_errors(tmp_path):
    raw_dir, chapters_raw, chapters, chapters_split, chapter_cards, eval_cards, sft_dataset = _write_ready_artifacts(tmp_path)
    write_jsonl(
        chapter_cards,
        [
            _card(
                "train_1",
                chapter_structure=[
                    {"step": "1", "name": "开场", "goal": "推进", "estimated_chars": "300"}
                ],
            ),
            _card("eval_1"),
        ],
    )

    summary = build_stage3_summary(
        raw_dir=raw_dir,
        chapters_raw=chapters_raw,
        chapters=chapters,
        chapters_split=chapters_split,
        chapter_cards=chapter_cards,
        eval_cards=eval_cards,
        sft_dataset=sft_dataset,
        smoke_dry_run={"exit_code": 0, "command": "dry-run", "stderr": ""},
        min_trainable_sft=2,
        min_eval_cards=1,
        preferred_eval_cards=1,
    )

    assert summary["decision"] == "blocked_missing_chapter_cards"
    assert summary["card_issues"]["source_leakage_errors"] == []
    assert any(
        "chapter_structure[0].step must be a positive integer" in error
        for error in summary["card_issues"]["render_errors"]
    )


def test_readiness_classifies_eval_structure_render_value_errors_as_render_errors(tmp_path):
    raw_dir, chapters_raw, chapters, chapters_split, chapter_cards, eval_cards, sft_dataset = _write_ready_artifacts(tmp_path)
    write_jsonl(
        eval_cards,
        [
            _card(
                "eval_1",
                chapter_structure=[
                    {"step": "1", "name": "开场", "goal": "推进", "estimated_chars": "300"}
                ],
            )
        ],
    )

    summary = build_stage3_summary(
        raw_dir=raw_dir,
        chapters_raw=chapters_raw,
        chapters=chapters,
        chapters_split=chapters_split,
        chapter_cards=chapter_cards,
        eval_cards=eval_cards,
        sft_dataset=sft_dataset,
        smoke_dry_run={"exit_code": 0, "command": "dry-run", "stderr": ""},
        min_trainable_sft=2,
        min_eval_cards=1,
        preferred_eval_cards=1,
    )

    assert summary["decision"] == "blocked_missing_chapter_cards"
    assert summary["eval_card_issues"]["source_leakage_errors"] == []
    assert any(
        "chapter_structure[0].step must be a positive integer" in error
        for error in summary["eval_card_issues"]["render_errors"]
    )


def test_empty_prompt_lists_warn_without_blocking_readiness(tmp_path):
    raw_dir, chapters_raw, chapters, chapters_split, chapter_cards, eval_cards, sft_dataset = _write_ready_artifacts(tmp_path)
    write_jsonl(chapter_cards, [_card("train_1", must_include=[], must_not_include=[]), _card("eval_1")])

    summary = build_stage3_summary(
        raw_dir,
        chapters_raw,
        chapters,
        chapters_split,
        chapter_cards,
        eval_cards,
        sft_dataset,
        smoke_dry_run={"exit_code": 0, "command": "dry-run", "stderr": ""},
        min_trainable_sft=2,
        min_eval_cards=1,
        preferred_eval_cards=1,
    )

    assert summary["decision"] == "ready_for_stage4_smoke_training"
    assert summary["card_issues"]["schema_errors"] == []
    assert summary["card_issues"]["cards_with_empty_lists"] == [
        {"id": "train_1", "field": "must_include"},
        {"id": "train_1", "field": "must_not_include"},
    ]
    assert "some chapter cards have empty must_include or must_not_include lists" in summary["warnings"]


def test_empty_sft_dataset_blocks_as_sft_empty(tmp_path):
    raw_dir, chapters_raw, chapters, chapters_split, chapter_cards, eval_cards, sft_dataset = _write_ready_artifacts(tmp_path)
    write_jsonl(sft_dataset, [])

    summary = build_stage3_summary(
        raw_dir,
        chapters_raw,
        chapters,
        chapters_split,
        chapter_cards,
        eval_cards,
        sft_dataset,
        smoke_dry_run={"exit_code": 0, "command": "dry-run", "stderr": ""},
        min_trainable_sft=2,
        min_eval_cards=1,
        preferred_eval_cards=1,
    )

    assert summary["decision"] == "blocked_sft_empty"


def test_empty_eval_cards_blocks_as_eval_missing(tmp_path):
    raw_dir, chapters_raw, chapters, chapters_split, chapter_cards, eval_cards, sft_dataset = _write_ready_artifacts(tmp_path)
    write_jsonl(eval_cards, [])

    summary = build_stage3_summary(
        raw_dir,
        chapters_raw,
        chapters,
        chapters_split,
        chapter_cards,
        eval_cards,
        sft_dataset,
        smoke_dry_run={"exit_code": 0, "command": "dry-run", "stderr": ""},
        min_trainable_sft=2,
        min_eval_cards=1,
        preferred_eval_cards=1,
    )

    assert summary["decision"] == "blocked_eval_missing"


def test_decide_stage3_status_derives_missing_eval_split_rows():
    split_rows = [{"id": "train_1", "split": "train", "quality_tag": "A"}]
    train_rows = [split_rows[0]]

    decision = decide_stage3_status(
        raw_files=1,
        split_rows=split_rows,
        train_rows=train_rows,
        card_rows=[_card("train_1")],
        card_issues={
            "missing_required_fields": [],
            "source_leakage_errors": [],
            "render_errors": [],
            "schema_errors": [],
            "non_object_rows": [],
            "unmatched_chapter_ids": [],
        },
        sft_rows=[{"input": "x"}],
        eval_rows=[_card("eval_1")],
        dry_run={"exit_code": 0},
        min_trainable_sft=1,
        min_eval_cards=1,
    )

    assert decision == "blocked_insufficient_chapters"


def test_missing_eval_split_rows_blocks_even_with_eval_cards(tmp_path):
    raw_dir, chapters_raw, chapters, chapters_split, chapter_cards, eval_cards, sft_dataset = _write_ready_artifacts(tmp_path)
    write_jsonl(
        chapters_split,
        [{"id": "train_1", "text": "这是训练章节正文。", "quality_tag": "A", "split": "train"}],
    )

    summary = build_stage3_summary(
        raw_dir,
        chapters_raw,
        chapters,
        chapters_split,
        chapter_cards,
        eval_cards,
        sft_dataset,
        smoke_dry_run={"exit_code": 0, "command": "dry-run", "stderr": ""},
        min_trainable_sft=2,
        min_eval_cards=1,
        preferred_eval_cards=1,
    )

    assert summary["decision"] == "blocked_insufficient_chapters"
    assert "chapters_split has no eval rows" in summary["blockers"]


def test_non_object_chapter_card_row_blocks_without_raising(tmp_path):
    raw_dir, chapters_raw, chapters, chapters_split, chapter_cards, eval_cards, sft_dataset = _write_ready_artifacts(tmp_path)
    write_jsonl(chapter_cards, [[], _card("train_1")])

    summary = build_stage3_summary(
        raw_dir,
        chapters_raw,
        chapters,
        chapters_split,
        chapter_cards,
        eval_cards,
        sft_dataset,
        smoke_dry_run={"exit_code": 0, "command": "dry-run", "stderr": ""},
        min_trainable_sft=2,
        min_eval_cards=1,
        preferred_eval_cards=1,
    )
    report = render_stage3_readiness_report(summary)

    assert summary["decision"] == "blocked_missing_chapter_cards"
    assert summary["card_issues"]["non_object_rows"] == ["row_1: card row must be a JSON object"]
    assert "non_object_rows" in report


def test_eval_card_source_leakage_blocks_separately(tmp_path):
    raw_dir, chapters_raw, chapters, chapters_split, chapter_cards, eval_cards, sft_dataset = _write_ready_artifacts(tmp_path)
    write_jsonl(
        eval_cards,
        [
            _card(
                "eval_1",
                previous_summary="评估卡泄漏：这是一段非常独特的评估原文，应该阻断。",
                source_text="这是一段非常独特的评估原文，不能进入提示词。",
            )
        ],
    )

    summary = build_stage3_summary(
        raw_dir,
        chapters_raw,
        chapters,
        chapters_split,
        chapter_cards,
        eval_cards,
        sft_dataset,
        smoke_dry_run={"exit_code": 0, "command": "dry-run", "stderr": ""},
        min_trainable_sft=2,
        min_eval_cards=1,
        preferred_eval_cards=1,
    )
    report = render_stage3_readiness_report(summary)

    assert summary["decision"] == "blocked_source_leakage"
    assert summary["card_issues"]["source_leakage_errors"] == []
    assert any("eval_1:" in error for error in summary["eval_card_issues"]["source_leakage_errors"])
    assert "Eval Card Issues" in report


def test_unmatched_trainable_chapter_card_id_blocks(tmp_path):
    raw_dir, chapters_raw, chapters, chapters_split, chapter_cards, eval_cards, sft_dataset = _write_ready_artifacts(tmp_path)
    write_jsonl(chapter_cards, [_card("missing_trainable"), _card("eval_1")])

    summary = build_stage3_summary(
        raw_dir,
        chapters_raw,
        chapters,
        chapters_split,
        chapter_cards,
        eval_cards,
        sft_dataset,
        smoke_dry_run={"exit_code": 0, "command": "dry-run", "stderr": ""},
        min_trainable_sft=2,
        min_eval_cards=1,
        preferred_eval_cards=1,
    )
    report = render_stage3_readiness_report(summary)

    assert summary["decision"] == "blocked_missing_chapter_cards"
    assert summary["card_issues"]["unmatched_chapter_ids"] == ["missing_trainable"]
    assert "missing_trainable" in report


def test_render_report_contains_required_readiness_lines(tmp_path):
    report = render_stage3_readiness_report(_summary(tmp_path))

    assert "# Stage 3 Data Readiness Report" in report
    assert "ready_for_stage4_smoke_training" in report
    assert "- SFT 样本数：2" in report


def test_cli_without_smoke_dry_run_writes_blocked_report(tmp_path):
    raw_dir, chapters_raw, chapters, chapters_split, chapter_cards, eval_cards, sft_dataset = _write_ready_artifacts(tmp_path)
    report = tmp_path / "reports" / "stage3_data_readiness_report.md"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_stage3_data_readiness.py",
            "--raw-dir",
            str(raw_dir),
            "--chapters-raw",
            str(chapters_raw),
            "--chapters",
            str(chapters),
            "--chapters-split",
            str(chapters_split),
            "--chapter-cards",
            str(chapter_cards),
            "--eval-cards",
            str(eval_cards),
            "--sft-dataset",
            str(sft_dataset),
            "--report",
            str(report),
            "--min-trainable-sft",
            "2",
            "--min-eval-cards",
            "1",
            "--preferred-eval-cards",
            "1",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert report.exists()
    report_text = report.read_text(encoding="utf-8")
    assert "blocked_stage2_dry_run_failed" in report_text
    assert "smoke dry-run has not been run" in report_text
