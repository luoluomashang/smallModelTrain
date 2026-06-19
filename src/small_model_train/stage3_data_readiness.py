"""Stage 3 data readiness checks before real GPU training."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from small_model_train.io_utils import read_jsonl
from small_model_train.sft_builder import render_sft_input
from small_model_train.text_utils import count_chinese_chars


REQUIRED_CARD_FIELDS = [
    "chapter_goal",
    "chapter_structure",
    "character_states",
    "must_include",
    "must_not_include",
    "target_word_count",
]


def _card_id(card: dict) -> str:
    return str(card.get("id") or "<missing id>")


def _count_raw_text_files(raw_dir: str | Path) -> int:
    root = Path(raw_dir)
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".txt", ".md"})


def _validate_card_schema(card: dict, row_label: str | None = None) -> list[str]:
    errors = []
    card_id = row_label or _card_id(card)

    for field in ("chapter_structure", "character_states"):
        value = card.get(field)
        if field not in card:
            continue
        if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
            errors.append(f"{card_id}: {field} must be a list of dictionaries")

    for field in ("must_include", "must_not_include"):
        value = card.get(field)
        if field not in card:
            continue
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            errors.append(f"{card_id}: {field} must be a list of strings")

    return errors


def _empty_card_issues() -> dict:
    return {
        "missing_required_fields": [],
        "source_leakage_errors": [],
        "render_errors": [],
        "schema_errors": [],
        "non_object_rows": [],
        "unmatched_chapter_ids": [],
        "cards_with_empty_lists": [],
    }


def _inspect_cards(cards: list, require_required_fields: bool) -> dict:
    issues = _empty_card_issues()

    for index, card in enumerate(cards, start=1):
        if not isinstance(card, dict):
            issues["non_object_rows"].append(f"row_{index}: card row must be a JSON object")
            continue

        if require_required_fields:
            missing_fields = [field for field in REQUIRED_CARD_FIELDS if field not in card]
            if missing_fields:
                issues["missing_required_fields"].append({"id": _card_id(card), "missing_fields": missing_fields})

        issues["schema_errors"].extend(_validate_card_schema(card))

        if require_required_fields:
            for field in ("must_include", "must_not_include"):
                if field in card and card[field] == []:
                    issues["cards_with_empty_lists"].append({"id": _card_id(card), "field": field})

        try:
            render_sft_input(card)
        except ValueError as exc:
            issues["source_leakage_errors"].append(f"{_card_id(card)}: {exc}")
        except (AttributeError, TypeError) as exc:
            issues["render_errors"].append(f"{_card_id(card)}: {type(exc).__name__}: {exc}")

    return issues


def inspect_chapter_cards(cards: list) -> dict:
    return _inspect_cards(cards, require_required_fields=True)


def inspect_eval_cards(cards: list) -> dict:
    issues = _inspect_cards(cards, require_required_fields=False)
    issues.pop("missing_required_fields")
    issues.pop("unmatched_chapter_ids")
    issues.pop("cards_with_empty_lists")
    return issues


def _with_unmatched_chapter_ids(card_issues: dict, cards: list, split_rows: list[dict]) -> dict:
    trainable_ids = {
        row.get("id")
        for row in split_rows
        if isinstance(row, dict) and row.get("split") == "train" and row.get("quality_tag") == "A"
    }
    split_ids = {row.get("id") for row in split_rows if isinstance(row, dict)}
    unmatched = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        card_id = card.get("id")
        if card_id in trainable_ids:
            continue
        if card_id not in split_ids:
            unmatched.append(str(card_id or "<missing id>"))
    card_issues["unmatched_chapter_ids"] = unmatched
    return card_issues


def _has_card_blocking_issues(card_issues: dict) -> bool:
    return bool(
        card_issues.get("missing_required_fields")
        or card_issues.get("render_errors")
        or card_issues.get("schema_errors")
        or card_issues.get("non_object_rows")
        or card_issues.get("unmatched_chapter_ids")
    )


def _has_eval_card_blocking_issues(eval_card_issues: dict | None) -> bool:
    if not eval_card_issues:
        return False
    return bool(
        eval_card_issues.get("render_errors")
        or eval_card_issues.get("schema_errors")
        or eval_card_issues.get("non_object_rows")
    )


def _has_source_leakage(card_issues: dict, eval_card_issues: dict | None = None) -> bool:
    return bool(
        card_issues.get("source_leakage_errors")
        or (eval_card_issues and eval_card_issues.get("source_leakage_errors"))
    )


def summarize_chapter_lengths(rows: list[dict]) -> dict:
    lengths = [count_chinese_chars(str(row.get("text", ""))) for row in rows]
    if not lengths:
        return {"min": 0, "max": 0, "avg": 0}
    return {
        "min": min(lengths),
        "max": max(lengths),
        "avg": round(sum(lengths) / len(lengths), 2),
    }


def decide_stage3_status(
    raw_files: int | list[Any],
    split_rows: int | list[dict],
    train_rows: int | list[dict],
    card_rows: int | list[dict],
    card_issues: dict,
    sft_rows: int | list[dict],
    eval_rows: int | list[dict],
    dry_run: dict,
    min_trainable_sft: int,
    min_eval_cards: int,
    eval_split_rows: int | list[dict] | None = None,
    eval_card_issues: dict | None = None,
) -> str:
    raw_count = len(raw_files) if not isinstance(raw_files, int) else raw_files
    split_count = len(split_rows) if not isinstance(split_rows, int) else split_rows
    train_count = len(train_rows) if not isinstance(train_rows, int) else train_rows
    card_count = len(card_rows) if not isinstance(card_rows, int) else card_rows
    sft_count = len(sft_rows) if not isinstance(sft_rows, int) else sft_rows
    eval_count = len(eval_rows) if not isinstance(eval_rows, int) else eval_rows
    if eval_split_rows is None:
        eval_split_count = (
            sum(1 for row in split_rows if isinstance(row, dict) and row.get("split") == "eval")
            if isinstance(split_rows, list)
            else 0
        )
    else:
        eval_split_count = len(eval_split_rows) if not isinstance(eval_split_rows, int) else eval_split_rows

    if raw_count == 0:
        return "blocked_missing_raw_text"
    if split_count == 0 or train_count == 0 or eval_split_count == 0 or (0 < sft_count < min_trainable_sft):
        return "blocked_insufficient_chapters"
    if card_count == 0 or _has_card_blocking_issues(card_issues) or _has_eval_card_blocking_issues(eval_card_issues):
        return "blocked_missing_chapter_cards"
    if _has_source_leakage(card_issues, eval_card_issues):
        return "blocked_source_leakage"
    if sft_count == 0:
        return "blocked_sft_empty"
    if eval_count < min_eval_cards:
        return "blocked_eval_missing"
    if dry_run.get("exit_code") != 0:
        return "blocked_stage2_dry_run_failed"
    return "ready_for_stage4_smoke_training"


def _build_blockers(
    raw_dir: str | Path,
    raw_text_file_count: int,
    split_count: int,
    train_count: int,
    eval_split_count: int,
    chapter_card_count: int,
    card_issues: dict,
    eval_card_issues: dict,
    sft_row_count: int,
    eval_card_count: int,
    smoke_dry_run: dict,
    min_trainable_sft: int,
    min_eval_cards: int,
) -> list[str]:
    blockers: list[str] = []
    raw_label = Path(raw_dir).as_posix()

    if raw_text_file_count == 0:
        blockers.append(f"{raw_label} has no .txt or .md files")
    if split_count == 0:
        blockers.append("chapters_split is missing or empty")
    if train_count == 0:
        blockers.append("chapters_split has no train rows")
    if eval_split_count == 0:
        blockers.append("chapters_split has no eval rows")
    if chapter_card_count == 0:
        blockers.append("chapter cards are missing")
    if card_issues.get("missing_required_fields"):
        blockers.append("chapter cards are missing required fields")
    if card_issues.get("non_object_rows"):
        blockers.append("chapter cards contain non-object JSONL rows")
    if card_issues.get("unmatched_chapter_ids"):
        blockers.append("chapter cards include ids that do not match split chapters")
    if card_issues.get("render_errors"):
        blockers.append("chapter cards have malformed fields that cannot render SFT inputs")
    if card_issues.get("schema_errors"):
        blockers.append("chapter cards have malformed schema fields")
    if card_issues.get("source_leakage_errors"):
        blockers.append("chapter cards contain source_text leakage in rendered SFT inputs")
    if eval_card_issues.get("non_object_rows"):
        blockers.append("eval cards contain non-object JSONL rows")
    if eval_card_issues.get("render_errors"):
        blockers.append("eval cards have malformed fields that cannot render SFT inputs")
    if eval_card_issues.get("schema_errors"):
        blockers.append("eval cards have malformed schema fields")
    if eval_card_issues.get("source_leakage_errors"):
        blockers.append("eval cards contain source_text leakage in rendered SFT inputs")
    if sft_row_count == 0:
        blockers.append("SFT dataset is empty")
    elif sft_row_count < min_trainable_sft:
        blockers.append(f"SFT dataset has {sft_row_count} rows, below minimum {min_trainable_sft}")
    if eval_card_count == 0:
        blockers.append("eval cards are missing")
    elif eval_card_count < min_eval_cards:
        blockers.append(f"eval cards has {eval_card_count} rows, below minimum {min_eval_cards}")
    if smoke_dry_run.get("exit_code") != 0:
        blockers.append(f"smoke dry-run failed with exit_code={smoke_dry_run.get('exit_code')}")

    return blockers


def _build_warnings(
    card_issues: dict,
    sft_row_count: int,
    eval_card_count: int,
    preferred_eval_cards: int,
) -> list[str]:
    warnings: list[str] = []
    if eval_card_count < preferred_eval_cards:
        warnings.append(f"eval cards has {eval_card_count} rows, below preferred {preferred_eval_cards}")
    if 0 < sft_row_count < 100:
        warnings.append(f"SFT dataset has {sft_row_count} rows, below preferred 100")
    if card_issues.get("cards_with_empty_lists"):
        warnings.append("some chapter cards have empty must_include or must_not_include lists")
    return warnings


def build_stage3_summary(
    raw_dir,
    chapters_raw,
    chapters,
    chapters_split,
    chapter_cards,
    eval_cards,
    sft_dataset,
    smoke_dry_run=None,
    min_trainable_sft=20,
    min_eval_cards=10,
    preferred_eval_cards=50,
) -> dict:
    if smoke_dry_run is None:
        smoke_dry_run = {"exit_code": None, "command": "", "stderr": "smoke dry-run has not been run"}

    chapters_raw_rows = read_jsonl(chapters_raw)
    chapter_rows = read_jsonl(chapters)
    split_rows = read_jsonl(chapters_split)
    card_rows = read_jsonl(chapter_cards)
    eval_card_rows = read_jsonl(eval_cards)
    sft_rows = read_jsonl(sft_dataset)

    train_rows = [row for row in split_rows if isinstance(row, dict) and row.get("split") == "train"]
    eval_split_rows = [row for row in split_rows if isinstance(row, dict) and row.get("split") == "eval"]
    raw_text_file_count = _count_raw_text_files(raw_dir)
    card_issues = _with_unmatched_chapter_ids(inspect_chapter_cards(card_rows), card_rows, split_rows)
    eval_card_issues = inspect_eval_cards(eval_card_rows)

    decision = decide_stage3_status(
        raw_text_file_count,
        split_rows,
        train_rows,
        card_rows,
        card_issues,
        sft_rows,
        eval_card_rows,
        smoke_dry_run,
        min_trainable_sft,
        min_eval_cards,
        eval_split_rows,
        eval_card_issues,
    )
    quality_tag_counts = dict(
        Counter(row.get("quality_tag", "<missing>") for row in split_rows if isinstance(row, dict))
    )
    blockers = _build_blockers(
        raw_dir,
        raw_text_file_count,
        len(split_rows),
        len(train_rows),
        len(eval_split_rows),
        len(card_rows),
        card_issues,
        eval_card_issues,
        len(sft_rows),
        len(eval_card_rows),
        smoke_dry_run,
        min_trainable_sft,
        min_eval_cards,
    )
    warnings = _build_warnings(card_issues, len(sft_rows), len(eval_card_rows), preferred_eval_cards)

    return {
        "decision": decision,
        "paths": {
            "raw_dir": str(raw_dir),
            "chapters_raw": str(chapters_raw),
            "chapters": str(chapters),
            "chapters_split": str(chapters_split),
            "chapter_cards": str(chapter_cards),
            "eval_cards": str(eval_cards),
            "sft_dataset": str(sft_dataset),
        },
        "raw_text_file_count": raw_text_file_count,
        "chapters_raw_count": len(chapters_raw_rows),
        "chapter_count": len(chapter_rows),
        "split_count": len(split_rows),
        "train_count": len(train_rows),
        "eval_split_count": len(eval_split_rows),
        "quality_tag_counts": quality_tag_counts,
        "chapter_length": summarize_chapter_lengths(chapter_rows),
        "chapter_card_count": len(card_rows),
        "eval_card_count": len(eval_card_rows),
        "sft_row_count": len(sft_rows),
        "card_issues": card_issues,
        "eval_card_issues": eval_card_issues,
        "smoke_dry_run": smoke_dry_run,
        "blockers": blockers,
        "warnings": warnings,
    }


def _format_mapping(mapping: dict) -> list[str]:
    if not mapping:
        return ["- 无"]
    return [f"- {key}：{value}" for key, value in mapping.items()]


def _format_items(items: list[Any]) -> list[str]:
    if not items:
        return ["- 无"]
    return [f"- {item}" for item in items]


def render_stage3_readiness_report(summary: dict) -> str:
    lines = [
        "# Stage 3 Data Readiness Report",
        "",
        "## Decision",
        f"- {summary['decision']}",
        "",
        "## Counts",
        f"- 原始文本文件数：{summary['raw_text_file_count']}",
        f"- chapters_raw 行数：{summary['chapters_raw_count']}",
        f"- chapters 行数：{summary['chapter_count']}",
        f"- split 行数：{summary['split_count']}",
        f"- train 行数：{summary['train_count']}",
        f"- eval split 行数：{summary['eval_split_count']}",
        f"- 章节卡数：{summary['chapter_card_count']}",
        f"- eval 卡数：{summary['eval_card_count']}",
        f"- SFT 样本数：{summary['sft_row_count']}",
        "",
        "## Paths",
    ]
    lines.extend(_format_mapping(summary.get("paths", {})))
    lines.extend(["", "## Quality Tags"])
    lines.extend(_format_mapping(summary.get("quality_tag_counts", {})))
    chapter_length = summary.get("chapter_length", {})
    lines.extend(
        [
            "",
            "## Chapter Lengths",
            f"- min：{chapter_length.get('min', 0)}",
            f"- max：{chapter_length.get('max', 0)}",
            f"- avg：{chapter_length.get('avg', 0)}",
            "",
            "## Card Issues",
            f"- missing_required_fields：{summary['card_issues'].get('missing_required_fields', [])}",
            f"- source_leakage_errors：{summary['card_issues'].get('source_leakage_errors', [])}",
            f"- render_errors：{summary['card_issues'].get('render_errors', [])}",
            f"- schema_errors：{summary['card_issues'].get('schema_errors', [])}",
            f"- non_object_rows：{summary['card_issues'].get('non_object_rows', [])}",
            f"- unmatched_chapter_ids：{summary['card_issues'].get('unmatched_chapter_ids', [])}",
            f"- cards_with_empty_lists：{summary['card_issues'].get('cards_with_empty_lists', [])}",
            "",
            "## Eval Card Issues",
            f"- source_leakage_errors：{summary.get('eval_card_issues', {}).get('source_leakage_errors', [])}",
            f"- render_errors：{summary.get('eval_card_issues', {}).get('render_errors', [])}",
            f"- schema_errors：{summary.get('eval_card_issues', {}).get('schema_errors', [])}",
            f"- non_object_rows：{summary.get('eval_card_issues', {}).get('non_object_rows', [])}",
            "",
            "## Smoke Dry-Run",
            f"- exit_code：{summary['smoke_dry_run'].get('exit_code')}",
            f"- command：{summary['smoke_dry_run'].get('command', '')}",
            f"- stderr：{summary['smoke_dry_run'].get('stderr', '')}",
            "",
            "## Blockers",
        ]
    )
    lines.extend(_format_items(summary.get("blockers", [])))
    lines.extend(["", "## Warnings"])
    lines.extend(_format_items(summary.get("warnings", [])))
    return "\n".join(lines) + "\n"
