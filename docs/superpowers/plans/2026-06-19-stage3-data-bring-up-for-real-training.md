# Stage 3 Data Bring-Up For Real Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Stage 3 data readiness tooling that turns real raw novel text plus chapter cards into validated SFT/eval assets and a clear go/no-go report for Stage 4 smoke training.

**Architecture:** Reuse the existing Stage 1 data scripts for ingestion, cleaning, splitting, style profiling, and SFT row construction. Add one focused readiness module under `src/small_model_train/` plus a thin CLI wrapper under `scripts/` to summarize real artifacts, detect blockers, run the Stage 2 smoke dry-run preflight, and write `reports/stage3_data_readiness_report.md`.

**Tech Stack:** Python 3.11 stdlib (`argparse`, `pathlib`, `subprocess`, `statistics`, `collections`), existing JSONL helpers, existing `sft_builder.render_sft_input` leakage guard, existing `text_utils.count_chinese_chars`, and `pytest`.

---

## File Structure

Create these files:

```text
src/small_model_train/stage3_data_readiness.py
scripts/check_stage3_data_readiness.py
tests/test_stage3_data_readiness.py
docs/stage3-data-bring-up-guide.zh.md
```

Modify these files:

```text
README.md
```

Responsibilities:

- `stage3_data_readiness.py`: Pure functions for reading Stage 3 artifacts, counting raw/clean/split/card/SFT/eval rows, checking required chapter-card fields, reusing the source leakage guard, choosing a Stage 3 decision code, and rendering the Markdown readiness report.
- `scripts/check_stage3_data_readiness.py`: CLI wrapper that optionally runs `scripts/run_sft_smoke.py --dry-run`, calls the readiness module, writes `reports/stage3_data_readiness_report.md`, and exits nonzero when Stage 4 should not start.
- `tests/test_stage3_data_readiness.py`: Unit and CLI tests for ready, blocked, malformed card, leakage, and report-rendering paths.
- `docs/stage3-data-bring-up-guide.zh.md`: Human-facing runbook for preparing real data from zero.
- `README.md`: Add a short Stage 3 command sequence and explain that real GPU training starts in Stage 4.

## Task 1: Stage 3 Readiness Summary Module

**Files:**
- Create: `src/small_model_train/stage3_data_readiness.py`
- Test: `tests/test_stage3_data_readiness.py`

- [ ] **Step 1: Write failing tests for ready, blocked, and report-rendering paths**

Create `tests/test_stage3_data_readiness.py`:

```python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from small_model_train.io_utils import write_jsonl
from small_model_train.stage3_data_readiness import (
    build_stage3_summary,
    render_stage3_readiness_report,
)


def complete_card(sample_id: str, source_text: str = "") -> dict:
    return {
        "id": sample_id,
        "style_contract": "只输出正文。",
        "previous_summary": "上一章交易没有谈拢。",
        "chapter_goal": "林默进入仓库并完成谈判。",
        "chapter_structure": [
            {"step": 1, "name": "入场", "goal": "交代地点和压力", "estimated_chars": "300-400"},
            {"step": 2, "name": "谈判", "goal": "让林默提出加钱", "estimated_chars": "500-600"},
        ],
        "character_states": [
            {"name": "林默", "state": "冷静但警惕", "speech_style": "短句，少解释"}
        ],
        "must_include": ["旧仓库", "加钱"],
        "must_not_include": ["真相大白"],
        "ending_hook": "箱子自己响了一下。",
        "target_word_count": "2000-2500中文汉字",
        "source_text": source_text,
    }


def write_stage3_assets(tmp_path: Path, *, card: dict | None = None) -> dict[str, Path]:
    raw_dir = tmp_path / "data_raw" / "novels"
    raw_dir.mkdir(parents=True)
    (raw_dir / "novel.txt").write_text("第1章 开始\n\n林默说加钱。\n", encoding="utf-8")

    chapters_raw = tmp_path / "data_clean" / "chapters_raw.jsonl"
    chapters = tmp_path / "data_clean" / "chapters.jsonl"
    chapters_split = tmp_path / "data_clean" / "chapters_split.jsonl"
    cards = tmp_path / "data_cards" / "chapter_cards.jsonl"
    eval_cards = tmp_path / "data_cards" / "eval_cards_20.jsonl"
    sft = tmp_path / "data_sft" / "sft_chapter_v1.jsonl"

    split_rows = [
        {
            "id": "train_1",
            "work_id": "work",
            "chapter_title": "第1章",
            "text": "训练正文" * 300,
            "char_count_zh": 1200,
            "quality_tag": "A",
            "split": "train",
        },
        {
            "id": "eval_1",
            "work_id": "work",
            "chapter_title": "第2章",
            "text": "评估正文" * 300,
            "char_count_zh": 1200,
            "quality_tag": "A",
            "split": "eval",
        },
    ]
    write_jsonl(chapters_raw, split_rows)
    write_jsonl(chapters, split_rows)
    write_jsonl(chapters_split, split_rows)
    write_jsonl(cards, [card or complete_card("train_1")])
    write_jsonl(eval_cards, [complete_card("eval_1")])
    write_jsonl(
        sft,
        [
            {
                "instruction": "你是作者的正文执行器。",
                "input": "章节卡输入",
                "output": "训练正文" * 300,
            },
            {
                "instruction": "你是作者的正文执行器。",
                "input": "章节卡输入二",
                "output": "训练正文二" * 300,
            },
        ],
    )
    return {
        "raw_dir": raw_dir,
        "chapters_raw": chapters_raw,
        "chapters": chapters,
        "chapters_split": chapters_split,
        "cards": cards,
        "eval_cards": eval_cards,
        "sft": sft,
    }


def test_build_stage3_summary_ready_when_assets_and_dry_run_pass(tmp_path: Path):
    paths = write_stage3_assets(tmp_path)

    summary = build_stage3_summary(
        raw_dir=paths["raw_dir"],
        chapters_raw=paths["chapters_raw"],
        chapters=paths["chapters"],
        chapters_split=paths["chapters_split"],
        chapter_cards=paths["cards"],
        eval_cards=paths["eval_cards"],
        sft_dataset=paths["sft"],
        smoke_dry_run={"exit_code": 0, "command": "python scripts/run_sft_smoke.py --dry-run", "stderr": ""},
        min_trainable_sft=2,
        min_eval_cards=1,
        preferred_eval_cards=1,
    )

    assert summary["decision"] == "ready_for_stage4_smoke_training"
    assert summary["raw_text_file_count"] == 1
    assert summary["sft_row_count"] == 2
    assert summary["eval_card_count"] == 1


def test_build_stage3_summary_blocks_missing_raw_text_first(tmp_path: Path):
    paths = write_stage3_assets(tmp_path)
    for file_path in paths["raw_dir"].glob("*"):
        file_path.unlink()

    summary = build_stage3_summary(
        raw_dir=paths["raw_dir"],
        chapters_raw=paths["chapters_raw"],
        chapters=paths["chapters"],
        chapters_split=paths["chapters_split"],
        chapter_cards=paths["cards"],
        eval_cards=paths["eval_cards"],
        sft_dataset=paths["sft"],
        smoke_dry_run={"exit_code": 0, "command": "dry", "stderr": ""},
        min_trainable_sft=2,
        min_eval_cards=1,
        preferred_eval_cards=1,
    )

    assert summary["decision"] == "blocked_missing_raw_text"
    assert "data_raw/novels has no .txt or .md files" in "\n".join(summary["blockers"])


def test_build_stage3_summary_reports_missing_card_fields(tmp_path: Path):
    bad_card = complete_card("train_1")
    bad_card.pop("chapter_goal")
    paths = write_stage3_assets(tmp_path, card=bad_card)

    summary = build_stage3_summary(
        raw_dir=paths["raw_dir"],
        chapters_raw=paths["chapters_raw"],
        chapters=paths["chapters"],
        chapters_split=paths["chapters_split"],
        chapter_cards=paths["cards"],
        eval_cards=paths["eval_cards"],
        sft_dataset=paths["sft"],
        smoke_dry_run={"exit_code": 0, "command": "dry", "stderr": ""},
        min_trainable_sft=2,
        min_eval_cards=1,
        preferred_eval_cards=1,
    )

    assert summary["decision"] == "blocked_missing_chapter_cards"
    assert summary["card_issues"]["missing_required_fields"] == [
        {"id": "train_1", "missing_fields": ["chapter_goal"]}
    ]


def test_build_stage3_summary_reports_source_text_leakage(tmp_path: Path):
    leaking_card = complete_card(
        "train_1",
        source_text="这是一段非常独特的原文句子，不能出现在提示词。",
    )
    leaking_card["previous_summary"] = "上一章他说：这是一段非常独特的原文句子。"
    paths = write_stage3_assets(tmp_path, card=leaking_card)

    summary = build_stage3_summary(
        raw_dir=paths["raw_dir"],
        chapters_raw=paths["chapters_raw"],
        chapters=paths["chapters"],
        chapters_split=paths["chapters_split"],
        chapter_cards=paths["cards"],
        eval_cards=paths["eval_cards"],
        sft_dataset=paths["sft"],
        smoke_dry_run={"exit_code": 0, "command": "dry", "stderr": ""},
        min_trainable_sft=2,
        min_eval_cards=1,
        preferred_eval_cards=1,
    )

    assert summary["decision"] == "blocked_source_leakage"
    assert "train_1" in summary["card_issues"]["source_leakage_errors"][0]


def test_build_stage3_summary_blocks_empty_sft_dataset(tmp_path: Path):
    paths = write_stage3_assets(tmp_path)
    write_jsonl(paths["sft"], [])

    summary = build_stage3_summary(
        raw_dir=paths["raw_dir"],
        chapters_raw=paths["chapters_raw"],
        chapters=paths["chapters"],
        chapters_split=paths["chapters_split"],
        chapter_cards=paths["cards"],
        eval_cards=paths["eval_cards"],
        sft_dataset=paths["sft"],
        smoke_dry_run={"exit_code": 0, "command": "dry", "stderr": ""},
        min_trainable_sft=2,
        min_eval_cards=1,
        preferred_eval_cards=1,
    )

    assert summary["decision"] == "blocked_sft_empty"


def test_build_stage3_summary_blocks_missing_eval_cards(tmp_path: Path):
    paths = write_stage3_assets(tmp_path)
    write_jsonl(paths["eval_cards"], [])

    summary = build_stage3_summary(
        raw_dir=paths["raw_dir"],
        chapters_raw=paths["chapters_raw"],
        chapters=paths["chapters"],
        chapters_split=paths["chapters_split"],
        chapter_cards=paths["cards"],
        eval_cards=paths["eval_cards"],
        sft_dataset=paths["sft"],
        smoke_dry_run={"exit_code": 0, "command": "dry", "stderr": ""},
        min_trainable_sft=2,
        min_eval_cards=1,
        preferred_eval_cards=1,
    )

    assert summary["decision"] == "blocked_eval_missing"


def test_render_stage3_readiness_report_contains_decision_and_counts(tmp_path: Path):
    paths = write_stage3_assets(tmp_path)
    summary = build_stage3_summary(
        raw_dir=paths["raw_dir"],
        chapters_raw=paths["chapters_raw"],
        chapters=paths["chapters"],
        chapters_split=paths["chapters_split"],
        chapter_cards=paths["cards"],
        eval_cards=paths["eval_cards"],
        sft_dataset=paths["sft"],
        smoke_dry_run={"exit_code": 0, "command": "dry", "stderr": ""},
        min_trainable_sft=2,
        min_eval_cards=1,
        preferred_eval_cards=1,
    )

    report = render_stage3_readiness_report(summary)

    assert "# Stage 3 Data Readiness Report" in report
    assert "ready_for_stage4_smoke_training" in report
    assert "- SFT 样本数：2" in report
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_stage3_data_readiness.py -v
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `small_model_train.stage3_data_readiness`.

- [ ] **Step 3: Implement the readiness module**

Create `src/small_model_train/stage3_data_readiness.py`:

```python
from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import mean
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


def build_stage3_summary(
    raw_dir: str | Path,
    chapters_raw: str | Path,
    chapters: str | Path,
    chapters_split: str | Path,
    chapter_cards: str | Path,
    eval_cards: str | Path,
    sft_dataset: str | Path,
    smoke_dry_run: dict[str, Any] | None = None,
    min_trainable_sft: int = 20,
    min_eval_cards: int = 10,
    preferred_eval_cards: int = 50,
) -> dict[str, Any]:
    raw_path = Path(raw_dir)
    raw_files = sorted(
        path
        for path in raw_path.rglob("*")
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}
    ) if raw_path.exists() else []
    chapters_raw_rows = read_jsonl(chapters_raw)
    chapter_rows = read_jsonl(chapters)
    split_rows = read_jsonl(chapters_split)
    card_rows = read_jsonl(chapter_cards)
    eval_rows = read_jsonl(eval_cards)
    sft_rows = read_jsonl(sft_dataset)
    train_rows = [row for row in split_rows if row.get("split") == "train"]
    eval_split_rows = [row for row in split_rows if row.get("split") == "eval"]
    quality_counts = Counter(str(row.get("quality_tag", "")) for row in split_rows)
    card_issues = inspect_chapter_cards(card_rows)
    dry_run = smoke_dry_run or {
        "exit_code": None,
        "command": "",
        "stderr": "smoke dry-run has not been run",
    }
    blockers: list[str] = []
    warnings: list[str] = []

    if not raw_files:
        blockers.append("data_raw/novels has no .txt or .md files")
    if not split_rows:
        blockers.append("data_clean/chapters_split.jsonl is missing or empty")
    if not train_rows:
        blockers.append("chapters_split.jsonl has no train rows")
    if not eval_split_rows:
        blockers.append("chapters_split.jsonl has no eval rows")
    if not card_rows:
        blockers.append("data_cards/chapter_cards.jsonl is missing or empty")
    if card_issues["missing_required_fields"]:
        blockers.append("chapter cards are missing required fields")
    if card_issues["source_leakage_errors"]:
        blockers.append("chapter cards leak source_text into rendered prompts")
    if not sft_rows:
        blockers.append("data_sft/sft_chapter_v1.jsonl is missing or empty")
    if len(sft_rows) < min_trainable_sft:
        blockers.append(f"SFT row count {len(sft_rows)} is below minimum {min_trainable_sft}")
    if not eval_rows:
        blockers.append("fixed eval cards file is missing or empty")
    if len(eval_rows) < min_eval_cards:
        blockers.append(f"eval card count {len(eval_rows)} is below minimum {min_eval_cards}")
    if dry_run.get("exit_code") != 0:
        blockers.append("Stage 2 smoke dry-run did not pass")

    if 0 < len(eval_rows) < preferred_eval_cards:
        warnings.append(f"eval card count {len(eval_rows)} is below preferred {preferred_eval_cards}")
    if 0 < len(sft_rows) < 100:
        warnings.append(f"SFT row count {len(sft_rows)} is below first 100-row smoke target")
    if card_issues["cards_with_empty_lists"]:
        warnings.append("some chapter cards have empty must_include or must_not_include lists")

    return {
        "decision": decide_stage3_status(
            raw_files=raw_files,
            split_rows=split_rows,
            train_rows=train_rows,
            card_rows=card_rows,
            card_issues=card_issues,
            sft_rows=sft_rows,
            eval_rows=eval_rows,
            dry_run=dry_run,
            min_trainable_sft=min_trainable_sft,
            min_eval_cards=min_eval_cards,
        ),
        "paths": {
            "raw_dir": str(raw_path),
            "chapters_raw": str(chapters_raw),
            "chapters": str(chapters),
            "chapters_split": str(chapters_split),
            "chapter_cards": str(chapter_cards),
            "eval_cards": str(eval_cards),
            "sft_dataset": str(sft_dataset),
        },
        "raw_text_file_count": len(raw_files),
        "chapters_raw_count": len(chapters_raw_rows),
        "chapter_count": len(chapter_rows),
        "split_count": len(split_rows),
        "train_count": len(train_rows),
        "eval_split_count": len(eval_split_rows),
        "quality_tag_counts": dict(sorted(quality_counts.items())),
        "chapter_length": summarize_chapter_lengths(split_rows),
        "chapter_card_count": len(card_rows),
        "eval_card_count": len(eval_rows),
        "sft_row_count": len(sft_rows),
        "card_issues": card_issues,
        "smoke_dry_run": dry_run,
        "blockers": blockers,
        "warnings": warnings,
    }


def decide_stage3_status(
    raw_files: list[Path],
    split_rows: list[dict],
    train_rows: list[dict],
    card_rows: list[dict],
    card_issues: dict[str, Any],
    sft_rows: list[dict],
    eval_rows: list[dict],
    dry_run: dict[str, Any],
    min_trainable_sft: int,
    min_eval_cards: int,
) -> str:
    if not raw_files:
        return "blocked_missing_raw_text"
    if not split_rows or not train_rows:
        return "blocked_insufficient_chapters"
    if not card_rows or card_issues["missing_required_fields"]:
        return "blocked_missing_chapter_cards"
    if card_issues["source_leakage_errors"]:
        return "blocked_source_leakage"
    if not sft_rows:
        return "blocked_sft_empty"
    if len(sft_rows) < min_trainable_sft:
        return "blocked_insufficient_chapters"
    if len(eval_rows) < min_eval_cards:
        return "blocked_eval_missing"
    if dry_run.get("exit_code") != 0:
        return "blocked_stage2_dry_run_failed"
    return "ready_for_stage4_smoke_training"


def inspect_chapter_cards(cards: list[dict]) -> dict[str, Any]:
    missing_required_fields = []
    source_leakage_errors = []
    cards_with_empty_lists = []
    for index, card in enumerate(cards, start=1):
        sample_id = str(card.get("id", f"row_{index}"))
        missing = [field for field in REQUIRED_CARD_FIELDS if field not in card]
        if missing:
            missing_required_fields.append({"id": sample_id, "missing_fields": missing})
        for field in ("must_include", "must_not_include"):
            if field in card and not card.get(field):
                cards_with_empty_lists.append({"id": sample_id, "field": field})
        try:
            render_sft_input(card)
        except ValueError as exc:
            source_leakage_errors.append(f"{sample_id}: {exc}")
    return {
        "missing_required_fields": missing_required_fields,
        "source_leakage_errors": source_leakage_errors,
        "cards_with_empty_lists": cards_with_empty_lists,
    }


def summarize_chapter_lengths(rows: list[dict]) -> dict[str, float | int]:
    lengths = [
        int(row.get("char_count_zh") or count_chinese_chars(row.get("text", "")))
        for row in rows
    ]
    if not lengths:
        return {"min": 0, "max": 0, "avg": 0}
    return {
        "min": min(lengths),
        "max": max(lengths),
        "avg": round(mean(lengths), 2),
    }


def render_stage3_readiness_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Stage 3 Data Readiness Report",
        "",
        f"- 结论：{summary['decision']}",
        f"- 原始文本文件数：{summary['raw_text_file_count']}",
        f"- raw 章节数：{summary['chapters_raw_count']}",
        f"- 清洗后章节数：{summary['chapter_count']}",
        f"- split 总数：{summary['split_count']}",
        f"- train 数量：{summary['train_count']}",
        f"- eval split 数量：{summary['eval_split_count']}",
        f"- 章节卡数量：{summary['chapter_card_count']}",
        f"- eval cards 数量：{summary['eval_card_count']}",
        f"- SFT 样本数：{summary['sft_row_count']}",
        "",
        "## Paths",
    ]
    for name, path in summary["paths"].items():
        lines.append(f"- {name}: {path}")
    lines.extend(["", "## Quality Tags"])
    if summary["quality_tag_counts"]:
        for tag, count in summary["quality_tag_counts"].items():
            lines.append(f"- {tag or '(empty)'}: {count}")
    else:
        lines.append("- 无")
    chapter_length = summary["chapter_length"]
    lines.extend(
        [
            "",
            "## Chapter Lengths",
            f"- min: {chapter_length['min']}",
            f"- max: {chapter_length['max']}",
            f"- avg: {chapter_length['avg']}",
            "",
            "## Chapter Card Issues",
        ]
    )
    card_issues = summary["card_issues"]
    if not any(card_issues.values()):
        lines.append("- 无")
    else:
        for item in card_issues["missing_required_fields"]:
            lines.append(f"- missing_required_fields: {item['id']} -> {', '.join(item['missing_fields'])}")
        for item in card_issues["source_leakage_errors"]:
            lines.append(f"- source_leakage: {item}")
        for item in card_issues["cards_with_empty_lists"]:
            lines.append(f"- empty_list: {item['id']} -> {item['field']}")
    lines.extend(["", "## Smoke Dry Run"])
    dry_run = summary["smoke_dry_run"]
    lines.append(f"- exit_code: {dry_run.get('exit_code')}")
    lines.append(f"- command: {dry_run.get('command', '')}")
    stderr = str(dry_run.get("stderr", "")).strip()
    lines.append(f"- stderr: {stderr if stderr else '无'}")
    lines.extend(["", "## Blockers"])
    if summary["blockers"]:
        for blocker in summary["blockers"]:
            lines.append(f"- {blocker}")
    else:
        lines.append("- 无")
    lines.extend(["", "## Warnings"])
    if summary["warnings"]:
        for warning in summary["warnings"]:
            lines.append(f"- {warning}")
    else:
        lines.append("- 无")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_stage3_data_readiness.py -v
```

Expected: PASS for all tests in `tests/test_stage3_data_readiness.py`.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/small_model_train/stage3_data_readiness.py tests/test_stage3_data_readiness.py
git commit -m "feat: add stage three data readiness summary"
```

Expected: commit succeeds.

## Task 2: Readiness CLI With Smoke Dry-Run Preflight

**Files:**
- Create: `scripts/check_stage3_data_readiness.py`
- Modify: `tests/test_stage3_data_readiness.py`

- [ ] **Step 1: Add CLI tests for report writing and non-ready exit code**

Append to `tests/test_stage3_data_readiness.py`:

```python
def test_check_stage3_data_readiness_cli_writes_report_and_blocks_without_dry_run(tmp_path: Path):
    paths = write_stage3_assets(tmp_path)
    report = tmp_path / "reports" / "stage3.md"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_stage3_data_readiness.py",
            "--raw-dir",
            str(paths["raw_dir"]),
            "--chapters-raw",
            str(paths["chapters_raw"]),
            "--chapters",
            str(paths["chapters"]),
            "--chapters-split",
            str(paths["chapters_split"]),
            "--chapter-cards",
            str(paths["cards"]),
            "--eval-cards",
            str(paths["eval_cards"]),
            "--sft-dataset",
            str(paths["sft"]),
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
        check=False,
        text=True,
    )

    assert result.returncode == 1
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "blocked_stage2_dry_run_failed" in text
    assert "smoke dry-run has not been run" in text
```

- [ ] **Step 2: Run the CLI test and verify it fails**

Run:

```powershell
python -m pytest tests/test_stage3_data_readiness.py::test_check_stage3_data_readiness_cli_writes_report_and_blocks_without_dry_run -v
```

Expected: FAIL because `scripts/check_stage3_data_readiness.py` does not exist.

- [ ] **Step 3: Implement the CLI wrapper**

Create `scripts/check_stage3_data_readiness.py`:

```python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.stage3_data_readiness import (
    build_stage3_summary,
    render_stage3_readiness_report,
)


def run_smoke_dry_run(args: argparse.Namespace) -> dict:
    command = [
        sys.executable,
        "scripts/run_sft_smoke.py",
        "--dry-run",
        "--config",
        args.config,
        "--model-dir",
        args.model_dir,
        "--sft-dataset",
        args.sft_dataset,
        "--eval-cards",
        args.eval_cards,
    ]
    completed = subprocess.run(command, capture_output=True, check=False, text=True)
    return {
        "exit_code": completed.returncode,
        "command": subprocess.list2cmdline(command),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data_raw/novels")
    parser.add_argument("--chapters-raw", default="data_clean/chapters_raw.jsonl")
    parser.add_argument("--chapters", default="data_clean/chapters.jsonl")
    parser.add_argument("--chapters-split", default="data_clean/chapters_split.jsonl")
    parser.add_argument("--chapter-cards", default="data_cards/chapter_cards.jsonl")
    parser.add_argument("--eval-cards", default="data_cards/eval_cards_20.jsonl")
    parser.add_argument("--sft-dataset", default="data_sft/sft_chapter_v1.jsonl")
    parser.add_argument("--report", default="reports/stage3_data_readiness_report.md")
    parser.add_argument("--config", default="configs/sft_qlora_qwen3_4b.yaml")
    parser.add_argument("--model-dir", default=r"E:\models\Qwen3-4B-Instruct-2507")
    parser.add_argument("--min-trainable-sft", type=int, default=20)
    parser.add_argument("--min-eval-cards", type=int, default=10)
    parser.add_argument("--preferred-eval-cards", type=int, default=50)
    parser.add_argument("--run-smoke-dry-run", action="store_true")
    args = parser.parse_args()

    smoke_dry_run = (
        run_smoke_dry_run(args)
        if args.run_smoke_dry_run
        else {"exit_code": None, "command": "", "stderr": "smoke dry-run has not been run"}
    )
    summary = build_stage3_summary(
        raw_dir=args.raw_dir,
        chapters_raw=args.chapters_raw,
        chapters=args.chapters,
        chapters_split=args.chapters_split,
        chapter_cards=args.chapter_cards,
        eval_cards=args.eval_cards,
        sft_dataset=args.sft_dataset,
        smoke_dry_run=smoke_dry_run,
        min_trainable_sft=args.min_trainable_sft,
        min_eval_cards=args.min_eval_cards,
        preferred_eval_cards=args.preferred_eval_cards,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_stage3_readiness_report(summary), encoding="utf-8")
    print(f"wrote Stage 3 readiness report to {report_path}")
    print(summary["decision"])
    return 0 if summary["decision"] == "ready_for_stage4_smoke_training" else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the CLI test and module tests**

Run:

```powershell
python -m pytest tests/test_stage3_data_readiness.py -v
```

Expected: PASS for all tests in `tests/test_stage3_data_readiness.py`.

- [ ] **Step 5: Run the full test suite**

Run:

```powershell
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

Run:

```powershell
git add scripts/check_stage3_data_readiness.py tests/test_stage3_data_readiness.py
git commit -m "feat: add stage three readiness cli"
```

Expected: commit succeeds.

## Task 3: Stage 3 User Guide And README Commands

**Files:**
- Create: `docs/stage3-data-bring-up-guide.zh.md`
- Modify: `README.md`

- [ ] **Step 1: Create the Stage 3 guide**

Create `docs/stage3-data-bring-up-guide.zh.md`:

````markdown
# 第三阶段：面向真实训练的数据准备指南

第三阶段的目标是把项目从“代码测试通过”推进到“真实训练数据可用”。本阶段不启动真实 GPU 训练；真实 smoke training 属于第四阶段。

## 1. 准备原始文本

把真实小说原稿放到：

```text
data_raw/novels/
```

支持 `.txt` 和 `.md`。建议第一批只放同一作者、同一目标风格的 A 类正文，避免混入废稿、设定、笔记和题材差异很大的旧文。

## 2. 生成清洗章节

```powershell
python scripts/ingest_raw_text.py --input-dir data_raw/novels --output data_clean/chapters_raw.jsonl
python scripts/clean_chapters.py --input data_clean/chapters_raw.jsonl --output data_clean/chapters.jsonl --min-chars 500 --max-chars 5000
```

如果清洗后章节过少，先补原稿或调整长度阈值。不要用空章节进入训练。

## 3. 固定 train/eval

章节数量不足 50 条 eval 时，先固定 20 条或更小的 eval 集：

```powershell
python scripts/split_train_eval.py --input data_clean/chapters.jsonl --output data_clean/chapters_split.jsonl --eval-output data_cards/eval_cards_20.jsonl --eval-count 20
```

章节数量足够时使用 50 条：

```powershell
python scripts/split_train_eval.py --input data_clean/chapters.jsonl --output data_clean/chapters_split.jsonl --eval-output data_cards/eval_cards_50.jsonl --eval-count 50
```

固定 eval cards 不能进入 SFT 训练。

## 4. 生成风格契约

```powershell
python scripts/build_style_contract.py --chapters data_clean/chapters_split.jsonl --contract-output style_contract.md --profile-output style_profile.json
```

`style_contract.md` 会进入章节卡输入。`style_profile.json` 用于复查统计分布。

## 5. 准备章节卡

手工或外部模型辅助准备：

```text
data_cards/chapter_cards.jsonl
```

每行至少包含：

```json
{"id":"chapter_id","style_contract":"只输出正文。","previous_summary":"上一章摘要。","chapter_goal":"本章目标。","chapter_structure":[{"step":1,"name":"入场","goal":"交代压力","estimated_chars":"300-400"}],"character_states":[{"name":"林默","state":"冷静但警惕","speech_style":"短句，少解释"}],"must_include":["旧仓库"],"must_not_include":["真相大白"],"ending_hook":"箱子响了一下。","target_word_count":"2000-2500中文汉字","source_text":"只用于离线溯源，不进入 prompt。"}
```

章节卡不能复制原文句子。`source_text` 可以保留作检查，但不能进入 prompt。

## 6. 构造 SFT 数据

```powershell
python scripts/build_sft_dataset.py --cards data_cards/chapter_cards.jsonl --chapters data_clean/chapters_split.jsonl --output data_sft/sft_chapter_v1.jsonl
```

如果脚本报 `source_text` 泄漏，先修章节卡，不要绕过检查。

## 7. 生成 Stage 3 数据验收报告

使用 20 条 eval cards 时：

```powershell
python scripts/check_stage3_data_readiness.py --eval-cards data_cards/eval_cards_20.jsonl --run-smoke-dry-run
```

使用 50 条 eval cards 时：

```powershell
python scripts/check_stage3_data_readiness.py --eval-cards data_cards/eval_cards_50.jsonl --run-smoke-dry-run
```

报告输出到：

```text
reports/stage3_data_readiness_report.md
```

只有报告结论为 `ready_for_stage4_smoke_training`，才进入第四阶段真实训练。
````

- [ ] **Step 2: Update README with a concise Stage 3 command sequence**

Insert this section before `## Stage 2 Training Execution` in `README.md`:

````markdown
## Stage 3 Data Bring-Up

Stage 3 prepares the first real data assets for training. It does not start real GPU training; it ends when the Stage 2 smoke dry-run can read the generated SFT and eval files.

```powershell
python scripts/ingest_raw_text.py --input-dir data_raw/novels --output data_clean/chapters_raw.jsonl
python scripts/clean_chapters.py --input data_clean/chapters_raw.jsonl --output data_clean/chapters.jsonl --min-chars 500 --max-chars 5000
python scripts/split_train_eval.py --input data_clean/chapters.jsonl --output data_clean/chapters_split.jsonl --eval-output data_cards/eval_cards_20.jsonl --eval-count 20
python scripts/build_style_contract.py --chapters data_clean/chapters_split.jsonl --contract-output style_contract.md --profile-output style_profile.json
python scripts/build_sft_dataset.py --cards data_cards/chapter_cards.jsonl --chapters data_clean/chapters_split.jsonl --output data_sft/sft_chapter_v1.jsonl
python scripts/check_stage3_data_readiness.py --eval-cards data_cards/eval_cards_20.jsonl --run-smoke-dry-run
```

When `reports/stage3_data_readiness_report.md` says `ready_for_stage4_smoke_training`, move to real Stage 4 smoke training.
````

- [ ] **Step 3: Run Markdown red-flag scans**

Run:

```powershell
$patterns = @("TB" + "D", "TO" + "DO", "place" + "holder", "not imple" + "mented", "fill" + " in")
rg -n -i ($patterns -join "|") docs/stage3-data-bring-up-guide.zh.md README.md
git diff --check
```

Expected: no red-flag matches in the new Stage 3 sections and no whitespace errors.

- [ ] **Step 4: Commit**

Run:

```powershell
git add docs/stage3-data-bring-up-guide.zh.md README.md
git commit -m "docs: add stage three data bring-up guide"
```

Expected: commit succeeds.

## Task 4: End-To-End Synthetic Verification And Final Checks

**Files:**
- Verify: `src/small_model_train/stage3_data_readiness.py`
- Verify: `scripts/check_stage3_data_readiness.py`
- Verify: `README.md`
- Verify: `docs/stage3-data-bring-up-guide.zh.md`

- [ ] **Step 1: Run targeted Stage 3 tests**

Run:

```powershell
python -m pytest tests/test_stage3_data_readiness.py -v
```

Expected: PASS for all Stage 3 readiness tests.

- [ ] **Step 2: Run the full test suite**

Run:

```powershell
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Run a synthetic CLI report check without real user data**

Use temporary files so the repository does not gain fake training data:

```powershell
$tmp = New-Item -ItemType Directory -Path ([System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), "stage3-readiness-" + [guid]::NewGuid()))
New-Item -ItemType Directory -Path "$tmp\data_raw\novels" | Out-Null
New-Item -ItemType Directory -Path "$tmp\data_clean" | Out-Null
New-Item -ItemType Directory -Path "$tmp\data_cards" | Out-Null
New-Item -ItemType Directory -Path "$tmp\data_sft" | Out-Null
Set-Content -LiteralPath "$tmp\data_raw\novels\novel.txt" -Encoding UTF8 -Value "第1章 开始`n`n林默说加钱。"
@'
{"id":"train_1","work_id":"work","chapter_title":"第1章","text":"训练正文训练正文训练正文","char_count_zh":1200,"quality_tag":"A","split":"train"}
{"id":"eval_1","work_id":"work","chapter_title":"第2章","text":"评估正文评估正文评估正文","char_count_zh":1200,"quality_tag":"A","split":"eval"}
'@ | Set-Content -LiteralPath "$tmp\data_clean\chapters_raw.jsonl" -Encoding UTF8
Copy-Item -LiteralPath "$tmp\data_clean\chapters_raw.jsonl" -Destination "$tmp\data_clean\chapters.jsonl"
Copy-Item -LiteralPath "$tmp\data_clean\chapters_raw.jsonl" -Destination "$tmp\data_clean\chapters_split.jsonl"
@'
{"id":"train_1","style_contract":"只输出正文。","previous_summary":"上一章交易没有谈拢。","chapter_goal":"林默进入仓库并完成谈判。","chapter_structure":[{"step":1,"name":"入场","goal":"交代地点和压力","estimated_chars":"300-400"}],"character_states":[{"name":"林默","state":"冷静但警惕","speech_style":"短句，少解释"}],"must_include":["旧仓库","加钱"],"must_not_include":["真相大白"],"ending_hook":"箱子自己响了一下。","target_word_count":"2000-2500中文汉字","source_text":"离线溯源文本。"}
'@ | Set-Content -LiteralPath "$tmp\data_cards\chapter_cards.jsonl" -Encoding UTF8
@'
{"id":"eval_1","style_contract":"只输出正文。","previous_summary":"上一章结束。","chapter_goal":"完成交易。","chapter_structure":[],"character_states":[],"must_include":["加钱"],"must_not_include":["真相"],"ending_hook":"箱子响了一下。","target_word_count":"2000-2500中文汉字"}
'@ | Set-Content -LiteralPath "$tmp\data_cards\eval_cards_20.jsonl" -Encoding UTF8
@'
{"instruction":"你是作者的正文执行器。","input":"章节卡输入","output":"训练正文训练正文训练正文"}
{"instruction":"你是作者的正文执行器。","input":"章节卡输入二","output":"训练正文二训练正文二训练正文二"}
'@ | Set-Content -LiteralPath "$tmp\data_sft\sft_chapter_v1.jsonl" -Encoding UTF8
python scripts/check_stage3_data_readiness.py --raw-dir "$tmp\data_raw\novels" --chapters-raw "$tmp\data_clean\chapters_raw.jsonl" --chapters "$tmp\data_clean\chapters.jsonl" --chapters-split "$tmp\data_clean\chapters_split.jsonl" --chapter-cards "$tmp\data_cards\chapter_cards.jsonl" --eval-cards "$tmp\data_cards\eval_cards_20.jsonl" --sft-dataset "$tmp\data_sft\sft_chapter_v1.jsonl" --report "$tmp\reports\stage3.md" --min-trainable-sft 2 --min-eval-cards 1 --preferred-eval-cards 1
Get-Content -LiteralPath "$tmp\reports\stage3.md"
```

Expected: the command exits 1 because smoke dry-run was intentionally not executed, and the report contains `blocked_stage2_dry_run_failed`. This confirms the CLI refuses to mark data ready without the Stage 2 dry-run preflight.

- [ ] **Step 4: Run final repository checks**

Run:

```powershell
git diff --check
git status --short
```

Expected: `git diff --check` exits 0. `git status --short` shows only intended files before the final commit, or clean after all task commits.

## Manual Stage 3 Execution Checklist With Real Data

After implementation, run these commands when real raw text and chapter cards are available:

```powershell
python scripts/ingest_raw_text.py --input-dir data_raw/novels --output data_clean/chapters_raw.jsonl
python scripts/clean_chapters.py --input data_clean/chapters_raw.jsonl --output data_clean/chapters.jsonl --min-chars 500 --max-chars 5000
python scripts/split_train_eval.py --input data_clean/chapters.jsonl --output data_clean/chapters_split.jsonl --eval-output data_cards/eval_cards_20.jsonl --eval-count 20
python scripts/build_style_contract.py --chapters data_clean/chapters_split.jsonl --contract-output style_contract.md --profile-output style_profile.json
python scripts/build_sft_dataset.py --cards data_cards/chapter_cards.jsonl --chapters data_clean/chapters_split.jsonl --output data_sft/sft_chapter_v1.jsonl
python scripts/check_stage3_data_readiness.py --eval-cards data_cards/eval_cards_20.jsonl --run-smoke-dry-run
```

If the available corpus can support 50 eval cards, use:

```powershell
python scripts/split_train_eval.py --input data_clean/chapters.jsonl --output data_clean/chapters_split.jsonl --eval-output data_cards/eval_cards_50.jsonl --eval-count 50
python scripts/check_stage3_data_readiness.py --eval-cards data_cards/eval_cards_50.jsonl --run-smoke-dry-run
```

Expected final Stage 3 artifacts:

```text
data_clean/chapters_raw.jsonl
data_clean/chapters.jsonl
data_clean/chapters_split.jsonl
style_contract.md
style_profile.json
data_cards/chapter_cards.jsonl
data_cards/eval_cards_20.jsonl or data_cards/eval_cards_50.jsonl
data_sft/sft_chapter_v1.jsonl
reports/stage3_data_readiness_report.md
```

## Self-Review

Spec coverage:

- Stage 3 starts from real raw text and existing Stage 1 scripts: covered by the manual checklist and README guide.
- The plan does not claim true GPU training: all commands stop at `run_sft_smoke.py --dry-run` inside Stage 3.
- Required artifacts are counted and reported by `stage3_data_readiness.py`.
- Missing raw text, insufficient chapters, missing cards, empty SFT data, source leakage, missing eval, and dry-run failure decisions are represented by decision codes.
- The minimum 20-50 SFT row direction is configurable with `--min-trainable-sft` and defaults to 20.
- Smaller fixed eval sets are supported through `--eval-cards`, `--min-eval-cards`, and `--preferred-eval-cards`.

Type consistency:

- Summary dictionaries consistently use `decision`, `paths`, count fields, `card_issues`, `smoke_dry_run`, `blockers`, and `warnings`.
- Chapter-card issues consistently expose `missing_required_fields`, `source_leakage_errors`, and `cards_with_empty_lists`.
- Smoke dry-run dictionaries consistently expose `exit_code`, `command`, `stdout`, and `stderr`, with `stdout` optional in report rendering.

Execution boundary:

- Unit tests use synthetic temporary data.
- Real `data_raw`, `data_clean`, `data_cards`, `data_sft`, `outputs`, and `reports` artifacts are produced only when the user supplies real data and runs the manual checklist.
- Stage 4 begins only after `reports/stage3_data_readiness_report.md` reports `ready_for_stage4_smoke_training`.
