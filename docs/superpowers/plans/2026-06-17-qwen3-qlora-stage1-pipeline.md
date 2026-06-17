# Qwen3 QLoRA Stage 1 Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first-stage data, scoring, reporting, and configuration pipeline for the Qwen3-4B-Instruct-2507 QLoRA whole-chapter novel executor.

**Architecture:** Core logic lives in `src/small_model_train/` as testable Python modules. Files under `scripts/` are thin CLI wrappers that call those modules. Config files and reports are plain YAML, JSONL, and Markdown so Codex can orchestrate each agent step without a long-running service.

**Tech Stack:** Python 3.10+, stdlib JSON/regex/pathlib/argparse, `pytest`, optional `pyyaml` only if already available for later training config validation.

---

## File Structure

Create these files:

```text
pyproject.toml
README.md
configs/sft_qlora_qwen3_4b.yaml
configs/infer_eval_qwen3_4b.yaml
src/small_model_train/__init__.py
src/small_model_train/io_utils.py
src/small_model_train/text_utils.py
src/small_model_train/chapter_splitter.py
src/small_model_train/dataset_split.py
src/small_model_train/style_profile.py
src/small_model_train/sft_builder.py
src/small_model_train/preference_builder.py
src/small_model_train/scoring.py
src/small_model_train/reporting.py
scripts/ingest_raw_text.py
scripts/clean_chapters.py
scripts/split_train_eval.py
scripts/build_style_contract.py
scripts/build_sft_dataset.py
scripts/build_preference_dataset.py
scripts/detect_ai_trace.py
scripts/score_outputs.py
scripts/evaluate_outputs.py
tests/test_text_utils.py
tests/test_chapter_splitter.py
tests/test_dataset_split.py
tests/test_style_profile.py
tests/test_sft_builder.py
tests/test_preference_builder.py
tests/test_scoring.py
tests/test_reporting.py
```

Directory responsibilities:

- `src/small_model_train/io_utils.py`: JSONL read/write and robust text reading.
- `src/small_model_train/text_utils.py`: text normalization, Chinese character counting, paragraphs, dialogue metrics, n-gram repetition.
- `src/small_model_train/chapter_splitter.py`: raw text cleaning and chapter extraction.
- `src/small_model_train/dataset_split.py`: deterministic train/eval split.
- `src/small_model_train/style_profile.py`: style statistics and contract template generation.
- `src/small_model_train/sft_builder.py`: chapter card to SFT prompt conversion.
- `src/small_model_train/preference_builder.py`: failed samples to preference candidates.
- `src/small_model_train/scoring.py`: hard gates, AI trace rules, rule metrics, failure labels.
- `src/small_model_train/reporting.py`: Markdown report generation.
- `scripts/*.py`: CLI entry points used by Codex agents.

## Task 1: Project Scaffold And Shared Text Utilities

**Files:**
- Create: `pyproject.toml`
- Create: `src/small_model_train/__init__.py`
- Create: `src/small_model_train/io_utils.py`
- Create: `src/small_model_train/text_utils.py`
- Test: `tests/test_text_utils.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_text_utils.py`:

```python
from pathlib import Path

from small_model_train.io_utils import read_jsonl, read_text_auto, write_jsonl
from small_model_train.text_utils import (
    count_chinese_chars,
    dialogue_ratio,
    normalize_newlines,
    paragraph_lengths,
    repeated_ngram_ratio,
)


def test_count_chinese_chars_ignores_ascii_and_punctuation():
    assert count_chinese_chars("第1章 Hello，林默说：加钱。") == 7


def test_normalize_newlines_and_paragraph_lengths():
    text = "第一段\r\n\r\n\r\n第二段\n\n第三段"
    normalized = normalize_newlines(text)
    assert normalized == "第一段\n\n第二段\n\n第三段"
    assert paragraph_lengths(normalized) == [3, 3, 3]


def test_dialogue_ratio_counts_dialogue_paragraphs():
    text = "林默看了她一眼。\n\n“这单加钱。”\n\n苏小满愣住：“多少？”"
    assert dialogue_ratio(text) == 2 / 3


def test_repeated_ngram_ratio_detects_repetition():
    text = "加钱加钱加钱加钱"
    assert repeated_ngram_ratio(text, n=2) > 0.5


def test_jsonl_round_trip(tmp_path: Path):
    path = tmp_path / "items.jsonl"
    rows = [{"id": "a", "text": "正文"}, {"id": "b", "text": "另一章"}]
    write_jsonl(path, rows)
    assert read_jsonl(path) == rows


def test_read_text_auto_handles_utf8_sig(tmp_path: Path):
    path = tmp_path / "novel.txt"
    path.write_text("正文", encoding="utf-8-sig")
    assert read_text_auto(path) == "正文"
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_text_utils.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'small_model_train'`.

- [ ] **Step 3: Create package scaffold and project config**

Create `pyproject.toml`:

```toml
[project]
name = "small-model-train"
version = "0.1.0"
description = "Qwen3 QLoRA whole-chapter novel executor training pipeline"
requires-python = ">=3.10"
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

Create `src/small_model_train/__init__.py`:

```python
"""Utilities for building and evaluating Qwen3 QLoRA novel executor datasets."""

__all__ = [
    "io_utils",
    "text_utils",
]
```

- [ ] **Step 4: Implement shared I/O helpers**

Create `src/small_model_train/io_utils.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")


def read_text_auto(path: str | Path) -> str:
    file_path = Path(path)
    last_error: UnicodeDecodeError | None = None
    for encoding in ENCODINGS:
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return file_path.read_text()


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    file_path = Path(path)
    if not file_path.exists():
        return rows
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{file_path}:{line_number} is not valid JSON") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
```

- [ ] **Step 5: Implement text utility functions**

Create `src/small_model_train/text_utils.py`:

```python
from __future__ import annotations

import re
from collections import Counter


CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")


def count_chinese_chars(text: str) -> int:
    return len(CHINESE_RE.findall(text))


def normalize_newlines(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def paragraphs(text: str) -> list[str]:
    normalized = normalize_newlines(text)
    if not normalized:
        return []
    return [part.strip() for part in normalized.split("\n\n") if part.strip()]


def paragraph_lengths(text: str) -> list[int]:
    return [count_chinese_chars(part) for part in paragraphs(text)]


def dialogue_ratio(text: str) -> float:
    parts = paragraphs(text)
    if not parts:
        return 0.0
    dialogue_count = sum(1 for part in parts if "“" in part or "”" in part or part.startswith('"'))
    return dialogue_count / len(parts)


def repeated_ngram_ratio(text: str, n: int = 4) -> float:
    chars = CHINESE_RE.findall(text)
    if len(chars) < n:
        return 0.0
    grams = ["".join(chars[index : index + n]) for index in range(len(chars) - n + 1)]
    if not grams:
        return 0.0
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(grams)
```

- [ ] **Step 6: Run the tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_text_utils.py -v
```

Expected: PASS for all tests in `tests/test_text_utils.py`.

- [ ] **Step 7: Commit**

Run:

```powershell
git add pyproject.toml src/small_model_train/__init__.py src/small_model_train/io_utils.py src/small_model_train/text_utils.py tests/test_text_utils.py
git commit -m "feat: add shared text utilities"
```

Expected: commit succeeds.

## Task 2: Raw Text Cleaning And Chapter Splitting

**Files:**
- Create: `src/small_model_train/chapter_splitter.py`
- Create: `scripts/ingest_raw_text.py`
- Create: `scripts/clean_chapters.py`
- Test: `tests/test_chapter_splitter.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_chapter_splitter.py`:

```python
from pathlib import Path

from small_model_train.chapter_splitter import clean_raw_text, split_chapters


def test_clean_raw_text_removes_author_notes_and_separators():
    text = "作者有话说：今天两更\n\n---\n\n第1章 开始\n\n正文"
    assert clean_raw_text(text) == "第1章 开始\n\n正文"


def test_split_chapters_extracts_numbered_chapters():
    text = "第1章 开始\n\n林默回来了。\n\n第2章 加钱\n\n苏小满愣住。"
    chapters = split_chapters(text, work_id="work_001")
    assert [chapter["id"] for chapter in chapters] == [
        "work_001_chapter_0001",
        "work_001_chapter_0002",
    ]
    assert chapters[0]["chapter_title"] == "第1章 开始"
    assert chapters[0]["char_count_zh"] == 6
    assert chapters[0]["quality_tag"] == "A"
    assert chapters[0]["split"] == "train"


def test_split_chapters_uses_single_chapter_for_untitled_text():
    chapters = split_chapters("林默回来了。", work_id="solo")
    assert len(chapters) == 1
    assert chapters[0]["chapter_title"] == "未命名章节"
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_chapter_splitter.py -v
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `small_model_train.chapter_splitter`.

- [ ] **Step 3: Implement chapter cleaning and splitting**

Create `src/small_model_train/chapter_splitter.py`:

```python
from __future__ import annotations

import re

from small_model_train.text_utils import count_chinese_chars, normalize_newlines


CHAPTER_TITLE_RE = re.compile(
    r"^(第[零一二三四五六七八九十百千万0-9]+[章节卷集部].*)$",
    re.MULTILINE,
)

AUTHOR_NOTE_RE = re.compile(r"^作者有话说[:：].*$", re.MULTILINE)
SEPARATOR_RE = re.compile(r"^\s*[-=*_]{3,}\s*$", re.MULTILINE)


def clean_raw_text(text: str) -> str:
    text = AUTHOR_NOTE_RE.sub("", text)
    text = SEPARATOR_RE.sub("", text)
    return normalize_newlines(text)


def split_chapters(
    text: str,
    work_id: str,
    quality_tag: str = "A",
    split: str = "train",
) -> list[dict]:
    cleaned = clean_raw_text(text)
    matches = list(CHAPTER_TITLE_RE.finditer(cleaned))
    if not matches:
        return [
            {
                "id": f"{work_id}_chapter_0001",
                "work_id": work_id,
                "chapter_title": "未命名章节",
                "text": cleaned,
                "char_count_zh": count_chinese_chars(cleaned),
                "quality_tag": quality_tag,
                "split": split,
            }
        ]

    chapters: list[dict] = []
    for index, match in enumerate(matches, start=1):
        start = match.end()
        end = matches[index].start() if index < len(matches) else len(cleaned)
        title = match.group(1).strip()
        body = normalize_newlines(cleaned[start:end])
        chapters.append(
            {
                "id": f"{work_id}_chapter_{index:04d}",
                "work_id": work_id,
                "chapter_title": title,
                "text": body,
                "char_count_zh": count_chinese_chars(body),
                "quality_tag": quality_tag,
                "split": split,
            }
        )
    return chapters
```

- [ ] **Step 4: Implement ingest CLI**

Create `scripts/ingest_raw_text.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from small_model_train.chapter_splitter import split_chapters
from small_model_train.io_utils import read_text_auto, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quality-tag", default="A")
    args = parser.parse_args()

    rows: list[dict] = []
    input_dir = Path(args.input_dir)
    for path in sorted(input_dir.rglob("*")):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        work_id = path.stem.replace(" ", "_")
        rows.extend(
            split_chapters(
                read_text_auto(path),
                work_id=work_id,
                quality_tag=args.quality_tag,
            )
        )
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} chapters to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Implement clean chapters CLI**

Create `scripts/clean_chapters.py`:

```python
from __future__ import annotations

import argparse

from small_model_train.io_utils import read_jsonl, write_jsonl
from small_model_train.text_utils import count_chinese_chars, normalize_newlines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-chars", type=int, default=500)
    parser.add_argument("--max-chars", type=int, default=5000)
    args = parser.parse_args()

    seen_texts: set[str] = set()
    cleaned_rows: list[dict] = []
    for row in read_jsonl(args.input):
        text = normalize_newlines(row.get("text", ""))
        char_count = count_chinese_chars(text)
        if char_count < args.min_chars or char_count > args.max_chars:
            continue
        if text in seen_texts:
            continue
        seen_texts.add(text)
        updated = dict(row)
        updated["text"] = text
        updated["char_count_zh"] = char_count
        cleaned_rows.append(updated)
    write_jsonl(args.output, cleaned_rows)
    print(f"wrote {len(cleaned_rows)} cleaned chapters to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_chapter_splitter.py -v
```

Expected: PASS for all tests in `tests/test_chapter_splitter.py`.

- [ ] **Step 7: Commit**

Run:

```powershell
git add src/small_model_train/chapter_splitter.py scripts/ingest_raw_text.py scripts/clean_chapters.py tests/test_chapter_splitter.py
git commit -m "feat: add raw chapter ingestion"
```

Expected: commit succeeds.

## Task 3: Deterministic Train And Eval Split

**Files:**
- Create: `src/small_model_train/dataset_split.py`
- Create: `scripts/split_train_eval.py`
- Test: `tests/test_dataset_split.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dataset_split.py`:

```python
from small_model_train.dataset_split import split_rows


def test_split_rows_is_deterministic_and_marks_split():
    rows = [{"id": f"chapter_{index:04d}", "text": "正文"} for index in range(10)]
    first = split_rows(rows, eval_count=3, seed=7)
    second = split_rows(rows, eval_count=3, seed=7)
    assert first == second
    assert sum(1 for row in first if row["split"] == "eval") == 3
    assert sum(1 for row in first if row["split"] == "train") == 7


def test_split_rows_keeps_source_fields():
    rows = [{"id": "a", "work_id": "w", "text": "正文", "quality_tag": "A"}]
    split = split_rows(rows, eval_count=1, seed=1)
    assert split[0]["work_id"] == "w"
    assert split[0]["quality_tag"] == "A"
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_dataset_split.py -v
```

Expected: FAIL with `ImportError` for `small_model_train.dataset_split`.

- [ ] **Step 3: Implement deterministic split logic**

Create `src/small_model_train/dataset_split.py`:

```python
from __future__ import annotations

import random


def split_rows(rows: list[dict], eval_count: int, seed: int = 20260617) -> list[dict]:
    if eval_count < 0:
        raise ValueError("eval_count must be >= 0")
    ids = [row["id"] for row in rows]
    rng = random.Random(seed)
    eval_ids = set(rng.sample(ids, k=min(eval_count, len(ids))))
    output: list[dict] = []
    for row in rows:
        updated = dict(row)
        updated["split"] = "eval" if row["id"] in eval_ids else "train"
        output.append(updated)
    return output
```

- [ ] **Step 4: Implement split CLI**

Create `scripts/split_train_eval.py`:

```python
from __future__ import annotations

import argparse

from small_model_train.dataset_split import split_rows
from small_model_train.io_utils import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--eval-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260617)
    args = parser.parse_args()

    rows = split_rows(read_jsonl(args.input), eval_count=args.eval_count, seed=args.seed)
    write_jsonl(args.output, rows)
    write_jsonl(args.eval_output, [row for row in rows if row["split"] == "eval"])
    print(f"wrote {len(rows)} split rows to {args.output}")
```

Append the script entry point to `scripts/split_train_eval.py`:

```python

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_dataset_split.py -v
```

Expected: PASS for all tests in `tests/test_dataset_split.py`.

- [ ] **Step 6: Commit**

Run:

```powershell
git add src/small_model_train/dataset_split.py scripts/split_train_eval.py tests/test_dataset_split.py
git commit -m "feat: add deterministic train eval split"
```

Expected: commit succeeds.

## Task 4: Style Profile And Contract Template

**Files:**
- Create: `src/small_model_train/style_profile.py`
- Create: `scripts/build_style_contract.py`
- Test: `tests/test_style_profile.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_style_profile.py`:

```python
from small_model_train.style_profile import build_style_profile, render_style_contract


def test_build_style_profile_counts_core_metrics():
    rows = [
        {"id": "c1", "text": "林默看了一眼。\n\n“加钱。”"},
        {"id": "c2", "text": "苏小满愣住。\n\n“多少？”"},
    ]
    profile = build_style_profile(rows)
    assert profile["chapter_count"] == 2
    assert profile["avg_chinese_chars"] > 0
    assert profile["avg_dialogue_ratio"] == 0.5


def test_render_style_contract_contains_project_rules():
    contract = render_style_contract({"avg_dialogue_ratio": 0.5, "avg_paragraph_chars": 8})
    assert "只输出正文" in contract
    assert "不要输出提纲" in contract
    assert "对话比例参考" in contract
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_style_profile.py -v
```

Expected: FAIL with `ImportError` for `small_model_train.style_profile`.

- [ ] **Step 3: Implement style profile logic**

Create `src/small_model_train/style_profile.py`:

```python
from __future__ import annotations

from statistics import mean

from small_model_train.text_utils import (
    count_chinese_chars,
    dialogue_ratio,
    paragraph_lengths,
)


def build_style_profile(rows: list[dict]) -> dict:
    texts = [row.get("text", "") for row in rows if row.get("text")]
    paragraph_counts = [length for text in texts for length in paragraph_lengths(text)]
    return {
        "chapter_count": len(texts),
        "avg_chinese_chars": round(mean([count_chinese_chars(text) for text in texts]), 2)
        if texts
        else 0,
        "avg_paragraph_chars": round(mean(paragraph_counts), 2) if paragraph_counts else 0,
        "avg_dialogue_ratio": round(mean([dialogue_ratio(text) for text in texts]), 4)
        if texts
        else 0,
    }


def render_style_contract(profile: dict) -> str:
    dialogue_percent = round(float(profile.get("avg_dialogue_ratio", 0)) * 100, 1)
    avg_paragraph_chars = profile.get("avg_paragraph_chars", 0)
    return "\n".join(
        [
            "【角色】",
            "你是作者的正文执行器，只负责根据章节执行卡写正文。",
            "",
            "【叙述原则】",
            "1. 句子朴素直接，动作承接优先于心理解释。",
            "2. 情绪通过动作、对白和反应表现，不写总结式升华。",
            "3. 主角视角跟随，不随意跳到全知视角。",
            f"4. 段落长度参考：平均约 {avg_paragraph_chars} 个中文汉字。",
            "",
            "【对白原则】",
            f"1. 对话比例参考：约 {dialogue_percent}%。",
            "2. 对话短、准、自然，不用长篇对白解释世界观。",
            "3. 允许省略、打断和反问。",
            "",
            "【禁止风格】",
            "1. 不写空气仿佛凝固了。",
            "2. 不写难以言喻的情绪涌上心头。",
            "3. 不写命运的齿轮开始转动。",
            "4. 不写嘴角勾起一抹弧度。",
            "5. 不写眼神逐渐坚定起来。",
            "",
            "【输出要求】",
            "只输出正文。不要输出提纲、小标题、解释、分析或提示语。",
        ]
    )
```

- [ ] **Step 4: Implement style contract CLI**

Create `scripts/build_style_contract.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from small_model_train.io_utils import read_jsonl
from small_model_train.style_profile import build_style_profile, render_style_contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapters", required=True)
    parser.add_argument("--contract-output", required=True)
    parser.add_argument("--profile-output", required=True)
    args = parser.parse_args()

    rows = [row for row in read_jsonl(args.chapters) if row.get("quality_tag") == "A"]
    profile = build_style_profile(rows)
    Path(args.profile_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.profile_output).write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    Path(args.contract_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.contract_output).write_text(render_style_contract(profile), encoding="utf-8")
    print(f"wrote style profile to {args.profile_output}")
    print(f"wrote style contract to {args.contract_output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_style_profile.py -v
```

Expected: PASS for all tests in `tests/test_style_profile.py`.

- [ ] **Step 6: Commit**

Run:

```powershell
git add src/small_model_train/style_profile.py scripts/build_style_contract.py tests/test_style_profile.py
git commit -m "feat: add style profile generation"
```

Expected: commit succeeds.

## Task 5: SFT Dataset Builder

**Files:**
- Create: `src/small_model_train/sft_builder.py`
- Create: `scripts/build_sft_dataset.py`
- Test: `tests/test_sft_builder.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sft_builder.py`:

```python
from small_model_train.sft_builder import build_sft_rows, render_sft_input


def test_render_sft_input_excludes_source_text():
    card = {
        "style_contract": "风格契约",
        "previous_summary": "上一章摘要",
        "chapter_goal": "本章目标",
        "target_word_count": "2000-2500中文汉字",
        "chapter_structure": [{"step": 1, "name": "开场", "goal": "引出冲突", "estimated_chars": "300-400"}],
        "character_states": [{"name": "林默", "state": "冷静", "speech_style": "短句"}],
        "must_include": ["加钱"],
        "must_not_include": ["提前揭露真相"],
        "ending_hook": "箱子响了一下",
        "source_text": "原文不能进入prompt",
    }
    text = render_sft_input(card)
    assert "原文不能进入prompt" not in text
    assert "只输出正文" in text
    assert "2000-2500中文汉字" in text


def test_build_sft_rows_pairs_cards_with_chapters():
    cards = [{"id": "c1", "style_contract": "契约", "previous_summary": "", "chapter_goal": "", "target_word_count": "2000-2500中文汉字", "chapter_structure": [], "character_states": [], "must_include": [], "must_not_include": [], "ending_hook": ""}]
    chapters = [{"id": "c1", "text": "正文"}]
    rows = build_sft_rows(cards, chapters)
    assert rows[0]["instruction"].startswith("你是作者的正文执行器")
    assert rows[0]["output"] == "正文"
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_sft_builder.py -v
```

Expected: FAIL with `ImportError` for `small_model_train.sft_builder`.

- [ ] **Step 3: Implement SFT builder logic**

Create `src/small_model_train/sft_builder.py`:

```python
from __future__ import annotations


INSTRUCTION = "你是作者的正文执行器。请严格根据章节执行卡，写出符合作者风格的一章正文。"


def _format_list(title: str, values: list[str]) -> str:
    body = "\n".join(f"- {value}" for value in values) if values else "- 无"
    return f"【{title}】\n{body}"


def _format_structure(items: list[dict]) -> str:
    if not items:
        return "【章节结构】\n- 无"
    lines = ["【章节结构】"]
    for item in items:
        step = item.get("step", "")
        name = item.get("name", "")
        goal = item.get("goal", "")
        chars = item.get("estimated_chars", "")
        lines.append(f"- {step}. {name}：{goal}（建议 {chars}）")
    return "\n".join(lines)


def _format_characters(items: list[dict]) -> str:
    if not items:
        return "【人物状态】\n- 无"
    lines = ["【人物状态】"]
    for item in items:
        lines.append(
            f"- {item.get('name', '')}：{item.get('state', '')}；说话方式：{item.get('speech_style', '')}"
        )
    return "\n".join(lines)


def render_sft_input(card: dict) -> str:
    sections = [
        "【风格契约】",
        card.get("style_contract", ""),
        "【前情摘要】",
        card.get("previous_summary", ""),
        "【本章目标】",
        card.get("chapter_goal", ""),
        _format_structure(card.get("chapter_structure", [])),
        _format_characters(card.get("character_states", [])),
        _format_list("必须出现", card.get("must_include", [])),
        _format_list("禁止事项", card.get("must_not_include", [])),
        "【章末钩子】",
        card.get("ending_hook", ""),
        "【目标字数】",
        card.get("target_word_count", "2000-2500中文汉字"),
        "【输出要求】",
        "只输出正文，不输出提纲、小标题、解释、分析或提示语。",
    ]
    return "\n".join(section for section in sections if section is not None)


def build_sft_rows(cards: list[dict], chapters: list[dict]) -> list[dict]:
    chapter_by_id = {chapter["id"]: chapter for chapter in chapters}
    rows: list[dict] = []
    for card in cards:
        chapter = chapter_by_id.get(card["id"])
        if not chapter:
            continue
        rows.append(
            {
                "instruction": INSTRUCTION,
                "input": render_sft_input(card),
                "output": chapter.get("text", ""),
            }
        )
    return rows
```

- [ ] **Step 4: Implement SFT builder CLI**

Create `scripts/build_sft_dataset.py`:

```python
from __future__ import annotations

import argparse

from small_model_train.io_utils import read_jsonl, write_jsonl
from small_model_train.sft_builder import build_sft_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", required=True)
    parser.add_argument("--chapters", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = build_sft_rows(read_jsonl(args.cards), read_jsonl(args.chapters))
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} SFT rows to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_sft_builder.py -v
```

Expected: PASS for all tests in `tests/test_sft_builder.py`.

- [ ] **Step 6: Commit**

Run:

```powershell
git add src/small_model_train/sft_builder.py scripts/build_sft_dataset.py tests/test_sft_builder.py
git commit -m "feat: add sft dataset builder"
```

Expected: commit succeeds.

## Task 6: Rule Scoring And AI Trace Detection

**Files:**
- Create: `src/small_model_train/scoring.py`
- Create: `scripts/detect_ai_trace.py`
- Create: `scripts/score_outputs.py`
- Test: `tests/test_scoring.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scoring.py`:

```python
from small_model_train.scoring import detect_ai_trace, score_output


def test_detect_ai_trace_counts_known_phrases():
    result = detect_ai_trace("空气仿佛凝固了，他心中涌起一股复杂的情绪。")
    assert result["count"] == 2
    assert "空气仿佛凝固了" in result["matches"]


def test_score_output_applies_hard_gates_and_failure_types():
    card = {
        "must_include": ["加钱"],
        "must_not_include": ["真相是他父亲"],
    }
    output = "林默说加钱。真相是他父亲。"
    score = score_output("case1", card, output)
    assert score["hard_gate_pass"] is False
    assert "forbidden_violation" in score["failure_types"]
    assert score["must_include_coverage"] == 1.0


def test_score_output_marks_length_short():
    score = score_output("case2", {"must_include": [], "must_not_include": []}, "太短。")
    assert "length_short" in score["failure_types"]
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_scoring.py -v
```

Expected: FAIL with `ImportError` for `small_model_train.scoring`.

- [ ] **Step 3: Implement scoring logic**

Create `src/small_model_train/scoring.py`:

```python
from __future__ import annotations

from small_model_train.text_utils import count_chinese_chars, repeated_ngram_ratio


AI_TRACE_PHRASES = [
    "空气仿佛凝固了",
    "难以言喻的情绪",
    "心中涌起一股复杂的情绪",
    "命运的齿轮开始转动",
    "眼神逐渐坚定起来",
    "嘴角勾起一抹弧度",
    "这一刻，他终于明白",
    "所有人都意识到",
    "一种前所未有的感觉",
]


def detect_ai_trace(text: str) -> dict:
    matches = [phrase for phrase in AI_TRACE_PHRASES if phrase in text]
    return {"count": len(matches), "matches": matches}


def _coverage(required: list[str], text: str) -> float:
    if not required:
        return 1.0
    hits = sum(1 for item in required if item and item in text)
    return hits / len(required)


def score_output(sample_id: str, card: dict, output: str) -> dict:
    char_count = count_chinese_chars(output)
    ai_trace = detect_ai_trace(output)
    repetition = repeated_ngram_ratio(output, n=4)
    must_include = card.get("must_include", [])
    must_not_include = card.get("must_not_include", [])
    include_coverage = _coverage(must_include, output)
    forbidden_hits = [item for item in must_not_include if item and item in output]

    failure_types: list[str] = []
    if char_count < 2000:
        failure_types.append("length_short")
    if char_count > 2500:
        failure_types.append("length_long")
    if any(marker in output for marker in ["【", "】", "章节结构", "以下是正文"]):
        failure_types.append("outline_leak")
    if include_coverage < 1.0:
        failure_types.append("must_include_missing")
    if forbidden_hits:
        failure_types.append("forbidden_violation")
    if repetition > 0.1:
        failure_types.append("repetition")
    if ai_trace["count"] > 0:
        failure_types.append("ai_trace")

    hard_gate_failures = {
        "length_short",
        "length_long",
        "outline_leak",
        "forbidden_violation",
        "repetition",
    }
    hard_gate_pass = not any(item in hard_gate_failures for item in failure_types)

    return {
        "id": sample_id,
        "char_count_zh": char_count,
        "hard_gate_pass": hard_gate_pass,
        "must_include_coverage": round(include_coverage, 4),
        "forbidden_hits": forbidden_hits,
        "ai_trace_count": ai_trace["count"],
        "ai_trace_matches": ai_trace["matches"],
        "repeated_ngram_ratio": round(repetition, 4),
        "failure_types": failure_types,
    }
```

- [ ] **Step 4: Implement AI trace CLI**

Create `scripts/detect_ai_trace.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from small_model_train.scoring import detect_ai_trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", required=True)
    args = parser.parse_args()
    text = Path(args.text_file).read_text(encoding="utf-8")
    print(json.dumps(detect_ai_trace(text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Implement score outputs CLI**

Create `scripts/score_outputs.py`:

```python
from __future__ import annotations

import argparse

from small_model_train.io_utils import read_jsonl, write_jsonl
from small_model_train.scoring import score_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", required=True)
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cards = {row["id"]: row for row in read_jsonl(args.cards)}
    scores = []
    for row in read_jsonl(args.outputs):
        sample_id = row["id"]
        text = row.get("output", row.get("text", ""))
        scores.append(score_output(sample_id, cards.get(sample_id, {}), text))
    write_jsonl(args.output, scores)
    print(f"wrote {len(scores)} scores to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_scoring.py -v
```

Expected: PASS for all tests in `tests/test_scoring.py`.

- [ ] **Step 7: Commit**

Run:

```powershell
git add src/small_model_train/scoring.py scripts/detect_ai_trace.py scripts/score_outputs.py tests/test_scoring.py
git commit -m "feat: add rule scoring"
```

Expected: commit succeeds.

## Task 7: Preference Candidate Builder

**Files:**
- Create: `src/small_model_train/preference_builder.py`
- Create: `scripts/build_preference_dataset.py`
- Test: `tests/test_preference_builder.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_preference_builder.py`:

```python
from small_model_train.preference_builder import build_preference_candidates


def test_build_preference_candidates_uses_failed_scores():
    cards = [{"id": "c1", "prompt": "卡", "style_contract": "契约"}]
    outputs = [{"id": "c1", "output": "坏正文"}]
    scores = [{"id": "c1", "hard_gate_pass": False, "failure_types": ["ai_trace"]}]
    rows = build_preference_candidates(cards, outputs, scores)
    assert rows == [
        {
            "id": "c1",
            "prompt": "卡",
            "rejected": "坏正文",
            "reject_type": "ai_trace",
            "chosen": "",
            "source": "failed_eval",
        }
    ]
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_preference_builder.py -v
```

Expected: FAIL with `ImportError` for `small_model_train.preference_builder`.

- [ ] **Step 3: Implement preference candidate builder**

Create `src/small_model_train/preference_builder.py`:

```python
from __future__ import annotations

from small_model_train.sft_builder import render_sft_input


def build_preference_candidates(
    cards: list[dict],
    outputs: list[dict],
    scores: list[dict],
) -> list[dict]:
    cards_by_id = {row["id"]: row for row in cards}
    outputs_by_id = {row["id"]: row for row in outputs}
    rows: list[dict] = []
    for score in scores:
        if score.get("hard_gate_pass", True):
            continue
        sample_id = score["id"]
        card = cards_by_id.get(sample_id, {})
        output = outputs_by_id.get(sample_id, {})
        failure_types = score.get("failure_types", [])
        reject_type = failure_types[0] if failure_types else "unknown"
        rows.append(
            {
                "id": sample_id,
                "prompt": card.get("prompt") or render_sft_input(card),
                "rejected": output.get("output", output.get("text", "")),
                "reject_type": reject_type,
                "chosen": "",
                "source": "failed_eval",
            }
        )
    return rows
```

- [ ] **Step 4: Implement preference CLI**

Create `scripts/build_preference_dataset.py`:

```python
from __future__ import annotations

import argparse

from small_model_train.io_utils import read_jsonl, write_jsonl
from small_model_train.preference_builder import build_preference_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", required=True)
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = build_preference_candidates(
        read_jsonl(args.cards),
        read_jsonl(args.outputs),
        read_jsonl(args.scores),
    )
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} preference candidates to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_preference_builder.py -v
```

Expected: PASS for all tests in `tests/test_preference_builder.py`.

- [ ] **Step 6: Commit**

Run:

```powershell
git add src/small_model_train/preference_builder.py scripts/build_preference_dataset.py tests/test_preference_builder.py
git commit -m "feat: add preference candidate builder"
```

Expected: commit succeeds.

## Task 8: Evaluation Report Generator

**Files:**
- Create: `src/small_model_train/reporting.py`
- Create: `scripts/evaluate_outputs.py`
- Test: `tests/test_reporting.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reporting.py`:

```python
from small_model_train.reporting import build_markdown_report, summarize_scores


def test_summarize_scores_counts_hard_gate_and_failures():
    scores = [
        {"id": "a", "hard_gate_pass": True, "failure_types": [], "char_count_zh": 2200},
        {"id": "b", "hard_gate_pass": False, "failure_types": ["length_short"], "char_count_zh": 900},
    ]
    summary = summarize_scores(scores)
    assert summary["sample_count"] == 2
    assert summary["hard_gate_pass_rate"] == 0.5
    assert summary["failure_counts"]["length_short"] == 1


def test_build_markdown_report_contains_decision():
    report = build_markdown_report(
        title="SFT v1",
        scores=[{"id": "a", "hard_gate_pass": True, "failure_types": [], "char_count_zh": 2200}],
        config_snapshot={"model": "qwen3"},
    )
    assert "# SFT v1" in report
    assert "是否进入下一阶段" in report
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_reporting.py -v
```

Expected: FAIL with `ImportError` for `small_model_train.reporting`.

- [ ] **Step 3: Implement reporting logic**

Create `src/small_model_train/reporting.py`:

```python
from __future__ import annotations

from collections import Counter
from statistics import mean


def summarize_scores(scores: list[dict]) -> dict:
    sample_count = len(scores)
    hard_gate_passes = sum(1 for score in scores if score.get("hard_gate_pass"))
    failure_counts = Counter(
        failure
        for score in scores
        for failure in score.get("failure_types", [])
    )
    avg_chars = mean([score.get("char_count_zh", 0) for score in scores]) if scores else 0
    return {
        "sample_count": sample_count,
        "hard_gate_pass_rate": round(hard_gate_passes / sample_count, 4)
        if sample_count
        else 0,
        "avg_chinese_chars": round(avg_chars, 2),
        "failure_counts": dict(failure_counts),
    }


def build_markdown_report(
    title: str,
    scores: list[dict],
    config_snapshot: dict | None = None,
) -> str:
    summary = summarize_scores(scores)
    worst = sorted(
        scores,
        key=lambda score: (score.get("hard_gate_pass", False), score.get("char_count_zh", 0)),
    )[:10]
    lines = [
        f"# {title}",
        "",
        "## 配置快照",
        "```json",
        str(config_snapshot or {}).replace("'", '"'),
        "```",
        "",
        "## 汇总",
        f"- 样本数：{summary['sample_count']}",
        f"- 硬门槛通过率：{summary['hard_gate_pass_rate']}",
        f"- 平均中文汉字数：{summary['avg_chinese_chars']}",
        "",
        "## 失败类型分布",
    ]
    for failure, count in sorted(summary["failure_counts"].items()):
        lines.append(f"- {failure}: {count}")
    lines.extend(["", "## 最差样本", ""])
    for score in worst:
        lines.append(f"- {score.get('id')}: {', '.join(score.get('failure_types', [])) or 'pass'}")
    decision = "可以进入下一阶段" if summary["hard_gate_pass_rate"] >= 0.65 else "继续修数据和配置"
    lines.extend(["", "## 是否进入下一阶段", decision])
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Implement evaluation CLI**

Create `scripts/evaluate_outputs.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from small_model_train.io_utils import read_jsonl
from small_model_train.reporting import build_markdown_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--title", default="Evaluation Report")
    parser.add_argument("--config-json", default="{}")
    args = parser.parse_args()

    config_snapshot = json.loads(args.config_json)
    report = build_markdown_report(args.title, read_jsonl(args.scores), config_snapshot)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report, encoding="utf-8")
    print(f"wrote report to {args.report}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_reporting.py -v
```

Expected: PASS for all tests in `tests/test_reporting.py`.

- [ ] **Step 6: Commit**

Run:

```powershell
git add src/small_model_train/reporting.py scripts/evaluate_outputs.py tests/test_reporting.py
git commit -m "feat: add evaluation reporting"
```

Expected: commit succeeds.

## Task 9: QLoRA And Inference Configs

**Files:**
- Create: `configs/sft_qlora_qwen3_4b.yaml`
- Create: `configs/infer_eval_qwen3_4b.yaml`
- Modify: `README.md`

- [ ] **Step 1: Create QLoRA SFT config**

Create `configs/sft_qlora_qwen3_4b.yaml`:

```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
template: qwen3
stage: sft
do_train: true
finetuning_type: lora
quantization_bit: 4
quantization_method: bitsandbytes

lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: all

dataset: sft_chapter_v1
dataset_dir: data_sft
cutoff_len: 8192
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 3.0e-5
num_train_epochs: 2

bf16: true
gradient_checkpointing: true
logging_steps: 10
save_steps: 200
save_total_limit: 3
output_dir: outputs/sft_v1
```

- [ ] **Step 2: Create fixed inference config**

Create `configs/infer_eval_qwen3_4b.yaml`:

```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
adapter_name_or_path: outputs/sft_v1
template: qwen3
max_new_tokens: 5120
temperature: 0.7
top_p: 0.8
top_k: 20
repetition_penalty: 1.05
```

- [ ] **Step 3: Create README with stage-one commands**

Create `README.md`:

````markdown
# Small Model Train

This project builds a Qwen3-4B-Instruct-2507 QLoRA pipeline for a Chinese whole-chapter novel executor.

## Stage 1 Pipeline

```powershell
python scripts/ingest_raw_text.py --input-dir data_raw/novels --output data_clean/chapters_raw.jsonl
python scripts/clean_chapters.py --input data_clean/chapters_raw.jsonl --output data_clean/chapters.jsonl --min-chars 500 --max-chars 5000
python scripts/split_train_eval.py --input data_clean/chapters.jsonl --output data_clean/chapters_split.jsonl --eval-output data_cards/eval_cards_50.jsonl --eval-count 50
python scripts/build_style_contract.py --chapters data_clean/chapters_split.jsonl --contract-output style_contract.md --profile-output style_profile.json
python scripts/build_sft_dataset.py --cards data_cards/chapter_cards.jsonl --chapters data_clean/chapters_split.jsonl --output data_sft/sft_chapter_v1.jsonl
```

## Evaluation

```powershell
python scripts/score_outputs.py --cards data_cards/eval_cards_50.jsonl --outputs outputs/baseline/generated.jsonl --output outputs/baseline/metrics.jsonl
python scripts/evaluate_outputs.py --scores outputs/baseline/metrics.jsonl --report reports/baseline_report.md --title "Baseline Report"
```

## Training Config

Use `configs/sft_qlora_qwen3_4b.yaml` as the first QLoRA SFT configuration. Start with a 100-sample smoke run before a full run on 500-1000 samples.
````

- [ ] **Step 4: Run all tests**

Run:

```powershell
python -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

Run:

```powershell
git add configs/sft_qlora_qwen3_4b.yaml configs/infer_eval_qwen3_4b.yaml README.md
git commit -m "docs: add qlora configs and usage"
```

Expected: commit succeeds.

## Task 10: Pipeline Smoke Test With Synthetic Data

**Files:**
- Create: `tests/test_pipeline_smoke.py`

- [ ] **Step 1: Write the smoke test**

Create `tests/test_pipeline_smoke.py`:

```python
from small_model_train.chapter_splitter import split_chapters
from small_model_train.dataset_split import split_rows
from small_model_train.preference_builder import build_preference_candidates
from small_model_train.reporting import build_markdown_report
from small_model_train.scoring import score_output
from small_model_train.sft_builder import build_sft_rows


def test_stage_one_pipeline_smoke():
    raw = "第1章 开始\n\n" + "林默说加钱。" * 260
    chapters = split_chapters(raw, work_id="work")
    split = split_rows(chapters, eval_count=1, seed=1)
    card = {
        "id": split[0]["id"],
        "style_contract": "只输出正文。",
        "previous_summary": "上一章结束。",
        "chapter_goal": "完成交易。",
        "target_word_count": "2000-2500中文汉字",
        "chapter_structure": [],
        "character_states": [],
        "must_include": ["加钱"],
        "must_not_include": ["真相"],
        "ending_hook": "箱子响了一下。",
    }
    sft_rows = build_sft_rows([card], split)
    assert len(sft_rows) == 1
    score = score_output(split[0]["id"], card, split[0]["text"])
    candidates = build_preference_candidates([card], [{"id": split[0]["id"], "output": split[0]["text"]}], [score])
    report = build_markdown_report("Smoke", [score], {"model": "qwen3"})
    assert "Smoke" in report
    assert isinstance(candidates, list)
```

- [ ] **Step 2: Run the smoke test**

Run:

```powershell
python -m pytest tests/test_pipeline_smoke.py -v
```

Expected: PASS for the smoke test.

- [ ] **Step 3: Run the full test suite**

Run:

```powershell
python -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

Run:

```powershell
git add tests/test_pipeline_smoke.py
git commit -m "test: add stage one pipeline smoke test"
```

Expected: commit succeeds.

## Self-Review

Spec coverage:

- Data cleaning and chapter splitting are covered by Tasks 1 and 2.
- Train/eval split is covered by Task 3.
- Style contract and profile are covered by Task 4.
- SFT dataset construction is covered by Task 5.
- Preference candidate construction is covered by Task 7.
- Rule scoring and AI trace detection are covered by Task 6.
- Evaluation reports are covered by Task 8.
- QLoRA and inference configs are covered by Task 9.
- End-to-end stage-one validation is covered by Task 10.

Type consistency:

- All JSONL rows use `id` as the join key.
- Generated outputs use `output` as the preferred text field and accept `text` as a fallback.
- Cards use `must_include`, `must_not_include`, `chapter_structure`, and `character_states` consistently.

Execution boundary:

- This plan prepares the QLoRA SFT configuration and data pipeline.
- This plan does not run full training.
- This plan does not implement local model inference yet; generated eval outputs are accepted as JSONL so inference can be wired after the first pipeline is stable.
