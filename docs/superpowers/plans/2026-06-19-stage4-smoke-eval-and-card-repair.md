# Stage 4 Smoke Eval And Card Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the first real smoke training result into a repeatable Stage 4 loop: repair chapter cards, re-run 50-sample smoke training with fixed prompts, run fixed eval inference, score outputs, and decide whether to expand to 100 or 500 samples.

**Architecture:** Stage 4 should treat data quality as the training entrance gate. Chapter cards are generated and validated by repo code, SFT and LLaMA-Factory metadata are rebuilt deterministically, then smoke training and eval are run from recorded commands with reports at every boundary.

**Tech Stack:** Python 3.11, pytest, JSONL, LLaMA-Factory 0.9.5, Qwen3-4B-Instruct-2507 local model, existing `small_model_train` modules and scripts.

---

## Current State

- Stage 3 readiness reports `ready_for_stage4_smoke_training`.
- A first real 50-row smoke run completed successfully and produced `outputs/sft_smoke/adapter_model.safetensors`.
- `reports/stage4_smoke_training_report.md` records `train_loss = 4.0140790939331055`, `total_steps = 4`, `exit_code = 0`.
- Adapter check passed for `outputs/sft_smoke`.
- A data quality issue was found: generated chapter card `chapter_structure` items used `beat`, while `sft_builder._format_structure()` expects `step` and `name`, rendering structure lines as `- . ：...`.
- `data_sft/dataset_info.json` was required by LLaMA-Factory and exists as a local generated artifact, but there is no repo script that creates it deterministically yet.

## File Map

- Modify: `src/small_model_train/sft_builder.py`
  - Keep SFT prompt rendering strict enough to catch malformed structure rows before training.
- Create: `src/small_model_train/chapter_cards.py`
  - Generate draft chapter cards from `chapters_split.jsonl`.
  - Normalize `chapter_structure` items into the canonical `step/name/goal/estimated_chars` schema.
  - Validate card fields before SFT construction.
- Create: `scripts/build_chapter_cards.py`
  - CLI for building `data_cards/chapter_cards.jsonl` from real train/A chapters.
- Modify: `scripts/build_sft_dataset.py`
  - Add optional `--dataset-info-output data_sft/dataset_info.json` to write LLaMA-Factory metadata.
- Modify: `src/small_model_train/stage3_data_readiness.py`
  - Treat malformed chapter structure as a blocking card schema error.
- Tests: `tests/test_chapter_cards.py`, `tests/test_sft_builder.py`, `tests/test_stage3_data_readiness.py`
  - Cover canonical card generation, structure rendering, malformed structure rejection, dataset metadata writing, and readiness blocking.
- Modify: `README.md`
  - Add Stage 4 command sequence with smoke eval commands and card repair gate.
- Local generated artifacts:
  - `data_cards/chapter_cards.jsonl`
  - `data_sft/sft_chapter_v1.jsonl`
  - `data_sft/dataset_info.json`
  - `outputs/sft_smoke/*`
  - `outputs/sft_smoke/generated.jsonl`
  - `outputs/sft_smoke/metrics.jsonl`
  - `reports/sft_smoke_eval_report.md`

---

## Task 1: Add Chapter Card Generation And Validation

**Files:**
- Create: `src/small_model_train/chapter_cards.py`
- Create: `tests/test_chapter_cards.py`

- [ ] **Step 1: Write failing tests for canonical structure generation**

Create `tests/test_chapter_cards.py`:

```python
from __future__ import annotations

import pytest

from small_model_train.chapter_cards import (
    build_draft_chapter_cards,
    normalize_chapter_structure,
    validate_chapter_card,
)


def _chapter(sample_id: str = "chapter-1", char_count: int = 2800) -> dict:
    return {
        "id": sample_id,
        "text": "正文" * 1400,
        "split": "train",
        "quality_tag": "A",
        "char_count_zh": char_count,
        "chapter_title": "第一章 测试",
    }


def test_normalize_chapter_structure_writes_step_and_name():
    items = [
        {"beat": "承接", "goal": "承接上一章压力。", "estimated_chars": 400},
        {"name": "转折", "goal": "完成本章选择。", "estimated_chars": "500"},
    ]

    normalized = normalize_chapter_structure(items)

    assert normalized == [
        {"step": 1, "name": "承接", "goal": "承接上一章压力。", "estimated_chars": "400"},
        {"step": 2, "name": "转折", "goal": "完成本章选择。", "estimated_chars": "500"},
    ]


def test_validate_chapter_card_rejects_empty_structure_labels():
    card = {
        "id": "chapter-1",
        "style_contract": "只输出正文。",
        "previous_summary": "前情摘要。",
        "chapter_goal": "推进冲突。",
        "chapter_structure": [{"beat": "承接", "goal": "推进。", "estimated_chars": 400}],
        "character_states": [{"name": "核心视角人物", "state": "谨慎", "speech_style": "短句"}],
        "must_include": ["清楚的开场状态"],
        "must_not_include": ["输出提纲或小标题"],
        "ending_hook": "留下下一步压力。",
        "target_word_count": "2500-3000中文汉字",
        "source_text": "正文" * 100,
    }

    with pytest.raises(ValueError, match="chapter_structure\\[0\\].step"):
        validate_chapter_card(card)


def test_build_draft_chapter_cards_uses_train_a_chapters_only():
    chapters = [
        _chapter("train-a", 2800),
        {**_chapter("eval-a", 2800), "id": "eval-a", "split": "eval"},
        {**_chapter("train-b", 2800), "id": "train-b", "quality_tag": "B"},
    ]

    cards = build_draft_chapter_cards(chapters, count=1, min_chars=2000, max_chars=3000)

    assert [card["id"] for card in cards] == ["train-a"]
    assert cards[0]["chapter_structure"][0]["step"] == 1
    assert cards[0]["chapter_structure"][0]["name"] == "承接"
    assert cards[0]["source_text"].startswith("正文")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_chapter_cards.py -q
```

Expected: FAIL because `small_model_train.chapter_cards` does not exist.

- [ ] **Step 3: Implement chapter card module**

Create `src/small_model_train/chapter_cards.py`:

```python
from __future__ import annotations

from typing import Any


STYLE_CONTRACT = (
    "只输出正文。以动作、场景反应和短对白推进章节，不输出提纲、小标题、分析或创作说明。"
    "保持节奏清晰，承接上一章压力，在本章结尾留下自然余波或下一步悬念。"
)

STRUCTURE_TEMPLATE = [
    ("承接", "承接上一章留下的压力，交代本章开场状态。", 0.15),
    ("入局", "让核心视角人物进入新的场景、关系变化或任务压力。", 0.20),
    ("试探", "通过行动和短对白推进矛盾，不急于解释设定。", 0.20),
    ("加压", "让局势出现阻碍、误判、代价或反向压力。", 0.20),
    ("转折", "完成本章最关键的信息变化、人物选择或关系推进。", 0.15),
    ("收束", "用余波、未完成动作或新压力把读者带向下一章。", 0.10),
]

REQUIRED_CARD_FIELDS = (
    "id",
    "style_contract",
    "previous_summary",
    "chapter_goal",
    "chapter_structure",
    "character_states",
    "must_include",
    "must_not_include",
    "ending_hook",
    "target_word_count",
)


def normalize_chapter_structure(items: list[dict[str, Any]]) -> list[dict[str, str | int]]:
    normalized = []
    for index, item in enumerate(items, start=1):
        name = str(item.get("name") or item.get("beat") or "").strip()
        goal = str(item.get("goal") or "").strip()
        estimated_chars = str(item.get("estimated_chars") or "").strip()
        normalized.append(
            {
                "step": int(item.get("step") or index),
                "name": name,
                "goal": goal,
                "estimated_chars": estimated_chars,
            }
        )
    return normalized


def validate_chapter_card(card: dict[str, Any]) -> None:
    for field in REQUIRED_CARD_FIELDS:
        if field not in card:
            raise ValueError(f"missing required field: {field}")
        if card[field] in ("", None, []):
            raise ValueError(f"empty required field: {field}")

    for index, item in enumerate(card["chapter_structure"]):
        if not item.get("step"):
            raise ValueError(f"chapter_structure[{index}].step is required")
        if not item.get("name"):
            raise ValueError(f"chapter_structure[{index}].name is required")
        if not item.get("goal"):
            raise ValueError(f"chapter_structure[{index}].goal is required")
        if not item.get("estimated_chars"):
            raise ValueError(f"chapter_structure[{index}].estimated_chars is required")

    for index, item in enumerate(card["character_states"]):
        if not item.get("name"):
            raise ValueError(f"character_states[{index}].name is required")
        if not item.get("state"):
            raise ValueError(f"character_states[{index}].state is required")
        if not item.get("speech_style"):
            raise ValueError(f"character_states[{index}].speech_style is required")


def build_draft_chapter_cards(
    chapters: list[dict[str, Any]],
    count: int,
    min_chars: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    candidates = [
        chapter
        for chapter in chapters
        if chapter.get("split") == "train"
        and chapter.get("quality_tag") == "A"
        and min_chars <= int(chapter.get("char_count_zh", 0)) <= max_chars
    ]
    cards = [_build_card(chapter) for chapter in candidates[:count]]
    for card in cards:
        validate_chapter_card(card)
    return cards


def _build_card(chapter: dict[str, Any]) -> dict[str, Any]:
    char_count = int(chapter.get("char_count_zh", 0))
    card = {
        "id": chapter["id"],
        "card_version": "draft-v2",
        "source_title": chapter.get("chapter_title", ""),
        "style_contract": STYLE_CONTRACT,
        "previous_summary": "上一章事件留下新的压力，本章从既有关系、目标和未解决冲突继续推进。",
        "chapter_goal": "围绕本章既定剧情推进主要冲突，让核心人物在观察、试探、行动和选择中完成阶段性变化。",
        "chapter_structure": _structure_for(char_count),
        "character_states": [
            {
                "name": "核心视角人物",
                "state": "带着上一章留下的压力进入本章，在观察、试探和行动中推进目标。",
                "speech_style": "短句优先，少解释，多用反应、动作和停顿承接情绪。",
            },
            {
                "name": "关键对手或阻力方",
                "state": "制造误判、压力或选择代价，迫使核心人物调整行动。",
                "speech_style": "保持信息克制，避免直接替作者解释设定。",
            },
        ],
        "must_include": ["清楚的开场状态", "可感知的中段压力升级", "章末余波或下一步悬念"],
        "must_not_include": ["输出提纲或小标题", "跳出正文解释创作意图", "大段复述世界观设定", "直接照抄原文句段"],
        "ending_hook": "在本章余波中留下下一章继续推进的压力、疑问或动作方向。",
        "target_word_count": _target_word_count(char_count),
        "source_text": chapter.get("text", ""),
    }
    card["chapter_structure"] = normalize_chapter_structure(card["chapter_structure"])
    return card


def _structure_for(char_count: int) -> list[dict[str, str | int]]:
    return [
        {
            "step": index,
            "name": name,
            "goal": goal,
            "estimated_chars": str(int(round(char_count * ratio / 50) * 50)),
        }
        for index, (name, goal, ratio) in enumerate(STRUCTURE_TEMPLATE, start=1)
    ]


def _target_word_count(char_count: int) -> str:
    if char_count <= 2500:
        return "2000-2500中文汉字"
    if char_count <= 3000:
        return "2500-3000中文汉字"
    return "3000-4000中文汉字"
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
python -m pytest tests/test_chapter_cards.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/small_model_train/chapter_cards.py tests/test_chapter_cards.py
git commit -m "feat: add chapter card generation and validation"
```

---

## Task 2: Make SFT Rendering Reject Malformed Structure Rows

**Files:**
- Modify: `src/small_model_train/sft_builder.py`
- Modify: `tests/test_sft_builder.py`

- [ ] **Step 1: Write failing tests for empty structure labels**

Append to `tests/test_sft_builder.py`:

```python
def test_render_sft_input_rejects_structure_without_step_or_name():
    card = {
        "style_contract": "契约",
        "previous_summary": "前情",
        "chapter_goal": "目标",
        "target_word_count": "2000-2500中文汉字",
        "chapter_structure": [{"beat": "承接", "goal": "推进", "estimated_chars": "300"}],
        "character_states": [{"name": "林默", "state": "冷静", "speech_style": "短句"}],
        "must_include": ["加钱"],
        "must_not_include": ["提前揭露真相"],
        "ending_hook": "箱子响了一下",
        "source_text": "",
    }

    with pytest.raises(ValueError, match="chapter_structure\\[0\\].step"):
        render_sft_input(card)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_sft_builder.py::test_render_sft_input_rejects_structure_without_step_or_name -q
```

Expected: FAIL because `_format_structure()` currently renders empty labels instead of rejecting malformed structure rows.

- [ ] **Step 3: Implement strict structure formatting**

Modify `_format_structure()` in `src/small_model_train/sft_builder.py`:

```python
def _format_structure(items: list[dict]) -> str:
    if not items:
        return "【章节结构】\n- 无"
    lines = ["【章节结构】"]
    for index, item in enumerate(items):
        step = item.get("step", "")
        name = item.get("name", "")
        goal = item.get("goal", "")
        chars = item.get("estimated_chars", "")
        if not step:
            raise ValueError(f"chapter_structure[{index}].step is required")
        if not name:
            raise ValueError(f"chapter_structure[{index}].name is required")
        if not goal:
            raise ValueError(f"chapter_structure[{index}].goal is required")
        if not chars:
            raise ValueError(f"chapter_structure[{index}].estimated_chars is required")
        lines.append(f"- {step}. {name}：{goal}（建议 {chars}）")
    return "\n".join(lines)
```

- [ ] **Step 4: Run targeted tests**

Run:

```powershell
python -m pytest tests/test_sft_builder.py tests/test_chapter_cards.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/small_model_train/sft_builder.py tests/test_sft_builder.py
git commit -m "fix: reject malformed chapter structure prompts"
```

---

## Task 3: Add CLI To Build Chapter Cards From Real Train Chapters

**Files:**
- Create: `scripts/build_chapter_cards.py`
- Modify: `tests/test_chapter_cards.py`

- [ ] **Step 1: Add CLI test**

Append to `tests/test_chapter_cards.py`:

```python
import subprocess
import sys

from small_model_train.io_utils import read_jsonl, write_jsonl


def test_build_chapter_cards_cli_writes_fixed_cards(tmp_path):
    chapters_path = tmp_path / "chapters_split.jsonl"
    output_path = tmp_path / "chapter_cards.jsonl"
    write_jsonl(chapters_path, [_chapter("train-a", 2800)])

    subprocess.run(
        [
            sys.executable,
            "scripts/build_chapter_cards.py",
            "--chapters",
            str(chapters_path),
            "--output",
            str(output_path),
            "--count",
            "1",
            "--min-chars",
            "2000",
            "--max-chars",
            "3000",
        ],
        check=True,
    )

    rows = read_jsonl(output_path)
    assert len(rows) == 1
    assert rows[0]["chapter_structure"][0]["step"] == 1
    assert rows[0]["chapter_structure"][0]["name"] == "承接"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_chapter_cards.py::test_build_chapter_cards_cli_writes_fixed_cards -q
```

Expected: FAIL because `scripts/build_chapter_cards.py` does not exist.

- [ ] **Step 3: Implement CLI**

Create `scripts/build_chapter_cards.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.chapter_cards import build_draft_chapter_cards
from small_model_train.io_utils import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapters", default="data_clean/chapters_split.jsonl")
    parser.add_argument("--output", default="data_cards/chapter_cards.jsonl")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--min-chars", type=int, default=2000)
    parser.add_argument("--max-chars", type=int, default=3000)
    args = parser.parse_args()

    cards = build_draft_chapter_cards(
        read_jsonl(args.chapters),
        count=args.count,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )
    write_jsonl(args.output, cards)
    print(f"wrote {len(cards)} chapter cards to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run:

```powershell
python -m pytest tests/test_chapter_cards.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add scripts/build_chapter_cards.py tests/test_chapter_cards.py
git commit -m "feat: add chapter card builder cli"
```

---

## Task 4: Generate LLaMA-Factory Dataset Metadata With SFT Build

**Files:**
- Modify: `scripts/build_sft_dataset.py`
- Modify: `tests/test_sft_builder.py`

- [ ] **Step 1: Add failing CLI test for dataset_info output**

Append to `tests/test_sft_builder.py`:

```python
def test_build_sft_dataset_cli_writes_llamafactory_dataset_info(tmp_path):
    cards_path = tmp_path / "cards.jsonl"
    chapters_path = tmp_path / "chapters.jsonl"
    output_path = tmp_path / "sft_chapter_v1.jsonl"
    dataset_info_path = tmp_path / "dataset_info.json"
    write_jsonl(
        cards_path,
        [
            {
                "id": "c1",
                "style_contract": "契约",
                "previous_summary": "前情",
                "chapter_goal": "目标",
                "target_word_count": "2000-2500中文汉字",
                "chapter_structure": [
                    {"step": 1, "name": "开场", "goal": "引出冲突", "estimated_chars": "300"}
                ],
                "character_states": [{"name": "林默", "state": "冷静", "speech_style": "短句"}],
                "must_include": ["加钱"],
                "must_not_include": ["提前揭露真相"],
                "ending_hook": "箱子响了一下",
            }
        ],
    )
    write_jsonl(chapters_path, [{"id": "c1", "text": "正文", "split": "train", "quality_tag": "A"}])

    subprocess.run(
        [
            sys.executable,
            "scripts/build_sft_dataset.py",
            "--cards",
            str(cards_path),
            "--chapters",
            str(chapters_path),
            "--output",
            str(output_path),
            "--dataset-info-output",
            str(dataset_info_path),
        ],
        check=True,
    )

    info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
    assert info["sft_chapter_v1"]["file_name"] == "sft_chapter_v1.jsonl"
    assert info["sft_chapter_v1"]["formatting"] == "alpaca"
    assert info["sft_chapter_v1"]["columns"] == {
        "prompt": "instruction",
        "query": "input",
        "response": "output",
    }
```

- [ ] **Step 2: Add missing import**

Add this import to the top of `tests/test_sft_builder.py`:

```python
import json
```

- [ ] **Step 3: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_sft_builder.py::test_build_sft_dataset_cli_writes_llamafactory_dataset_info -q
```

Expected: FAIL because `--dataset-info-output` is not supported.

- [ ] **Step 4: Implement dataset metadata output**

Modify `scripts/build_sft_dataset.py`:

```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.io_utils import read_jsonl, write_jsonl
from small_model_train.sft_builder import build_sft_rows


def _write_dataset_info(path: str | Path, output_path: str | Path) -> None:
    info_path = Path(path)
    info_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_name = Path(output_path).stem
    info = {
        dataset_name: {
            "file_name": Path(output_path).name,
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        }
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", required=True)
    parser.add_argument("--chapters", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-info-output")
    args = parser.parse_args()

    rows = build_sft_rows(read_jsonl(args.cards), read_jsonl(args.chapters))
    write_jsonl(args.output, rows)
    if args.dataset_info_output:
        _write_dataset_info(args.dataset_info_output, args.output)
    print(f"wrote {len(rows)} SFT rows to {args.output}")
```

- [ ] **Step 5: Run targeted tests**

Run:

```powershell
python -m pytest tests/test_sft_builder.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add scripts/build_sft_dataset.py tests/test_sft_builder.py
git commit -m "feat: write llamafactory dataset metadata"
```

---

## Task 5: Make Readiness Block Malformed Chapter Cards

**Files:**
- Modify: `src/small_model_train/stage3_data_readiness.py`
- Modify: `tests/test_stage3_data_readiness.py`

- [ ] **Step 1: Add failing readiness test**

Append to `tests/test_stage3_data_readiness.py`:

```python
def test_readiness_blocks_malformed_chapter_structure(tmp_path):
    raw_dir, chapters_raw, chapters, chapters_split, chapter_cards, eval_cards, sft_dataset = _write_ready_artifacts(tmp_path)
    rows = read_jsonl(chapter_cards)
    rows[0]["chapter_structure"] = [{"beat": "承接", "goal": "推进", "estimated_chars": "300"}]
    write_jsonl(chapter_cards, rows)

    summary = build_stage3_readiness_summary(
        raw_dir=raw_dir,
        chapters_raw_path=chapters_raw,
        chapters_path=chapters,
        chapters_split_path=chapters_split,
        chapter_cards_path=chapter_cards,
        eval_cards_path=eval_cards,
        sft_dataset_path=sft_dataset,
        smoke_dry_run={"exit_code": 0, "command": "dry-run", "stderr": ""},
    )

    assert summary["decision"] == "blocked_missing_chapter_cards"
    assert summary["card_issues"]["schema_errors"]
    assert "chapter_structure[0].step" in summary["card_issues"]["schema_errors"][0]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_stage3_data_readiness.py::test_readiness_blocks_malformed_chapter_structure -q
```

Expected: FAIL because readiness does not validate structure internals strictly enough.

- [ ] **Step 3: Integrate card validator**

Modify card inspection in `src/small_model_train/stage3_data_readiness.py` so each chapter card calls:

```python
from small_model_train.chapter_cards import validate_chapter_card
```

and appends schema errors with the card id:

```python
try:
    validate_chapter_card(card)
except ValueError as exc:
    issues["schema_errors"].append(f"{card.get('id', '<missing-id>')}: {exc}")
```

Keep existing source leakage, render, unmatched id, and empty list checks.

- [ ] **Step 4: Run Stage 3 tests**

Run:

```powershell
python -m pytest tests/test_stage3_data_readiness.py tests/test_chapter_cards.py tests/test_sft_builder.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/small_model_train/stage3_data_readiness.py tests/test_stage3_data_readiness.py
git commit -m "fix: block malformed chapter cards in readiness"
```

---

## Task 6: Regenerate Fixed 50-Row Data And Re-Run Stage 3 Readiness

**Files:**
- Generated local data only.

- [ ] **Step 1: Regenerate chapter cards with fixed schema**

Run:

```powershell
python scripts/build_chapter_cards.py --chapters data_clean/chapters_split.jsonl --output data_cards/chapter_cards.jsonl --count 50 --min-chars 2000 --max-chars 3000
```

Expected:

```text
wrote 50 chapter cards to data_cards/chapter_cards.jsonl
```

- [ ] **Step 2: Rebuild SFT and dataset metadata**

Run:

```powershell
python scripts/build_sft_dataset.py --cards data_cards/chapter_cards.jsonl --chapters data_clean/chapters_split.jsonl --output data_sft/sft_chapter_v1.jsonl --dataset-info-output data_sft/dataset_info.json
```

Expected:

```text
wrote 50 SFT rows to data_sft/sft_chapter_v1.jsonl
```

- [ ] **Step 3: Confirm rendered structure labels are not empty**

Run:

```powershell
$env:PYTHONPATH='src'
@'
from small_model_train.io_utils import read_jsonl
row = read_jsonl("data_sft/sft_chapter_v1.jsonl")[0]
lines = [line for line in row["input"].splitlines() if line.startswith("- 1.")]
print(lines[0])
'@ | python -
```

Expected output includes:

```text
- 1. 承接：
```

- [ ] **Step 4: Re-run readiness**

Run:

```powershell
python scripts/check_stage3_data_readiness.py --eval-cards data_cards/eval_cards_50.jsonl --run-smoke-dry-run
```

Expected:

```text
decision: ready_for_stage4_smoke_training
```

- [ ] **Step 5: Commit code only**

Generated data remains ignored. Commit code and docs from prior tasks only:

```powershell
git status --short
```

Expected: no tracked generated data. If only code/docs changes remain, continue. If `data_*`, `outputs`, `logs`, `reports`, or `mlflow.db` appear as tracked changes, stop and fix `.gitignore` before committing.

---

## Task 7: Re-Run Fixed 50-Sample Smoke Training

**Files:**
- Generated local training artifacts only.

- [ ] **Step 1: Preflight model, environment, and dry-run**

Run:

```powershell
python scripts/check_local_model.py --model-dir E:\models\Qwen3-4B-Instruct-2507 --report reports/model_check_report.md
python scripts/check_training_env.py --report reports/training_env_report.md
python scripts/run_sft_smoke.py --eval-cards data_cards/eval_cards_50.jsonl --dry-run
```

Expected:

```text
llamafactory-cli train outputs\sft_smoke\training_config_snapshot.yaml
```

- [ ] **Step 2: Run real smoke training**

Run:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$env:WANDB_DISABLED='true'
python scripts/run_sft_smoke.py --eval-cards data_cards/eval_cards_50.jsonl
```

Expected:

```text
llamafactory-cli train outputs\sft_smoke\training_config_snapshot.yaml
```

and process exit code `0`.

- [ ] **Step 3: Check adapter**

Run:

```powershell
python scripts/check_adapter.py --adapter-dir outputs/sft_smoke --report reports/sft_smoke_report.md --title "SFT Smoke Adapter Check"
```

Expected report contains:

```text
decision: 允许进入下一步
```

- [ ] **Step 4: Record metrics**

Run:

```powershell
Get-Content outputs\sft_smoke\train_results.json
Get-Content outputs\sft_smoke\trainer_log.jsonl
```

Expected:

```text
"epoch": 1.0
```

and `trainer_log.jsonl` contains `total_steps` equal to the smoke run steps.

---

## Task 8: Run Fixed Eval Inference On Smoke Adapter

**Files:**
- Generated local eval outputs only.

- [ ] **Step 1: Dry-run eval prompt generation**

Run:

```powershell
python scripts/run_eval_inference.py --cards data_cards/eval_cards_50.jsonl --adapter-dir outputs/sft_smoke --output outputs/sft_smoke/generated_dry_run.jsonl --model-name sft_smoke --dry-run
```

Expected: `outputs/sft_smoke/generated_dry_run.jsonl` contains 50 rows.

- [ ] **Step 2: Run real eval inference**

Run:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
python scripts/run_eval_inference.py --cards data_cards/eval_cards_50.jsonl --adapter-dir outputs/sft_smoke --output outputs/sft_smoke/generated.jsonl --model-name sft_smoke --event-log logs/training/sft_smoke_eval_events.jsonl --stderr-log logs/training/sft_smoke_eval_stderr.log --stdout-log logs/training/sft_smoke_eval_stdout.log
```

Expected: exit code `0`, event log ends with status `ok`, and `outputs/sft_smoke/generated.jsonl` has 50 rows.

- [ ] **Step 3: Score fixed eval outputs**

Run:

```powershell
python scripts/score_outputs.py --cards data_cards/eval_cards_50.jsonl --outputs outputs/sft_smoke/generated.jsonl --output outputs/sft_smoke/metrics.jsonl
```

Expected:

```text
wrote 50 scores to outputs/sft_smoke/metrics.jsonl
```

- [ ] **Step 4: Build eval report**

Run:

```powershell
python scripts/evaluate_outputs.py --scores outputs/sft_smoke/metrics.jsonl --report reports/sft_smoke_eval_report.md --title "SFT Smoke Eval Report"
```

Expected:

```text
wrote report to reports/sft_smoke_eval_report.md
```

---

## Task 9: Decide Whether To Expand To 100 Samples

**Files:**
- Modify: `docs/stage4-decision-log.zh.md`

- [ ] **Step 1: Create decision log**

Create `docs/stage4-decision-log.zh.md`:

```markdown
# Stage 4 Decision Log

## 2026-06-19 Fixed 50-Sample Smoke

- 数据：50 条 fixed chapter cards，50 条 fixed eval cards。
- 训练：`python scripts/run_sft_smoke.py --eval-cards data_cards/eval_cards_50.jsonl`
- Adapter 检查：见 `reports/sft_smoke_report.md`。
- Eval 推理：见 `outputs/sft_smoke/generated.jsonl`。
- 评分报告：见 `reports/sft_smoke_eval_report.md`。

## Decision

- 如果 smoke adapter 检查失败：停止扩样，回到训练环境或数据问题诊断。
- 如果 eval 推理失败：停止扩样，先修 eval worker、显存参数或 adapter 加载。
- 如果 eval 输出存在明显格式问题：先修章节卡和输出约束，再重训 50 条。
- 如果 eval 输出格式可用但质量弱：扩到 100 条，继续观察训练曲线和固定 eval 变化。
- 如果 eval 输出已经稳定：扩到 100 条，再准备 500 条正式 v1。
```

- [ ] **Step 2: Fill decision using actual evidence**

Edit the `Decision` section after Task 8 completes. Use only values from:

```text
reports/sft_smoke_report.md
reports/sft_smoke_eval_report.md
outputs/sft_smoke/train_results.json
outputs/sft_smoke/metrics.jsonl
logs/training/sft_smoke_eval_events.jsonl
```

- [ ] **Step 3: Commit decision log**

```powershell
git add docs/stage4-decision-log.zh.md
git commit -m "docs: record stage four smoke decision"
```

---

## Task 10: Document Stage 4 Runbook

**Files:**
- Modify: `README.md`
- Create: `docs/stage4-smoke-eval-guide.zh.md`

- [ ] **Step 1: Add Chinese runbook**

Create `docs/stage4-smoke-eval-guide.zh.md`:

````markdown
# 第四阶段：真实 Smoke Training 与固定 Eval 指南

第四阶段的目标不是直接冲 500 条正式训练，而是把真实训练闭环跑通：修复章节卡，重建 SFT，真实 smoke training，adapter 检查，固定 eval 推理，规则评分，然后决定是否扩到 100 条。

## 1. 修复并生成章节卡

```powershell
python scripts/build_chapter_cards.py --chapters data_clean/chapters_split.jsonl --output data_cards/chapter_cards.jsonl --count 50 --min-chars 2000 --max-chars 3000
```

章节卡必须包含 `chapter_goal`、`chapter_structure`、`character_states`、`must_include`、`must_not_include`、`target_word_count`。其中 `chapter_structure` 每一项必须有 `step`、`name`、`goal`、`estimated_chars`。

## 2. 重建 SFT 和 LLaMA-Factory metadata

```powershell
python scripts/build_sft_dataset.py --cards data_cards/chapter_cards.jsonl --chapters data_clean/chapters_split.jsonl --output data_sft/sft_chapter_v1.jsonl --dataset-info-output data_sft/dataset_info.json
```

## 3. 第三阶段 readiness 复核

```powershell
python scripts/check_stage3_data_readiness.py --eval-cards data_cards/eval_cards_50.jsonl --run-smoke-dry-run
```

只有输出 `ready_for_stage4_smoke_training` 才继续。

## 4. 真实 smoke training

```powershell
python scripts/check_local_model.py --model-dir E:\models\Qwen3-4B-Instruct-2507 --report reports/model_check_report.md
python scripts/check_training_env.py --report reports/training_env_report.md
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$env:WANDB_DISABLED='true'
python scripts/run_sft_smoke.py --eval-cards data_cards/eval_cards_50.jsonl
python scripts/check_adapter.py --adapter-dir outputs/sft_smoke --report reports/sft_smoke_report.md --title "SFT Smoke Adapter Check"
```

## 5. 固定 eval 推理与评分

```powershell
python scripts/run_eval_inference.py --cards data_cards/eval_cards_50.jsonl --adapter-dir outputs/sft_smoke --output outputs/sft_smoke/generated.jsonl --model-name sft_smoke --event-log logs/training/sft_smoke_eval_events.jsonl --stderr-log logs/training/sft_smoke_eval_stderr.log --stdout-log logs/training/sft_smoke_eval_stdout.log
python scripts/score_outputs.py --cards data_cards/eval_cards_50.jsonl --outputs outputs/sft_smoke/generated.jsonl --output outputs/sft_smoke/metrics.jsonl
python scripts/evaluate_outputs.py --scores outputs/sft_smoke/metrics.jsonl --report reports/sft_smoke_eval_report.md --title "SFT Smoke Eval Report"
```

## 6. 扩样决策

如果 adapter 检查、eval 推理和评分报告都完成，再决定是否扩到 100 条。不要在章节卡结构字段仍然异常时扩样。
````

- [ ] **Step 2: Add README Stage 4 section**

Add this section to `README.md` after Stage 3:

````markdown
## Stage 4 Smoke Eval

Stage 4 starts only after Stage 3 reports `ready_for_stage4_smoke_training`. It repairs generated chapter cards, runs real smoke training, checks the LoRA adapter, then runs fixed eval inference and scoring.

```powershell
python scripts/build_chapter_cards.py --chapters data_clean/chapters_split.jsonl --output data_cards/chapter_cards.jsonl --count 50 --min-chars 2000 --max-chars 3000
python scripts/build_sft_dataset.py --cards data_cards/chapter_cards.jsonl --chapters data_clean/chapters_split.jsonl --output data_sft/sft_chapter_v1.jsonl --dataset-info-output data_sft/dataset_info.json
python scripts/check_stage3_data_readiness.py --eval-cards data_cards/eval_cards_50.jsonl --run-smoke-dry-run
python scripts/run_sft_smoke.py --eval-cards data_cards/eval_cards_50.jsonl
python scripts/check_adapter.py --adapter-dir outputs/sft_smoke --report reports/sft_smoke_report.md --title "SFT Smoke Adapter Check"
python scripts/run_eval_inference.py --cards data_cards/eval_cards_50.jsonl --adapter-dir outputs/sft_smoke --output outputs/sft_smoke/generated.jsonl --model-name sft_smoke
python scripts/score_outputs.py --cards data_cards/eval_cards_50.jsonl --outputs outputs/sft_smoke/generated.jsonl --output outputs/sft_smoke/metrics.jsonl
python scripts/evaluate_outputs.py --scores outputs/sft_smoke/metrics.jsonl --report reports/sft_smoke_eval_report.md --title "SFT Smoke Eval Report"
```

See `docs/stage4-smoke-eval-guide.zh.md` for the full Chinese runbook.
````

- [ ] **Step 3: Run docs grep checks**

Run:

```powershell
rg -n "beat|\\. ：|eval_cards_20" README.md docs/stage4-smoke-eval-guide.zh.md docs/stage4-decision-log.zh.md
```

Expected: no `beat` or `- . ：` references. `eval_cards_20` may remain only in older Stage 3 sections, not in Stage 4.

- [ ] **Step 4: Commit docs**

```powershell
git add README.md docs/stage4-smoke-eval-guide.zh.md
git commit -m "docs: add stage four smoke eval runbook"
```

---

## Task 11: Full Verification Gate

**Files:**
- No code changes.

- [ ] **Step 1: Run targeted test suite**

Run:

```powershell
python -m pytest tests/test_chapter_cards.py tests/test_sft_builder.py tests/test_stage3_data_readiness.py tests/test_stage2_inference.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full test suite**

Run:

```powershell
python -m pytest -q
```

Expected: PASS.

- [ ] **Step 3: Verify real local artifacts**

Run:

```powershell
python scripts/check_stage3_data_readiness.py --eval-cards data_cards/eval_cards_50.jsonl --run-smoke-dry-run
python scripts/check_adapter.py --adapter-dir outputs/sft_smoke --report reports/sft_smoke_report.md --title "SFT Smoke Adapter Check"
```

Expected:

```text
decision: ready_for_stage4_smoke_training
wrote adapter report to reports\sft_smoke_report.md
```

and `reports/sft_smoke_report.md` contains `decision: 允许进入下一步`.

- [ ] **Step 4: Verify generated eval rows and scores**

Run:

```powershell
$env:PYTHONPATH='src'
@'
from small_model_train.io_utils import read_jsonl
print("generated", len(read_jsonl("outputs/sft_smoke/generated.jsonl")))
print("metrics", len(read_jsonl("outputs/sft_smoke/metrics.jsonl")))
'@ | python -
```

Expected:

```text
generated 50
metrics 50
```

- [ ] **Step 5: Final status check**

Run:

```powershell
git status --short --ignored
```

Expected:

- Tracked changes are committed or intentionally left for review.
- Ignored generated directories include `data_cards/`, `data_clean/`, `data_raw/`, `data_sft/`, `logs/`, `outputs/`, `reports/`, `mlflow.db`, `style_contract.md`, and `style_profile.json`.

---

## Execution Notes

- The first completed smoke training proves the training stack can run, but its prompt quality is compromised by malformed structure labels. Treat that run as infrastructure proof, not a quality baseline.
- Re-run smoke training after card repair before judging model output.
- Do not expand to 100 or 500 samples until `chapter_structure` renders as numbered named beats in SFT inputs.
- Keep generated data, adapters, logs, reports, and `mlflow.db` ignored.
- If real eval inference OOMs, reduce eval generation settings in `src/small_model_train/stage2_inference.py` only after recording the failure logs.

## Self-Review

- Spec coverage: Stage 4 sequence covers model/environment checks, real smoke training, adapter check, fixed eval inference, scoring report, and expansion decision. Chapter card repair is a required pre-training gate.
- Placeholder scan: The plan contains exact file paths, code snippets, commands, and expected outputs. No TBD/TODO placeholders are present.
- Type consistency: `chapter_structure` uses `step`, `name`, `goal`, and `estimated_chars` consistently across generator, validator, renderer, readiness, and tests.
