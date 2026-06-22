# Male Webnovel Agent Review Acceptance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Stage 4.1 acceptance gate from `docs/superpowers/specs/2026-06-21-agent-review-acceptance-system-design.md` so the project can reject bad male-webnovel chapter outputs before any training expansion.

**Architecture:** Keep the small model scoped as a prose executor: external systems provide plot, structure, conflict, payoff, and ending-hook fields through execution cards; this work validates whether generated prose executes those cards. Add a deterministic rule gate, an agent-review vote coordinator, and a CLI that can import or mock reviewer rows; do not change the training algorithm, introduce DPO/ORPO/SimPO, or redesign SFT.

**Tech Stack:** Python 3.11, JSONL artifact files, existing `small_model_train` package, `pytest`, existing script style under `scripts/`.

---

## File Structure

- Create `src/small_model_train/execution_cards.py`: execution-card schema validation and platform profile constants.
- Modify `src/small_model_train/sft_builder.py`: render optional external-control fields into the prompt so the prose executor can see them.
- Modify `src/small_model_train/stage2_inference.py`: require execution-card schema when loading eval cards.
- Modify `scripts/run_eval_inference.py`: fail early when eval cards are raw chapter rows instead of execution cards.
- Create `src/small_model_train/quality_rules.py`: deterministic residue, repetition, ending, and male-webnovel execution-risk checks.
- Modify `src/small_model_train/scoring.py`: merge deterministic quality rules into `score_output`.
- Create `src/small_model_train/agent_review.py`: review-row validation, majority-vote aggregation, and Markdown reporting.
- Create `scripts/run_agent_review.py`: command-line coordinator for `--backend mock` and `--reviews-import` mode.
- Modify `src/small_model_train/stage4_quality.py`: summarize optional agent-review decisions in Stage 4 reports.
- Modify `scripts/build_stage4_quality_report.py`: accept optional agent vote rows and include final decision context.
- Add tests:
  - `tests/test_execution_cards.py`
  - `tests/test_quality_rules.py`
  - `tests/test_agent_review.py`
  - extend `tests/test_stage2_inference.py`
  - extend `tests/test_sft_builder.py`
  - extend `tests/test_scoring.py`
  - extend `tests/test_stage4_quality.py`
- Modify docs:
  - `docs/stage4-1-quality-eval-guide.zh.md`
  - `docs/stage4-decision-log.zh.md`

## Task 1: Execution-Card Schema Guard

**Files:**
- Create: `src/small_model_train/execution_cards.py`
- Test: `tests/test_execution_cards.py`

- [ ] **Step 1: Write failing schema tests**

Create `tests/test_execution_cards.py`:

```python
from __future__ import annotations

import pytest

from small_model_train.execution_cards import (
    DEFAULT_TARGET_PLATFORM,
    validate_execution_card,
    validate_execution_cards,
)


def _valid_card() -> dict:
    return {
        "id": "case1",
        "target_platform": DEFAULT_TARGET_PLATFORM,
        "genre_tags": ["urban", "system"],
        "style_contract": "短句推进，口语化，男频爽文节奏。",
        "previous_summary": "林默刚拿到系统任务。",
        "chapter_goal": "林默必须在晚宴上证明自己。",
        "chapter_structure": [
            {
                "step": 1,
                "name": "开场压迫",
                "goal": "岳家众人当众羞辱林默。",
                "estimated_chars": "500-700",
            }
        ],
        "conflict_beat": "岳家要求林默认错，林默反手提出赌约。",
        "payoff_beat": "林默用系统奖励拿出证据，第一次压住对方。",
        "must_include": ["赌约", "证据"],
        "must_not_include": ["作者说明"],
        "ending_hook": "门外传来真正买家的声音。",
        "target_word_count": "2000-2500中文汉字",
    }


def test_validate_execution_card_accepts_complete_card():
    assert validate_execution_card(_valid_card()) == _valid_card()


def test_validate_execution_card_blocks_raw_eval_chapter():
    raw_card = {
        "id": "case1",
        "work_id": "book1",
        "chapter_title": "第1章",
        "text": "原文正文",
        "quality_tag": "A",
        "split": "eval",
    }

    with pytest.raises(ValueError) as excinfo:
        validate_execution_card(raw_card)

    assert "missing execution-card fields" in str(excinfo.value)
    assert "style_contract" in str(excinfo.value)


def test_validate_execution_card_blocks_unknown_platform():
    card = _valid_card()
    card["target_platform"] = "traditional_literary"

    with pytest.raises(ValueError) as excinfo:
        validate_execution_card(card)

    assert "unknown target_platform" in str(excinfo.value)


def test_validate_execution_card_blocks_empty_genre_tags():
    card = _valid_card()
    card["genre_tags"] = []

    with pytest.raises(ValueError) as excinfo:
        validate_execution_card(card)

    assert "genre_tags must be a non-empty list" in str(excinfo.value)


def test_validate_execution_cards_reports_row_number():
    rows = [_valid_card(), {"id": "bad"}]

    with pytest.raises(ValueError) as excinfo:
        validate_execution_cards(rows)

    assert "row 2" in str(excinfo.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_execution_cards.py -q
```

Expected: fail during import because `small_model_train.execution_cards` does not exist.

- [ ] **Step 3: Implement execution-card validation**

Create `src/small_model_train/execution_cards.py`:

```python
from __future__ import annotations

from typing import Any


DEFAULT_TARGET_PLATFORM = "hybrid_fanqie_qidian"
VALID_TARGET_PLATFORMS = {"fanqie", "qidian", DEFAULT_TARGET_PLATFORM}
RUBRIC_VERSION = "male_webnovel_v1"

REQUIRED_EXECUTION_FIELDS = (
    "id",
    "target_platform",
    "genre_tags",
    "style_contract",
    "chapter_goal",
    "chapter_structure",
    "conflict_beat",
    "payoff_beat",
    "must_include",
    "must_not_include",
    "ending_hook",
    "target_word_count",
)


def validate_execution_card(card: dict[str, Any]) -> dict[str, Any]:
    missing = [
        field
        for field in REQUIRED_EXECUTION_FIELDS
        if field not in card or card.get(field) in (None, "")
    ]
    if missing:
        raise ValueError(
            "missing execution-card fields: " + ", ".join(sorted(missing))
        )

    target_platform = card.get("target_platform")
    if target_platform not in VALID_TARGET_PLATFORMS:
        raise ValueError(f"unknown target_platform: {target_platform}")

    genre_tags = card.get("genre_tags")
    if not isinstance(genre_tags, list) or not genre_tags:
        raise ValueError("genre_tags must be a non-empty list")
    if not all(isinstance(tag, str) and tag.strip() for tag in genre_tags):
        raise ValueError("genre_tags must contain non-empty strings")

    chapter_structure = card.get("chapter_structure")
    if not isinstance(chapter_structure, list) or not chapter_structure:
        raise ValueError("chapter_structure must be a non-empty list")

    for field in ("must_include", "must_not_include"):
        values = card.get(field)
        if not isinstance(values, list):
            raise ValueError(f"{field} must be a list")

    return card


def validate_execution_cards(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for index, row in enumerate(rows, start=1):
        try:
            validate_execution_card(row)
        except ValueError as exc:
            raise ValueError(f"row {index}: {exc}") from exc
    return rows
```

- [ ] **Step 4: Run schema tests**

Run:

```powershell
python -m pytest tests/test_execution_cards.py -q
```

Expected: `5 passed`.

- [ ] **Step 5: Commit**

```powershell
git add src/small_model_train/execution_cards.py tests/test_execution_cards.py
git commit -m "feat: validate male webnovel execution cards"
```

## Task 2: Prompt Rendering and Eval Loading Guard

**Files:**
- Modify: `src/small_model_train/sft_builder.py`
- Modify: `src/small_model_train/stage2_inference.py`
- Modify: `scripts/run_eval_inference.py`
- Test: `tests/test_sft_builder.py`
- Test: `tests/test_stage2_inference.py`

- [ ] **Step 1: Write failing prompt-rendering test**

Append to `tests/test_sft_builder.py`:

```python
def test_render_sft_input_includes_external_control_beats():
    card = {
        "style_contract": "男频短句，强冲突。",
        "previous_summary": "林默被逼到晚宴角落。",
        "chapter_goal": "林默必须当场破局。",
        "chapter_structure": [
            {
                "step": 1,
                "name": "压迫",
                "goal": "岳家逼他低头。",
                "estimated_chars": "500-700",
            }
        ],
        "character_states": [],
        "conflict_beat": "岳家当众羞辱，林默提出反赌。",
        "payoff_beat": "林默拿出合同证据，让对方第一次失声。",
        "must_include": ["合同证据"],
        "must_not_include": ["作者说明"],
        "ending_hook": "门外响起真正买家的声音。",
        "target_word_count": "2000-2500中文汉字",
    }

    rendered = render_sft_input(card)

    assert "【冲突推进】\n岳家当众羞辱，林默提出反赌。" in rendered
    assert "【爽点兑现】\n林默拿出合同证据，让对方第一次失声。" in rendered
```

- [ ] **Step 2: Write failing eval-card loading test**

Append to `tests/test_stage2_inference.py`:

```python
import pytest


def test_load_eval_cards_requires_execution_card_schema(tmp_path):
    cards_path = tmp_path / "raw_eval_cards.jsonl"
    cards_path.write_text(
        '{"id":"case1","text":"原文","quality_tag":"A","split":"eval"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        load_eval_cards(cards_path)

    assert "missing execution-card fields" in str(excinfo.value)
```

- [ ] **Step 3: Run focused tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_sft_builder.py::test_render_sft_input_includes_external_control_beats tests/test_stage2_inference.py::test_load_eval_cards_requires_execution_card_schema -q
```

Expected: both tests fail.

- [ ] **Step 4: Render external-control beats**

In `src/small_model_train/sft_builder.py`, inside `render_sft_input`, insert these sections after `chapter_goal` and before `_format_structure(...)`:

```python
        "【冲突推进】",
        card.get("conflict_beat", ""),
        "【爽点兑现】",
        card.get("payoff_beat", ""),
```

The surrounding block becomes:

```python
        "【本章目标】",
        card.get("chapter_goal", ""),
        "【冲突推进】",
        card.get("conflict_beat", ""),
        "【爽点兑现】",
        card.get("payoff_beat", ""),
        _format_structure(card.get("chapter_structure", [])),
```

- [ ] **Step 5: Enforce eval execution-card loading**

Modify `src/small_model_train/stage2_inference.py` imports:

```python
from small_model_train.execution_cards import validate_execution_cards
```

Replace `load_eval_cards` with:

```python
def load_eval_cards(path: str | Path) -> list[dict]:
    cards_path = Path(path)
    if not cards_path.exists():
        raise ValueError(f"cards file is missing: {cards_path}")

    rows = read_jsonl(cards_path)
    if not rows:
        raise ValueError(f"cards file has no rows: {cards_path}")

    return validate_execution_cards(rows)
```

- [ ] **Step 6: Run focused tests**

Run:

```powershell
python -m pytest tests/test_sft_builder.py::test_render_sft_input_includes_external_control_beats tests/test_stage2_inference.py::test_load_eval_cards_requires_execution_card_schema -q
```

Expected: `2 passed`.

- [ ] **Step 7: Run inference-related tests**

Run:

```powershell
python -m pytest tests/test_stage2_inference.py tests/test_sft_builder.py -q
```

Expected: all selected tests pass. If existing tests create eval cards, update those fixtures to include the fields from Task 1 `_valid_card()`.

- [ ] **Step 8: Commit**

```powershell
git add src/small_model_train/sft_builder.py src/small_model_train/stage2_inference.py tests/test_sft_builder.py tests/test_stage2_inference.py
git commit -m "feat: require execution cards for eval inference"
```

## Task 3: Deterministic Quality Rules

**Files:**
- Create: `src/small_model_train/quality_rules.py`
- Modify: `src/small_model_train/scoring.py`
- Test: `tests/test_quality_rules.py`
- Test: `tests/test_scoring.py`

- [ ] **Step 1: Write failing quality-rule tests**

Create `tests/test_quality_rules.py`:

```python
from __future__ import annotations

from small_model_train.quality_rules import detect_quality_issues


def test_detect_quality_issues_flags_markdown_and_disclaimer():
    result = detect_quality_issues(
        {},
        "> 林默走进大厅。\n作为AI，我无法保证以上内容完全准确。",
    )

    assert "markdown_residue" in result["issues"]
    assert "disclaimer_residue" in result["issues"]


def test_detect_quality_issues_flags_semantic_repetition():
    text = (
        "林默终于明白自己不能退。他知道自己必须向前。他清楚自己不能退缩。"
        "林默终于明白自己不能退。他知道自己必须向前。他清楚自己不能退缩。"
        "林默终于明白自己不能退。他知道自己必须向前。他清楚自己不能退缩。"
    )

    result = detect_quality_issues({}, text)

    assert "semantic_repetition" in result["issues"]


def test_detect_quality_issues_flags_weak_ending():
    result = detect_quality_issues({}, "林默看向窗外，心里有了某种决定")

    assert "unnatural_ending" in result["issues"]


def test_detect_quality_issues_flags_missing_external_payoff_terms():
    card = {"payoff_beat": "合同证据让岳家闭嘴", "ending_hook": "真正买家出现"}
    result = detect_quality_issues(card, "林默沉默了很久，最后只是握紧拳头。")

    assert "no_visible_payoff" in result["issues"]
    assert "weak_ending_hook" in result["issues"]
```

- [ ] **Step 2: Write failing scoring integration test**

Append to `tests/test_scoring.py`:

```python
def test_score_output_merges_quality_rule_failures():
    card = {
        "must_include": [],
        "must_not_include": [],
        "payoff_beat": "合同证据让岳家闭嘴",
        "ending_hook": "真正买家出现",
    }
    output = "> 林默沉默了很久，最后只是握紧拳头。"

    score = score_output("case-quality", card, output)

    assert "markdown_residue" in score["failure_types"]
    assert "no_visible_payoff" in score["failure_types"]
    assert score["hard_gate_pass"] is False
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_quality_rules.py tests/test_scoring.py::test_score_output_merges_quality_rule_failures -q
```

Expected: fail during import because `small_model_train.quality_rules` does not exist.

- [ ] **Step 4: Implement deterministic quality rules**

Create `src/small_model_train/quality_rules.py`:

```python
from __future__ import annotations

import re
from collections import Counter
from typing import Any

from small_model_train.text_utils import count_chinese_chars


MARKDOWN_RE = re.compile(r"(^|\n)\s*(>|#{1,6}\s+|[-*]\s+|\d+[.、]\s+)")
DISCLAIMER_MARKERS = ("作为AI", "我无法", "不能保证", "仅供参考")
META_EVALUATION_MARKERS = ("以下是正文", "最终确认", "检查清单", "本章完成", "符合要求")
PROSE_END_RE = re.compile(r"[。！？!?…’”）)]$")
CHINESE_RUN_RE = re.compile(r"[\u4e00-\u9fff]{5,}")
GENERIC_PHRASES = (
    "终于明白",
    "轻声说",
    "像是某种",
    "深吸一口气",
    "空气仿佛凝固",
)


def detect_quality_issues(card: dict[str, Any], text: str) -> dict[str, Any]:
    issues: list[str] = []
    details: dict[str, Any] = {}

    if MARKDOWN_RE.search(text):
        issues.append("markdown_residue")
    if any(marker in text for marker in DISCLAIMER_MARKERS):
        issues.append("disclaimer_residue")
    if any(marker in text for marker in META_EVALUATION_MARKERS):
        issues.append("meta_evaluation_residue")
    if text.strip() and not PROSE_END_RE.search(text.strip()):
        issues.append("unnatural_ending")

    repeated = _repeated_chinese_runs(text)
    if repeated:
        issues.append("semantic_repetition")
        details["repeated_runs"] = repeated[:5]

    generic_hits = [phrase for phrase in GENERIC_PHRASES if text.count(phrase) >= 2]
    if generic_hits:
        issues.append("generic_ai_phrase")
        details["generic_phrase_hits"] = generic_hits

    payoff_beat = str(card.get("payoff_beat", "")).strip()
    if payoff_beat and _coverage_terms(payoff_beat, text) < 0.34:
        issues.append("no_visible_payoff")

    ending_hook = str(card.get("ending_hook", "")).strip()
    if ending_hook and _coverage_terms(ending_hook, text[-200:]) < 0.34:
        issues.append("weak_ending_hook")

    if count_chinese_chars(text) >= 2450 and "semantic_repetition" in issues:
        issues.append("padding_to_length")

    return {"issues": sorted(set(issues)), "details": details}


def _repeated_chinese_runs(text: str) -> list[str]:
    runs = CHINESE_RUN_RE.findall(text)
    windows: list[str] = []
    for run in runs:
        for index in range(0, max(len(run) - 7, 0) + 1):
            windows.append(run[index : index + 8])
    counts = Counter(windows)
    return [value for value, count in counts.items() if count >= 3]


def _coverage_terms(source: str, text: str) -> float:
    terms = [term for term in CHINESE_RUN_RE.findall(source) if len(term) >= 2]
    if not terms:
        return 1.0
    hits = sum(1 for term in terms if term in text)
    return hits / len(terms)
```

- [ ] **Step 5: Integrate quality rules into scoring**

Modify `src/small_model_train/scoring.py` imports:

```python
from small_model_train.quality_rules import detect_quality_issues
```

Inside `score_output`, after `forbidden_hits`:

```python
    quality = detect_quality_issues(card, output)
```

After existing failure appends, add:

```python
    failure_types.extend(
        issue for issue in quality["issues"] if issue not in failure_types
    )
```

Replace `hard_gate_failures` with:

```python
    hard_gate_failures = {
        "length_short",
        "length_long",
        "outline_leak",
        "forbidden_violation",
        "must_include_missing",
        "repetition",
        "markdown_residue",
        "disclaimer_residue",
        "meta_evaluation_residue",
        "semantic_repetition",
        "padding_to_length",
        "unnatural_ending",
        "no_visible_payoff",
        "weak_ending_hook",
    }
```

Add this field to the returned dict:

```python
        "quality_rule_details": quality["details"],
```

- [ ] **Step 6: Run quality-rule tests**

Run:

```powershell
python -m pytest tests/test_quality_rules.py tests/test_scoring.py -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit**

```powershell
git add src/small_model_train/quality_rules.py src/small_model_train/scoring.py tests/test_quality_rules.py tests/test_scoring.py
git commit -m "feat: add deterministic male webnovel quality rules"
```

## Task 4: Agent Review Data Model and Aggregation

**Files:**
- Create: `src/small_model_train/agent_review.py`
- Test: `tests/test_agent_review.py`

- [ ] **Step 1: Write failing aggregation tests**

Create `tests/test_agent_review.py`:

```python
from __future__ import annotations

import pytest

from small_model_train.agent_review import (
    aggregate_agent_reviews,
    render_agent_review_report,
    validate_review_row,
)
from small_model_train.execution_cards import DEFAULT_TARGET_PLATFORM, RUBRIC_VERSION


def _review(sample_id: str, reviewer: str, passed: bool, severity: str = "none") -> dict:
    return {
        "id": sample_id,
        "target_platform": DEFAULT_TARGET_PLATFORM,
        "genre_tags": ["urban"],
        "rubric_version": RUBRIC_VERSION,
        "reviewer": reviewer,
        "pass": passed,
        "severity": severity,
        "issues": [] if passed else ["semantic_repetition"],
        "evidence": [
            {
                "type": "summary",
                "location": "middle",
                "note": "重复解释，没有新的正文推进。",
            }
        ],
        "recommendation": "accept" if passed else "reject_or_retry",
        "confidence": "high",
    }


def test_validate_review_row_accepts_known_reviewer():
    row = _review("case1", "readthrough_structure", True)

    assert validate_review_row(row) == row


def test_validate_review_row_blocks_unknown_reviewer():
    row = _review("case1", "literary_beauty", True)

    with pytest.raises(ValueError) as excinfo:
        validate_review_row(row)

    assert "unknown reviewer" in str(excinfo.value)


def test_aggregate_agent_reviews_passes_two_of_three():
    rows = [
        _review("case1", "readthrough_structure", True),
        _review("case1", "male_genre_payoff", True),
        _review("case1", "platform_style_compliance", False, "major"),
    ]

    summary, votes = aggregate_agent_reviews(
        expected_ids=["case1"],
        review_rows=rows,
        target_platform=DEFAULT_TARGET_PLATFORM,
        rubric_version=RUBRIC_VERSION,
    )

    assert votes[0]["agent_gate_pass"] is True
    assert votes[0]["requires_human_arbitration"] is False
    assert summary["decision"] == "ready_for_human_spot_check"


def test_aggregate_agent_reviews_blocks_one_of_three():
    rows = [
        _review("case1", "readthrough_structure", True),
        _review("case1", "male_genre_payoff", False, "major"),
        _review("case1", "platform_style_compliance", False, "major"),
    ]

    summary, votes = aggregate_agent_reviews(
        expected_ids=["case1"],
        review_rows=rows,
        target_platform=DEFAULT_TARGET_PLATFORM,
        rubric_version=RUBRIC_VERSION,
    )

    assert votes[0]["agent_gate_pass"] is False
    assert summary["decision"] == "blocked_by_agent_review"


def test_aggregate_agent_reviews_sends_blocker_to_arbitration():
    rows = [
        _review("case1", "readthrough_structure", True),
        _review("case1", "male_genre_payoff", True),
        _review("case1", "platform_style_compliance", False, "blocker"),
    ]

    summary, votes = aggregate_agent_reviews(
        expected_ids=["case1"],
        review_rows=rows,
        target_platform=DEFAULT_TARGET_PLATFORM,
        rubric_version=RUBRIC_VERSION,
    )

    assert votes[0]["agent_gate_pass"] is True
    assert votes[0]["requires_human_arbitration"] is True
    assert summary["decision"] == "blocked_by_human_arbitration"


def test_aggregate_agent_reviews_blocks_incomplete_reviews():
    rows = [_review("case1", "readthrough_structure", True)]

    summary, votes = aggregate_agent_reviews(
        expected_ids=["case1"],
        review_rows=rows,
        target_platform=DEFAULT_TARGET_PLATFORM,
        rubric_version=RUBRIC_VERSION,
    )

    assert votes[0]["review_count"] == 1
    assert votes[0]["agent_gate_pass"] is False
    assert summary["missing_review_ids"] == ["case1"]
    assert summary["decision"] == "blocked_incomplete_agent_review"


def test_render_agent_review_report_omits_generated_text():
    rows = [
        _review("case1", "readthrough_structure", True),
        _review("case1", "male_genre_payoff", True),
        _review("case1", "platform_style_compliance", True),
    ]
    summary, votes = aggregate_agent_reviews(
        expected_ids=["case1"],
        review_rows=rows,
        target_platform=DEFAULT_TARGET_PLATFORM,
        rubric_version=RUBRIC_VERSION,
    )

    report = render_agent_review_report("Agent Review", summary, votes)

    assert "# Agent Review" in report
    assert "ready_for_next_expansion" in report
    assert "重复解释，没有新的正文推进" not in report
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_agent_review.py -q
```

Expected: fail during import because `small_model_train.agent_review` does not exist.

- [ ] **Step 3: Implement agent review aggregation**

Create `src/small_model_train/agent_review.py`:

```python
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from small_model_train.execution_cards import VALID_TARGET_PLATFORMS


REVIEWERS = {
    "readthrough_structure",
    "male_genre_payoff",
    "platform_style_compliance",
}
SEVERITIES = {"none", "minor", "major", "blocker"}
REQUIRED_REVIEW_FIELDS = (
    "id",
    "target_platform",
    "genre_tags",
    "rubric_version",
    "reviewer",
    "pass",
    "severity",
    "issues",
    "evidence",
    "recommendation",
    "confidence",
)


def validate_review_row(row: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_REVIEW_FIELDS if field not in row]
    if missing:
        raise ValueError("missing review fields: " + ", ".join(sorted(missing)))
    if row["reviewer"] not in REVIEWERS:
        raise ValueError(f"unknown reviewer: {row['reviewer']}")
    if row["severity"] not in SEVERITIES:
        raise ValueError(f"unknown severity: {row['severity']}")
    if row["target_platform"] not in VALID_TARGET_PLATFORMS:
        raise ValueError(f"unknown target_platform: {row['target_platform']}")
    if type(row["pass"]) is not bool:
        raise ValueError("pass must be boolean")
    for field in ("genre_tags", "issues", "evidence"):
        if not isinstance(row[field], list):
            raise ValueError(f"{field} must be a list")
    return row


def aggregate_agent_reviews(
    expected_ids: list[str],
    review_rows: list[dict[str, Any]],
    target_platform: str,
    rubric_version: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reviews_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    malformed_rows: list[str] = []

    for index, row in enumerate(review_rows, start=1):
        try:
            validated = validate_review_row(row)
        except ValueError as exc:
            malformed_rows.append(f"row {index}: {exc}")
            continue
        reviews_by_id[str(validated["id"])].append(validated)

    votes = [
        _vote_for_sample(sample_id, reviews_by_id.get(sample_id, []))
        for sample_id in expected_ids
    ]
    issue_counts = Counter(issue for vote in votes for issue in vote["issues"])
    missing_review_ids = [
        vote["id"] for vote in votes if vote["review_count"] < len(REVIEWERS)
    ]
    blocked_ids = [vote["id"] for vote in votes if not vote["agent_gate_pass"]]
    arbitration_ids = [
        vote["id"] for vote in votes if vote["requires_human_arbitration"]
    ]

    decision = _batch_decision(missing_review_ids, blocked_ids, arbitration_ids, votes)
    summary = {
        "target_platform": target_platform,
        "rubric_version": rubric_version,
        "expected_rows": len(expected_ids),
        "reviewed_rows": len({row["id"] for row in review_rows if "id" in row}),
        "missing_review_ids": missing_review_ids,
        "malformed_review_rows": malformed_rows,
        "agent_gate_pass": decision in {
            "ready_for_human_spot_check",
            "ready_for_next_expansion",
        },
        "blocked_ids": blocked_ids,
        "arbitration_ids": arbitration_ids,
        "issue_counts": dict(sorted(issue_counts.items())),
        "decision": decision,
    }
    return summary, votes


def _vote_for_sample(sample_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    reviewers = sorted({row["reviewer"] for row in rows})
    pass_votes = sum(1 for row in rows if row["pass"])
    fail_votes = sum(1 for row in rows if not row["pass"])
    blocker_votes = sum(1 for row in rows if row["severity"] == "blocker")
    issues = sorted({issue for row in rows for issue in row.get("issues", [])})
    complete = len(reviewers) == len(REVIEWERS)
    agent_gate_pass = complete and pass_votes >= 2
    return {
        "id": sample_id,
        "target_platform": rows[0]["target_platform"] if rows else "",
        "review_count": len(rows),
        "pass_votes": pass_votes,
        "fail_votes": fail_votes,
        "blocker_votes": blocker_votes,
        "agent_gate_pass": agent_gate_pass,
        "requires_human_arbitration": blocker_votes > 0,
        "issues": issues,
        "reviewers": reviewers,
    }


def _batch_decision(
    missing_review_ids: list[str],
    blocked_ids: list[str],
    arbitration_ids: list[str],
    votes: list[dict[str, Any]],
) -> str:
    if missing_review_ids:
        return "blocked_incomplete_agent_review"
    if blocked_ids:
        return "blocked_by_agent_review"
    if arbitration_ids:
        return "blocked_by_human_arbitration"
    if any(vote["pass_votes"] == 2 for vote in votes):
        return "ready_for_human_spot_check"
    return "ready_for_next_expansion"


def render_agent_review_report(
    title: str,
    summary: dict[str, Any],
    votes: list[dict[str, Any]],
) -> str:
    lines = [
        f"# {title}",
        "",
        "## Decision",
        f"- {summary['decision']}",
        f"- target_platform: {summary['target_platform']}",
        f"- rubric_version: {summary['rubric_version']}",
        "",
        "## Rows",
        f"- expected: {summary['expected_rows']}",
        f"- reviewed: {summary['reviewed_rows']}",
        "",
        "## Issue Counts",
    ]
    if summary["issue_counts"]:
        for issue, count in summary["issue_counts"].items():
            lines.append(f"- {issue}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "## Blocked Samples"])
    lines.append("- " + ", ".join(summary["blocked_ids"]) if summary["blocked_ids"] else "- none")

    lines.extend(["", "## Arbitration Samples"])
    lines.append("- " + ", ".join(summary["arbitration_ids"]) if summary["arbitration_ids"] else "- none")

    lines.extend(["", "## Sample Votes"])
    for vote in votes:
        issue_text = ", ".join(vote["issues"]) or "none"
        lines.append(
            f"- {vote['id']}: pass={vote['pass_votes']}/3; "
            f"fail={vote['fail_votes']}; blocker={vote['blocker_votes']}; "
            f"issues={issue_text}"
        )
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run aggregation tests**

Run:

```powershell
python -m pytest tests/test_agent_review.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/small_model_train/agent_review.py tests/test_agent_review.py
git commit -m "feat: aggregate agent review votes"
```

## Task 5: Agent Review CLI

**Files:**
- Create: `scripts/run_agent_review.py`
- Test: `tests/test_agent_review_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/test_agent_review_cli.py`:

```python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from small_model_train.execution_cards import DEFAULT_TARGET_PLATFORM, RUBRIC_VERSION
from small_model_train.io_utils import read_jsonl, write_jsonl


def _card(sample_id: str = "case1") -> dict:
    return {
        "id": sample_id,
        "target_platform": DEFAULT_TARGET_PLATFORM,
        "genre_tags": ["urban"],
        "style_contract": "短句推进，男频爽文。",
        "chapter_goal": "林默当场破局。",
        "chapter_structure": [
            {
                "step": 1,
                "name": "压迫",
                "goal": "岳家逼他认错。",
                "estimated_chars": "500-700",
            }
        ],
        "conflict_beat": "岳家羞辱，林默反手立赌约。",
        "payoff_beat": "合同证据让岳家闭嘴。",
        "must_include": ["合同证据"],
        "must_not_include": ["作者说明"],
        "ending_hook": "真正买家出现。",
        "target_word_count": "2000-2500中文汉字",
    }


def _review(sample_id: str, reviewer: str, passed: bool) -> dict:
    return {
        "id": sample_id,
        "target_platform": DEFAULT_TARGET_PLATFORM,
        "genre_tags": ["urban"],
        "rubric_version": RUBRIC_VERSION,
        "reviewer": reviewer,
        "pass": passed,
        "severity": "none" if passed else "major",
        "issues": [] if passed else ["semantic_repetition"],
        "evidence": [{"type": "summary", "location": "middle", "note": "重复"}],
        "recommendation": "accept" if passed else "reject_or_retry",
        "confidence": "high",
    }


def test_run_agent_review_import_mode_writes_votes_and_report(tmp_path: Path):
    cards_path = tmp_path / "cards.jsonl"
    outputs_path = tmp_path / "generated.jsonl"
    metrics_path = tmp_path / "metrics.jsonl"
    reviews_path = tmp_path / "reviews_in.jsonl"
    output_reviews_path = tmp_path / "reviews_out.jsonl"
    votes_path = tmp_path / "votes.jsonl"
    report_path = tmp_path / "report.md"
    write_jsonl(cards_path, [_card()])
    write_jsonl(outputs_path, [{"id": "case1", "output": "正文"}])
    write_jsonl(metrics_path, [{"id": "case1", "hard_gate_pass": True, "failure_types": []}])
    write_jsonl(
        reviews_path,
        [
            _review("case1", "readthrough_structure", True),
            _review("case1", "male_genre_payoff", True),
            _review("case1", "platform_style_compliance", True),
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_agent_review.py",
            "--cards",
            str(cards_path),
            "--outputs",
            str(outputs_path),
            "--metrics",
            str(metrics_path),
            "--target-platform",
            DEFAULT_TARGET_PLATFORM,
            "--reviews-import",
            str(reviews_path),
            "--output",
            str(output_reviews_path),
            "--votes-output",
            str(votes_path),
            "--report",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert len(read_jsonl(output_reviews_path)) == 3
    assert read_jsonl(votes_path)[0]["agent_gate_pass"] is True
    assert "ready_for_next_expansion" in report_path.read_text(encoding="utf-8")


def test_run_agent_review_mock_mode_exits_nonzero_on_failed_metrics(tmp_path: Path):
    cards_path = tmp_path / "cards.jsonl"
    outputs_path = tmp_path / "generated.jsonl"
    metrics_path = tmp_path / "metrics.jsonl"
    reviews_path = tmp_path / "reviews_out.jsonl"
    votes_path = tmp_path / "votes.jsonl"
    report_path = tmp_path / "report.md"
    write_jsonl(cards_path, [_card()])
    write_jsonl(outputs_path, [{"id": "case1", "output": "正文"}])
    write_jsonl(
        metrics_path,
        [{"id": "case1", "hard_gate_pass": False, "failure_types": ["semantic_repetition"]}],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_agent_review.py",
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
            str(reviews_path),
            "--votes-output",
            str(votes_path),
            "--report",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert read_jsonl(votes_path)[0]["agent_gate_pass"] is False
    assert "blocked_by_agent_review" in report_path.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run CLI tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_agent_review_cli.py -q
```

Expected: fail because `scripts/run_agent_review.py` does not exist.

- [ ] **Step 3: Implement CLI**

Create `scripts/run_agent_review.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.agent_review import (
    REVIEWERS,
    aggregate_agent_reviews,
    render_agent_review_report,
)
from small_model_train.execution_cards import RUBRIC_VERSION, validate_execution_cards
from small_model_train.io_utils import read_jsonl, write_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", required=True)
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--target-platform", required=True)
    parser.add_argument("--rubric-version", default=RUBRIC_VERSION)
    parser.add_argument("--backend", choices=["mock"], default="mock")
    parser.add_argument("--reviews-import")
    parser.add_argument("--output", required=True)
    parser.add_argument("--votes-output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--title", default="Stage 4 Agent Review Report")
    args = parser.parse_args(argv)

    cards = validate_execution_cards(read_jsonl(args.cards))
    expected_ids = [str(card["id"]) for card in cards]
    metrics = read_jsonl(args.metrics)

    if args.reviews_import:
        review_rows = read_jsonl(args.reviews_import)
    else:
        review_rows = _mock_reviews(cards, metrics, args.target_platform, args.rubric_version)

    summary, votes = aggregate_agent_reviews(
        expected_ids=expected_ids,
        review_rows=review_rows,
        target_platform=args.target_platform,
        rubric_version=args.rubric_version,
    )

    write_jsonl(args.output, review_rows)
    write_jsonl(args.votes_output, votes)
    report = render_agent_review_report(args.title, summary, votes)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(f"wrote {len(review_rows)} agent reviews to {args.output}")
    print(f"wrote {len(votes)} agent vote rows to {args.votes_output}")
    print(f"wrote agent review report to {args.report}")
    return 0 if summary["decision"] in {"ready_for_human_spot_check", "ready_for_next_expansion"} else 1


def _mock_reviews(
    cards: list[dict],
    metrics: list[dict],
    target_platform: str,
    rubric_version: str,
) -> list[dict]:
    metrics_by_id = {str(row.get("id", "")): row for row in metrics}
    rows: list[dict] = []
    for card in cards:
        sample_id = str(card["id"])
        metric = metrics_by_id.get(sample_id, {})
        passed = bool(metric.get("hard_gate_pass"))
        issues = list(metric.get("failure_types", []))
        for reviewer in sorted(REVIEWERS):
            rows.append(
                {
                    "id": sample_id,
                    "target_platform": target_platform,
                    "genre_tags": list(card.get("genre_tags", [])),
                    "rubric_version": rubric_version,
                    "reviewer": reviewer,
                    "pass": passed,
                    "severity": "none" if passed else "major",
                    "issues": [] if passed else issues,
                    "evidence": [
                        {
                            "type": "summary",
                            "location": "rule-gate",
                            "note": "mock backend mirrors deterministic rule-gate status",
                        }
                    ],
                    "recommendation": "accept" if passed else "reject_or_retry",
                    "confidence": "high",
                }
            )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI tests**

Run:

```powershell
python -m pytest tests/test_agent_review_cli.py -q
```

Expected: all CLI tests pass.

- [ ] **Step 5: Commit**

```powershell
git add scripts/run_agent_review.py tests/test_agent_review_cli.py
git commit -m "feat: add agent review coordinator cli"
```

## Task 6: Stage 4 Report Integration

**Files:**
- Modify: `src/small_model_train/stage4_quality.py`
- Modify: `scripts/build_stage4_quality_report.py`
- Test: `tests/test_stage4_quality.py`

- [ ] **Step 1: Write failing report integration test**

Append to `tests/test_stage4_quality.py`:

```python
def test_render_quality_budget_report_includes_agent_review_summary():
    summary = summarize_quality_budget(
        cards=[_card("a")],
        generated_rows=[{"id": "a", "output": "正文", "params": {"max_new_tokens": 1024}}],
        metric_rows=[
            {
                "id": "a",
                "hard_gate_pass": True,
                "char_count_zh": 2200,
                "failure_types": [],
            }
        ],
        agent_summary={
            "decision": "ready_for_human_spot_check",
            "agent_gate_pass": True,
            "blocked_ids": [],
            "arbitration_ids": [],
            "issue_counts": {"semantic_repetition": 0},
        },
    )

    report = render_quality_budget_report("Stage 4.1", summary)

    assert "## Agent Review" in report
    assert "ready_for_human_spot_check" in report
    assert summary["decision"] == "ready_for_human_spot_check"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_stage4_quality.py::test_render_quality_budget_report_includes_agent_review_summary -q
```

Expected: fail because `summarize_quality_budget` does not accept `agent_summary`.

- [ ] **Step 3: Add optional agent summary to Stage 4 quality summary**

Change `summarize_quality_budget` signature in `src/small_model_train/stage4_quality.py`:

```python
def summarize_quality_budget(
    cards: list[dict[str, Any]],
    generated_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    agent_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
```

After setting `summary["recommendation"]`, add:

```python
    if agent_summary is not None:
        summary["agent_review"] = agent_summary
        summary["decision"] = _combined_decision(summary["decision"], agent_summary)
        summary["recommendation"] = _recommendation(summary["decision"])
```

Add helper:

```python
def _combined_decision(rule_decision: str, agent_summary: dict[str, Any]) -> str:
    if rule_decision not in {"ready_for_full_50_long_eval", "ready_for_next_expansion"}:
        return rule_decision
    return str(agent_summary.get("decision", "rules_pass_agent_pending"))
```

Extend `_recommendation` dict:

```python
        "ready_for_human_spot_check": "Agent gate 通过但存在 2/3 投票样本，先完成人工抽检记录。",
        "ready_for_next_expansion": "Rule gate 与 agent gate 均通过，可以作为下一阶段扩展证据。",
        "blocked_by_agent_review": "Agent gate 阻断，先复盘 blocked_ids 与 issue_counts。",
        "blocked_by_human_arbitration": "存在 blocker vote，先完成 human arbitration。",
        "blocked_incomplete_agent_review": "Agent review 不完整，补齐三类 reviewer 后再判断。",
        "rules_pass_agent_pending": "Rule gate 已过，等待 agent review。",
```

- [ ] **Step 4: Render agent section**

In `render_quality_budget_report`, after the hard gate section and before failure counts, insert:

```python
    agent_review = summary.get("agent_review")
    lines.extend(["", "## Agent Review"])
    if agent_review:
        lines.extend(
            [
                f"- decision: {agent_review['decision']}",
                f"- agent_gate_pass: {agent_review['agent_gate_pass']}",
                f"- blocked_ids: {', '.join(agent_review.get('blocked_ids', [])) or 'none'}",
                f"- arbitration_ids: {', '.join(agent_review.get('arbitration_ids', [])) or 'none'}",
            ]
        )
    else:
        lines.append("- pending")
```

- [ ] **Step 5: Wire optional CLI argument**

Modify `scripts/build_stage4_quality_report.py` parser:

```python
    parser.add_argument("--agent-summary")
```

Before calling `summarize_quality_budget`, read the optional JSONL summary:

```python
    agent_summary = None
    if args.agent_summary:
        rows = read_jsonl(args.agent_summary)
        if rows:
            agent_summary = rows[0]
```

Pass it:

```python
    summary = summarize_quality_budget(
        read_jsonl(args.cards),
        read_jsonl(args.generated),
        read_jsonl(args.metrics),
        agent_summary=agent_summary,
    )
```

- [ ] **Step 6: Run Stage 4 tests**

Run:

```powershell
python -m pytest tests/test_stage4_quality.py -q
```

Expected: all Stage 4 quality tests pass.

- [ ] **Step 7: Commit**

```powershell
git add src/small_model_train/stage4_quality.py scripts/build_stage4_quality_report.py tests/test_stage4_quality.py
git commit -m "feat: include agent gate in stage4 reports"
```

## Task 7: Documentation and Final Verification

**Files:**
- Modify: `docs/stage4-1-quality-eval-guide.zh.md`
- Modify: `docs/stage4-decision-log.zh.md`

- [ ] **Step 1: Update Stage 4.1 guide command sequence**

In `docs/stage4-1-quality-eval-guide.zh.md`, add a section after scoring:

````markdown
## Agent Review Gate

Stage 4.1 不再只看 rule metrics。`score_outputs.py` 完成后，必须运行 agent review coordinator。当前第一版支持 mock 与 import 两种模式：

```powershell
python scripts/run_agent_review.py `
  --cards data_cards/eval_execution_cards_50.jsonl `
  --outputs outputs/sft_smoke/generated.jsonl `
  --metrics outputs/sft_smoke/metrics.jsonl `
  --target-platform hybrid_fanqie_qidian `
  --backend mock `
  --output outputs/sft_smoke/agent_reviews.jsonl `
  --votes-output outputs/sft_smoke/agent_votes.jsonl `
  --report reports/stage4_agent_review_report.md
```

真实 subagent 审核完成后，用 `--reviews-import` 导入三类 reviewer 的 JSONL 结果。只有 rule gate 与 agent gate 都通过，且 blocker arbitration 已处理，Stage 4.1 才能作为扩展训练证据。
````

- [ ] **Step 2: Update decision log status**

In `docs/stage4-decision-log.zh.md`, add a note under the revoked full50 decision:

```markdown
## Agent Review Acceptance Gate

- design: `docs/superpowers/specs/2026-06-21-agent-review-acceptance-system-design.md`
- implementation plan: `docs/superpowers/plans/2026-06-21-male-webnovel-agent-review-acceptance.md`
- scope: 验收正文执行质量，不改变本阶段训练方法；外部执行卡负责情节、结构、冲突、爽点和章末钩子。
- required before expansion: execution-card schema guard, deterministic quality rules, agent majority gate, and recorded human arbitration for blocker votes.
```

- [ ] **Step 3: Run full unit suite**

Run:

```powershell
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 4: Run whitespace verification**

Run:

```powershell
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 5: Commit docs**

```powershell
git add docs/stage4-1-quality-eval-guide.zh.md docs/stage4-decision-log.zh.md
git commit -m "docs: document agent review acceptance gate"
```

## Self-Review Checklist

- Spec coverage:
  - Execution-card schema mismatch is covered by Tasks 1 and 2.
  - Deterministic blind spots are covered by Task 3.
  - Agent majority voting, blocker arbitration, and incomplete reviews are covered by Task 4.
  - CLI import/mock workflows are covered by Task 5.
  - Stage 4 reporting is covered by Task 6.
  - Documentation and final verification are covered by Task 7.
- Scope boundary:
  - This plan does not change QLoRA, LoRA rank, learning rate, SFT packing, DPO, ORPO, SimPO, KTO, or data expansion.
  - The small model remains a prose executor. It is judged on executing external cards, not inventing plot plans.
- Verification:
  - Each implementation task has a failing-test step, a pass step, and a commit step.
  - Final verification uses `python -m pytest -q` and `git diff --check`.
