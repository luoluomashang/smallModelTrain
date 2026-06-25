# Stage 5C Formal Execution Cards And Data Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Stage 5C so formal SFT consumes approved `ChapterExecutionCard` assets with stable data splits, leakage checks, near-duplicate checks, and dataset manifests, while closing the remaining Stage 5B documentation command mismatch.

**Architecture:** Keep command entrypoints in `scripts/` and put reusable schema, compiler, renderer, validation, split, dedup, and manifest logic in focused modules under `src/small_model_train/`. Preserve the existing draft-card smoke/dev path behind `--allow-draft-cards`; make the default formal path stricter by requiring formal cards, approved/frozen StyleContract JSON, one approved/frozen card per trainable chapter, and a dataset manifest.

**Tech Stack:** Python 3.10+, pytest, JSON/JSONL, Markdown docs, stdlib `hashlib`, current `small_model_train` package, existing script/test layout.

---

## Scope Check

This plan implements the approved spec:

- `docs/superpowers/specs/2026-06-25-stage5c-formal-execution-cards-data-integrity-design.md`

It includes the Stage 5B documentation closure because the runtime behavior already changed in Stage 5B. It does not implement author feedback, same-plot revisions, rejection sampling, DPO, experiment matrices, or larger training.

Current expected baseline before this plan:

```powershell
python -m pytest -q
```

Expected current result: `424 passed`.

---

## File Map

- Create: `src/small_model_train/schemas/__init__.py`
  - Marks schema helpers as an importable package.
- Create: `src/small_model_train/schemas/chapter_execution_card.py`
  - Defines `ChapterExecutionCard` validation, canonical hashes, status rules, source text hashing, and JSON read/write helpers.
- Create: `src/small_model_train/cards/__init__.py`
  - Marks card helpers as an importable package.
- Create: `src/small_model_train/cards/card_compiler.py`
  - Converts current draft cards plus chapters plus StyleContract assets into reviewed formal-card candidates.
- Create: `src/small_model_train/cards/card_renderer.py`
  - Renders a formal card with a supplied StyleContract into the same prompt surface used by SFT.
- Create: `src/small_model_train/cards/card_validator.py`
  - Validates batches of formal cards against chapters and StyleContract assets, including one-card-per-chapter and leakage checks.
- Create: `src/small_model_train/data/__init__.py`
  - Marks data-integrity helpers as an importable package.
- Create: `src/small_model_train/data/dedup.py`
  - Computes deterministic Chinese-character shingle fingerprints and flags near duplicates.
- Create: `src/small_model_train/data/dataset_manifest.py`
  - Builds and writes formal SFT dataset manifests.
- Modify: `src/small_model_train/dataset_split.py`
  - Add grouped deterministic `train` / `validation` / `sealed` split helpers.
- Modify: `src/small_model_train/sft_builder.py`
  - Add formal-card SFT row builder; keep draft-card builder for `--allow-draft-cards`.
- Modify: `scripts/build_sft_dataset.py`
  - Route `--allow-draft-cards` to the existing draft path; route formal mode to the new formal-card path; write optional dataset manifest.
- Create: `scripts/compile_chapter_execution_cards.py`
  - CLI helper for producing reviewed formal-card candidates from draft cards.
- Modify docs:
  - `README.md`
  - `docs/index.zh.md`
  - `docs/project-map.zh.md`
  - `docs/pipeline-flow.zh.md`
  - `docs/stage1-pipeline-guide.zh.md`
  - `docs/stage3-data-bring-up-guide.zh.md`
  - `docs/stage4-smoke-eval-guide.zh.md`
  - `docs/zero-start.zh.md`
  - Create `docs/stage5c-formal-execution-cards-data-integrity.zh.md`
- Tests:
  - Create `tests/test_chapter_execution_card.py`
  - Create `tests/test_card_compiler.py`
  - Create `tests/test_card_validator.py`
  - Create `tests/test_dataset_manifest.py`
  - Modify `tests/test_dataset_split.py`
  - Modify `tests/test_sft_builder.py`

---

## Task 0: Stage 5B Documentation Command Closure

**Files:**
- Modify: `README.md`
- Modify: `docs/pipeline-flow.zh.md`
- Modify: `docs/stage1-pipeline-guide.zh.md`
- Modify: `docs/stage3-data-bring-up-guide.zh.md`
- Modify: `docs/stage4-smoke-eval-guide.zh.md`
- Modify: `docs/zero-start.zh.md`

- [ ] **Step 1: Find stale public draft SFT commands**

Run:

```powershell
rg --pcre2 -n "build_sft_dataset.py(?!.*(--allow-draft-cards|--style-contract-json))" README.md docs --glob "!docs/superpowers/**"
```

Expected before this task: matches in the public guides for old `build_sft_dataset.py` examples.

- [ ] **Step 2: Update smoke/dev examples to include explicit draft override**

In each public guide that builds `data_sft/sft_chapter_v1.jsonl` from `data_cards/chapter_cards.jsonl`, change the command to include `--allow-draft-cards`.

Use this exact shape for draft smoke/dev examples:

```powershell
python scripts/build_sft_dataset.py --cards data_cards/chapter_cards.jsonl --chapters data_clean/chapters_split.jsonl --output data_sft/sft_chapter_v1.jsonl --dataset-info-output data_sft/dataset_info.json --allow-draft-cards
```

If a doc command lacks `--dataset-info-output`, keep its existing output arguments and append only:

```powershell
--allow-draft-cards
```

- [ ] **Step 3: Add a short explanation beside the first changed command in each guide**

Use this wording in Chinese where the command appears:

```markdown
`chapter_cards.jsonl` 仍是 smoke/dev 草稿卡路径，所以这里必须显式使用 `--allow-draft-cards`。formal SFT 需要改用 approved/frozen 的 `ChapterExecutionCard` 文件，并传入 `--style-contract-json`。
```

- [ ] **Step 4: Verify no stale public draft commands remain**

Run:

```powershell
rg --pcre2 -n "build_sft_dataset.py(?!.*(--allow-draft-cards|--style-contract-json))" README.md docs --glob "!docs/superpowers/**"
```

Expected: no matches for runnable public commands. Mentions in code-design tables may remain only if they are file-name references rather than commands.

- [ ] **Step 5: Run docs scan for the intended command split**

Run:

```powershell
rg -n "allow-draft-cards|style-contract-json|ChapterExecutionCard|chapter_execution_cards_approved" README.md docs --glob "!docs/superpowers/**"
```

Expected: public docs mention draft override and formal StyleContract binding.

- [ ] **Step 6: Commit Task 0**

Run:

```bash
git add README.md docs/pipeline-flow.zh.md docs/stage1-pipeline-guide.zh.md docs/stage3-data-bring-up-guide.zh.md docs/stage4-smoke-eval-guide.zh.md docs/zero-start.zh.md
git commit -m "docs: clarify draft sft command gate"
```

---

## Task 1: ChapterExecutionCard Schema

**Files:**
- Create: `src/small_model_train/schemas/__init__.py`
- Create: `src/small_model_train/schemas/chapter_execution_card.py`
- Create: `tests/test_chapter_execution_card.py`

- [ ] **Step 1: Write failing schema tests**

Create `tests/test_chapter_execution_card.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest


STYLE_HASH = "a" * 64
SOURCE_HASH = "b" * 64


def _card(**overrides) -> dict:
    from small_model_train.schemas.chapter_execution_card import (
        build_chapter_execution_card,
    )

    card = build_chapter_execution_card(
        card_id="card-c1-v1",
        chapter_id="c1",
        card_status="approved",
        style_contract_id="contract-v1",
        style_contract_sha256=STYLE_HASH,
        source_chapter_text="这一章用于计算来源哈希。",
        target_platform="hybrid_fanqie_qidian",
        genre_tags=["xuanhuan", "system"],
        hard_constraints={
            "must_include": ["旧仓库", "加钱"],
            "must_not_include": ["作者说明"],
            "continuity_facts": ["林默上一章已经接下委托"],
            "forbidden_future_facts": ["最终幕后身份"],
            "style_bans": ["不要输出提纲"],
        },
        execution_plan={
            "chapter_goal": "林默进入旧仓库并完成谈判。",
            "conflict_beat": "周衡压价并试探林默底线。",
            "payoff_beat": "林默用箱子异常逼对方退让。",
            "chapter_structure": [
                {"step": 1, "name": "入场", "goal": "交代压力", "estimated_chars": "300-500"}
            ],
            "character_states": [
                {"name": "林默", "state": "警惕", "speech_style": "短句"}
            ],
            "ending_hook": "箱子自己响了一下。",
            "target_word_count": "2000-2500中文汉字",
        },
        creative_space={
            "optional_sensory_details": ["雨声", "铁锈味"],
            "optional_dialogue_moves": ["短暂停顿"],
            "optional_micro_conflicts": ["临时加价"],
            "allowed_scene_expansion": ["仓库外等待的人群"],
        },
        provenance={
            "source_card_id": "draft-c1",
            "compiler_version": "stage5c_v1",
            "created_at": "2026-06-25T00:00:00Z",
            "reviewer": "author",
            "reviewed_at": "2026-06-25T01:00:00Z",
            "review_notes": "批准。",
            "group_id": "group-c1",
            "split": "train",
        },
    )
    card.update(overrides)
    if overrides:
        from small_model_train.schemas.chapter_execution_card import (
            canonical_card_sha256,
        )

        card["card_sha256"] = canonical_card_sha256(card)
    return card


def test_validate_chapter_execution_card_accepts_valid_card():
    from small_model_train.schemas.chapter_execution_card import (
        is_card_approved_for_formal_sft,
        validate_chapter_execution_card,
    )

    card = validate_chapter_execution_card(_card())

    assert card["schema_version"] == 1
    assert len(card["card_sha256"]) == 64
    assert len(card["source_chapter_sha256"]) == 64
    assert is_card_approved_for_formal_sft(card) is True


def test_card_hash_excludes_card_sha256():
    from small_model_train.schemas.chapter_execution_card import canonical_card_sha256

    card = _card()
    original = card["card_sha256"]
    card["card_sha256"] = "0" * 64

    assert canonical_card_sha256(card) == original


@pytest.mark.parametrize("status", ["draft", "reviewed", "rejected"])
def test_non_formal_status_is_not_formal(status: str):
    from small_model_train.schemas.chapter_execution_card import (
        is_card_approved_for_formal_sft,
    )

    assert is_card_approved_for_formal_sft(_card(card_status=status)) is False


def test_invalid_status_is_rejected():
    from small_model_train.schemas.chapter_execution_card import (
        validate_chapter_execution_card,
    )

    with pytest.raises(ValueError, match="card_status"):
        validate_chapter_execution_card(_card(card_status="pending"))


def test_missing_nested_required_field_is_rejected():
    from small_model_train.schemas.chapter_execution_card import (
        canonical_card_sha256,
        validate_chapter_execution_card,
    )

    card = _card()
    del card["execution_plan"]["chapter_goal"]
    card["card_sha256"] = canonical_card_sha256(card)

    with pytest.raises(ValueError, match="execution_plan.chapter_goal"):
        validate_chapter_execution_card(card)


def test_hash_mismatch_is_rejected():
    from small_model_train.schemas.chapter_execution_card import (
        validate_chapter_execution_card,
    )

    card = _card()
    card["execution_plan"]["chapter_goal"] = "偷偷改目标。"

    with pytest.raises(ValueError, match="card_sha256 mismatch"):
        validate_chapter_execution_card(card)


def test_read_and_write_chapter_execution_cards_round_trip(tmp_path: Path):
    from small_model_train.schemas.chapter_execution_card import (
        read_chapter_execution_cards,
        write_chapter_execution_cards,
    )

    path = tmp_path / "cards.jsonl"
    write_chapter_execution_cards(path, [_card()])

    assert read_chapter_execution_cards(path)[0]["card_id"] == "card-c1-v1"
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["card_id"] == "card-c1-v1"
```

- [ ] **Step 2: Run schema tests and verify failure**

Run:

```powershell
python -m pytest tests/test_chapter_execution_card.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'small_model_train.schemas'`.

- [ ] **Step 3: Create schema package marker**

Create `src/small_model_train/schemas/__init__.py`:

```python
"""Schema helpers for formal training assets."""
```

- [ ] **Step 4: Implement schema module**

Create `src/small_model_train/schemas/chapter_execution_card.py`:

```python
from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1
CARD_STATUSES = {"draft", "reviewed", "approved", "frozen", "rejected"}
FORMAL_CARD_STATUSES = {"approved", "frozen"}
LOWER_HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_TOP_LEVEL_FIELDS = (
    "schema_version",
    "card_id",
    "chapter_id",
    "card_status",
    "style_contract_id",
    "style_contract_sha256",
    "source_chapter_sha256",
    "card_sha256",
    "target_platform",
    "genre_tags",
    "hard_constraints",
    "execution_plan",
    "creative_space",
    "provenance",
)
REQUIRED_HARD_CONSTRAINT_FIELDS = (
    "must_include",
    "must_not_include",
    "continuity_facts",
    "forbidden_future_facts",
    "style_bans",
)
REQUIRED_EXECUTION_PLAN_FIELDS = (
    "chapter_goal",
    "conflict_beat",
    "payoff_beat",
    "chapter_structure",
    "character_states",
    "ending_hook",
    "target_word_count",
)
REQUIRED_CREATIVE_SPACE_FIELDS = (
    "optional_sensory_details",
    "optional_dialogue_moves",
    "optional_micro_conflicts",
    "allowed_scene_expansion",
)
REQUIRED_PROVENANCE_FIELDS = (
    "source_card_id",
    "compiler_version",
    "created_at",
    "reviewer",
    "reviewed_at",
    "review_notes",
    "group_id",
    "split",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_card_sha256(card: dict[str, Any]) -> str:
    canonical = copy.deepcopy(card)
    canonical.pop("card_sha256", None)
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_chapter_execution_card(
    *,
    card_id: str,
    chapter_id: str,
    card_status: str,
    style_contract_id: str,
    style_contract_sha256: str,
    source_chapter_text: str,
    target_platform: str,
    genre_tags: list[str],
    hard_constraints: dict[str, Any],
    execution_plan: dict[str, Any],
    creative_space: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    card = {
        "schema_version": SCHEMA_VERSION,
        "card_id": card_id,
        "chapter_id": chapter_id,
        "card_status": card_status,
        "style_contract_id": style_contract_id,
        "style_contract_sha256": style_contract_sha256,
        "source_chapter_sha256": text_sha256(source_chapter_text),
        "card_sha256": "",
        "target_platform": target_platform,
        "genre_tags": list(genre_tags),
        "hard_constraints": copy.deepcopy(hard_constraints),
        "execution_plan": copy.deepcopy(execution_plan),
        "creative_space": copy.deepcopy(creative_space),
        "provenance": copy.deepcopy(provenance),
    }
    card["card_sha256"] = canonical_card_sha256(card)
    return validate_chapter_execution_card(card)


def validate_chapter_execution_card(card: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(card, dict):
        raise ValueError("chapter execution card must be a JSON object")
    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in card:
            raise ValueError(f"{field} is required")

    if card["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    _require_non_empty_string(card, "card_id")
    _require_non_empty_string(card, "chapter_id")
    _require_non_empty_string(card, "style_contract_id")
    _require_sha(card, "style_contract_sha256")
    _require_sha(card, "source_chapter_sha256")
    _require_sha(card, "card_sha256")
    _require_non_empty_string(card, "target_platform")
    if card["card_status"] not in CARD_STATUSES:
        raise ValueError("card_status must be one of: " + ", ".join(sorted(CARD_STATUSES)))
    _require_string_list(card, "genre_tags", allow_empty=False)

    _validate_section("hard_constraints", card["hard_constraints"], REQUIRED_HARD_CONSTRAINT_FIELDS)
    for field in REQUIRED_HARD_CONSTRAINT_FIELDS:
        _require_string_list(card["hard_constraints"], field, allow_empty=True, section="hard_constraints")

    _validate_section("execution_plan", card["execution_plan"], REQUIRED_EXECUTION_PLAN_FIELDS)
    for field in ("chapter_goal", "conflict_beat", "payoff_beat", "ending_hook", "target_word_count"):
        _require_non_empty_string(card["execution_plan"], field, section="execution_plan")
    _validate_chapter_structure(card["execution_plan"]["chapter_structure"])
    _validate_character_states(card["execution_plan"]["character_states"])

    _validate_section("creative_space", card["creative_space"], REQUIRED_CREATIVE_SPACE_FIELDS)
    for field in REQUIRED_CREATIVE_SPACE_FIELDS:
        _require_string_list(card["creative_space"], field, allow_empty=True, section="creative_space")

    _validate_section("provenance", card["provenance"], REQUIRED_PROVENANCE_FIELDS)
    for field in REQUIRED_PROVENANCE_FIELDS:
        if not isinstance(card["provenance"][field], str):
            raise ValueError(f"provenance.{field} must be a string")

    expected = canonical_card_sha256(card)
    if card["card_sha256"] != expected:
        raise ValueError("card_sha256 mismatch")
    return card


def validate_chapter_execution_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for index, card in enumerate(cards, start=1):
        try:
            validate_chapter_execution_card(card)
        except ValueError as exc:
            raise ValueError(f"row {index}: {exc}") from exc
    return cards


def is_card_approved_for_formal_sft(card: dict[str, Any]) -> bool:
    validated = validate_chapter_execution_card(card)
    return validated["card_status"] in FORMAL_CARD_STATUSES


def read_chapter_execution_cards(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    cards: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{file_path}:{line_number} is not valid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{file_path}:{line_number} is not a JSON object")
            cards.append(validate_chapter_execution_card(row))
    return cards


def write_chapter_execution_cards(path: str | Path, cards: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for card in cards:
            validated = validate_chapter_execution_card(card)
            handle.write(json.dumps(validated, ensure_ascii=False) + "\n")


def _validate_section(section_name: str, value: Any, required_fields: tuple[str, ...]) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{section_name} must be a JSON object")
    for field in required_fields:
        if field not in value:
            raise ValueError(f"{section_name}.{field} is required")


def _require_non_empty_string(values: dict[str, Any], field: str, *, section: str = "") -> None:
    value = values.get(field)
    label = f"{section}.{field}" if section else field
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")


def _require_sha(values: dict[str, Any], field: str) -> None:
    value = values.get(field)
    if not isinstance(value, str) or LOWER_HEX_SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a 64-character lowercase hex string")


def _require_string_list(
    values: dict[str, Any],
    field: str,
    *,
    allow_empty: bool,
    section: str = "",
) -> None:
    value = values.get(field)
    label = f"{section}.{field}" if section else field
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    if not allow_empty and not value:
        raise ValueError(f"{label} must be a non-empty list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{label} must contain non-empty strings")


def _validate_chapter_structure(items: Any) -> None:
    if not isinstance(items, list) or not items:
        raise ValueError("execution_plan.chapter_structure must be a non-empty list")
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"execution_plan.chapter_structure[{index}] must be a JSON object")
        step = item.get("step")
        if type(step) is not int or step < 1:
            raise ValueError(f"execution_plan.chapter_structure[{index}].step must be a positive integer")
        for field in ("name", "goal", "estimated_chars"):
            value = item.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"execution_plan.chapter_structure[{index}].{field} must be a non-empty string")


def _validate_character_states(items: Any) -> None:
    if not isinstance(items, list) or not items:
        raise ValueError("execution_plan.character_states must be a non-empty list")
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"execution_plan.character_states[{index}] must be a JSON object")
        for field in ("name", "state", "speech_style"):
            value = item.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"execution_plan.character_states[{index}].{field} must be a non-empty string")
```

- [ ] **Step 5: Run schema tests**

Run:

```powershell
python -m pytest tests/test_chapter_execution_card.py -q
```

Expected: pass.

- [ ] **Step 6: Commit Task 1**

Run:

```bash
git add src/small_model_train/schemas/__init__.py src/small_model_train/schemas/chapter_execution_card.py tests/test_chapter_execution_card.py
git commit -m "feat: define formal chapter execution card schema"
```

---

## Task 2: Formal Card Renderer And Compiler

**Files:**
- Create: `src/small_model_train/cards/__init__.py`
- Create: `src/small_model_train/cards/card_renderer.py`
- Create: `src/small_model_train/cards/card_compiler.py`
- Create: `tests/test_card_compiler.py`

- [ ] **Step 1: Write failing renderer/compiler tests**

Create `tests/test_card_compiler.py`:

```python
from __future__ import annotations

import subprocess
import sys

import pytest

from small_model_train.io_utils import read_jsonl, write_jsonl


def _style_contract() -> dict:
    from small_model_train.style_contract import build_style_contract_asset

    return build_style_contract_asset(
        style_contract_id="contract-v1",
        approval_status="approved",
        source_corpus={
            "path": "chapters.jsonl",
            "sha256": "b" * 64,
            "quality_filter": "quality_tag=A",
            "row_count": 1,
            "selected_rows": 1,
            "split_summary": {"train": 1},
        },
        profile_metrics={
            "chapter_count": 1,
            "avg_dialogue_ratio": 0.1,
            "avg_paragraph_chars": 20,
            "ai_taste": {"phrase_hits": {}, "total_hits": 0, "hits_per_10k_chars": 0},
        },
    )


def _draft_card(**overrides) -> dict:
    card = {
        "id": "c1",
        "style_contract": "旧风格契约",
        "previous_summary": "上一章林默接下委托。",
        "chapter_goal": "林默进入旧仓库并完成谈判。",
        "chapter_structure": [
            {"step": 1, "name": "入场", "goal": "交代仓库压力", "estimated_chars": "300-500"}
        ],
        "character_states": [
            {"name": "林默", "state": "警惕", "speech_style": "短句"}
        ],
        "conflict_beat": "周衡压价。",
        "payoff_beat": "林默发现箱子异常。",
        "must_include": ["旧仓库", "加钱"],
        "must_not_include": ["作者说明"],
        "ending_hook": "箱子自己响了一下。",
        "target_word_count": "2000-2500中文汉字",
    }
    card.update(overrides)
    return card


def _chapter() -> dict:
    return {
        "id": "c1",
        "text": "这一章用于计算来源哈希。",
        "split": "train",
        "quality_tag": "A",
    }


def test_compile_draft_card_outputs_reviewed_formal_candidate():
    from small_model_train.cards.card_compiler import compile_chapter_execution_card

    card = compile_chapter_execution_card(
        draft_card=_draft_card(),
        chapter=_chapter(),
        style_contract=_style_contract(),
        group_id="group-c1",
        split="train",
    )

    assert card["card_id"] == "card-c1-v1"
    assert card["chapter_id"] == "c1"
    assert card["card_status"] == "reviewed"
    assert card["style_contract_id"] == "contract-v1"
    assert card["hard_constraints"]["must_include"] == ["旧仓库", "加钱"]
    assert card["execution_plan"]["conflict_beat"] == "周衡压价。"
    assert card["creative_space"]["allowed_scene_expansion"]


def test_compile_rejects_abstract_only_card():
    from small_model_train.cards.card_compiler import compile_chapter_execution_card

    with pytest.raises(ValueError, match="abstract-only"):
        compile_chapter_execution_card(
            draft_card=_draft_card(
                chapter_goal="节奏紧凑，写得爽一点，减少 AI 味。",
                conflict_beat="",
                payoff_beat="",
                must_include=[],
                ending_hook="",
            ),
            chapter=_chapter(),
            style_contract=_style_contract(),
            group_id="group-c1",
            split="train",
        )


def test_render_chapter_execution_input_uses_style_contract_and_formal_sections():
    from small_model_train.cards.card_compiler import compile_chapter_execution_card
    from small_model_train.cards.card_renderer import render_chapter_execution_input

    contract = _style_contract()
    formal_card = compile_chapter_execution_card(
        draft_card=_draft_card(),
        chapter=_chapter(),
        style_contract=contract,
        group_id="group-c1",
        split="train",
    )

    rendered = render_chapter_execution_input(formal_card, contract)

    assert "【风格契约】" in rendered
    assert contract["prompt_rules"]["style_contract_text"] in rendered
    assert "【本章目标】\n林默进入旧仓库并完成谈判。" in rendered
    assert "【创作自由】" in rendered
    assert "这一章用于计算来源哈希" not in rendered


def test_compile_chapter_execution_cards_cli_writes_reviewed_cards(tmp_path):
    from small_model_train.style_contract import write_style_contract_asset

    cards_path = tmp_path / "draft_cards.jsonl"
    chapters_path = tmp_path / "chapters.jsonl"
    contract_path = tmp_path / "style_contract.json"
    output_path = tmp_path / "formal_cards.jsonl"
    write_jsonl(cards_path, [_draft_card()])
    write_jsonl(chapters_path, [_chapter()])
    write_style_contract_asset(contract_path, _style_contract())

    result = subprocess.run(
        [
            sys.executable,
            "scripts/compile_chapter_execution_cards.py",
            "--cards",
            str(cards_path),
            "--chapters",
            str(chapters_path),
            "--style-contract-json",
            str(contract_path),
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    rows = read_jsonl(output_path)
    assert rows[0]["card_status"] == "reviewed"
    assert rows[0]["chapter_id"] == "c1"
```

- [ ] **Step 2: Run compiler tests and verify failure**

Run:

```powershell
python -m pytest tests/test_card_compiler.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'small_model_train.cards'`.

- [ ] **Step 3: Create card package marker**

Create `src/small_model_train/cards/__init__.py`:

```python
"""Formal chapter-card helpers."""
```

- [ ] **Step 4: Implement formal renderer**

Create `src/small_model_train/cards/card_renderer.py`:

```python
from __future__ import annotations

from typing import Any

from small_model_train.prompt_renderer import render_execution_input
from small_model_train.schemas.chapter_execution_card import validate_chapter_execution_card
from small_model_train.style_contract import validate_style_contract_asset


def formal_card_to_prompt_card(card: dict[str, Any], style_contract: dict[str, Any]) -> dict[str, Any]:
    validated_card = validate_chapter_execution_card(card)
    validated_contract = validate_style_contract_asset(style_contract)
    plan = validated_card["execution_plan"]
    constraints = validated_card["hard_constraints"]
    creative = validated_card["creative_space"]
    creative_lines = []
    for label, field in (
        ("感官细节", "optional_sensory_details"),
        ("对白变化", "optional_dialogue_moves"),
        ("微冲突", "optional_micro_conflicts"),
        ("场景扩展", "allowed_scene_expansion"),
    ):
        values = creative.get(field, [])
        body = "、".join(values) if values else "无"
        creative_lines.append(f"{label}：{body}")
    return {
        "id": validated_card["chapter_id"],
        "style_contract": validated_contract["prompt_rules"]["style_contract_text"],
        "previous_summary": "见连续性事实：" + "；".join(constraints.get("continuity_facts", [])),
        "chapter_goal": plan["chapter_goal"],
        "conflict_beat": plan["conflict_beat"],
        "payoff_beat": plan["payoff_beat"],
        "chapter_structure": plan["chapter_structure"],
        "character_states": plan["character_states"],
        "must_include": constraints["must_include"],
        "must_not_include": constraints["must_not_include"] + constraints["forbidden_future_facts"] + constraints["style_bans"],
        "ending_hook": plan["ending_hook"],
        "target_word_count": plan["target_word_count"],
        "creative_space_text": "\n".join(creative_lines),
    }


def render_chapter_execution_input(card: dict[str, Any], style_contract: dict[str, Any]) -> str:
    prompt_card = formal_card_to_prompt_card(card, style_contract)
    rendered = render_execution_input(prompt_card)
    return rendered + "\n【创作自由】\n" + prompt_card["creative_space_text"]
```

- [ ] **Step 5: Implement compiler**

Create `src/small_model_train/cards/card_compiler.py`:

```python
from __future__ import annotations

from typing import Any

from small_model_train.schemas.chapter_execution_card import build_chapter_execution_card
from small_model_train.style_contract import validate_style_contract_asset

COMPILER_VERSION = "stage5c_v1"
ABSTRACT_ONLY_PHRASES = ("节奏紧凑", "写得爽一点", "减少 AI 味", "更有代入感")


def compile_chapter_execution_card(
    *,
    draft_card: dict[str, Any],
    chapter: dict[str, Any],
    style_contract: dict[str, Any],
    group_id: str,
    split: str,
    card_status: str = "reviewed",
) -> dict[str, Any]:
    contract = validate_style_contract_asset(style_contract)
    chapter_id = str(chapter.get("id") or "")
    draft_id = str(draft_card.get("id") or "")
    if not chapter_id or draft_id != chapter_id:
        raise ValueError(f"draft card id must match chapter id: card={draft_id!r}, chapter={chapter_id!r}")
    _reject_abstract_only_card(draft_card)
    execution_plan = {
        "chapter_goal": _string(draft_card, "chapter_goal"),
        "conflict_beat": _string(draft_card, "conflict_beat"),
        "payoff_beat": _string(draft_card, "payoff_beat"),
        "chapter_structure": _normalize_structure(draft_card.get("chapter_structure", [])),
        "character_states": _normalize_character_states(draft_card.get("character_states", [])),
        "ending_hook": _string(draft_card, "ending_hook"),
        "target_word_count": _string(draft_card, "target_word_count", "2000-2500中文汉字"),
    }
    hard_constraints = {
        "must_include": _string_list(draft_card.get("must_include", [])),
        "must_not_include": _string_list(draft_card.get("must_not_include", [])),
        "continuity_facts": [_string(draft_card, "previous_summary")] if _string(draft_card, "previous_summary") else [],
        "forbidden_future_facts": [],
        "style_bans": contract.get("ai_taste_guardrails", {}).get("banned_phrases", []),
    }
    creative_space = {
        "optional_sensory_details": ["场景气味", "动作停顿"],
        "optional_dialogue_moves": ["短问短答", "沉默打断"],
        "optional_micro_conflicts": ["临时阻碍", "误判代价"],
        "allowed_scene_expansion": ["不改变硬约束的环境细节"],
    }
    provenance = {
        "source_card_id": draft_id,
        "compiler_version": COMPILER_VERSION,
        "created_at": "2026-06-25T00:00:00Z",
        "reviewer": "",
        "reviewed_at": "",
        "review_notes": "",
        "group_id": group_id,
        "split": split,
    }
    return build_chapter_execution_card(
        card_id=f"card-{chapter_id}-v1",
        chapter_id=chapter_id,
        card_status=card_status,
        style_contract_id=contract["style_contract_id"],
        style_contract_sha256=contract["contract_sha256"],
        source_chapter_text=str(chapter.get("text") or ""),
        target_platform=str(draft_card.get("target_platform") or "hybrid_fanqie_qidian"),
        genre_tags=_string_list(draft_card.get("genre_tags", ["male_webnovel"])),
        hard_constraints=hard_constraints,
        execution_plan=execution_plan,
        creative_space=creative_space,
        provenance=provenance,
    )


def _reject_abstract_only_card(card: dict[str, Any]) -> None:
    concrete_values = (
        _string(card, "conflict_beat"),
        _string(card, "payoff_beat"),
        _string(card, "ending_hook"),
        " ".join(_string_list(card.get("must_include", []))),
    )
    if any(value.strip() for value in concrete_values):
        return
    goal = _string(card, "chapter_goal")
    if goal and any(phrase in goal for phrase in ABSTRACT_ONLY_PHRASES):
        raise ValueError(f"abstract-only card cannot become formal card: {goal}")
    raise ValueError("abstract-only card cannot become formal card: missing executable facts")


def _string(values: dict[str, Any], field: str, default: str = "") -> str:
    value = values.get(field, default)
    return str(value).strip() if value is not None else default


def _string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _normalize_structure(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list) or not items:
        raise ValueError("chapter_structure is required for formal card compilation")
    normalized = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"chapter_structure[{index - 1}] must be a JSON object")
        normalized.append(
            {
                "step": int(item.get("step") or index),
                "name": str(item.get("name") or "").strip(),
                "goal": str(item.get("goal") or "").strip(),
                "estimated_chars": str(item.get("estimated_chars") or "").strip(),
            }
        )
    return normalized


def _normalize_character_states(items: Any) -> list[dict[str, str]]:
    if not isinstance(items, list) or not items:
        raise ValueError("character_states is required for formal card compilation")
    normalized = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("character_states items must be JSON objects")
        normalized.append(
            {
                "name": str(item.get("name") or "").strip(),
                "state": str(item.get("state") or "").strip(),
                "speech_style": str(item.get("speech_style") or "").strip(),
            }
        )
    return normalized
```

- [ ] **Step 6: Add compiler CLI**

Create `scripts/compile_chapter_execution_cards.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.cards.card_compiler import compile_chapter_execution_card
from small_model_train.io_utils import read_jsonl
from small_model_train.schemas.chapter_execution_card import write_chapter_execution_cards
from small_model_train.style_contract import read_style_contract_asset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", required=True)
    parser.add_argument("--chapters", required=True)
    parser.add_argument("--style-contract-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    draft_cards = read_jsonl(args.cards)
    chapters = {row["id"]: row for row in read_jsonl(args.chapters)}
    style_contract = read_style_contract_asset(args.style_contract_json)
    formal_cards = []
    try:
        for draft_card in draft_cards:
            chapter_id = draft_card.get("id")
            chapter = chapters.get(chapter_id)
            if chapter is None:
                raise ValueError(f"chapter not found for draft card: {chapter_id}")
            formal_cards.append(
                compile_chapter_execution_card(
                    draft_card=draft_card,
                    chapter=chapter,
                    style_contract=style_contract,
                    group_id=str(chapter.get("group_id") or chapter_id),
                    split=str(chapter.get("split") or "train"),
                )
            )
        write_chapter_execution_cards(args.output, formal_cards)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"wrote {len(formal_cards)} chapter execution cards to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: Run compiler tests**

Run:

```powershell
python -m pytest tests/test_card_compiler.py -q
```

Expected: pass.

- [ ] **Step 8: Commit Task 2**

Run:

```bash
git add src/small_model_train/cards/__init__.py src/small_model_train/cards/card_renderer.py src/small_model_train/cards/card_compiler.py scripts/compile_chapter_execution_cards.py tests/test_card_compiler.py
git commit -m "feat: compile formal chapter execution cards"
```

---

## Task 3: Stable Grouped Split And Near-Duplicate Checks

**Files:**
- Modify: `src/small_model_train/dataset_split.py`
- Create: `src/small_model_train/data/__init__.py`
- Create: `src/small_model_train/data/dedup.py`
- Modify: `tests/test_dataset_split.py`

- [ ] **Step 1: Add failing split and dedup tests**

Append to `tests/test_dataset_split.py`:

```python

def test_split_grouped_rows_is_deterministic_and_non_overlapping():
    from small_model_train.dataset_split import split_grouped_rows

    rows = [{"id": f"chapter_{index:04d}", "text": f"正文{index}"} for index in range(12)]
    first = split_grouped_rows(rows, validation_count=2, sealed_count=3, seed=9)
    second = split_grouped_rows(rows, validation_count=2, sealed_count=3, seed=9)

    assert first == second
    assert sum(1 for row in first if row["split"] == "validation") == 2
    assert sum(1 for row in first if row["split"] == "sealed") == 3
    assert sum(1 for row in first if row["split"] == "train") == 7
    assert len({row["group_id"] for row in first}) == 12
    assert all(len(row["group_sha256"]) == 64 for row in first)


def test_split_grouped_rows_rejects_negative_counts():
    from small_model_train.dataset_split import split_grouped_rows

    with pytest.raises(ValueError, match="validation_count"):
        split_grouped_rows([{"id": "a", "text": "正文"}], validation_count=-1, sealed_count=0)
    with pytest.raises(ValueError, match="sealed_count"):
        split_grouped_rows([{"id": "a", "text": "正文"}], validation_count=0, sealed_count=-1)


def test_find_near_duplicate_pairs_flags_high_overlap():
    from small_model_train.data.dedup import find_near_duplicate_pairs

    rows = [
        {"id": "train_1", "split": "train", "text": "林默走进旧仓库发现箱子正在响动"},
        {"id": "sealed_1", "split": "sealed", "text": "林默走进旧仓库发现箱子正在响动"},
        {"id": "sealed_2", "split": "sealed", "text": "另一条完全不同的章节内容"},
    ]

    pairs = find_near_duplicate_pairs(rows, threshold=0.8, shingle_size=4)

    assert pairs == [
        {
            "left_id": "train_1",
            "left_split": "train",
            "right_id": "sealed_1",
            "right_split": "sealed",
            "overlap": 1.0,
        }
    ]
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
python -m pytest tests/test_dataset_split.py::test_split_grouped_rows_is_deterministic_and_non_overlapping tests/test_dataset_split.py::test_split_grouped_rows_rejects_negative_counts tests/test_dataset_split.py::test_find_near_duplicate_pairs_flags_high_overlap -q
```

Expected: fail because `split_grouped_rows` and `small_model_train.data.dedup` do not exist.

- [ ] **Step 3: Create data package marker**

Create `src/small_model_train/data/__init__.py`:

```python
"""Data integrity helpers for formal SFT assets."""
```

- [ ] **Step 4: Add grouped split helper**

Append to `src/small_model_train/dataset_split.py`:

```python

import hashlib


def split_grouped_rows(
    rows: list[dict],
    validation_count: int,
    sealed_count: int,
    seed: int = 20260625,
) -> list[dict]:
    if validation_count < 0:
        raise ValueError("validation_count must be >= 0")
    if sealed_count < 0:
        raise ValueError("sealed_count must be >= 0")
    ranked = sorted(
        enumerate(rows),
        key=lambda item: _split_rank(item[1], seed, item[0]),
    )
    validation_indexes = {index for index, _row in ranked[:validation_count]}
    sealed_indexes = {
        index
        for index, _row in ranked[validation_count : validation_count + sealed_count]
    }
    output = []
    for index, row in enumerate(rows):
        updated = dict(row)
        if index in validation_indexes:
            split = "validation"
        elif index in sealed_indexes:
            split = "sealed"
        else:
            split = "train"
        group_sha = _group_sha256(row, seed, index)
        updated["split"] = split
        updated["group_id"] = f"group-{group_sha[:16]}"
        updated["group_sha256"] = group_sha
        output.append(updated)
    return output


def _split_rank(row: dict, seed: int, index: int) -> str:
    return _group_sha256(row, seed, index)


def _group_sha256(row: dict, seed: int, index: int) -> str:
    chapter_id = str(row.get("id") or index)
    text = str(row.get("text") or "")
    payload = f"{seed}\n{index}\n{chapter_id}\n{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

- [ ] **Step 5: Implement dedup helper**

Create `src/small_model_train/data/dedup.py`:

```python
from __future__ import annotations

import re
from typing import Any

CHINESE_RE = re.compile(r"[\u4e00-\u9fff]+")


def normalized_chinese_text(text: str) -> str:
    return "".join(CHINESE_RE.findall(text))


def chinese_shingles(text: str, shingle_size: int = 12) -> set[str]:
    normalized = normalized_chinese_text(text)
    if not normalized:
        return set()
    if len(normalized) <= shingle_size:
        return {normalized}
    return {
        normalized[index : index + shingle_size]
        for index in range(0, len(normalized) - shingle_size + 1)
    }


def jaccard_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0
    return round(len(left & right) / len(left | right), 4)


def find_near_duplicate_pairs(
    rows: list[dict[str, Any]],
    *,
    threshold: float = 0.82,
    shingle_size: int = 12,
) -> list[dict[str, Any]]:
    fingerprints = [
        (
            str(row.get("id") or ""),
            str(row.get("split") or ""),
            chinese_shingles(str(row.get("text") or ""), shingle_size),
        )
        for row in rows
    ]
    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(fingerprints):
        for right in fingerprints[left_index + 1 :]:
            left_id, left_split, left_shingles = left
            right_id, right_split, right_shingles = right
            if left_split == right_split:
                continue
            overlap = jaccard_overlap(left_shingles, right_shingles)
            if overlap >= threshold:
                pairs.append(
                    {
                        "left_id": left_id,
                        "left_split": left_split,
                        "right_id": right_id,
                        "right_split": right_split,
                        "overlap": overlap,
                    }
                )
    return pairs
```

- [ ] **Step 6: Run split and dedup tests**

Run:

```powershell
python -m pytest tests/test_dataset_split.py -q
```

Expected: pass.

- [ ] **Step 7: Commit Task 3**

Run:

```bash
git add src/small_model_train/dataset_split.py src/small_model_train/data/__init__.py src/small_model_train/data/dedup.py tests/test_dataset_split.py
git commit -m "feat: add stable grouped split and dedup checks"
```

---

## Task 4: Formal Card Batch Validator And Leakage Checks

**Files:**
- Create: `src/small_model_train/cards/card_validator.py`
- Create: `tests/test_card_validator.py`

- [ ] **Step 1: Write failing validator tests**

Create `tests/test_card_validator.py`:

```python
from __future__ import annotations

import pytest


def _style_contract() -> dict:
    from small_model_train.style_contract import build_style_contract_asset

    return build_style_contract_asset(
        style_contract_id="contract-v1",
        approval_status="approved",
        source_corpus={
            "path": "chapters.jsonl",
            "sha256": "b" * 64,
            "quality_filter": "quality_tag=A",
            "row_count": 2,
            "selected_rows": 2,
            "split_summary": {"train": 2},
        },
        profile_metrics={
            "chapter_count": 1,
            "avg_dialogue_ratio": 0.1,
            "avg_paragraph_chars": 20,
            "ai_taste": {"phrase_hits": {}, "total_hits": 0, "hits_per_10k_chars": 0},
        },
    )


def _formal_card(chapter_id: str, text: str = "这一章用于计算来源哈希。", **overrides) -> dict:
    from small_model_train.schemas.chapter_execution_card import build_chapter_execution_card

    contract = _style_contract()
    card = build_chapter_execution_card(
        card_id=f"card-{chapter_id}-v1",
        chapter_id=chapter_id,
        card_status="approved",
        style_contract_id=contract["style_contract_id"],
        style_contract_sha256=contract["contract_sha256"],
        source_chapter_text=text,
        target_platform="hybrid_fanqie_qidian",
        genre_tags=["xuanhuan"],
        hard_constraints={
            "must_include": ["旧仓库"],
            "must_not_include": ["作者说明"],
            "continuity_facts": ["上一章压力还在"],
            "forbidden_future_facts": ["未来真相"],
            "style_bans": [],
        },
        execution_plan={
            "chapter_goal": "林默进入旧仓库。",
            "conflict_beat": "对手压价。",
            "payoff_beat": "林默反制。",
            "chapter_structure": [
                {"step": 1, "name": "入场", "goal": "交代压力", "estimated_chars": "300"}
            ],
            "character_states": [
                {"name": "林默", "state": "警惕", "speech_style": "短句"}
            ],
            "ending_hook": "箱子响了一下。",
            "target_word_count": "2000-2500中文汉字",
        },
        creative_space={
            "optional_sensory_details": [],
            "optional_dialogue_moves": [],
            "optional_micro_conflicts": [],
            "allowed_scene_expansion": [],
        },
        provenance={
            "source_card_id": chapter_id,
            "compiler_version": "stage5c_v1",
            "created_at": "2026-06-25T00:00:00Z",
            "reviewer": "",
            "reviewed_at": "",
            "review_notes": "",
            "group_id": f"group-{chapter_id}",
            "split": "train",
        },
    )
    card.update(overrides)
    if overrides:
        from small_model_train.schemas.chapter_execution_card import canonical_card_sha256

        card["card_sha256"] = canonical_card_sha256(card)
    return card


def test_validate_formal_card_batch_accepts_one_card_per_train_chapter():
    from small_model_train.cards.card_validator import validate_formal_card_batch

    chapters = [{"id": "c1", "text": "这一章用于计算来源哈希。", "split": "train", "quality_tag": "A"}]
    result = validate_formal_card_batch([_formal_card("c1")], chapters, _style_contract())

    assert result["passed"] is True
    assert result["errors"] == []
    assert result["card_by_chapter_id"]["c1"]["card_id"] == "card-c1-v1"


def test_validate_formal_card_batch_rejects_missing_train_chapter_card():
    from small_model_train.cards.card_validator import validate_formal_card_batch

    chapters = [{"id": "c1", "text": "正文", "split": "train", "quality_tag": "A"}]
    result = validate_formal_card_batch([], chapters, _style_contract())

    assert result["passed"] is False
    assert "missing formal card for train chapter: c1" in result["errors"]


def test_validate_formal_card_batch_rejects_duplicate_cards():
    from small_model_train.cards.card_validator import validate_formal_card_batch

    chapters = [{"id": "c1", "text": "这一章用于计算来源哈希。", "split": "train", "quality_tag": "A"}]
    duplicate = _formal_card("c1", card_id="card-c1-v2")
    result = validate_formal_card_batch([_formal_card("c1"), duplicate], chapters, _style_contract())

    assert result["passed"] is False
    assert "duplicate formal cards for chapter c1" in "\n".join(result["errors"])


def test_validate_formal_card_batch_rejects_source_hash_mismatch():
    from small_model_train.cards.card_validator import validate_formal_card_batch

    chapters = [{"id": "c1", "text": "不同正文", "split": "train", "quality_tag": "A"}]
    result = validate_formal_card_batch([_formal_card("c1")], chapters, _style_contract())

    assert result["passed"] is False
    assert "source_chapter_sha256 mismatch: c1" in "\n".join(result["errors"])


def test_validate_formal_card_batch_rejects_future_context_leakage():
    from small_model_train.cards.card_validator import validate_formal_card_batch

    card = _formal_card("c1")
    card["hard_constraints"]["must_include"] = ["未来章节独有内容非常明显"]
    from small_model_train.schemas.chapter_execution_card import canonical_card_sha256

    card["card_sha256"] = canonical_card_sha256(card)
    chapters = [
        {"id": "c1", "text": "这一章用于计算来源哈希。", "split": "train", "quality_tag": "A"},
        {"id": "c2", "text": "未来章节独有内容非常明显，应当禁止泄漏。", "split": "sealed", "quality_tag": "A"},
    ]

    result = validate_formal_card_batch([card], chapters, _style_contract())

    assert result["passed"] is False
    assert "future-context leakage" in "\n".join(result["errors"])
```

- [ ] **Step 2: Run validator tests and verify failure**

Run:

```powershell
python -m pytest tests/test_card_validator.py -q
```

Expected: fail because `small_model_train.cards.card_validator` does not exist.

- [ ] **Step 3: Implement validator**

Create `src/small_model_train/cards/card_validator.py`:

```python
from __future__ import annotations

from typing import Any

from small_model_train.cards.card_renderer import render_chapter_execution_input
from small_model_train.schemas.chapter_execution_card import (
    FORMAL_CARD_STATUSES,
    text_sha256,
    validate_chapter_execution_card,
)
from small_model_train.style_contract import validate_style_contract_asset
from small_model_train.text_utils import count_chinese_chars

LEAK_MIN_CHARS = 12


def validate_formal_card_batch(
    cards: list[dict[str, Any]],
    chapters: list[dict[str, Any]],
    style_contract: dict[str, Any],
    *,
    require_all_train_chapters: bool = True,
) -> dict[str, Any]:
    contract = validate_style_contract_asset(style_contract)
    errors: list[str] = []
    chapter_by_id = {str(chapter.get("id")): chapter for chapter in chapters}
    train_chapter_ids = {
        str(chapter.get("id"))
        for chapter in chapters
        if chapter.get("split") == "train" and chapter.get("quality_tag") == "A"
    }
    card_by_chapter_id: dict[str, dict[str, Any]] = {}

    for raw_card in cards:
        try:
            card = validate_chapter_execution_card(raw_card)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        chapter_id = card["chapter_id"]
        if card["card_status"] not in FORMAL_CARD_STATUSES:
            errors.append(f"formal card must be approved or frozen: {card['card_id']}")
            continue
        if card["style_contract_id"] != contract["style_contract_id"]:
            errors.append(f"style_contract_id mismatch: {card['card_id']}")
        if card["style_contract_sha256"] != contract["contract_sha256"]:
            errors.append(f"style_contract_sha256 mismatch: {card['card_id']}")
        chapter = chapter_by_id.get(chapter_id)
        if chapter is None:
            errors.append(f"formal card points to missing chapter: {chapter_id}")
            continue
        expected_source_hash = text_sha256(str(chapter.get("text") or ""))
        if card["source_chapter_sha256"] != expected_source_hash:
            errors.append(f"source_chapter_sha256 mismatch: {chapter_id}")
        if chapter_id in card_by_chapter_id:
            errors.append(
                "duplicate formal cards for chapter "
                f"{chapter_id}: {card_by_chapter_id[chapter_id]['card_id']}, {card['card_id']}"
            )
        else:
            card_by_chapter_id[chapter_id] = card
        errors.extend(_leakage_errors(card, chapter, chapters, contract))

    if require_all_train_chapters:
        for chapter_id in sorted(train_chapter_ids - set(card_by_chapter_id)):
            errors.append(f"missing formal card for train chapter: {chapter_id}")

    return {
        "passed": not errors,
        "errors": errors,
        "card_by_chapter_id": card_by_chapter_id,
    }


def _leakage_errors(
    card: dict[str, Any],
    chapter: dict[str, Any],
    chapters: list[dict[str, Any]],
    style_contract: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    rendered = render_chapter_execution_input(card, style_contract)
    target_fragment = _find_leakage_fragment(rendered, str(chapter.get("text") or ""))
    if target_fragment:
        errors.append(f"target-text leakage: {card['card_id']}: {target_fragment}")
    for other in chapters:
        if other.get("id") == chapter.get("id"):
            continue
        if other.get("split") not in {"validation", "sealed", "eval"}:
            continue
        future_fragment = _find_leakage_fragment(rendered, str(other.get("text") or ""))
        if future_fragment:
            errors.append(f"future-context leakage: {card['card_id']}: {future_fragment}")
    return errors


def _find_leakage_fragment(rendered_input: str, source_text: str) -> str | None:
    if count_chinese_chars(source_text) < LEAK_MIN_CHARS:
        return None
    chinese_runs = []
    current = []
    for char in source_text:
        if "\u4e00" <= char <= "\u9fff":
            current.append(char)
        elif current:
            chinese_runs.append("".join(current))
            current = []
    if current:
        chinese_runs.append("".join(current))
    for run in chinese_runs:
        if len(run) < LEAK_MIN_CHARS:
            continue
        for start in range(0, len(run) - LEAK_MIN_CHARS + 1):
            fragment = run[start : start + LEAK_MIN_CHARS]
            if fragment in rendered_input:
                return fragment
    return None
```

- [ ] **Step 4: Run validator tests**

Run:

```powershell
python -m pytest tests/test_card_validator.py -q
```

Expected: pass.

- [ ] **Step 5: Commit Task 4**

Run:

```bash
git add src/small_model_train/cards/card_validator.py tests/test_card_validator.py
git commit -m "feat: validate formal chapter cards"
```

---

## Task 5: Dataset Manifest Builder

**Files:**
- Create: `src/small_model_train/data/dataset_manifest.py`
- Create: `tests/test_dataset_manifest.py`

- [ ] **Step 1: Write failing dataset manifest tests**

Create `tests/test_dataset_manifest.py`:

```python
from __future__ import annotations

import json
from pathlib import Path


def test_build_dataset_manifest_records_hashes_and_formal_flag(tmp_path: Path):
    from small_model_train.data.dataset_manifest import build_dataset_manifest, write_dataset_manifest
    from small_model_train.io_utils import write_jsonl

    sft_path = tmp_path / "sft.jsonl"
    chapters_path = tmp_path / "chapters.jsonl"
    cards_path = tmp_path / "cards.jsonl"
    style_path = tmp_path / "style.json"
    manifest_path = tmp_path / "manifest.json"
    write_jsonl(sft_path, [{"instruction": "i", "input": "x", "output": "y"}])
    write_jsonl(chapters_path, [{"id": "c1", "text": "正文", "split": "train"}])
    write_jsonl(cards_path, [{"card_id": "card-c1-v1", "chapter_id": "c1", "card_sha256": "a" * 64}])
    style_path.write_text(json.dumps({"style_contract_id": "contract-v1"}) + "\n", encoding="utf-8")

    manifest = build_dataset_manifest(
        sft_dataset_path=sft_path,
        chapters_path=chapters_path,
        cards_path=cards_path,
        style_contract_path=style_path,
        style_contract={"style_contract_id": "contract-v1", "contract_sha256": "b" * 64},
        split_manifest={"counts": {"train": 1}},
        card_hashes={"card-c1-v1": "a" * 64},
        chapter_hashes={"c1": "c" * 64},
        leakage_report={"passed": True, "errors": []},
        near_duplicate_report=[],
        formal_dataset=True,
    )
    write_dataset_manifest(manifest_path, manifest)

    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded["row_count"] == 1
    assert loaded["formal_dataset"] is True
    assert loaded["style_contract_id"] == "contract-v1"
    assert loaded["card_hashes"]["card-c1-v1"] == "a" * 64
```

- [ ] **Step 2: Run dataset manifest test and verify failure**

Run:

```powershell
python -m pytest tests/test_dataset_manifest.py -q
```

Expected: fail because `small_model_train.data.dataset_manifest` does not exist.

- [ ] **Step 3: Implement dataset manifest module**

Create `src/small_model_train/data/dataset_manifest.py`:

```python
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from small_model_train.artifact_manifest import file_sha256, summarize_jsonl_artifact

SCHEMA_VERSION = 1


def build_dataset_manifest(
    *,
    sft_dataset_path: str | Path,
    chapters_path: str | Path,
    cards_path: str | Path,
    style_contract_path: str | Path,
    style_contract: dict[str, Any],
    split_manifest: dict[str, Any],
    card_hashes: dict[str, str],
    chapter_hashes: dict[str, str],
    leakage_report: dict[str, Any],
    near_duplicate_report: list[dict[str, Any]],
    formal_dataset: bool,
) -> dict[str, Any]:
    sft_summary = summarize_jsonl_artifact(
        sft_dataset_path,
        label="sft_dataset",
        validate_sft_dataset_schema=True,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "sft_dataset_path": str(sft_dataset_path),
        "sft_dataset_sha256": sft_summary["sha256"],
        "row_count": sft_summary["row_count"],
        "chapters_path": str(chapters_path),
        "chapters_sha256": file_sha256(chapters_path),
        "cards_path": str(cards_path),
        "cards_sha256": file_sha256(cards_path),
        "style_contract_id": style_contract["style_contract_id"],
        "style_contract_sha256": style_contract["contract_sha256"],
        "style_contract_path": str(style_contract_path),
        "style_contract_file_sha256": file_sha256(style_contract_path),
        "split_manifest": split_manifest,
        "card_hashes": card_hashes,
        "chapter_hashes": chapter_hashes,
        "leakage_report": leakage_report,
        "near_duplicate_report": near_duplicate_report,
        "formal_dataset": bool(formal_dataset),
    }


def write_dataset_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
```

- [ ] **Step 4: Run dataset manifest tests**

Run:

```powershell
python -m pytest tests/test_dataset_manifest.py -q
```

Expected: pass.

- [ ] **Step 5: Commit Task 5**

Run:

```bash
git add src/small_model_train/data/dataset_manifest.py tests/test_dataset_manifest.py
git commit -m "feat: build formal sft dataset manifests"
```

---

## Task 6: Formal SFT Builder And CLI Integration

**Files:**
- Modify: `src/small_model_train/sft_builder.py`
- Modify: `scripts/build_sft_dataset.py`
- Modify: `tests/test_sft_builder.py`

- [ ] **Step 1: Add failing formal SFT builder tests**

Append to `tests/test_sft_builder.py`:

```python

def _formal_style_contract_for_stage5c(source_sha256: str = "b" * 64) -> dict:
    from small_model_train.style_contract import build_style_contract_asset

    return build_style_contract_asset(
        style_contract_id="contract-v1",
        approval_status="approved",
        source_corpus={
            "path": "chapters.jsonl",
            "sha256": source_sha256,
            "quality_filter": "quality_tag=A",
            "row_count": 1,
            "selected_rows": 1,
            "split_summary": {"train": 1},
        },
        profile_metrics={
            "chapter_count": 1,
            "avg_dialogue_ratio": 0.1,
            "avg_paragraph_chars": 20,
            "ai_taste": {"phrase_hits": {}, "total_hits": 0, "hits_per_10k_chars": 0},
        },
    )


def _formal_card_for_stage5c(chapter_id: str = "c1", text: str = "这一章用于计算来源哈希。") -> dict:
    from small_model_train.schemas.chapter_execution_card import build_chapter_execution_card

    contract = _formal_style_contract_for_stage5c()
    return build_chapter_execution_card(
        card_id=f"card-{chapter_id}-v1",
        chapter_id=chapter_id,
        card_status="approved",
        style_contract_id=contract["style_contract_id"],
        style_contract_sha256=contract["contract_sha256"],
        source_chapter_text=text,
        target_platform="hybrid_fanqie_qidian",
        genre_tags=["xuanhuan"],
        hard_constraints={
            "must_include": ["旧仓库"],
            "must_not_include": ["作者说明"],
            "continuity_facts": ["上一章压力还在"],
            "forbidden_future_facts": [],
            "style_bans": [],
        },
        execution_plan={
            "chapter_goal": "林默进入旧仓库。",
            "conflict_beat": "对手压价。",
            "payoff_beat": "林默反制。",
            "chapter_structure": [
                {"step": 1, "name": "入场", "goal": "交代压力", "estimated_chars": "300"}
            ],
            "character_states": [
                {"name": "林默", "state": "警惕", "speech_style": "短句"}
            ],
            "ending_hook": "箱子响了一下。",
            "target_word_count": "2000-2500中文汉字",
        },
        creative_space={
            "optional_sensory_details": [],
            "optional_dialogue_moves": [],
            "optional_micro_conflicts": [],
            "allowed_scene_expansion": [],
        },
        provenance={
            "source_card_id": chapter_id,
            "compiler_version": "stage5c_v1",
            "created_at": "2026-06-25T00:00:00Z",
            "reviewer": "",
            "reviewed_at": "",
            "review_notes": "",
            "group_id": f"group-{chapter_id}",
            "split": "train",
        },
    )


def test_build_formal_sft_rows_requires_one_formal_card_per_train_chapter():
    from small_model_train.sft_builder import build_formal_sft_rows

    chapters = [{"id": "c1", "text": "这一章用于计算来源哈希。", "split": "train", "quality_tag": "A"}]
    rows = build_formal_sft_rows([_formal_card_for_stage5c()], chapters, _formal_style_contract_for_stage5c())

    assert rows[0]["output"] == "这一章用于计算来源哈希。"
    assert "【创作自由】" in rows[0]["input"]


def test_build_formal_sft_rows_rejects_missing_card():
    from small_model_train.sft_builder import build_formal_sft_rows

    chapters = [{"id": "c1", "text": "这一章用于计算来源哈希。", "split": "train", "quality_tag": "A"}]

    with pytest.raises(ValueError, match="missing formal card for train chapter: c1"):
        build_formal_sft_rows([], chapters, _formal_style_contract_for_stage5c())
```

- [ ] **Step 2: Add failing CLI manifest test**

Append to `tests/test_sft_builder.py`:

```python

def test_build_sft_dataset_cli_formal_cards_write_manifest(tmp_path):
    from small_model_train.artifact_manifest import file_sha256
    from small_model_train.io_utils import write_jsonl
    from small_model_train.style_contract import write_style_contract_asset

    chapters_path = tmp_path / "chapters.jsonl"
    cards_path = tmp_path / "formal_cards.jsonl"
    contract_path = tmp_path / "style_contract.json"
    output_path = tmp_path / "sft.jsonl"
    manifest_path = tmp_path / "sft_manifest.json"
    chapter_text = "这一章用于计算来源哈希。"
    write_jsonl(chapters_path, [{"id": "c1", "text": chapter_text, "split": "train", "quality_tag": "A"}])
    contract = _formal_style_contract_for_stage5c(source_sha256=file_sha256(chapters_path))
    write_style_contract_asset(contract_path, contract)
    formal_card = _formal_card_for_stage5c("c1", chapter_text)
    formal_card["style_contract_sha256"] = contract["contract_sha256"]
    from small_model_train.schemas.chapter_execution_card import canonical_card_sha256

    formal_card["card_sha256"] = canonical_card_sha256(formal_card)
    write_jsonl(cards_path, [formal_card])

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_sft_dataset.py",
            "--cards",
            str(cards_path),
            "--chapters",
            str(chapters_path),
            "--output",
            str(output_path),
            "--style-contract-json",
            str(contract_path),
            "--dataset-manifest-output",
            str(manifest_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert read_jsonl(output_path)[0]["output"] == chapter_text
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["formal_dataset"] is True
    assert manifest["row_count"] == 1
    assert manifest["style_contract_sha256"] == contract["contract_sha256"]
```

- [ ] **Step 3: Run new SFT tests and verify failure**

Run:

```powershell
python -m pytest tests/test_sft_builder.py::test_build_formal_sft_rows_requires_one_formal_card_per_train_chapter tests/test_sft_builder.py::test_build_formal_sft_rows_rejects_missing_card tests/test_sft_builder.py::test_build_sft_dataset_cli_formal_cards_write_manifest -q
```

Expected: fail because `build_formal_sft_rows` and `--dataset-manifest-output` do not exist.

- [ ] **Step 4: Add formal row builder**

Modify `src/small_model_train/sft_builder.py` imports:

```python
from small_model_train.cards.card_renderer import render_chapter_execution_input
from small_model_train.cards.card_validator import validate_formal_card_batch
```

Add below `build_sft_rows()`:

```python

def build_formal_sft_rows(
    cards: list[dict[str, Any]],
    chapters: list[dict[str, Any]],
    style_contract: dict[str, Any],
    *,
    require_all_train_chapters: bool = True,
) -> list[dict[str, str]]:
    validation = validate_formal_card_batch(
        cards,
        chapters,
        style_contract,
        require_all_train_chapters=require_all_train_chapters,
    )
    if not validation["passed"]:
        raise ValueError("\n".join(validation["errors"]))
    card_by_chapter_id = validation["card_by_chapter_id"]
    rows: list[dict[str, str]] = []
    for chapter in chapters:
        if not _is_trainable_chapter(chapter):
            continue
        chapter_id = chapter["id"]
        card = card_by_chapter_id.get(chapter_id)
        if card is None:
            continue
        rows.append(
            {
                "instruction": INSTRUCTION,
                "input": render_chapter_execution_input(card, style_contract),
                "output": chapter.get("text", ""),
            }
        )
    return rows
```

- [ ] **Step 5: Wire formal CLI path and manifest**

Modify `scripts/build_sft_dataset.py` imports:

```python
from small_model_train.cards.card_validator import validate_formal_card_batch
from small_model_train.data.dataset_manifest import build_dataset_manifest, write_dataset_manifest
from small_model_train.data.dedup import find_near_duplicate_pairs
from small_model_train.schemas.chapter_execution_card import text_sha256
from small_model_train.sft_builder import build_formal_sft_rows, build_sft_rows
```

Add parser arg:

```python
parser.add_argument("--dataset-manifest-output")
```

Replace the row-building branch with:

```python
        cards = read_jsonl(args.cards)
        chapters = read_jsonl(args.chapters)
        if args.style_contract_json:
            style_contract = read_style_contract_asset(args.style_contract_json)
            expected_chapters_sha256 = style_contract["source_corpus"]["sha256"]
            actual_chapters_sha256 = file_sha256(args.chapters)
            if actual_chapters_sha256 != expected_chapters_sha256:
                raise ValueError(
                    "chapters sha256 does not match style contract "
                    "source_corpus.sha256"
                )
            rows = build_formal_sft_rows(cards, chapters, style_contract)
        elif args.allow_draft_cards:
            style_contract = None
            rows = build_sft_rows(
                cards,
                chapters,
                require_approved_cards=False,
                style_contract=None,
            )
        else:
            raise ValueError("style contract JSON is required for formal SFT")
```

After `write_jsonl(args.output, rows)`, add:

```python
    if args.dataset_manifest_output:
        if style_contract is None:
            raise SystemExit("--dataset-manifest-output requires --style-contract-json")
        validation = validate_formal_card_batch(cards, chapters, style_contract)
        manifest = build_dataset_manifest(
            sft_dataset_path=args.output,
            chapters_path=args.chapters,
            cards_path=args.cards,
            style_contract_path=args.style_contract_json,
            style_contract=style_contract,
            split_manifest={
                "counts": {
                    split: sum(1 for row in chapters if row.get("split") == split)
                    for split in sorted({str(row.get("split")) for row in chapters})
                }
            },
            card_hashes={
                card["card_id"]: card["card_sha256"]
                for card in cards
                if isinstance(card.get("card_id"), str)
            },
            chapter_hashes={
                str(chapter["id"]): text_sha256(str(chapter.get("text") or ""))
                for chapter in chapters
            },
            leakage_report={"passed": validation["passed"], "errors": validation["errors"]},
            near_duplicate_report=find_near_duplicate_pairs(chapters),
            formal_dataset=True,
        )
        write_dataset_manifest(args.dataset_manifest_output, manifest)
```

- [ ] **Step 6: Run focused SFT tests**

Run:

```powershell
python -m pytest tests/test_sft_builder.py -q
```

Expected after the implementation edits: failures remain only in legacy CLI tests that still pass old approved chapter-card rows to formal mode. Update those tests exactly as follows:

- In `test_build_sft_dataset_cli_rejects_draft_cards_by_default_and_allows_with_flag`, keep the first subprocess formal call but change the stderr assertion to:

```python
assert "card_id is required" in result.stderr
```

- In `test_build_sft_dataset_cli_accepts_matching_approved_contract`, replace the old `card = _approved_sft_card(...)` block with:

```python
card = _formal_card_for_stage5c("c1", "正文")
card["style_contract_sha256"] = contract["contract_sha256"]
from small_model_train.schemas.chapter_execution_card import canonical_card_sha256

card["card_sha256"] = canonical_card_sha256(card)
```

Then rerun:

```powershell
python -m pytest tests/test_sft_builder.py -q
```

Expected: pass.

- [ ] **Step 7: Commit Task 6**

Run:

```bash
git add src/small_model_train/sft_builder.py scripts/build_sft_dataset.py tests/test_sft_builder.py
git commit -m "feat: require formal cards for formal sft"
```

---

## Task 7: Stage 5C Runbook And Index Docs

**Files:**
- Create: `docs/stage5c-formal-execution-cards-data-integrity.zh.md`
- Modify: `README.md`
- Modify: `docs/index.zh.md`
- Modify: `docs/project-map.zh.md`
- Modify: `docs/pipeline-flow.zh.md`

- [ ] **Step 1: Create Stage 5C runbook**

Create `docs/stage5c-formal-execution-cards-data-integrity.zh.md`:

````markdown
# Stage 5C 正式章节执行卡与数据完整性指南

## 目标

Stage 5C 把草稿章节卡升级为可审阅、可批准、可哈希、可追踪的 `ChapterExecutionCard`。它不扩样，不自动批准卡，也不进入作者反馈或实验矩阵。

## 草稿卡与正式卡

`data_cards/chapter_cards.jsonl` 是 smoke/dev 草稿卡。使用它构建 SFT 数据时必须显式传入：

```powershell
python scripts/build_sft_dataset.py --cards data_cards/chapter_cards.jsonl --chapters data_clean/chapters_split.jsonl --output data_sft/sft_chapter_v1.jsonl --dataset-info-output data_sft/dataset_info.json --allow-draft-cards
```

正式训练使用 `ChapterExecutionCard`：

```powershell
python scripts/compile_chapter_execution_cards.py --cards data_cards/chapter_cards.jsonl --chapters data_clean/chapters_split.jsonl --style-contract-json data_style/style_contract_author_main_v1.json --output data_cards/chapter_execution_cards_reviewed.jsonl
```

编译结果默认是 `reviewed`，不能直接进入 formal SFT。人工审阅后，可以把通过的卡改成 `approved` 或 `frozen`，并重新计算 `card_sha256`。

## Formal SFT

formal SFT 必须使用 approved/frozen 的正式卡和 approved/frozen 的 StyleContract：

```powershell
python scripts/build_sft_dataset.py --cards data_cards/chapter_execution_cards_approved.jsonl --chapters data_clean/chapters_split.jsonl --output data_sft/sft_chapter_formal.jsonl --dataset-info-output data_sft/dataset_info_formal.json --style-contract-json data_style/style_contract_author_main_v1.json --dataset-manifest-output data_sft/sft_chapter_formal_manifest.json
```

命令会检查：

- 每个 train/A 章节恰好有一张 approved/frozen 正式卡。
- 卡的 StyleContract id/hash 与 JSON 一致。
- 卡的 source chapter hash 与章节正文一致。
- prompt 不泄露目标正文、source_text、validation/sealed 文本片段。
- dataset manifest 记录 dataset、card、chapter、split 和 StyleContract hash。

## Sealed 数据边界

`sealed` split 只能用于后续最终证明，不进入训练，也不用于调参。Stage 5C 的 split manifest 用来证明 train、validation、sealed 之间没有重叠。

## Stage 5C 不证明什么

- 不证明模型质量已经提升。
- 不扩大训练规模。
- 不做 same-plot 作者修订。
- 不做 rejection sampling、DPO 或实验矩阵。
````

- [ ] **Step 2: Update README stage guide list**

In `README.md`, under existing Stage 5B guide, add:

```markdown
- [Stage 5C 正式章节执行卡与数据完整性指南](docs/stage5c-formal-execution-cards-data-integrity.zh.md)
```

- [ ] **Step 3: Update docs index**

In `docs/index.zh.md`, add Stage 5C after Stage 5B in both route list and stage guide list:

```markdown
- [Stage 5C 正式章节执行卡与数据完整性指南](stage5c-formal-execution-cards-data-integrity.zh.md)：解释 ChapterExecutionCard、正式卡审批、数据 split、泄漏检查和 dataset manifest。
```

- [ ] **Step 4: Update project map**

In `docs/project-map.zh.md`, add:

```markdown
- `data_cards/chapter_execution_cards_reviewed.jsonl`：Stage 5C 编译出的 reviewed 正式卡候选。
- `data_cards/chapter_execution_cards_approved.jsonl`：人工批准或冻结后可进入 formal SFT 的正式卡。
- `data_sft/*_manifest.json`：Stage 5C formal SFT dataset manifest，记录 dataset、card、chapter、split 和 StyleContract hash。
```

- [ ] **Step 5: Update pipeline flow**

In `docs/pipeline-flow.zh.md`, add a Stage 5C subsection after StyleContract generation and before formal SFT:

````markdown
Stage 5C 起，formal SFT 不再直接使用 `chapter_cards.jsonl`。先编译 reviewed 正式卡候选：

```powershell
python scripts/compile_chapter_execution_cards.py --cards data_cards/chapter_cards.jsonl --chapters data_clean/chapters_split.jsonl --style-contract-json data_style/style_contract_author_main_v1.json --output data_cards/chapter_execution_cards_reviewed.jsonl
```

人工审阅并批准后，formal SFT 使用 `chapter_execution_cards_approved.jsonl` 和 `--dataset-manifest-output`。
````

- [ ] **Step 6: Run docs scan**

Run:

```powershell
rg -n "Stage 5C|ChapterExecutionCard|chapter_execution_cards|dataset-manifest-output|allow-draft-cards|style-contract-json" README.md docs --glob "!docs/superpowers/**"
```

Expected: Stage 5C docs and public flow docs mention the new assets and flags.

- [ ] **Step 7: Commit Task 7**

Run:

```bash
git add docs/stage5c-formal-execution-cards-data-integrity.zh.md README.md docs/index.zh.md docs/project-map.zh.md docs/pipeline-flow.zh.md
git commit -m "docs: add stage5c formal card runbook"
```

---

## Task 8: Final Verification

**Files:**
- No new files.

- [ ] **Step 1: Run focused new tests**

Run:

```powershell
python -m pytest tests/test_chapter_execution_card.py tests/test_card_compiler.py tests/test_card_validator.py tests/test_dataset_manifest.py tests/test_dataset_split.py tests/test_sft_builder.py -q
```

Expected: pass.

- [ ] **Step 2: Run full suite**

Run:

```powershell
python -m pytest -q
```

Expected: pass.

- [ ] **Step 3: Verify stale public draft commands are gone**

Run:

```powershell
rg --pcre2 -n "build_sft_dataset.py(?!.*(--allow-draft-cards|--style-contract-json))" README.md docs --glob "!docs/superpowers/**"
```

Expected: no runnable stale draft commands.

- [ ] **Step 4: Verify draft command still works in smoke/dev mode**

Run:

```powershell
python scripts/build_sft_dataset.py --cards data_cards/chapter_cards.jsonl --chapters data_clean/chapters_split.jsonl --output outputs/stage5c_draft_sft_probe.jsonl --allow-draft-cards
```

Expected: exit code 0 and `outputs/stage5c_draft_sft_probe.jsonl` is written.

- [ ] **Step 5: Verify formal mode rejects draft cards**

Run:

```powershell
python scripts/build_sft_dataset.py --cards data_cards/chapter_cards.jsonl --chapters data_clean/chapters_split.jsonl --output outputs/stage5c_draft_formal_should_fail.jsonl --style-contract-json data_style/style_contract_author_main_v1.json
```

Expected: nonzero exit. The error should mention missing formal card fields such as `card_id` or `chapter_id`, or a StyleContract status/source mismatch if the local StyleContract is not approved for formal use.

- [ ] **Step 6: Verify diff and status**

Run:

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors. Status should show no tracked changes after commits; ignored `outputs/` probe files may be absent from status.

---

## Stage 5C Exit Criteria

Stage 5C is complete only when all of these are true:

- Full pytest suite passes.
- Public docs no longer show draft SFT commands without `--allow-draft-cards`.
- Formal SFT refuses generic draft cards.
- Formal SFT refuses missing, duplicate, unapproved, or hash-mismatched formal cards.
- Every formal SFT row can be traced to exactly one formal card, one source chapter, and one StyleContract hash.
- Train, validation, and sealed split helpers are deterministic and non-overlapping.
- Target/source/future leakage checks are machine-verifiable.
- Near-duplicate checks flag obvious split contamination.
- Formal SFT can write a dataset manifest before training.
- Stage 5C docs explain the operating sequence and boundaries.

---

## Self-Review

- Spec coverage: Task 0 covers 5B documentation closure. Tasks 1-2 cover formal schema, compiler, lifecycle, and renderer. Tasks 3-4 cover grouped splits, leakage checks, and near-duplicate checks. Tasks 5-6 cover dataset manifest and formal SFT admission. Task 7 covers docs. Task 8 covers verification.
- Scope control: The plan does not implement author feedback, same-plot revision data, rejection sampling, DPO, experiment matrices, or larger training.
- Type consistency: Formal cards use `card_id`, `chapter_id`, `card_status`, `style_contract_id`, `style_contract_sha256`, `source_chapter_sha256`, and `card_sha256` consistently. Dataset manifests use `style_contract_sha256` for the canonical contract hash and `style_contract_file_sha256` for the JSON file hash.
