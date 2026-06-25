# Stage 5B Style Contract Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Stage 5B Style Contract Closure so style guidance becomes a reviewable, hash-bound asset and formal training cannot proceed without valid execution cards, dataset provenance, and an approved or frozen StyleContract.

**Architecture:** Keep `scripts/` as command entrypoints and put reusable validation, hashing, metrics, and manifest logic in `src/small_model_train/`. First harden Stage 5A training input evidence, then introduce structured StyleContract assets, bind them into SFT construction, and record provenance in training manifests.

**Tech Stack:** Python 3.10+, pytest, JSON/JSONL, Markdown docs, stdlib `hashlib`, existing `small_model_train` package, existing LLaMA-Factory wrapper scripts.

---

## Scope Check

This plan implements Stage 5B only, with one required Stage 5A precondition fix. It does not implement Stage 5C formal card compilation, sealed-test split, near-duplicate detection, author feedback loops, DPO, or experiment matrices.

Reference spec:

- `docs/superpowers/specs/2026-06-24-stage5b-style-contract-closure-design.md`

Current baseline before this plan:

- `python -m pytest -q`
- Expected current baseline: `367 passed`

---

## File Map

- Create: `src/small_model_train/artifact_manifest.py`
  - Read JSONL rows for evidence, compute file sha256, validate execution-card files, and summarize SFT dataset files.
- Modify: `src/small_model_train/run_manifest.py`
  - Accept optional dataset, eval-card, style-contract, and `formal_evidence` fields.
- Modify: `scripts/run_sft_train.py`
  - Validate eval-card schema before training, write expanded manifest provenance, and accept `--style-contract-json`.
- Modify: `scripts/run_sft_smoke.py`
  - Validate eval-card schema before smoke command construction.
- Modify: `tests/test_stage2_training.py`
  - Cover raw eval-card rejection and expanded manifest fields.
- Modify: `src/small_model_train/style_profile.py`
  - Add deterministic distribution, punctuation, sentence, AI-taste, and filter metrics.
- Create: `src/small_model_train/style_contract.py`
  - Build, validate, hash, render, read, and summarize StyleContract assets.
- Modify: `scripts/build_style_contract.py`
  - Generate StyleContract JSON, Markdown, and metrics from one in-memory asset.
- Modify: `tests/test_style_profile.py`
  - Cover expanded metrics and upgraded CLI behavior.
- Create: `tests/test_style_contract.py`
  - Cover schema validation, approval status, canonical hash, and read/write helpers.
- Modify: `src/small_model_train/sft_builder.py`
  - Validate formal SFT cards against a selected StyleContract.
- Modify: `scripts/build_sft_dataset.py`
  - Add `--style-contract-json` and enforce approved/frozen contracts in formal mode.
- Modify: `tests/test_sft_builder.py`
  - Cover pending/approved/frozen/hash-mismatch formal SFT behavior.
- Modify: `.gitignore`
  - Ignore generated `data_style/`.
- Create: `docs/stage5b-style-contract-closure.zh.md`
  - Stage 5B runbook.
- Modify: `README.md`
- Modify: `docs/index.zh.md`
- Modify: `docs/project-map.zh.md`
- Modify: `docs/pipeline-flow.zh.md`
  - Link and explain Stage 5B assets.

---

## Task 0: Harden Eval-Card Evidence Gate

**Files:**
- Create: `src/small_model_train/artifact_manifest.py`
- Modify: `src/small_model_train/run_manifest.py`
- Modify: `scripts/run_sft_train.py`
- Modify: `scripts/run_sft_smoke.py`
- Modify: `tests/test_stage2_training.py`

- [ ] **Step 1: Add failing artifact manifest and eval schema tests**

Append to `tests/test_stage2_training.py`:

```python
def _execution_card(sample_id: str = "case1") -> dict[str, Any]:
    return {
        "id": sample_id,
        "target_platform": "hybrid_fanqie_qidian",
        "genre_tags": ["xuanhuan", "system"],
        "style_contract": "短句推进，强冲突。",
        "chapter_goal": "主角发现任务并反击。",
        "chapter_structure": [
            {"step": 1, "name": "压迫", "goal": "建立冲突", "estimated_chars": "800"}
        ],
        "conflict_beat": "旧势力当众羞辱主角。",
        "payoff_beat": "主角用证据完成反击。",
        "must_include": ["系统面板"],
        "must_not_include": ["女频误会流"],
        "ending_hook": "新的任务出现。",
        "target_word_count": "2000-2500中文汉字",
    }


def test_summarize_jsonl_artifact_records_sha_rows_and_schema(tmp_path: Path):
    from small_model_train.artifact_manifest import summarize_jsonl_artifact

    cards_path = tmp_path / "eval_cards.jsonl"
    write_jsonl(cards_path, [_execution_card("case1")])

    summary = summarize_jsonl_artifact(
        cards_path,
        label="eval_cards",
        validate_execution_card_schema=True,
    )

    assert summary["path"] == str(cards_path)
    assert summary["sha256"]
    assert summary["row_count"] == 1
    assert summary["schema"]["name"] == "execution_cards"
    assert summary["schema"]["valid"] is True
    assert summary["schema"]["errors"] == []


def test_summarize_jsonl_artifact_rejects_raw_eval_cards_when_schema_required(tmp_path: Path):
    from small_model_train.artifact_manifest import summarize_jsonl_artifact

    raw_cards = tmp_path / "eval_cards_50.jsonl"
    write_jsonl(raw_cards, [{"id": "case1", "text": "原文", "quality_tag": "A", "split": "eval"}])

    summary = summarize_jsonl_artifact(
        raw_cards,
        label="eval_cards",
        validate_execution_card_schema=True,
    )

    assert summary["schema"]["valid"] is False
    assert "missing execution-card fields" in "\n".join(summary["schema"]["errors"])
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
python -m pytest tests/test_stage2_training.py::test_summarize_jsonl_artifact_records_sha_rows_and_schema tests/test_stage2_training.py::test_summarize_jsonl_artifact_rejects_raw_eval_cards_when_schema_required -q
```

Expected: fail with `ModuleNotFoundError: No module named 'small_model_train.artifact_manifest'`.

- [ ] **Step 3: Implement artifact summaries**

Create `src/small_model_train/artifact_manifest.py`:

```python
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from small_model_train.execution_cards import validate_execution_cards
from small_model_train.io_utils import read_jsonl


def file_sha256(path: str | Path) -> str:
    artifact_path = Path(path)
    digest = hashlib.sha256()
    with artifact_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_jsonl_artifact(
    path: str | Path,
    *,
    label: str,
    validate_execution_card_schema: bool = False,
) -> dict[str, Any]:
    artifact_path = Path(path)
    summary: dict[str, Any] = {
        "label": label,
        "path": str(artifact_path),
        "exists": artifact_path.exists(),
        "sha256": "",
        "row_count": 0,
        "schema": {
            "name": "execution_cards" if validate_execution_card_schema else "jsonl",
            "valid": False,
            "errors": [],
        },
    }
    if not artifact_path.exists():
        summary["schema"]["errors"].append(f"{label} is missing: {artifact_path}")
        return summary
    if artifact_path.stat().st_size == 0:
        summary["sha256"] = file_sha256(artifact_path)
        summary["schema"]["errors"].append(f"{label} is empty: {artifact_path}")
        return summary

    summary["sha256"] = file_sha256(artifact_path)
    try:
        rows = read_jsonl(artifact_path)
    except ValueError as exc:
        summary["schema"]["errors"].append(str(exc))
        return summary

    summary["row_count"] = len(rows)
    if validate_execution_card_schema:
        try:
            validate_execution_cards(rows)
        except ValueError as exc:
            summary["schema"]["errors"].append(str(exc))
            return summary

    summary["schema"]["valid"] = True
    return summary
```

- [ ] **Step 4: Run artifact tests**

Run:

```powershell
python -m pytest tests/test_stage2_training.py::test_summarize_jsonl_artifact_records_sha_rows_and_schema tests/test_stage2_training.py::test_summarize_jsonl_artifact_rejects_raw_eval_cards_when_schema_required -q
```

Expected: pass.

- [ ] **Step 5: Add failing training CLI schema-gate test**

Append to `tests/test_stage2_training.py`:

```python
def test_run_sft_train_dry_run_rejects_raw_eval_cards_before_manifest(
    monkeypatch,
    tmp_path: Path,
):
    from scripts import run_sft_train

    sft_dataset = tmp_path / "data" / "sft.jsonl"
    raw_eval_cards = tmp_path / "data" / "eval_cards_50.jsonl"
    sft_dataset.parent.mkdir(parents=True)
    sft_dataset.write_text("{}\n", encoding="utf-8")
    write_jsonl(raw_eval_cards, [{"id": "case1", "text": "原文", "quality_tag": "A", "split": "eval"}])

    model_report = tmp_path / "reports" / "model.json"
    env_report = tmp_path / "reports" / "env.json"
    write_json_preflight(model_report, kind="model", passed=True)
    write_json_preflight(env_report, kind="environment", passed=True)
    write_valid_adapter(tmp_path / "outputs" / "sft_smoke")

    output_dir = tmp_path / "outputs" / "sft_v1"

    def fail_build_train_run(**_kwargs):
        raise AssertionError("schema gate must fail before command construction")

    monkeypatch.setattr(run_sft_train, "build_train_run", fail_build_train_run)
    monkeypatch.setattr(
        run_sft_train.sys,
        "argv",
        [
            "run_sft_train.py",
            "--dry-run",
            "--sft-dataset",
            str(sft_dataset),
            "--eval-cards",
            str(raw_eval_cards),
            "--model-report-json",
            str(model_report),
            "--env-report-json",
            str(env_report),
            "--smoke-adapter-dir",
            str(tmp_path / "outputs" / "sft_smoke"),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert run_sft_train.main() == 1
    assert not (output_dir / "run_manifest.json").exists()
```

- [ ] **Step 6: Run CLI schema-gate test and verify failure**

Run:

```powershell
python -m pytest tests/test_stage2_training.py::test_run_sft_train_dry_run_rejects_raw_eval_cards_before_manifest -q
```

Expected: fail because `run_sft_train.py` currently only checks existence and non-empty files.

- [ ] **Step 7: Enforce execution-card schema in training inputs**

Modify `src/small_model_train/stage2_training.py`:

```python
from small_model_train.artifact_manifest import summarize_jsonl_artifact
```

Replace `validate_training_inputs()` with:

```python
def validate_training_inputs(
    sft_dataset: str | Path,
    eval_cards: str | Path,
) -> dict[str, Any]:
    errors = []
    for label, raw_path in (("SFT dataset", sft_dataset),):
        path = Path(raw_path)
        if not path.exists():
            errors.append(f"{label} is missing: {path}")
        elif path.stat().st_size == 0:
            errors.append(f"{label} is empty: {path}")

    eval_summary = summarize_jsonl_artifact(
        eval_cards,
        label="eval cards",
        validate_execution_card_schema=True,
    )
    if not eval_summary["schema"]["valid"]:
        errors.extend(eval_summary["schema"]["errors"])

    return {
        "passed": not errors,
        "errors": errors,
        "artifacts": {"eval_cards": eval_summary},
    }
```

Existing tests that only assert `result["passed"]` and `result["errors"]` remain compatible.

- [ ] **Step 8: Run training validation tests**

Run:

```powershell
python -m pytest tests/test_stage2_training.py::test_validate_training_inputs_reports_missing_files tests/test_stage2_training.py::test_validate_training_inputs_reports_empty_files tests/test_stage2_training.py::test_run_sft_train_dry_run_rejects_raw_eval_cards_before_manifest -q
```

Expected: pass after updating old missing/empty eval-card assertions if their error wording now includes schema details.

- [ ] **Step 9: Add failing smoke schema-gate test**

Append to `tests/test_stage2_training.py`:

```python
def test_run_sft_smoke_rejects_raw_eval_cards_before_command(monkeypatch, tmp_path: Path):
    from scripts import run_sft_smoke

    sft_dataset = tmp_path / "data" / "sft.jsonl"
    raw_eval_cards = tmp_path / "data" / "eval_cards_50.jsonl"
    sft_dataset.parent.mkdir(parents=True)
    sft_dataset.write_text("{}\n", encoding="utf-8")
    write_jsonl(raw_eval_cards, [{"id": "case1", "text": "原文", "quality_tag": "A", "split": "eval"}])

    def fail_build_train_run(**_kwargs):
        raise AssertionError("schema gate must fail before smoke command construction")

    monkeypatch.setattr(run_sft_smoke, "build_train_run", fail_build_train_run)
    monkeypatch.setattr(
        run_sft_smoke.sys,
        "argv",
        [
            "run_sft_smoke.py",
            "--dry-run",
            "--sft-dataset",
            str(sft_dataset),
            "--eval-cards",
            str(raw_eval_cards),
            "--output-dir",
            str(tmp_path / "outputs" / "sft_smoke"),
        ],
    )

    assert run_sft_smoke.main() == 1
```

- [ ] **Step 10: Run smoke schema-gate test**

Run:

```powershell
python -m pytest tests/test_stage2_training.py::test_run_sft_smoke_rejects_raw_eval_cards_before_command -q
```

Expected: pass because `run_sft_smoke.py` uses `validate_training_inputs()`.

- [ ] **Step 11: Add manifest artifact fields**

Modify `src/small_model_train/run_manifest.py` signature:

```python
def build_run_manifest(
    *,
    run_name: str,
    command: Sequence[str],
    training_exit_code: int,
    model_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    preflight_reports: dict[str, Any],
    adapter_check: dict[str, Any],
    passed: bool,
    repo_root: str | Path | None = None,
    sft_dataset: dict[str, Any] | None = None,
    eval_cards: dict[str, Any] | None = None,
    style_contract: dict[str, Any] | None = None,
    formal_evidence: bool = False,
) -> dict[str, Any]:
```

Add these keys to the returned manifest:

```python
"sft_dataset": sft_dataset,
"eval_cards": eval_cards,
"style_contract": style_contract,
"formal_evidence": bool(formal_evidence),
```

- [ ] **Step 12: Pass artifact fields from `run_sft_train.py`**

In `scripts/run_sft_train.py`, after input validation passes, keep:

```python
input_artifacts = validation.get("artifacts", {})
```

Before writing the manifest, add:

```python
from small_model_train.artifact_manifest import summarize_jsonl_artifact
```

Build SFT summary:

```python
sft_dataset_summary = summarize_jsonl_artifact(
    args.sft_dataset,
    label="sft_dataset",
    validate_execution_card_schema=False,
)
eval_cards_summary = input_artifacts.get("eval_cards")
formal_evidence = (
    not args.dry_run
    and training_exit_code == 0
    and adapter_check.get("passed") is True
    and eval_cards_summary is not None
    and eval_cards_summary.get("schema", {}).get("valid") is True
)
```

Pass `sft_dataset=sft_dataset_summary`, `eval_cards=eval_cards_summary`, and `formal_evidence=formal_evidence` to `build_run_manifest()`.

- [ ] **Step 13: Add manifest assertions**

Update `test_run_sft_train_dry_run_writes_manifest_without_output_adapter()` in `tests/test_stage2_training.py` so `eval_cards` is a valid execution-card JSONL:

```python
write_jsonl(eval_cards, [_execution_card("case1")])
```

Add assertions:

```python
assert manifest["sft_dataset"]["row_count"] == 1
assert manifest["eval_cards"]["schema"]["valid"] is True
assert manifest["formal_evidence"] is False
```

- [ ] **Step 14: Run Task 0 tests**

Run:

```powershell
python -m pytest tests/test_stage2_training.py -q
```

Expected: pass.

- [ ] **Step 15: Commit Task 0**

Run:

```bash
git add src/small_model_train/artifact_manifest.py src/small_model_train/run_manifest.py src/small_model_train/stage2_training.py scripts/run_sft_train.py scripts/run_sft_smoke.py tests/test_stage2_training.py
git commit -m "fix: require execution-card eval evidence"
```

---

## Task 1: Expand Style Profile Metrics

**Files:**
- Modify: `src/small_model_train/style_profile.py`
- Modify: `tests/test_style_profile.py`

- [ ] **Step 1: Add failing expanded metric tests**

Add to `tests/test_style_profile.py`:

```python
def test_build_style_profile_reports_distributions_and_ai_trace_metrics():
    rows = [
        {"id": "a", "quality_tag": "A", "text": "林默点头。空气仿佛凝固了。\n\n“加钱。”"},
        {"id": "b", "quality_tag": "B", "text": "这行也参与函数级统计。"},
    ]

    profile = build_style_profile(rows)

    assert profile["chapter_count"] == 2
    assert profile["chinese_chars"]["min"] > 0
    assert profile["chinese_chars"]["p50"] > 0
    assert profile["paragraph_chars"]["avg"] > 0
    assert profile["dialogue_ratio"]["avg"] >= 0
    assert profile["sentence_chars"]["p90"] > 0
    assert profile["punctuation_density"]["。"] > 0
    assert profile["ai_taste"]["phrase_hits"]["空气仿佛凝固了"] == 1
    assert profile["source_filter"]["selected_rows"] == 2


def test_build_style_profile_handles_empty_input():
    profile = build_style_profile([])

    assert profile["chapter_count"] == 0
    assert profile["chinese_chars"]["avg"] == 0
    assert profile["paragraph_chars"]["p90"] == 0
    assert profile["ai_taste"]["total_hits"] == 0
```

- [ ] **Step 2: Run expanded metric tests and verify failure**

Run:

```powershell
python -m pytest tests/test_style_profile.py::test_build_style_profile_reports_distributions_and_ai_trace_metrics tests/test_style_profile.py::test_build_style_profile_handles_empty_input -q
```

Expected: fail because `build_style_profile()` returns flat average fields only.

- [ ] **Step 3: Implement deterministic distributions**

Modify `src/small_model_train/style_profile.py` imports:

```python
import re

from small_model_train.scoring import AI_TRACE_PHRASES
```

Add helpers above `build_style_profile()`:

```python
SENTENCE_RE = re.compile(r"[^。！？!?]+[。！？!?]?")
PUNCTUATION_MARKS = ("。", "，", "、", "；", "：", "！", "？", "“", "”", "…")


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return round(float(ordered[index]), 4)


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0, "max": 0, "avg": 0, "p50": 0, "p90": 0}
    return {
        "min": round(float(min(values)), 4),
        "max": round(float(max(values)), 4),
        "avg": round(mean(values), 4),
        "p50": _percentile(values, 0.5),
        "p90": _percentile(values, 0.9),
    }


def _sentence_lengths(text: str) -> list[int]:
    lengths = []
    for match in SENTENCE_RE.finditer(text.replace("\n", "")):
        sentence = match.group(0).strip()
        if sentence:
            lengths.append(count_chinese_chars(sentence))
    return [length for length in lengths if length > 0]


def _punctuation_density(texts: list[str]) -> dict[str, float]:
    total_chars = sum(count_chinese_chars(text) for text in texts)
    if total_chars <= 0:
        return {mark: 0 for mark in PUNCTUATION_MARKS}
    return {
        mark: round(sum(text.count(mark) for text in texts) / total_chars, 6)
        for mark in PUNCTUATION_MARKS
    }


def _ai_taste_metrics(texts: list[str]) -> dict:
    phrase_hits = {
        phrase: sum(text.count(phrase) for text in texts)
        for phrase in AI_TRACE_PHRASES
    }
    total_hits = sum(phrase_hits.values())
    total_chars = sum(count_chinese_chars(text) for text in texts)
    return {
        "phrase_hits": phrase_hits,
        "total_hits": total_hits,
        "hits_per_10k_chars": round(total_hits / total_chars * 10000, 4) if total_chars else 0,
    }
```

- [ ] **Step 4: Replace `build_style_profile()` body**

Use:

```python
def build_style_profile(rows: list[dict]) -> dict:
    texts = [row.get("text", "") for row in rows if row.get("text")]
    chapter_chars = [count_chinese_chars(text) for text in texts]
    paragraph_counts = [length for text in texts for length in paragraph_lengths(text)]
    dialogue_ratios = [dialogue_ratio(text) for text in texts]
    sentence_counts = [length for text in texts for length in _sentence_lengths(text)]
    return {
        "chapter_count": len(texts),
        "avg_chinese_chars": round(mean(chapter_chars), 2) if chapter_chars else 0,
        "avg_paragraph_chars": round(mean(paragraph_counts), 2) if paragraph_counts else 0,
        "avg_dialogue_ratio": round(mean(dialogue_ratios), 4) if dialogue_ratios else 0,
        "chinese_chars": _distribution([float(value) for value in chapter_chars]),
        "paragraph_chars": _distribution([float(value) for value in paragraph_counts]),
        "dialogue_ratio": _distribution([float(value) for value in dialogue_ratios]),
        "sentence_chars": _distribution([float(value) for value in sentence_counts]),
        "punctuation_density": _punctuation_density(texts),
        "ai_taste": _ai_taste_metrics(texts),
        "source_filter": {
            "total_rows": len(rows),
            "selected_rows": len(texts),
            "skipped_rows": len(rows) - len(texts),
            "quality_filter": "provided_rows",
        },
    }
```

Keep the legacy flat fields so older docs/tests remain compatible.

- [ ] **Step 5: Update `render_style_contract()` to use nested metrics**

Modify `render_style_contract()`:

```python
dialogue_ratio = profile.get("dialogue_ratio", {})
dialogue_percent = round(float(dialogue_ratio.get("avg", profile.get("avg_dialogue_ratio", 0))) * 100, 1)
paragraph_stats = profile.get("paragraph_chars", {})
avg_paragraph_chars = paragraph_stats.get("avg", profile.get("avg_paragraph_chars", 0))
```

Add one diagnostic line under `【禁止风格】`:

```python
f"6. 当前语料 AI 味短语命中约 {profile.get('ai_taste', {}).get('hits_per_10k_chars', 0)} 次/万字，生成时应继续压低。"
```

- [ ] **Step 6: Run style profile tests**

Run:

```powershell
python -m pytest tests/test_style_profile.py -q
```

Expected: pass.

- [ ] **Step 7: Commit Task 1**

Run:

```bash
git add src/small_model_train/style_profile.py tests/test_style_profile.py
git commit -m "feat: expand style profile metrics"
```

---

## Task 2: Add StyleContract Asset Module

**Files:**
- Create: `src/small_model_train/style_contract.py`
- Create: `tests/test_style_contract.py`

- [ ] **Step 1: Write failing StyleContract tests**

Create `tests/test_style_contract.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from small_model_train.style_contract import (
    APPROVED_FORMAL_STATUSES,
    build_style_contract_asset,
    canonical_style_contract_sha256,
    is_contract_approved_for_formal_sft,
    read_style_contract_asset,
    render_style_contract_markdown,
    validate_style_contract_asset,
    write_style_contract_asset,
)


def _profile() -> dict:
    return {
        "chapter_count": 2,
        "avg_chinese_chars": 1200,
        "avg_paragraph_chars": 80,
        "avg_dialogue_ratio": 0.42,
        "chinese_chars": {"min": 1000, "max": 1400, "avg": 1200, "p50": 1000, "p90": 1400},
        "paragraph_chars": {"min": 20, "max": 120, "avg": 80, "p50": 60, "p90": 120},
        "dialogue_ratio": {"min": 0.2, "max": 0.6, "avg": 0.42, "p50": 0.4, "p90": 0.6},
        "sentence_chars": {"min": 5, "max": 30, "avg": 12, "p50": 10, "p90": 30},
        "punctuation_density": {"。": 0.02},
        "ai_taste": {"phrase_hits": {"空气仿佛凝固了": 0}, "total_hits": 0, "hits_per_10k_chars": 0},
        "source_filter": {"total_rows": 2, "selected_rows": 2, "skipped_rows": 0, "quality_filter": "quality_tag=A"},
    }


def _asset(status: str = "pending_review") -> dict:
    return build_style_contract_asset(
        style_contract_id="author_main_v1",
        approval_status=status,
        source_corpus={
            "path": "data_clean/chapters_split.jsonl",
            "sha256": "a" * 64,
            "quality_filter": "quality_tag=A",
            "row_count": 2,
            "selected_rows": 2,
            "split_summary": {"train": 2},
        },
        profile_metrics=_profile(),
        author_notes="",
    )


def test_build_style_contract_asset_defaults_to_hash_bound_pending_review():
    asset = _asset()

    assert asset["schema_version"] == 1
    assert asset["style_contract_id"] == "author_main_v1"
    assert asset["approval_status"] == "pending_review"
    assert len(asset["contract_sha256"]) == 64
    assert canonical_style_contract_sha256(asset) == asset["contract_sha256"]
    assert validate_style_contract_asset(asset) == asset
    assert is_contract_approved_for_formal_sft(asset) is False


@pytest.mark.parametrize("status", sorted(APPROVED_FORMAL_STATUSES))
def test_approved_and_frozen_contracts_are_formal(status: str):
    assert is_contract_approved_for_formal_sft(_asset(status)) is True


@pytest.mark.parametrize("status", ["", "pending", "approved_by_author"])
def test_invalid_approval_status_is_rejected(status: str):
    asset = _asset()
    asset["approval_status"] = status
    asset["contract_sha256"] = canonical_style_contract_sha256(asset)

    with pytest.raises(ValueError, match="approval_status"):
        validate_style_contract_asset(asset)


def test_contract_hash_mismatch_is_rejected():
    asset = _asset()
    asset["prompt_rules"]["output"] = "tampered"

    with pytest.raises(ValueError, match="contract_sha256 mismatch"):
        validate_style_contract_asset(asset)


def test_style_contract_read_write_roundtrip(tmp_path: Path):
    path = tmp_path / "style_contract.json"
    asset = _asset("approved")

    write_style_contract_asset(path, asset)
    loaded = read_style_contract_asset(path)

    assert loaded == asset


def test_render_style_contract_markdown_is_human_reviewable():
    markdown = render_style_contract_markdown(_asset())

    assert "# Style Contract author_main_v1" in markdown
    assert "approval_status: pending_review" in markdown
    assert "只输出正文" in markdown
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
python -m pytest tests/test_style_contract.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'small_model_train.style_contract'`.

- [ ] **Step 3: Implement StyleContract module**

Create `src/small_model_train/style_contract.py`:

```python
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from small_model_train.style_profile import render_style_contract

SCHEMA_VERSION = 1
APPROVAL_STATUSES = {"draft", "pending_review", "approved", "frozen", "rejected"}
APPROVED_FORMAL_STATUSES = {"approved", "frozen"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def canonical_style_contract_sha256(asset: dict[str, Any]) -> str:
    import hashlib

    payload = copy.deepcopy(asset)
    payload.pop("contract_sha256", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_style_contract_asset(
    *,
    style_contract_id: str,
    approval_status: str,
    source_corpus: dict[str, Any],
    profile_metrics: dict[str, Any],
    author_notes: str = "",
    created_at: str | None = None,
) -> dict[str, Any]:
    prompt_text = render_style_contract(profile_metrics)
    asset = {
        "schema_version": SCHEMA_VERSION,
        "style_contract_id": style_contract_id,
        "approval_status": approval_status,
        "contract_sha256": "",
        "created_at": created_at or utc_now_iso(),
        "source_corpus": dict(source_corpus),
        "profile_metrics": dict(profile_metrics),
        "prompt_rules": {
            "system_role": "你是作者的正文执行器，只负责根据章节执行卡写正文。",
            "style_contract_text": prompt_text,
            "output": "只输出正文。不要输出提纲、小标题、解释、分析或提示语。",
        },
        "ai_taste_guardrails": {
            "banned_phrases": list(profile_metrics.get("ai_taste", {}).get("phrase_hits", {}).keys()),
            "policy": "生成时继续压低 AI 味短语、总结式升华和模板化转折。",
        },
        "author_notes": author_notes,
        "review": {
            "reviewer": "",
            "reviewed_at": "",
            "review_notes": "",
        },
    }
    asset["contract_sha256"] = canonical_style_contract_sha256(asset)
    return validate_style_contract_asset(asset)


def validate_style_contract_asset(asset: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(asset, dict):
        raise ValueError("style contract asset must be a JSON object")
    required = {
        "schema_version",
        "style_contract_id",
        "approval_status",
        "contract_sha256",
        "created_at",
        "source_corpus",
        "profile_metrics",
        "prompt_rules",
        "ai_taste_guardrails",
        "author_notes",
        "review",
    }
    missing = sorted(field for field in required if field not in asset)
    if missing:
        raise ValueError("style contract missing fields: " + ", ".join(missing))
    if asset["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"style contract schema_version must be {SCHEMA_VERSION}")
    if not isinstance(asset["style_contract_id"], str) or not asset["style_contract_id"].strip():
        raise ValueError("style_contract_id must be a non-empty string")
    if asset["approval_status"] not in APPROVAL_STATUSES:
        raise ValueError("approval_status must be one of: " + ", ".join(sorted(APPROVAL_STATUSES)))
    for field in ("source_corpus", "profile_metrics", "prompt_rules", "ai_taste_guardrails", "review"):
        if not isinstance(asset[field], dict):
            raise ValueError(f"{field} must be an object")
    expected_hash = canonical_style_contract_sha256(asset)
    if asset["contract_sha256"] != expected_hash:
        raise ValueError(
            "contract_sha256 mismatch: "
            f"expected {expected_hash}, got {asset['contract_sha256']}"
        )
    return asset


def is_contract_approved_for_formal_sft(asset: dict[str, Any]) -> bool:
    validate_style_contract_asset(asset)
    return asset["approval_status"] in APPROVED_FORMAL_STATUSES


def write_style_contract_asset(path: str | Path, asset: dict[str, Any]) -> None:
    validated = validate_style_contract_asset(asset)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(validated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_style_contract_asset(path: str | Path) -> dict[str, Any]:
    raw_path = Path(path)
    try:
        asset = json.loads(raw_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"style contract invalid JSON: {raw_path}: {exc}") from exc
    return validate_style_contract_asset(asset)


def render_style_contract_markdown(asset: dict[str, Any]) -> str:
    validated = validate_style_contract_asset(asset)
    source = validated["source_corpus"]
    lines = [
        f"# Style Contract {validated['style_contract_id']}",
        "",
        f"- approval_status: {validated['approval_status']}",
        f"- contract_sha256: {validated['contract_sha256']}",
        f"- source_path: {source.get('path', '')}",
        f"- source_sha256: {source.get('sha256', '')}",
        f"- selected_rows: {source.get('selected_rows', '')}",
        "",
        "## Prompt Rules",
        validated["prompt_rules"]["style_contract_text"],
        "",
        "## Author Notes",
        validated.get("author_notes", "") or "无",
    ]
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run StyleContract tests**

Run:

```powershell
python -m pytest tests/test_style_contract.py -q
```

Expected: pass.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add src/small_model_train/style_contract.py tests/test_style_contract.py
git commit -m "feat: add style contract asset model"
```

---

## Task 3: Upgrade StyleContract CLI

**Files:**
- Modify: `scripts/build_style_contract.py`
- Modify: `tests/test_style_profile.py`
- Modify: `.gitignore`

- [ ] **Step 1: Add failing CLI tests for three artifacts**

Append to `tests/test_style_profile.py`:

```python
def test_build_style_contract_script_writes_json_markdown_and_metrics(tmp_path):
    chapters_path = tmp_path / "chapters.jsonl"
    contract_json_path = tmp_path / "data_style" / "style_contract_author_main_v1.json"
    contract_md_path = tmp_path / "style_contract.md"
    metrics_path = tmp_path / "data_style" / "style_metrics_author_main_v1.json"
    write_jsonl(
        chapters_path,
        [
            {"id": "a", "quality_tag": "A", "split": "train", "text": "林默点头。\n\n“成交。”"},
            {"id": "b", "quality_tag": "B", "split": "train", "text": "这行不应该参与统计。"},
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_style_contract.py",
            "--chapters",
            str(chapters_path),
            "--contract-json-output",
            str(contract_json_path),
            "--contract-output",
            str(contract_md_path),
            "--metrics-output",
            str(metrics_path),
            "--style-contract-id",
            "author_main_v1",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    asset = json.loads(contract_json_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert asset["approval_status"] == "pending_review"
    assert asset["style_contract_id"] == "author_main_v1"
    assert asset["contract_sha256"]
    assert asset["source_corpus"]["selected_rows"] == 1
    assert metrics["chapter_count"] == 1
    assert "# Style Contract author_main_v1" in contract_md_path.read_text(encoding="utf-8")


def test_build_style_contract_rejects_duplicate_outputs(tmp_path):
    chapters_path = tmp_path / "chapters.jsonl"
    output_path = tmp_path / "same.json"
    write_jsonl(chapters_path, [{"id": "a", "quality_tag": "A", "text": "正文"}])

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_style_contract.py",
            "--chapters",
            str(chapters_path),
            "--contract-json-output",
            str(output_path),
            "--contract-output",
            str(output_path),
            "--metrics-output",
            str(tmp_path / "metrics.json"),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    assert "output paths must be distinct" in result.stderr
```

- [ ] **Step 2: Run CLI tests and verify failure**

Run:

```powershell
python -m pytest tests/test_style_profile.py::test_build_style_contract_script_writes_json_markdown_and_metrics tests/test_style_profile.py::test_build_style_contract_rejects_duplicate_outputs -q
```

Expected: fail because the CLI does not yet support `--contract-json-output` or `--metrics-output`.

- [ ] **Step 3: Add `data_style/` to `.gitignore`**

Modify `.gitignore`:

```gitignore
data_style/
```

- [ ] **Step 4: Upgrade build_style_contract CLI**

Replace `scripts/build_style_contract.py` with:

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

from small_model_train.artifact_manifest import file_sha256
from small_model_train.io_utils import read_jsonl
from small_model_train.style_contract import (
    build_style_contract_asset,
    render_style_contract_markdown,
    write_style_contract_asset,
)
from small_model_train.style_profile import build_style_profile


def _distinct_paths(*paths: str | None) -> bool:
    resolved = [Path(path).resolve() for path in paths if path]
    return len(resolved) == len(set(resolved))


def _split_summary(rows: list[dict]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for row in rows:
        split = str(row.get("split", "unknown"))
        summary[split] = summary.get(split, 0) + 1
    return dict(sorted(summary.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapters", required=True)
    parser.add_argument("--contract-json-output", default="data_style/style_contract_author_main_v1.json")
    parser.add_argument("--contract-output", required=True)
    parser.add_argument("--metrics-output", default="data_style/style_metrics_author_main_v1.json")
    parser.add_argument("--profile-output")
    parser.add_argument("--style-contract-id", default="author_main_v1")
    parser.add_argument(
        "--approval-status",
        default="pending_review",
        choices=["draft", "pending_review", "approved", "frozen", "rejected"],
    )
    parser.add_argument("--author-notes", default="")
    args = parser.parse_args()

    if not _distinct_paths(args.contract_json_output, args.contract_output, args.metrics_output, args.profile_output):
        parser.error("output paths must be distinct")

    all_rows = read_jsonl(args.chapters)
    rows = [row for row in all_rows if row.get("quality_tag") == "A"]
    profile = build_style_profile(rows)
    profile["source_filter"] = {
        "total_rows": len(all_rows),
        "selected_rows": len(rows),
        "skipped_rows": len(all_rows) - len(rows),
        "quality_filter": "quality_tag=A",
    }
    source_corpus = {
        "path": str(args.chapters),
        "sha256": file_sha256(args.chapters),
        "quality_filter": "quality_tag=A",
        "row_count": len(all_rows),
        "selected_rows": len(rows),
        "split_summary": _split_summary(rows),
    }
    asset = build_style_contract_asset(
        style_contract_id=args.style_contract_id,
        approval_status=args.approval_status,
        source_corpus=source_corpus,
        profile_metrics=profile,
        author_notes=args.author_notes,
    )

    write_style_contract_asset(args.contract_json_output, asset)
    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    contract_path = Path(args.contract_output)
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(render_style_contract_markdown(asset), encoding="utf-8")
    if args.profile_output:
        profile_path = Path(args.profile_output)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote style contract JSON to {args.contract_json_output}")
    print(f"wrote style contract Markdown to {args.contract_output}")
    print(f"wrote style metrics to {args.metrics_output}")


if __name__ == "__main__":
    main()
```

`--profile-output` is kept as a legacy alias so existing docs/tests can migrate gradually.

- [ ] **Step 5: Update legacy CLI test**

In `test_build_style_contract_script_writes_profile_and_contract_for_a_rows()`, keep the existing command but add:

```python
"--contract-json-output",
str(tmp_path / "style_contract.json"),
"--metrics-output",
str(tmp_path / "style_metrics.json"),
```

The existing `--profile-output` assertion should continue to pass.

- [ ] **Step 6: Run style CLI tests**

Run:

```powershell
python -m pytest tests/test_style_profile.py tests/test_style_contract.py -q
```

Expected: pass.

- [ ] **Step 7: Commit Task 3**

Run:

```bash
git add .gitignore scripts/build_style_contract.py tests/test_style_profile.py
git commit -m "feat: generate style contract asset bundle"
```

---

## Task 4: Enforce StyleContract in Formal SFT

**Files:**
- Modify: `src/small_model_train/sft_builder.py`
- Modify: `scripts/build_sft_dataset.py`
- Modify: `tests/test_sft_builder.py`

- [ ] **Step 1: Add failing SFT contract gate tests**

Append to `tests/test_sft_builder.py`:

```python
def _style_contract_asset(status: str = "approved") -> dict:
    from small_model_train.style_contract import build_style_contract_asset

    return build_style_contract_asset(
        style_contract_id="contract-v1",
        approval_status=status,
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


def test_build_sft_rows_rejects_pending_style_contract_in_formal_mode():
    card = _approved_sft_card()
    contract = _style_contract_asset("pending_review")

    with pytest.raises(ValueError, match="approved or frozen"):
        build_sft_rows(
            [card],
            [_train_chapter()],
            require_approved_cards=True,
            style_contract=contract,
        )


def test_build_sft_rows_rejects_style_contract_hash_mismatch():
    contract = _style_contract_asset("approved")
    card = _approved_sft_card(style_contract_sha256="c" * 64)

    with pytest.raises(ValueError, match="style_contract_sha256 mismatch"):
        build_sft_rows(
            [card],
            [_train_chapter()],
            require_approved_cards=True,
            style_contract=contract,
        )


def test_build_sft_rows_accepts_matching_approved_style_contract():
    contract = _style_contract_asset("approved")
    card = _approved_sft_card(
        style_contract_id=contract["style_contract_id"],
        style_contract_sha256=contract["contract_sha256"],
    )

    rows = build_sft_rows(
        [card],
        [_train_chapter()],
        require_approved_cards=True,
        style_contract=contract,
    )

    assert rows[0]["output"] == "正文"
```

- [ ] **Step 2: Run SFT contract tests and verify failure**

Run:

```powershell
python -m pytest tests/test_sft_builder.py::test_build_sft_rows_rejects_pending_style_contract_in_formal_mode tests/test_sft_builder.py::test_build_sft_rows_rejects_style_contract_hash_mismatch tests/test_sft_builder.py::test_build_sft_rows_accepts_matching_approved_style_contract -q
```

Expected: fail because `build_sft_rows()` has no `style_contract` argument.

- [ ] **Step 3: Add formal style contract validation**

Modify `src/small_model_train/sft_builder.py` imports:

```python
from typing import Any

from small_model_train.style_contract import is_contract_approved_for_formal_sft, validate_style_contract_asset
```

Add helper:

```python
def _require_card_matches_style_contract(card: dict, style_contract: dict[str, Any]) -> None:
    contract = validate_style_contract_asset(style_contract)
    card_id = card.get("id", "<missing id>")
    if not is_contract_approved_for_formal_sft(contract):
        raise ValueError(
            "style contract approval_status must be approved or frozen for formal SFT: "
            f"{contract['style_contract_id']}"
        )
    if card.get("style_contract_id") != contract["style_contract_id"]:
        raise ValueError(
            "style_contract_id mismatch for formal SFT: "
            f"{card_id}: card={card.get('style_contract_id')!r}, contract={contract['style_contract_id']!r}"
        )
    if card.get("style_contract_sha256") != contract["contract_sha256"]:
        raise ValueError(
            "style_contract_sha256 mismatch for formal SFT: "
            f"{card_id}: card={card.get('style_contract_sha256')!r}, contract={contract['contract_sha256']!r}"
        )
```

Change `build_sft_rows()` signature:

```python
def build_sft_rows(
    cards: list[dict],
    chapters: list[dict],
    require_approved_cards: bool = False,
    style_contract: dict[str, Any] | None = None,
) -> list[dict]:
```

Inside the loop after `_require_approved_card(card)`:

```python
if style_contract is not None:
    _require_card_matches_style_contract(card, style_contract)
elif require_approved_cards:
    raise ValueError("style contract JSON is required for formal SFT")
```

- [ ] **Step 4: Run SFT unit tests**

Run:

```powershell
python -m pytest tests/test_sft_builder.py -q
```

Expected: pass after updating existing formal tests to pass `style_contract=_style_contract_asset("approved")` only where they expect formal success. Tests that validate missing status/id/hash may continue without a contract if they expect the card gate to fail first.

- [ ] **Step 5: Add failing CLI tests for `--style-contract-json`**

Append to `tests/test_sft_builder.py`:

```python
def test_build_sft_dataset_cli_requires_style_contract_json_for_formal(tmp_path):
    cards_path = tmp_path / "cards.jsonl"
    chapters_path = tmp_path / "chapters.jsonl"
    output_path = tmp_path / "sft.jsonl"
    card = _approved_sft_card()
    write_jsonl(cards_path, [card])
    write_jsonl(chapters_path, [_train_chapter()])

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
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "style contract JSON is required for formal SFT" in result.stderr
    assert not output_path.exists()


def test_build_sft_dataset_cli_accepts_matching_approved_contract(tmp_path):
    from small_model_train.style_contract import write_style_contract_asset

    contract = _style_contract_asset("approved")
    cards_path = tmp_path / "cards.jsonl"
    chapters_path = tmp_path / "chapters.jsonl"
    contract_path = tmp_path / "style_contract.json"
    output_path = tmp_path / "sft.jsonl"
    card = _approved_sft_card(
        style_contract_id=contract["style_contract_id"],
        style_contract_sha256=contract["contract_sha256"],
    )
    write_jsonl(cards_path, [card])
    write_jsonl(chapters_path, [_train_chapter()])
    write_style_contract_asset(contract_path, contract)

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
            "--style-contract-json",
            str(contract_path),
        ],
        check=True,
    )

    assert read_jsonl(output_path)[0]["output"] == "正文"
```

- [ ] **Step 6: Run CLI tests and verify failure**

Run:

```powershell
python -m pytest tests/test_sft_builder.py::test_build_sft_dataset_cli_requires_style_contract_json_for_formal tests/test_sft_builder.py::test_build_sft_dataset_cli_accepts_matching_approved_contract -q
```

Expected: fail because `build_sft_dataset.py` has no `--style-contract-json`.

- [ ] **Step 7: Wire `--style-contract-json` into CLI**

Modify `scripts/build_sft_dataset.py` imports:

```python
from small_model_train.style_contract import read_style_contract_asset
```

Add parser arg:

```python
parser.add_argument("--style-contract-json")
```

Before `build_sft_rows()`:

```python
style_contract = None
if args.style_contract_json:
    style_contract = read_style_contract_asset(args.style_contract_json)
elif not args.allow_draft_cards:
    raise ValueError("style contract JSON is required for formal SFT")
```

Wrap the main row build in `try/except ValueError` so CLI errors print cleanly:

```python
try:
    rows = build_sft_rows(
        read_jsonl(args.cards),
        read_jsonl(args.chapters),
        require_approved_cards=not args.allow_draft_cards,
        style_contract=style_contract,
    )
except ValueError as exc:
    print(str(exc), file=sys.stderr)
    raise SystemExit(1) from exc
```

- [ ] **Step 8: Run SFT tests**

Run:

```powershell
python -m pytest tests/test_sft_builder.py -q
```

Expected: pass.

- [ ] **Step 9: Commit Task 4**

Run:

```bash
git add src/small_model_train/sft_builder.py scripts/build_sft_dataset.py tests/test_sft_builder.py
git commit -m "feat: gate formal sft on style contracts"
```

---

## Task 5: Record StyleContract Provenance In Training Manifests

**Files:**
- Modify: `scripts/run_sft_train.py`
- Modify: `src/small_model_train/run_manifest.py`
- Modify: `tests/test_stage2_training.py`

- [ ] **Step 1: Add failing manifest test for style contract provenance**

Append to `tests/test_stage2_training.py`:

```python
def test_run_sft_train_dry_run_records_style_contract_provenance(
    monkeypatch,
    tmp_path: Path,
):
    from scripts import run_sft_train
    from small_model_train.style_contract import build_style_contract_asset, write_style_contract_asset

    sft_dataset = tmp_path / "data" / "sft.jsonl"
    eval_cards = tmp_path / "data" / "eval.jsonl"
    contract_path = tmp_path / "data_style" / "style_contract.json"
    sft_dataset.parent.mkdir(parents=True)
    sft_dataset.write_text("{}\n", encoding="utf-8")
    write_jsonl(eval_cards, [_execution_card("case1")])
    contract = build_style_contract_asset(
        style_contract_id="author_main_v1",
        approval_status="approved",
        source_corpus={
            "path": "data_clean/chapters_split.jsonl",
            "sha256": "a" * 64,
            "quality_filter": "quality_tag=A",
            "row_count": 1,
            "selected_rows": 1,
            "split_summary": {"train": 1},
        },
        profile_metrics={
            "chapter_count": 1,
            "avg_chinese_chars": 1200,
            "avg_paragraph_chars": 80,
            "avg_dialogue_ratio": 0.3,
            "chinese_chars": {"min": 1200, "max": 1200, "avg": 1200, "p50": 1200, "p90": 1200},
            "paragraph_chars": {"min": 80, "max": 80, "avg": 80, "p50": 80, "p90": 80},
            "dialogue_ratio": {"min": 0.3, "max": 0.3, "avg": 0.3, "p50": 0.3, "p90": 0.3},
            "sentence_chars": {"min": 12, "max": 12, "avg": 12, "p50": 12, "p90": 12},
            "punctuation_density": {"。": 0.02},
            "ai_taste": {"phrase_hits": {"空气仿佛凝固了": 0}, "total_hits": 0, "hits_per_10k_chars": 0},
            "source_filter": {
                "total_rows": 1,
                "selected_rows": 1,
                "skipped_rows": 0,
                "quality_filter": "quality_tag=A",
            },
        },
    )
    write_style_contract_asset(contract_path, contract)

    model_report = tmp_path / "reports" / "model.json"
    env_report = tmp_path / "reports" / "env.json"
    write_json_preflight(model_report, kind="model", passed=True)
    write_json_preflight(env_report, kind="environment", passed=True)
    write_valid_adapter(tmp_path / "outputs" / "sft_smoke")

    output_dir = tmp_path / "outputs" / "sft_v1"
    config_path = output_dir / "training_config_snapshot.yaml"

    def fake_build_train_run(**kwargs):
        config_path.parent.mkdir(parents=True)
        config_path.write_text("output_dir: sft_v1\n", encoding="utf-8")
        return {
            "name": kwargs["name"],
            "config_path": str(config_path),
            "command": ["llamafactory-cli", "train", str(config_path)],
        }

    monkeypatch.setattr(run_sft_train, "build_train_run", fake_build_train_run)
    monkeypatch.setattr(
        run_sft_train,
        "run_training_dry",
        lambda _run: {
            "exit_code": 0,
            "command_text": f"llamafactory-cli train {config_path}",
            "error": {"error_type": "none", "suggestion": "dry-run"},
        },
    )
    monkeypatch.setattr(
        run_sft_train.sys,
        "argv",
        [
            "run_sft_train.py",
            "--dry-run",
            "--sft-dataset",
            str(sft_dataset),
            "--eval-cards",
            str(eval_cards),
            "--model-report-json",
            str(model_report),
            "--env-report-json",
            str(env_report),
            "--smoke-adapter-dir",
            str(tmp_path / "outputs" / "sft_smoke"),
            "--style-contract-json",
            str(contract_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert run_sft_train.main() == 0
    manifest = json.loads((output_dir / "run_manifest.json").read_text("utf-8"))
    assert manifest["style_contract"]["style_contract_id"] == contract["style_contract_id"]
    assert manifest["style_contract"]["contract_sha256"] == contract["contract_sha256"]
    assert manifest["style_contract"]["approval_status"] == "approved"
    assert manifest["style_contract"]["schema"]["valid"] is True
    assert manifest["formal_evidence"] is False
```

- [ ] **Step 2: Run manifest style test and verify failure**

Run:

```powershell
python -m pytest tests/test_stage2_training.py::test_run_sft_train_dry_run_records_style_contract_provenance -q
```

Expected: fail because `run_sft_train.py` has no `--style-contract-json`.

- [ ] **Step 3: Add style contract manifest summary helper**

Modify `scripts/run_sft_train.py` imports:

```python
from small_model_train.artifact_manifest import file_sha256, summarize_jsonl_artifact
from small_model_train.style_contract import read_style_contract_asset
```

Add helper:

```python
def _style_contract_for_manifest(path: str | Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    try:
        asset = read_style_contract_asset(path)
    except ValueError as exc:
        return {
            "path": str(path),
            "schema": {"valid": False, "errors": [str(exc)]},
        }
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "style_contract_id": asset["style_contract_id"],
        "contract_sha256": asset["contract_sha256"],
        "approval_status": asset["approval_status"],
        "schema": {"valid": True, "errors": []},
    }
```

- [ ] **Step 4: Wire CLI arg and manifest**

In `scripts/run_sft_train.py` parser, add:

```python
parser.add_argument("--style-contract-json")
```

Before manifest write:

```python
style_contract_summary = _style_contract_for_manifest(args.style_contract_json)
if style_contract_summary is not None and not style_contract_summary["schema"]["valid"]:
    for error in style_contract_summary["schema"]["errors"]:
        print(error, file=sys.stderr)
    return 1
```

Pass `style_contract=style_contract_summary` to `build_run_manifest()`.

Update `formal_evidence` to require:

```python
and style_contract_summary is not None
and style_contract_summary.get("schema", {}).get("valid") is True
and style_contract_summary.get("approval_status") in {"approved", "frozen"}
```

- [ ] **Step 5: Run manifest tests**

Run:

```powershell
python -m pytest tests/test_stage2_training.py -q
```

Expected: pass.

- [ ] **Step 6: Commit Task 5**

Run:

```bash
git add scripts/run_sft_train.py src/small_model_train/run_manifest.py tests/test_stage2_training.py
git commit -m "feat: record style contract provenance"
```

---

## Task 6: Stage 5B Documentation

**Files:**
- Create: `docs/stage5b-style-contract-closure.zh.md`
- Modify: `README.md`
- Modify: `docs/index.zh.md`
- Modify: `docs/project-map.zh.md`
- Modify: `docs/pipeline-flow.zh.md`

- [ ] **Step 1: Write Stage 5B runbook**

Create `docs/stage5b-style-contract-closure.zh.md`:

````markdown
# Stage 5B StyleContract 闭环指南

## 目标

Stage 5B 把风格契约从根目录里的普通 Markdown 文本升级为可审阅、可哈希、可追踪的 StyleContract 资产。它不扩样、不批准章节卡，也不替代真人风格审阅。

## 生成 StyleContract

```powershell
python scripts/build_style_contract.py --chapters data_clean/chapters_split.jsonl --contract-json-output data_style/style_contract_author_main_v1.json --contract-output style_contract.md --metrics-output data_style/style_metrics_author_main_v1.json --style-contract-id author_main_v1
```

默认状态是 `pending_review`。这表示资产可以被审阅和 smoke/dev 引用，但不能进入 formal SFT。

## 审阅状态

- `pending_review`：默认状态，等待人工检查。
- `approved`：允许 formal SFT 使用。
- `frozen`：允许 formal SFT 使用，并表示同 ID 资产不应被覆盖。
- `draft` / `rejected`：不能进入 formal SFT。

## Formal SFT

formal SFT 必须显式传入 approved 或 frozen StyleContract：

```powershell
python scripts/build_sft_dataset.py --cards data_cards/chapter_cards_approved.jsonl --chapters data_clean/chapters_split.jsonl --output data_sft/sft_chapter_formal.jsonl --dataset-info-output data_sft/dataset_info_formal.json --style-contract-json data_style/style_contract_author_main_v1.json
```

如果 contract 仍是 `pending_review`，命令应失败。卡里的 `style_contract_id` 和 `style_contract_sha256` 必须与 JSON 完全一致。

## 训练 Manifest

`run_sft_train.py` 会在 manifest 中记录：

- SFT dataset 路径、sha256、行数。
- eval cards 路径、sha256、schema 校验结果。
- StyleContract 路径、id、hash、approval status。
- `formal_evidence` 是否为真实 formal 证据。

dry-run 可以验证命令构造，但不会产生 formal evidence。

## Stage 5B 不证明什么

- 不自动批准任何 StyleContract。
- 不把现有草稿章节卡变成正式执行卡。
- 不实现 Stage 5C 的 Card Compiler。
- 不做 sealed test、group split 或近重复检查。
- 不允许直接扩到 100/500 样本。
````

- [ ] **Step 2: Update README**

In `README.md`, under “现有阶段指南”, add:

```markdown
- [Stage 5B StyleContract 闭环指南](docs/stage5b-style-contract-closure.zh.md)
```

- [ ] **Step 3: Update docs index**

In `docs/index.zh.md`, add Stage 5B after Stage 5A in both route and stage guide sections:

```markdown
- [Stage 5B StyleContract 闭环指南](stage5b-style-contract-closure.zh.md)：解释 StyleContract JSON、Markdown、metrics、approval status 和 formal SFT 绑定。
```

- [ ] **Step 4: Update project map**

In `docs/project-map.zh.md`, add `data_style/` to generated artifact locations:

```markdown
- `data_style/`：Stage 5B 生成的 StyleContract JSON 和 style metrics。
```

- [ ] **Step 5: Update pipeline flow**

In `docs/pipeline-flow.zh.md`, replace references that imply `style_contract.md` alone is the contract source with:

```markdown
Stage 5B 起，风格资产由 `scripts/build_style_contract.py` 生成三件套：`data_style/style_contract_author_main_v1.json`、`style_contract.md`、`data_style/style_metrics_author_main_v1.json`。JSON 是 formal SFT 的机器门禁源，Markdown 只用于人工审阅。
```

- [ ] **Step 6: Run docs reference scan**

Run:

```powershell
rg -n "Stage 5B|style_contract_author_main_v1|data_style|pending_review|style-contract-json" README.md docs
```

Expected: README, docs index, project map, pipeline flow, and Stage 5B runbook mention the new flow.

- [ ] **Step 7: Commit Task 6**

Run:

```bash
git add docs/stage5b-style-contract-closure.zh.md README.md docs/index.zh.md docs/project-map.zh.md docs/pipeline-flow.zh.md
git commit -m "docs: add stage5b style contract runbook"
```

---

## Task 7: Verification

**Files:**
- No additional files.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
python -m pytest tests/test_style_contract.py tests/test_style_profile.py tests/test_sft_builder.py tests/test_stage2_training.py -q
```

Expected: pass.

- [ ] **Step 2: Run full suite**

Run:

```powershell
python -m pytest -q
```

Expected: pass.

- [ ] **Step 3: Generate a pending StyleContract asset locally**

Run:

```powershell
python scripts/build_style_contract.py --chapters data_clean/chapters_split.jsonl --contract-json-output data_style/style_contract_author_main_v1.json --contract-output style_contract.md --metrics-output data_style/style_metrics_author_main_v1.json --style-contract-id author_main_v1
```

Expected:

- `data_style/style_contract_author_main_v1.json` exists.
- `style_contract.md` exists.
- `data_style/style_metrics_author_main_v1.json` exists.
- JSON `approval_status` is `pending_review`.

- [ ] **Step 4: Inspect generated asset**

Run:

```powershell
@'
import json
from pathlib import Path
path = Path("data_style/style_contract_author_main_v1.json")
asset = json.loads(path.read_text(encoding="utf-8"))
print({
    "style_contract_id": asset["style_contract_id"],
    "approval_status": asset["approval_status"],
    "hash_len": len(asset["contract_sha256"]),
    "selected_rows": asset["source_corpus"]["selected_rows"],
})
raise SystemExit(0 if asset["approval_status"] == "pending_review" and len(asset["contract_sha256"]) == 64 else 1)
'@ | python -
```

Expected: exit code 0 and `approval_status` is `pending_review`.

- [ ] **Step 5: Verify raw eval cards are rejected**

Run:

```powershell
python scripts/run_sft_train.py --dry-run --config configs/sft_qlora_qwen3_4b.yaml --sft-dataset data_sft/sft_chapter_v1.jsonl --eval-cards data_cards/eval_cards_50.jsonl --model-report-json reports/model_check_report.json --env-report-json reports/training_env_report.json --output-dir outputs/stage5b_eval_schema_probe --smoke-adapter-dir outputs/sft_smoke
```

Expected: nonzero exit and an execution-card schema error mentioning missing execution-card fields. No `outputs/stage5b_eval_schema_probe/run_manifest.json` should be written.

- [ ] **Step 6: Verify pending contract blocks formal SFT**

Run:

```powershell
python scripts/build_sft_dataset.py --cards data_cards/chapter_cards.jsonl --chapters data_clean/chapters_split.jsonl --output outputs/stage5b_pending_should_fail.jsonl --style-contract-json data_style/style_contract_author_main_v1.json
```

Expected: nonzero exit because the generated contract is `pending_review` and current cards are draft/unapproved.

- [ ] **Step 7: Final git diff check**

Run:

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors. Status should show only intentional generated local artifacts in ignored directories or no tracked changes after commits.

---

## Stage 5B Exit Criteria

Stage 5B is complete only when all of these are true:

- Full pytest suite passes.
- `run_sft_train.py` and `run_sft_smoke.py` reject raw eval-card files that do not match execution-card schema.
- Training manifests record SFT dataset, eval cards, style contract provenance, and `formal_evidence`.
- `scripts/build_style_contract.py` writes StyleContract JSON, Markdown, and metrics from the same source data.
- Generated StyleContract defaults to `pending_review`.
- Formal SFT refuses missing, pending, rejected, draft, or hash-mismatched StyleContracts.
- Formal SFT accepts approved/frozen matching StyleContracts.
- `data_style/` is ignored as generated local artifact storage.
- Stage 5B docs explain the operating sequence and boundaries.

---

## Self-Review

- Spec coverage: Tasks cover Stage 5A schema/manifest hardening, expanded style metrics, StyleContract JSON/Markdown/metrics, formal SFT gate, manifest provenance, generated local asset, and docs.
- Marker scan: no unresolved implementation markers remain; every code step provides concrete snippets, files, commands, and expected output.
- Type consistency: StyleContract fields are consistently `style_contract_id`, `contract_sha256`, and `approval_status`; card fields remain `style_contract_id` and `style_contract_sha256`; manifests use `style_contract.contract_sha256` and `formal_evidence`.
