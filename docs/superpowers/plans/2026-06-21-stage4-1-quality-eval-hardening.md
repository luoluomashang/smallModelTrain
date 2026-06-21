# Stage 4.1 Quality Eval Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible Stage 4.1 quality-eval tooling before any 100/500-sample expansion.

**Architecture:** Keep GPU-heavy inference in the existing eval worker. Add lightweight deterministic tooling around it: subset selection, budget summary reporting, tracked 6144 smoke config, and a Chinese runbook.

**Tech Stack:** Python 3.11, pytest, JSONL, Markdown docs, existing `small_model_train` scripts and modules.

---

## File Map

- Create: `src/small_model_train/stage4_quality.py`
  - Deterministic subset selection, budget summary, outline marker analysis, Markdown report rendering.
- Create: `tests/test_stage4_quality.py`
  - Unit and CLI regression tests for Stage 4.1 helpers.
- Create: `scripts/build_eval_quality_subset.py`
  - CLI for writing `data_cards/eval_cards_quality_subset.jsonl`.
- Create: `scripts/build_stage4_quality_report.py`
  - CLI for writing `reports/stage4_1_quality_eval_budget_report.md`.
- Create: `configs/sft_qlora_qwen3_4b_smoke_6144.yaml`
  - Tracked 6144 cutoff smoke config.
- Create: `docs/stage4-1-quality-eval-guide.zh.md`
  - Stage 4.1 runbook.
- Modify: `README.md`
  - Add Stage 4.1 command sequence.

---

## Task 1: Add Stage 4.1 Quality Helpers

**Files:**
- Create: `src/small_model_train/stage4_quality.py`
- Create: `tests/test_stage4_quality.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_stage4_quality.py` with tests for:

```python
from small_model_train.stage4_quality import (
    detect_outline_markers,
    render_quality_budget_report,
    select_quality_subset,
    summarize_quality_budget,
)


def _card(sample_id):
    return {"id": sample_id, "chapter_goal": f"goal {sample_id}"}


def test_select_quality_subset_prioritizes_outline_leak_metrics():
    cards = [_card("a"), _card("b"), _card("c")]
    metrics = [
        {"id": "a", "failure_types": ["length_short"], "char_count_zh": 300},
        {"id": "b", "failure_types": ["length_short", "outline_leak"], "char_count_zh": 280},
        {"id": "c", "failure_types": [], "char_count_zh": 2200},
    ]

    subset = select_quality_subset(cards, metrics, count=2)

    assert [row["id"] for row in subset] == ["b", "a"]


def test_select_quality_subset_falls_back_to_eval_order_without_metrics():
    assert [row["id"] for row in select_quality_subset([_card("a"), _card("b")], [], 1)] == ["a"]


def test_detect_outline_markers_reports_known_markers():
    assert detect_outline_markers("以下是正文：【章节结构】") == ["【", "】", "章节结构", "以下是正文"]


def test_summarize_quality_budget_counts_rows_tokens_and_failures():
    cards = [_card("a"), _card("b")]
    generated = [
        {"id": "a", "output": "短文", "params": {"max_new_tokens": 1024}},
        {"id": "b", "output": "【章节结构】", "params": {"max_new_tokens": 1024}},
    ]
    metrics = [
        {"id": "a", "hard_gate_pass": False, "char_count_zh": 300, "failure_types": ["length_short"]},
        {"id": "b", "hard_gate_pass": False, "char_count_zh": 320, "failure_types": ["length_short", "outline_leak"]},
    ]

    summary = summarize_quality_budget(cards, generated, metrics)

    assert summary["expected_rows"] == 2
    assert summary["generated_rows"] == 2
    assert summary["metrics_rows"] == 2
    assert summary["max_new_tokens"] == [1024]
    assert summary["failure_counts"] == {"length_short": 2, "outline_leak": 1}
    assert summary["decision"] == "blocked_length_short"


def test_render_quality_budget_report_does_not_include_generated_text():
    cards = [_card("a")]
    generated = [{"id": "a", "output": "以下是正文：秘密文本", "params": {"max_new_tokens": 1024}}]
    metrics = [{"id": "a", "hard_gate_pass": False, "char_count_zh": 300, "failure_types": ["outline_leak"]}]
    summary = summarize_quality_budget(cards, generated, metrics)

    report = render_quality_budget_report("Stage 4.1", summary)

    assert "# Stage 4.1" in report
    assert "outline_leak: 1" in report
    assert "以下是正文：秘密文本" not in report
```

- [ ] **Step 2: Run failing tests**

Run: `python -m pytest tests/test_stage4_quality.py -q`

Expected: fail with `ModuleNotFoundError` for `small_model_train.stage4_quality`.

- [ ] **Step 3: Implement minimal helper module**

Create `src/small_model_train/stage4_quality.py` with:

```python
OUTLINE_MARKERS = ("【", "】", "章节结构", "以下是正文")

def select_quality_subset(cards, metrics, count):
    metrics_by_id = {row.get("id"): row for row in metrics}
    indexed = list(enumerate(cards))
    def key(item):
        index, card = item
        failures = set(metrics_by_id.get(card.get("id"), {}).get("failure_types", []))
        priority = 0 if "outline_leak" in failures else 1 if failures else 2
        return (priority, index)
    return [card for _, card in sorted(indexed, key=key)[:count]]

def detect_outline_markers(text):
    return [marker for marker in OUTLINE_MARKERS if marker in text]
```

Then add summary and Markdown rendering using `collections.Counter` and averages from metric rows.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_stage4_quality.py -q`

Expected: pass.

---

## Task 2: Add Stage 4.1 CLI Wrappers

**Files:**
- Create: `scripts/build_eval_quality_subset.py`
- Create: `scripts/build_stage4_quality_report.py`
- Modify: `tests/test_stage4_quality.py`

- [ ] **Step 1: Add failing CLI tests**

Add tests that call:

```powershell
python scripts/build_eval_quality_subset.py --cards cards.jsonl --metrics metrics.jsonl --output subset.jsonl --count 1
python scripts/build_stage4_quality_report.py --cards subset.jsonl --generated generated.jsonl --metrics metrics.jsonl --report report.md --title "Stage 4.1"
```

Expected outputs:

- `subset.jsonl` contains the prioritized outline-leak card.
- `report.md` contains `# Stage 4.1` and the decision.

- [ ] **Step 2: Run failing CLI tests**

Run: `python -m pytest tests/test_stage4_quality.py -q`

Expected: fail because scripts do not exist.

- [ ] **Step 3: Implement CLI scripts**

Use existing `read_jsonl` and `write_jsonl` helpers. Both scripts should insert `src` into `sys.path`, parse arguments with `argparse`, and print the output path.

- [ ] **Step 4: Run targeted tests**

Run: `python -m pytest tests/test_stage4_quality.py -q`

Expected: pass.

---

## Task 3: Add 6144 Smoke Config And Docs

**Files:**
- Create: `configs/sft_qlora_qwen3_4b_smoke_6144.yaml`
- Create: `docs/stage4-1-quality-eval-guide.zh.md`
- Modify: `README.md`

- [ ] **Step 1: Add tracked 6144 config**

Create a smoke config with `cutoff_len: 6144`, `num_train_epochs: 1`, `logging_steps: 5`, `save_steps: 50`, `max_samples: 100`, and `output_dir: outputs/sft_smoke`.

- [ ] **Step 2: Add Stage 4.1 runbook**

Document commands for:

```powershell
python scripts/run_sft_smoke.py --config configs/sft_qlora_qwen3_4b_smoke_6144.yaml --eval-cards data_cards/eval_cards_50.jsonl
python scripts/build_eval_quality_subset.py --cards data_cards/eval_cards_50.jsonl --metrics outputs/sft_smoke/metrics.jsonl --output data_cards/eval_cards_quality_subset.jsonl --count 8
python scripts/run_eval_inference.py --cards data_cards/eval_cards_quality_subset.jsonl --adapter-dir outputs/sft_smoke --output outputs/sft_smoke/generated_subset_1024.jsonl --model-name sft_smoke_subset_1024 --max-new-tokens 1024
python scripts/score_outputs.py --cards data_cards/eval_cards_quality_subset.jsonl --outputs outputs/sft_smoke/generated_subset_1024.jsonl --output outputs/sft_smoke/metrics_subset_1024.jsonl
python scripts/build_stage4_quality_report.py --cards data_cards/eval_cards_quality_subset.jsonl --generated outputs/sft_smoke/generated_subset_1024.jsonl --metrics outputs/sft_smoke/metrics_subset_1024.jsonl --report reports/stage4_1_quality_eval_budget_report.md --title "Stage 4.1 Quality Eval Budget Report"
```

- [ ] **Step 3: Update README**

Add a short Stage 4.1 section linking the runbook.

---

## Task 4: Generate Local Stage 4.1 Baseline Artifacts

**Files:**
- Generated ignored artifacts under `data_cards/` and `reports/`.

- [ ] **Step 1: Build local quality subset**

Run:

```powershell
python scripts/build_eval_quality_subset.py --cards data_cards/eval_cards_50.jsonl --metrics outputs/sft_smoke/metrics.jsonl --output data_cards/eval_cards_quality_subset.jsonl --count 8
```

Expected: writes 8 cards.

- [ ] **Step 2: Build current 256-token baseline report**

Run:

```powershell
python scripts/build_stage4_quality_report.py --cards data_cards/eval_cards_50.jsonl --generated outputs/sft_smoke/generated.jsonl --metrics outputs/sft_smoke/metrics.jsonl --report reports/stage4_1_quality_eval_budget_report.md --title "Stage 4.1 Quality Eval Budget Report"
```

Expected: report decision is `blocked_length_short`.

---

## Task 5: Verification

**Files:**
- No new files.

- [ ] **Step 1: Run targeted tests**

Run: `python -m pytest tests/test_stage4_quality.py -q`

Expected: pass.

- [ ] **Step 2: Run full suite**

Run: `python -m pytest -q`

Expected: pass.

- [ ] **Step 3: Check docs for stale retry path**

Run: `rg -n "sft_smoke_retry_6144|Stage 4.1|stage4-1" README.md docs configs scripts src tests`

Expected: Stage 4.1 docs point to the tracked config. Older Stage 4 historical docs may mention the ignored retry path only as history.

## Self-Review

- Spec coverage: every chosen design item maps to a task.
- Placeholder scan: no TBD/TODO placeholders are used.
- Type consistency: `failure_types`, `char_count_zh`, `hard_gate_pass`, `output`, and `params.max_new_tokens` match existing JSONL schemas.
