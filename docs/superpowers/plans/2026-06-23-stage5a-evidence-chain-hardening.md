# Stage 5A Evidence Chain Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make training, inference, scoring, review, and formal SFT admission produce trustworthy, replayable evidence before expanding data volume or changing model technique.

**Architecture:** Keep the current `scripts/` as thin command entrypoints and put reusable logic in `src/small_model_train/`. Centralize prompt rendering, preserve raw generation records, require machine-readable preflight reports, isolate rule-projection review from real quality decisions, and add formal-mode gates that stop draft cards from entering real SFT runs.

**Tech Stack:** Python 3.10+, pytest, JSON/JSONL, Markdown docs, stdlib `hashlib`, existing Transformers/PEFT worker path, existing LLaMA-Factory config flow.

---

## Scope Check

This plan implements the next stage only: **Stage 5A / evidence-chain hardening**. It covers the P0 items from the uploaded design review that must be fixed before larger training runs:

- training and inference prompt consistency
- raw generation evidence preservation
- structured preflight and run manifest
- adapter success evidence beyond file existence
- mock review isolation
- draft-card blocking for formal SFT

It intentionally does not implement the full StyleContract system, Card Compiler, group split, near-duplicate detection, rejection sampling, CPT, or DPO. Those are follow-on Stage 5B/5C/5D plans after Stage 5A makes evidence reliable.

Full multi-stage roadmap:

- `docs/superpowers/plans/2026-06-23-small-model-train-full-roadmap.md`

Current verified baseline before this plan:

- `python -m pytest -q`
- Expected baseline at plan time: `284 passed`

---

## File Map

- Create: `src/small_model_train/prompt_renderer.py`
  - Own the system instruction, execution input renderer, chat messages, tokenizer chat-template application, and prompt hash.
- Modify: `src/small_model_train/sft_builder.py`
  - Reuse `prompt_renderer` instead of owning prompt section rendering.
- Modify: `src/small_model_train/stage2_inference.py`
  - Reuse prompt renderer, add sanitization-with-events helpers, and build raw-first generation records.
- Modify: `scripts/stage2_eval_worker.py`
  - Generate with the shared prompt renderer, write raw/sanitized evidence fields, record seed and token counts.
- Modify: `scripts/run_eval_inference.py`
  - Pass `--seed` to the worker and dry-run records.
- Modify: `scripts/score_outputs.py`
  - Prefer `raw_output` when present.
- Create: `src/small_model_train/preflight_reports.py`
  - Read, write, and validate machine-readable preflight reports.
- Create: `src/small_model_train/run_manifest.py`
  - Build and write training run manifests.
- Modify: `scripts/check_local_model.py`
  - Add JSON report output next to the Markdown report.
- Modify: `scripts/check_training_env.py`
  - Add JSON report output next to the Markdown report.
- Modify: `scripts/run_sft_train.py`
  - Require JSON preflight reports unless an explicit legacy flag is passed; write run manifest.
- Modify: `src/small_model_train/chapter_cards.py`
  - Mark auto-generated cards as draft-only with approval and style-contract provenance fields.
- Modify: `src/small_model_train/sft_builder.py`
  - Add formal-mode card approval gates.
- Modify: `scripts/build_sft_dataset.py`
  - Add `--allow-draft-cards`; formal mode is the default.
- Modify: `scripts/run_agent_review.py`
  - Rename generated mock reviews to rule projection and prevent projection-only results from producing release-ready decisions.
- Modify: `src/small_model_train/agent_review.py`
  - Validate optional `review_backend` and evidence span fields without breaking imported historical rows.
- Modify: `src/small_model_train/stage4_quality.py`
  - Preserve `rules_pass_agent_pending` behavior for projection-only review summaries.
- Create: `tests/test_prompt_renderer.py`
  - Prompt parity and hash tests.
- Modify: `tests/test_sft_builder.py`
  - Formal card gate tests.
- Modify: `tests/test_stage2_inference.py`
  - Raw/sanitized generation record and seed tests.
- Modify: `tests/test_scoring.py`
  - `raw_output` scoring preference tests.
- Modify: `tests/test_stage2_training.py`
  - JSON preflight and run manifest tests.
- Modify: `tests/test_agent_review_cli.py`
  - Rule-projection isolation tests.
- Create: `docs/stage5a-evidence-chain-hardening.zh.md`
  - Human runbook for the new sequence.
- Modify: `README.md`
  - Link the Stage 5A runbook and update preflight commands.

---

## Task 1: Centralize Prompt Rendering

**Files:**
- Create: `src/small_model_train/prompt_renderer.py`
- Modify: `src/small_model_train/sft_builder.py`
- Modify: `src/small_model_train/stage2_inference.py`
- Create: `tests/test_prompt_renderer.py`
- Modify: `tests/test_sft_builder.py`
- Modify: `tests/test_stage2_inference.py`

- [ ] **Step 1: Write failing prompt renderer tests**

Create `tests/test_prompt_renderer.py`:

```python
from small_model_train.prompt_renderer import (
    SYSTEM_PROMPT,
    build_chat_messages,
    prompt_sha256,
    render_execution_input,
    render_model_input_prefix,
)


def _card() -> dict:
    return {
        "id": "case-1",
        "style_contract": "短句推进，动作细节清楚。",
        "previous_summary": "主角刚抵达旧城。",
        "chapter_goal": "让主角发现密室钥匙。",
        "conflict_beat": "必须在管家返回前打开暗格。",
        "payoff_beat": "铜钥匙和密道图一起出现。",
        "chapter_structure": [
            {
                "step": 1,
                "name": "搜寻",
                "goal": "在书房找到线索",
                "estimated_chars": "800",
            }
        ],
        "character_states": [
            {
                "name": "林照",
                "state": "谨慎但兴奋",
                "speech_style": "简短直接",
            }
        ],
        "must_include": ["铜钥匙", "雨声"],
        "must_not_include": ["解释设定"],
        "ending_hook": "门后传来第二个人的呼吸声。",
        "target_word_count": "2000-2500中文汉字",
        "source_text": "这是一段不应该泄漏到提示词里的原文内容而且足够长",
    }


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        rendered = "\n".join(
            f"<{message['role']}>\n{message['content']}" for message in messages
        )
        if add_generation_prompt:
            rendered += "\n<assistant>\n"
        return rendered


def test_render_execution_input_contains_card_fields_without_source_text():
    rendered = render_execution_input(_card())

    assert "让主角发现密室钥匙" in rendered
    assert "短句推进，动作细节清楚。" in rendered
    assert "铜钥匙" in rendered
    assert "雨声" in rendered
    assert "不应该泄漏到提示词" not in rendered


def test_build_chat_messages_uses_single_system_prompt_and_user_card():
    messages = build_chat_messages(_card())

    assert messages == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_execution_input(_card())},
    ]


def test_render_model_input_prefix_uses_tokenizer_chat_template():
    prefix = render_model_input_prefix(_card(), FakeTokenizer())

    assert prefix.startswith("<system>\n你是作者的正文执行器")
    assert "让主角发现密室钥匙" in prefix
    assert prefix.endswith("<assistant>\n")


def test_prompt_hash_is_stable_for_same_card():
    first = prompt_sha256(render_model_input_prefix(_card(), FakeTokenizer()))
    second = prompt_sha256(render_model_input_prefix(_card(), FakeTokenizer()))

    assert first == second
    assert len(first) == 64
```

- [ ] **Step 2: Run prompt tests and verify failure**

Run:

```powershell
python -m pytest tests/test_prompt_renderer.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'small_model_train.prompt_renderer'`.

- [ ] **Step 3: Implement `prompt_renderer.py`**

Create `src/small_model_train/prompt_renderer.py`:

```python
from __future__ import annotations

import hashlib
from typing import Any


SYSTEM_PROMPT = "你是作者的正文执行器。严格执行章节卡，并保持指定作者风格。"


def _format_list(title: str, values: list[str]) -> str:
    body = "\n".join(f"- {value}" for value in values) if values else "- 无"
    return f"【{title}】\n{body}"


def _format_structure(items: list[dict[str, Any]]) -> str:
    if not items:
        return "【章节结构】\n- 无"
    lines = ["【章节结构】"]
    for index, item in enumerate(items):
        step = item.get("step")
        name = item.get("name")
        goal = item.get("goal")
        chars = item.get("estimated_chars")
        if type(step) is not int or step < 1:
            raise ValueError(f"chapter_structure[{index}].step must be a positive integer")
        if not name:
            raise ValueError(f"chapter_structure[{index}].name is required")
        if not goal:
            raise ValueError(f"chapter_structure[{index}].goal is required")
        if not isinstance(chars, str) or not chars.strip():
            raise ValueError(f"chapter_structure[{index}].estimated_chars must be a non-empty string")
        lines.append(f"- {step}. {name}：{goal}（建议 {chars}）")
    return "\n".join(lines)


def _format_characters(items: list[dict[str, Any]]) -> str:
    if not items:
        return "【人物状态】\n- 无"
    lines = ["【人物状态】"]
    for item in items:
        lines.append(
            f"- {item.get('name', '')}：{item.get('state', '')}；说话方式：{item.get('speech_style', '')}"
        )
    return "\n".join(lines)


def render_execution_input(card: dict[str, Any]) -> str:
    sections = [
        "【风格契约】",
        card.get("style_contract", ""),
        "【前情摘要】",
        card.get("previous_summary", ""),
        "【本章目标】",
        card.get("chapter_goal", ""),
        "【冲突推进】",
        card.get("conflict_beat", ""),
        "【爽点兑现】",
        card.get("payoff_beat", ""),
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


def build_chat_messages(card: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_execution_input(card)},
    ]


def render_model_input_prefix(card: dict[str, Any], tokenizer: Any | None = None) -> str:
    messages = build_chat_messages(card)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return SYSTEM_PROMPT + "\n\n" + render_execution_input(card) + "\n\n"


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Update `sft_builder.py` to reuse the renderer**

Change `src/small_model_train/sft_builder.py`:

```python
from small_model_train.prompt_renderer import SYSTEM_PROMPT, render_execution_input

INSTRUCTION = SYSTEM_PROMPT
```

Keep `_find_source_text_leak()` in `sft_builder.py`. Replace the body of `render_sft_input()` with:

```python
def render_sft_input(card: dict) -> str:
    rendered_input = render_execution_input(card)
    leak = _find_source_text_leak(rendered_input, card.get("source_text", ""))
    if leak:
        raise ValueError(f"{SOURCE_LEAK_ERROR_PREFIX}: {leak}")
    return rendered_input
```

- [ ] **Step 5: Update `stage2_inference.py` prompt entrypoints**

Change `render_eval_prompt(card)` to call `render_model_input_prefix(card)` when no tokenizer is passed, and add a tokenizer-aware helper:

```python
from small_model_train.prompt_renderer import (
    prompt_sha256,
    render_execution_input,
    render_model_input_prefix,
)


def render_eval_prompt(card: dict) -> str:
    return render_execution_input(card)


def render_eval_model_input(card: dict, tokenizer: Any | None = None) -> str:
    return render_model_input_prefix(card, tokenizer)
```

Dry-run output remains short and human-readable by using `render_eval_prompt(card)[:80]`.

- [ ] **Step 6: Run prompt and existing SFT/inference tests**

Run:

```powershell
python -m pytest tests/test_prompt_renderer.py tests/test_sft_builder.py tests/test_stage2_inference.py -q
```

Expected: pass.

---

## Task 2: Preserve Raw Generation Evidence

**Files:**
- Modify: `src/small_model_train/stage2_inference.py`
- Modify: `scripts/stage2_eval_worker.py`
- Modify: `scripts/run_eval_inference.py`
- Modify: `scripts/score_outputs.py`
- Modify: `tests/test_stage2_inference.py`
- Modify: `tests/test_scoring.py`

- [ ] **Step 1: Add failing generation record tests**

Add to `tests/test_stage2_inference.py`:

```python
from small_model_train.stage2_inference import (
    build_generation_row,
    sanitize_generated_output_with_events,
)


def test_sanitize_generated_output_with_events_keeps_audit_trail():
    sanitized, events = sanitize_generated_output_with_events(
        "【章节结构】\n- 承接：开场\n正文一句。"
    )

    assert sanitized == "正文一句。"
    assert events[0]["type"] == "drop_meta_line"
    assert events[0]["reason"] == "outline_or_meta_marker"


def test_build_generation_row_preserves_raw_and_sanitized_outputs():
    params = {"temperature": 0.7, "seed": 20260623}

    row = build_generation_row(
        sample_id="eval-1",
        raw_output="【章节结构】\n正文",
        model="sft_v1",
        params=params,
        sanitized_output="正文",
        sanitizer_events=[{"type": "drop_meta_line", "reason": "outline_or_meta_marker"}],
        finish_reason="unknown",
        generated_tokens=7,
        prompt_hash="a" * 64,
        adapter_dir="outputs/sft_v1",
    )

    assert row["id"] == "eval-1"
    assert row["output"] == "【章节结构】\n正文"
    assert row["raw_output"] == "【章节结构】\n正文"
    assert row["sanitized_output"] == "正文"
    assert row["sanitizer_events"][0]["type"] == "drop_meta_line"
    assert row["finish_reason"] == "unknown"
    assert row["generated_tokens"] == 7
    assert row["prompt_sha256"] == "a" * 64
    assert row["adapter_dir"] == "outputs/sft_v1"
    params["temperature"] = 0.1
    assert row["params"]["temperature"] == 0.7
```

Add to `tests/test_scoring.py`:

```python
def test_score_outputs_cli_prefers_raw_output_over_sanitized_output(tmp_path: Path):
    cards_path = tmp_path / "cards.jsonl"
    outputs_path = tmp_path / "generated.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    write_jsonl(cards_path, [_execution_card("case1")])
    write_jsonl(
        outputs_path,
        [
            {
                "id": "case1",
                "raw_output": "【章节结构】\n正文",
                "sanitized_output": "正文",
                "output": "正文",
            }
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/score_outputs.py",
            "--cards",
            str(cards_path),
            "--outputs",
            str(outputs_path),
            "--output",
            str(scores_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    score = read_jsonl(scores_path)[0]
    assert "outline_leak" in score["failure_types"]
```

- [ ] **Step 2: Run targeted tests and verify failure**

Run:

```powershell
python -m pytest tests/test_stage2_inference.py::test_sanitize_generated_output_with_events_keeps_audit_trail tests/test_stage2_inference.py::test_build_generation_row_preserves_raw_and_sanitized_outputs tests/test_scoring.py::test_score_outputs_cli_prefers_raw_output_over_sanitized_output -q
```

Expected: fail because `sanitize_generated_output_with_events()` and the expanded generation row schema do not exist.

- [ ] **Step 3: Add `sanitize_generated_output_with_events()`**

In `src/small_model_train/stage2_inference.py`, keep `sanitize_generated_output()` for backward compatibility and add:

```python
def sanitize_generated_output_with_events(
    text: str,
    max_chinese_chars: int | None = DEFAULT_MAX_CHINESE_CHARS,
) -> tuple[str, list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    lines: list[str] = []
    in_meta_block = False

    for line_number, raw_line in enumerate(
        text.replace("\r\n", "\n").replace("\r", "\n").split("\n"),
        start=1,
    ):
        stripped = raw_line.strip()
        if not stripped:
            if lines and lines[-1] != "":
                lines.append("")
            in_meta_block = False
            continue

        has_heading = bool(OUTLINE_HEADING_RE.search(stripped))
        has_outline_bracket = "【" in stripped or "】" in stripped
        has_meta_marker = any(marker in stripped for marker in META_DIRECTIVE_MARKERS)
        is_list_item = bool(LIST_ITEM_RE.match(stripped))
        is_separator = bool(SEPARATOR_RE.match(stripped))

        if has_heading or has_outline_bracket or has_meta_marker:
            events.append(
                {
                    "type": "drop_meta_line",
                    "line_number": line_number,
                    "reason": "outline_or_meta_marker",
                    "preview": stripped[:80],
                }
            )
            in_meta_block = True
            continue
        if is_separator:
            events.append(
                {
                    "type": "drop_separator_line",
                    "line_number": line_number,
                    "reason": "separator",
                    "preview": stripped[:80],
                }
            )
            in_meta_block = False
            continue
        if is_list_item:
            events.append(
                {
                    "type": "drop_list_line",
                    "line_number": line_number,
                    "reason": "list_item",
                    "preview": stripped[:80],
                }
            )
            continue
        if in_meta_block and len(stripped) <= 60 and not PROSE_END_RE.search(stripped):
            events.append(
                {
                    "type": "drop_meta_continuation",
                    "line_number": line_number,
                    "reason": "meta_block_continuation",
                    "preview": stripped[:80],
                }
            )
            continue

        in_meta_block = False
        lines.append(stripped)

    while lines and lines[-1] == "":
        lines.pop()
    joined = "\n".join(lines).strip()
    capped = _cap_chinese_chars(joined, max_chinese_chars)
    if capped != joined:
        events.append(
            {
                "type": "cap_chinese_chars",
                "reason": "max_chinese_chars",
                "max_chinese_chars": max_chinese_chars,
            }
        )
    return capped, events
```

Change `sanitize_generated_output()` to:

```python
def sanitize_generated_output(
    text: str,
    max_chinese_chars: int | None = DEFAULT_MAX_CHINESE_CHARS,
) -> str:
    sanitized, _events = sanitize_generated_output_with_events(text, max_chinese_chars)
    return sanitized
```

- [ ] **Step 4: Expand `build_generation_row()`**

Replace the function in `stage2_inference.py` with:

```python
def build_generation_row(
    sample_id: str,
    raw_output: str,
    model: str,
    params: dict,
    sanitized_output: str | None = None,
    sanitizer_events: list[dict[str, Any]] | None = None,
    finish_reason: str = "unknown",
    generated_tokens: int = 0,
    prompt_hash: str = "",
    adapter_dir: str = "",
) -> dict[str, Any]:
    sanitized = raw_output if sanitized_output is None else sanitized_output
    return {
        "id": sample_id,
        "output": raw_output,
        "raw_output": raw_output,
        "sanitized_output": sanitized,
        "model": model,
        "adapter_dir": adapter_dir,
        "params": dict(params),
        "finish_reason": finish_reason,
        "generated_tokens": generated_tokens,
        "prompt_sha256": prompt_hash,
        "sanitizer_events": list(sanitizer_events or []),
    }
```

Keep `output` equal to `raw_output` so older reports still read a text field while scoring sees raw model behavior.

- [ ] **Step 5: Update `stage2_eval_worker.py` to write raw-first records**

In `scripts/stage2_eval_worker.py`:

- import `prompt_sha256`, `render_eval_model_input`, and `sanitize_generated_output_with_events`
- add `parser.add_argument("--seed", type=int, default=20260623)`
- add `params["seed"] = args.seed` after `build_inference_params(...)`
- after importing Transformers, call:

```python
import torch

torch.manual_seed(args.seed)
```

Replace prompt creation and decoding with:

```python
prompt = render_eval_model_input(card, tokenizer)
inputs = tokenizer(prompt, return_tensors="pt")
...
raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
sanitized_output, sanitizer_events = sanitize_generated_output_with_events(raw_output)
sample_id = str(card.get("id", ""))
append_generation_row(
    args.output,
    build_generation_row(
        sample_id=sample_id,
        raw_output=raw_output,
        model=args.model_name,
        params=params,
        sanitized_output=sanitized_output,
        sanitizer_events=sanitizer_events,
        generated_tokens=int(new_tokens.shape[-1]),
        prompt_hash=prompt_sha256(prompt),
        adapter_dir=args.adapter_dir,
    ),
)
```

- [ ] **Step 6: Update `run_eval_inference.py` dry-run and command builder**

Add `--seed` to the launcher, dry-run params, and worker command. Dry-run rows should call:

```python
rows.append(
    build_generation_row(
        sample_id=sample_id,
        raw_output="[DRY RUN] " + render_eval_prompt(card)[:80],
        model=model_name,
        params=params,
        sanitized_output="[DRY RUN] " + render_eval_prompt(card)[:80],
        prompt_hash="",
    )
)
```

- [ ] **Step 7: Update scoring to prefer raw output**

Change `scripts/score_outputs.py`:

```python
text = row.get("raw_output", row.get("output", row.get("text", "")))
```

- [ ] **Step 8: Run targeted inference and scoring tests**

Run:

```powershell
python -m pytest tests/test_stage2_inference.py tests/test_scoring.py -q
```

Expected: pass after updating existing schema assertions from `{"id", "output", "model", "params"}` to include the new audit fields.

---

## Task 3: Require Structured Preflight Reports And Write Run Manifests

**Files:**
- Create: `src/small_model_train/preflight_reports.py`
- Create: `src/small_model_train/run_manifest.py`
- Modify: `scripts/check_local_model.py`
- Modify: `scripts/check_training_env.py`
- Modify: `scripts/run_sft_train.py`
- Modify: `tests/test_stage2_training.py`

- [ ] **Step 1: Add failing structured preflight tests**

Add to `tests/test_stage2_training.py`:

```python
def _write_preflight(path: Path, kind: str, passed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": kind,
                "passed": passed,
                "checked_at": "2026-06-23T00:00:00Z",
                "errors": [] if passed else [f"{kind} failed"],
                "warnings": [],
                "payload": {"fingerprint": kind},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_validate_full_training_prerequisites_reads_json_passed_state(tmp_path: Path):
    from scripts.run_sft_train import validate_full_training_prerequisites

    model_report_json = tmp_path / "reports" / "model.json"
    env_report_json = tmp_path / "reports" / "env.json"
    smoke_adapter = _valid_adapter_dir(tmp_path / "outputs" / "sft_smoke")
    _write_preflight(model_report_json, "model", passed=False)
    _write_preflight(env_report_json, "environment", passed=True)

    result = validate_full_training_prerequisites(
        model_report_json=model_report_json,
        env_report_json=env_report_json,
        smoke_adapter_dir=smoke_adapter,
    )

    assert result["passed"] is False
    assert "model preflight failed" in "\n".join(result["errors"])


def test_run_manifest_records_training_result_and_adapter_check(tmp_path: Path):
    from small_model_train.run_manifest import build_run_manifest

    manifest = build_run_manifest(
        run_name="sft_v1",
        command=["llamafactory-cli", "train", "snapshot.yaml"],
        exit_code=0,
        model_dir="E:/models/Qwen3-4B-Instruct-2507",
        output_dir="outputs/sft_v1",
        config_path="outputs/sft_v1/training_config_snapshot.yaml",
        preflight_reports=[
            {"kind": "model", "passed": True, "payload": {"fingerprint": "model"}},
            {"kind": "environment", "passed": True, "payload": {"fingerprint": "env"}},
        ],
        adapter_check={"passed": True, "errors": []},
    )

    assert manifest["schema_version"] == 1
    assert manifest["run_name"] == "sft_v1"
    assert manifest["training_exit_code"] == 0
    assert manifest["adapter_check"]["passed"] is True
```

If the test file does not already have `_valid_adapter_dir`, add:

```python
def _valid_adapter_dir(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    header = b"{}"
    (path / "adapter_model.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header
    )
    (path / "training_config_snapshot.yaml").write_text("output_dir: adapter\n", encoding="utf-8")
    return path
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
python -m pytest tests/test_stage2_training.py::test_validate_full_training_prerequisites_reads_json_passed_state tests/test_stage2_training.py::test_run_manifest_records_training_result_and_adapter_check -q
```

Expected: fail because `model_report_json`, `env_report_json`, and `run_manifest` support do not exist.

- [ ] **Step 3: Implement `preflight_reports.py`**

Create `src/small_model_train/preflight_reports.py`:

```python
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_preflight_report(
    kind: str,
    passed: bool,
    payload: dict[str, Any],
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": kind,
        "passed": bool(passed),
        "checked_at": utc_now_iso(),
        "errors": list(errors or []),
        "warnings": list(warnings or []),
        "payload": dict(payload),
    }


def write_preflight_report(path: str | Path, report: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_preflight_report(path: str | Path, expected_kind: str) -> dict[str, Any]:
    report_path = Path(path)
    if not report_path.is_file():
        raise ValueError(f"{expected_kind} preflight report is missing: {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{expected_kind} preflight report invalid JSON: {exc}") from exc
    if not isinstance(report, dict):
        raise ValueError(f"{expected_kind} preflight report must be a JSON object")
    if report.get("schema_version") != 1:
        raise ValueError(f"{expected_kind} preflight report schema_version must be 1")
    if report.get("kind") != expected_kind:
        raise ValueError(f"{expected_kind} preflight report kind mismatch: {report.get('kind')}")
    if type(report.get("passed")) is not bool:
        raise ValueError(f"{expected_kind} preflight report passed must be boolean")
    for field in ("errors", "warnings"):
        if not isinstance(report.get(field), list):
            raise ValueError(f"{expected_kind} preflight report {field} must be a list")
    if not isinstance(report.get("payload"), dict):
        raise ValueError(f"{expected_kind} preflight report payload must be an object")
    return report
```

- [ ] **Step 4: Add JSON output to model and environment checks**

In `scripts/check_local_model.py`, add:

```python
parser.add_argument("--json-output", default="reports/model_check_report.json")
```

After `result` is final:

```python
from small_model_train.preflight_reports import build_preflight_report, write_preflight_report

json_report = build_preflight_report(
    kind="model",
    passed=result["passed"],
    payload=result,
    errors=list(result.get("errors", [])),
)
write_preflight_report(args.json_output, json_report)
print(f"wrote JSON report to {args.json_output}")
```

In `scripts/check_training_env.py`, add:

```python
parser.add_argument("--json-output", default="reports/training_env_report.json")
```

After `snapshot` is collected:

```python
errors = list(snapshot.get("recommendation", {}).get("blocking_reasons", []))
json_report = build_preflight_report(
    kind="environment",
    passed=bool(snapshot["recommendation"]["allow_training"]),
    payload=snapshot,
    errors=errors,
)
write_preflight_report(args.json_output, json_report)
print(f"wrote JSON report to {args.json_output}")
```

- [ ] **Step 5: Update `run_sft_train.py` prerequisites**

Change `validate_full_training_prerequisites()` signature:

```python
def validate_full_training_prerequisites(
    model_report_json: str | Path,
    env_report_json: str | Path,
    smoke_adapter_dir: str | Path,
) -> dict[str, object]:
```

Inside it:

```python
from small_model_train.preflight_reports import read_preflight_report

errors = []
reports = []
for kind, path in (("model", model_report_json), ("environment", env_report_json)):
    try:
        report = read_preflight_report(path, kind)
    except ValueError as exc:
        errors.append(str(exc))
        continue
    reports.append(report)
    if not report["passed"]:
        errors.append(f"{kind} preflight failed: {path}")
        errors.extend(str(error) for error in report["errors"])
```

Keep the existing `check_adapter_dir(smoke_adapter_dir)` block.

Add CLI args:

```python
parser.add_argument("--model-report-json", default="reports/model_check_report.json")
parser.add_argument("--env-report-json", default="reports/training_env_report.json")
```

Remove Markdown report existence from the default training gate.

- [ ] **Step 6: Implement `run_manifest.py`**

Create `src/small_model_train/run_manifest.py`:

```python
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def current_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def build_run_manifest(
    run_name: str,
    command: list[str],
    exit_code: int,
    model_dir: str,
    output_dir: str,
    config_path: str,
    preflight_reports: list[dict[str, Any]],
    adapter_check: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": _now_iso(),
        "git_commit": current_git_commit(),
        "run_name": run_name,
        "command": list(command),
        "training_exit_code": int(exit_code),
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "config_path": str(config_path),
        "preflight_reports": list(preflight_reports),
        "adapter_check": dict(adapter_check),
        "passed": int(exit_code) == 0 and adapter_check.get("passed") is True,
    }


def write_run_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
```

- [ ] **Step 7: Write run manifest from `run_sft_train.py`**

After `result = ...`, run adapter check against `args.output_dir` when `result["exit_code"] == 0`. Write:

```python
manifest_path = Path(args.output_dir) / "run_manifest.json"
```

Use the actual command, exit code, config path, model dir, output dir, loaded JSON preflight reports, and adapter check. If the adapter check fails after a zero training exit, return `1`.

- [ ] **Step 8: Run training-related tests**

Run:

```powershell
python -m pytest tests/test_stage2_training.py tests/test_stage2_model_check.py tests/test_stage2_env_check.py tests/test_stage2_adapter.py -q
```

Expected: pass.

---

## Task 4: Isolate Rule Projection From Real Review Decisions

**Files:**
- Modify: `scripts/run_agent_review.py`
- Modify: `src/small_model_train/agent_review.py`
- Modify: `src/small_model_train/stage4_quality.py`
- Modify: `tests/test_agent_review.py`
- Modify: `tests/test_agent_review_cli.py`
- Modify: `tests/test_stage4_quality.py`

- [ ] **Step 1: Add failing rule-projection CLI test**

Add to `tests/test_agent_review_cli.py`:

```python
def test_rule_projection_backend_never_returns_expansion_ready_decision(tmp_path: Path):
    cards_path, outputs_path, metrics_path = _write_ready_review_inputs(tmp_path)
    reviews_path = tmp_path / "reviews.jsonl"
    votes_path = tmp_path / "votes.jsonl"
    summary_path = tmp_path / "summary.jsonl"
    report_path = tmp_path / "report.md"

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
            "rule_projection",
            "--output",
            str(reviews_path),
            "--votes-output",
            str(votes_path),
            "--summary-output",
            str(summary_path),
            "--report",
            str(report_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    summary = read_jsonl(summary_path)[0]
    assert summary["decision"] == "rules_pass_agent_pending"
    assert summary["review_backend"] == "rule_projection"
    assert summary["agent_gate_pass"] is False
```

If `_write_ready_review_inputs()` does not exist, create it in the test file by writing one valid execution card, one output row, and one metric row with `hard_gate_pass: True`.

- [ ] **Step 2: Run CLI test and verify failure**

Run:

```powershell
python -m pytest tests/test_agent_review_cli.py::test_rule_projection_backend_never_returns_expansion_ready_decision -q
```

Expected: fail because `rule_projection` is not an accepted backend and ready decisions are still possible for generated mock rows.

- [ ] **Step 3: Rename mock generation to rule projection**

In `scripts/run_agent_review.py`:

- rename `_mock_reviews()` to `_rule_projection_reviews()`
- change `parser.add_argument("--backend", choices=["mock"], default="mock")` to:

```python
parser.add_argument("--backend", choices=["rule_projection", "mock"], default="rule_projection")
```

- when `args.backend == "mock"`, set `backend = "rule_projection"` internally so old commands keep working but output is labeled correctly
- add `"review_backend": "rule_projection"` to each generated review row
- change evidence text to `"rule projection derived from Stage 4 hard gate metrics"`

- [ ] **Step 4: Force projection-only summaries to pending**

After `aggregate_agent_reviews(...)` in `run_agent_review.py`, add:

```python
if backend == "rule_projection":
    summary["review_backend"] = "rule_projection"
    summary["projection_only"] = True
    summary["agent_gate_pass"] = False
    if summary["decision"] in {"ready_for_human_spot_check", "ready_for_next_expansion"}:
        summary["decision"] = "rules_pass_agent_pending"
```

Keep imported real review behavior unchanged.

- [ ] **Step 5: Let `stage4_quality` validate projection-only summaries**

Add `review_backend` and `projection_only` as optional fields in `validate_agent_summary()`. Ensure the existing check accepts:

```python
decision == "rules_pass_agent_pending"
agent_gate_pass is False
projection_only is True
```

- [ ] **Step 6: Run review tests**

Run:

```powershell
python -m pytest tests/test_agent_review.py tests/test_agent_review_cli.py tests/test_stage4_quality.py -q
```

Expected: pass.

---

## Task 5: Block Draft Cards From Formal SFT

**Files:**
- Modify: `src/small_model_train/chapter_cards.py`
- Modify: `src/small_model_train/sft_builder.py`
- Modify: `scripts/build_sft_dataset.py`
- Modify: `tests/test_chapter_cards.py`
- Modify: `tests/test_sft_builder.py`

- [ ] **Step 1: Add failing draft-card metadata test**

Add to `tests/test_chapter_cards.py`:

```python
def test_build_draft_chapter_cards_marks_cards_as_draft_only():
    cards = build_draft_chapter_cards([_chapter("train-a", 2800)], count=1, min_chars=2000, max_chars=3000)

    assert cards[0]["draft_only"] is True
    assert cards[0]["approval_status"] == "draft"
    assert cards[0]["style_contract_id"] == "inline-draft-v0"
    assert len(cards[0]["style_contract_sha256"]) == 64
```

- [ ] **Step 2: Add failing formal SFT gate tests**

Add to `tests/test_sft_builder.py`:

```python
def test_build_sft_rows_rejects_draft_cards_in_formal_mode():
    cards = [
        {
            "id": "c1",
            "style_contract": "契约",
            "previous_summary": "前情",
            "chapter_goal": "目标",
            "conflict_beat": "",
            "payoff_beat": "",
            "chapter_structure": [{"step": 1, "name": "承接", "goal": "推进", "estimated_chars": "300"}],
            "character_states": [{"name": "主角", "state": "警惕", "speech_style": "短句"}],
            "must_include": [],
            "must_not_include": [],
            "ending_hook": "门响了。",
            "target_word_count": "2000-2500中文汉字",
            "draft_only": True,
            "approval_status": "draft",
        }
    ]
    chapters = [{"id": "c1", "text": "正文", "split": "train", "quality_tag": "A"}]

    with pytest.raises(ValueError, match="draft card cannot enter formal SFT"):
        build_sft_rows(cards, chapters, require_approved_cards=True)


def test_build_sft_rows_accepts_approved_cards_in_formal_mode():
    card = _valid_card("c1")
    card["draft_only"] = False
    card["approval_status"] = "approved"
    card["style_contract_id"] = "author-main-v1"
    card["style_contract_sha256"] = "a" * 64
    chapters = [{"id": "c1", "text": "正文", "split": "train", "quality_tag": "A"}]

    rows = build_sft_rows([card], chapters, require_approved_cards=True)

    assert len(rows) == 1
```

Use the existing card helper in `tests/test_sft_builder.py`; if it has a different name, extend that helper with the extra fields in this test only.

- [ ] **Step 3: Run tests and verify failure**

Run:

```powershell
python -m pytest tests/test_chapter_cards.py::test_build_draft_chapter_cards_marks_cards_as_draft_only tests/test_sft_builder.py::test_build_sft_rows_rejects_draft_cards_in_formal_mode tests/test_sft_builder.py::test_build_sft_rows_accepts_approved_cards_in_formal_mode -q
```

Expected: fail because draft metadata and formal gate do not exist.

- [ ] **Step 4: Mark generated cards as draft-only**

In `src/small_model_train/chapter_cards.py`, import `hashlib` and add to `_build_card()`:

```python
style_hash = hashlib.sha256(STYLE_CONTRACT.encode("utf-8")).hexdigest()
```

Add these fields to the card:

```python
"draft_only": True,
"approval_status": "draft",
"style_contract_id": "inline-draft-v0",
"style_contract_sha256": style_hash,
```

Do not require these fields in `REQUIRED_CARD_FIELDS` yet, because historical cards in local data should still produce clear SFT errors instead of failing old card validation paths.

- [ ] **Step 5: Add formal gate to SFT builder**

In `src/small_model_train/sft_builder.py`, add:

```python
APPROVED_CARD_STATUSES = {"approved", "frozen"}


def _validate_formal_card(card: dict) -> None:
    if card.get("draft_only") is True:
        raise ValueError(f"draft card cannot enter formal SFT: {card.get('id', '')}")
    if card.get("approval_status") not in APPROVED_CARD_STATUSES:
        raise ValueError(f"card approval_status must be approved or frozen: {card.get('id', '')}")
    if not card.get("style_contract_id"):
        raise ValueError(f"style_contract_id is required for formal SFT: {card.get('id', '')}")
    style_hash = card.get("style_contract_sha256")
    if not isinstance(style_hash, str) or len(style_hash) != 64:
        raise ValueError(f"style_contract_sha256 must be a SHA-256 hex digest: {card.get('id', '')}")
```

Change `build_sft_rows()`:

```python
def build_sft_rows(
    cards: list[dict],
    chapters: list[dict],
    require_approved_cards: bool = False,
) -> list[dict]:
```

Inside the card loop:

```python
if require_approved_cards:
    _validate_formal_card(card)
```

- [ ] **Step 6: Make formal SFT the CLI default**

In `scripts/build_sft_dataset.py`, add:

```python
parser.add_argument("--allow-draft-cards", action="store_true")
```

Call:

```python
rows = build_sft_rows(
    read_jsonl(args.cards),
    read_jsonl(args.chapters),
    require_approved_cards=not args.allow_draft_cards,
)
```

Update smoke commands in docs to pass `--allow-draft-cards`. Formal training docs must omit that flag.

- [ ] **Step 7: Run card and SFT tests**

Run:

```powershell
python -m pytest tests/test_chapter_cards.py tests/test_sft_builder.py -q
```

Expected: pass after updating existing CLI tests to pass `--allow-draft-cards` where they use generated draft cards.

---

## Task 6: Document Stage 5A Operating Sequence

**Files:**
- Create: `docs/stage5a-evidence-chain-hardening.zh.md`
- Modify: `README.md`
- Modify: `docs/index.zh.md`

- [ ] **Step 1: Write the Stage 5A runbook**

Create `docs/stage5a-evidence-chain-hardening.zh.md`:

````markdown
# Stage 5A 证据链修正指南

Stage 5A 的目标是证明训练、推理、评分和审阅结果可复现、可追溯、不可被清洗器或 mock review 掩盖。

## 入口命令

```powershell
python scripts/check_local_model.py --model-dir E:\models\Qwen3-4B-Instruct-2507 --report reports/model_check_report.md --json-output reports/model_check_report.json
python scripts/check_training_env.py --report reports/training_env_report.md --json-output reports/training_env_report.json
python scripts/build_sft_dataset.py --cards data_cards/chapter_cards.jsonl --chapters data_clean/chapters_split.jsonl --output data_sft/sft_chapter_v1.jsonl --dataset-info-output data_sft/dataset_info.json --allow-draft-cards
python scripts/run_sft_smoke.py --config configs/sft_qlora_qwen3_4b_smoke_6144.yaml --eval-cards data_cards/eval_cards_50.jsonl
python scripts/run_eval_inference.py --cards data_cards/eval_execution_cards_50.jsonl --adapter-dir outputs/sft_smoke --output outputs/sft_smoke/generated_raw.jsonl --model-name sft_smoke --max-new-tokens 1024 --seed 20260623
python scripts/score_outputs.py --cards data_cards/eval_execution_cards_50.jsonl --outputs outputs/sft_smoke/generated_raw.jsonl --output outputs/sft_smoke/metrics_raw.jsonl
python scripts/run_agent_review.py --cards data_cards/eval_execution_cards_50.jsonl --outputs outputs/sft_smoke/generated_raw.jsonl --metrics outputs/sft_smoke/metrics_raw.jsonl --target-platform hybrid_fanqie_qidian --backend rule_projection --output outputs/sft_smoke/reviews_projection.jsonl --votes-output outputs/sft_smoke/votes_projection.jsonl --summary-output outputs/sft_smoke/review_projection_summary.jsonl --report reports/stage5a_rule_projection_report.md
```

## 成功标志

- JSON preflight 的 `passed` 必须为 `true`。
- 生成 JSONL 必须包含 `raw_output`、`sanitized_output`、`prompt_sha256`、`generated_tokens`、`seed`。
- `score_outputs.py` 默认使用 `raw_output`。
- `rule_projection` 只能产生 `rules_pass_agent_pending`，不能产生扩量或发布结论。
- 正式 SFT 不允许 `draft_only: true` 的章节卡进入数据集。
- `outputs/<run>/run_manifest.json` 能追溯模型、环境、配置、adapter 检查和训练退出码。
````

- [ ] **Step 2: Update README links**

In `README.md`, add Stage 5A to the “现有阶段指南” list:

```markdown
- [Stage 5A 证据链修正指南](docs/stage5a-evidence-chain-hardening.zh.md)
```

Update preflight examples to include `--json-output`.

- [ ] **Step 3: Update docs index**

In `docs/index.zh.md`, add the same link under the stage guide section.

- [ ] **Step 4: Check docs references**

Run:

```powershell
rg -n "json-output|allow-draft-cards|rule_projection|raw_output|stage5a" README.md docs scripts src tests
```

Expected: Stage 5A docs and tests mention the new flags and raw evidence fields.

---

## Task 7: Verification

**Files:**
- No additional files.

- [ ] **Step 1: Run focused test groups**

Run:

```powershell
python -m pytest tests/test_prompt_renderer.py tests/test_sft_builder.py tests/test_stage2_inference.py tests/test_scoring.py tests/test_stage2_training.py tests/test_agent_review.py tests/test_agent_review_cli.py tests/test_stage4_quality.py -q
```

Expected: pass.

- [ ] **Step 2: Run the full suite**

Run:

```powershell
python -m pytest -q
```

Expected: pass.

- [ ] **Step 3: Run dry-run commands without GPU**

Run:

```powershell
python scripts/check_local_model.py --model-dir E:\models\Qwen3-4B-Instruct-2507 --report reports/model_check_report.md --json-output reports/model_check_report.json --skip-transformers-load
python scripts/run_eval_inference.py --cards data_cards/eval_execution_cards_50.jsonl --output outputs/dry/generated_raw.jsonl --model-name dry_eval --dry-run --seed 20260623
python scripts/score_outputs.py --cards data_cards/eval_execution_cards_50.jsonl --outputs outputs/dry/generated_raw.jsonl --output outputs/dry/metrics_raw.jsonl
```

Expected:

- model check writes Markdown and JSON reports
- dry eval writes generation rows with raw/sanitized fields
- scoring reads the dry generation file without schema errors

- [ ] **Step 4: Inspect evidence fields**

Run:

```powershell
@'
import json
from pathlib import Path
path = Path("outputs/dry/generated_raw.jsonl")
row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
required = {"raw_output", "sanitized_output", "params", "prompt_sha256", "generated_tokens", "sanitizer_events"}
missing = sorted(required - set(row))
print({"missing": missing, "id": row.get("id"), "seed": row.get("params", {}).get("seed")})
raise SystemExit(1 if missing else 0)
'@ | python -
```

Expected: `missing` is `[]` and `seed` is `20260623`.

---

## Stage 5A Exit Criteria

Stage 5A is complete only when all of these are true:

- Full pytest suite passes.
- Training and inference prompt construction share the same renderer.
- Eval worker writes raw model output before any sanitizer changes.
- Scoring uses `raw_output` by default.
- Sanitizer changes are recorded as events and cannot hide model failures from scoring.
- `run_sft_train.py` reads JSON preflight reports and checks `passed == true`.
- Successful training writes `run_manifest.json`.
- A zero-exit training run with an invalid adapter is treated as failed.
- Rule projection is labeled as projection and cannot produce `ready_for_next_expansion`.
- Formal SFT rejects draft-only cards unless the caller explicitly chooses the smoke-only `--allow-draft-cards` path.

---

## Follow-On Plans After Stage 5A

- Stage 5B: StyleContract closure with approved JSON/Markdown assets, richer style metrics, contract hash binding, adapter manifest binding, and author review tables.
- Stage 5C: Formal ChapterExecutionCard compiler with hard constraints, free-space fields, approval lifecycle, leakage checks, group split, sealed test, and near-duplicate checks.
- Stage 5D: AI-taste defect taxonomy, same-plot author revisions, rejection sampling SFT, local rewrite records, and small preference optimization experiments.

---

## Self-Review

- Spec coverage: this plan maps the uploaded P0 recommendations to executable tasks for prompt consistency, raw evidence, structured preflight, adapter evidence, mock review isolation, and draft-card blocking.
- Placeholder scan: no task depends on unfilled placeholders; each task names concrete files, commands, expected results, and representative code.
- Type consistency: generation rows use `raw_output`, `sanitized_output`, `output`, `params`, `prompt_sha256`, `generated_tokens`, and `sanitizer_events`; review summaries use `decision`, `agent_gate_pass`, `review_backend`, and `projection_only`; formal cards use `draft_only`, `approval_status`, `style_contract_id`, and `style_contract_sha256`.
