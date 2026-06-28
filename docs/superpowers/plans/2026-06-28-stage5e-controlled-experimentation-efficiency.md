# Stage 5E Controlled Experimentation And Efficiency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Stage 5E controlled experimentation scaffolding so model/data/PEFT/generation changes can be compared one primary variable at a time with paired evidence and reproducible manifests.

**Architecture:** Stage 5E starts from the Stage 5D entry gate, not from training code. The first implementation creates experiment manifests, paired-eval summaries, dry-run experiment-matrix commands, and documentation; actual expensive training remains gated by explicit manifests and later operator execution. Preference-optimization methods stay listed as future experiment categories and are not run by this plan.

**Tech Stack:** Python 3.10+, pytest, JSON/JSONL artifacts, existing `small_model_train` modules, CLI scripts under `scripts/`.

---

## Scope Check

This plan begins Stage 5E after the local Stage 5D gate passed:

```powershell
python scripts/check_stage5e_entry.py --summary reports/stage5d_review_summary.json --review-records data_review/stage5d_review_records.jsonl --revisions data_review/stage5d_revisions.jsonl --rejection-sampling-rows data_sft/stage5d_rejection_sampling_sft.jsonl --preference-rows data_pref/stage5d_same_plot_preference.jsonl --generation-records outputs/stage5d_generation_records.jsonl --output reports/stage5e_entry_check.json
```

Required entry evidence before executing this plan:

- `reports/stage5e_entry_check.json` contains `"passed": true`.
- Full `python -m pytest -q` passes.
- Stage 5D local artifacts remain candidate data only and do not claim model-quality improvement.

This plan does not run DPO, SimPO, ORPO, KTO, reward-model training, or large experiment batches. It builds the control plane and reports needed before those techniques can be evaluated safely.

---

## File Map

- Create: `src/small_model_train/evaluation/__init__.py`
  - Package marker for Stage 5E evaluation helpers.
- Create: `src/small_model_train/evaluation/experiment_manifest.py`
  - Build and validate Stage 5E experiment manifests, entry-gate references, artifact hashes, and one-primary-variable constraints.
- Create: `src/small_model_train/evaluation/paired_eval.py`
  - Pair baseline/candidate outputs by id, combine deterministic metrics with optional review judgments, and render JSON/Markdown summaries.
- Create: `scripts/build_stage5e_experiment_manifest.py`
  - CLI for writing the initial Stage 5E experiment manifest.
- Create: `scripts/build_paired_eval_report.py`
  - CLI for writing paired-eval summary JSON and Markdown report.
- Create: `scripts/run_experiment_matrix.py`
  - CLI that validates a manifest and writes dry-run training/eval commands for a controlled experiment matrix.
- Create: `tests/test_stage5e_experiment_manifest.py`
  - Manifest schema, gate, hash, and one-variable tests.
- Create: `tests/test_paired_eval.py`
  - Pairing, winner, regression, and report tests.
- Create: `tests/test_run_experiment_matrix.py`
  - Dry-run command generation tests.
- Create: `docs/stage5e-controlled-experimentation-efficiency.zh.md`
  - Operator runbook for Stage 5E.
- Modify: `README.md`
  - Add the Stage 5E runbook link.
- Modify: `docs/index.zh.md`
  - Add the Stage 5E runbook link.
- Modify: `docs/pipeline-flow.zh.md`
  - Add the Stage 5E flow after Stage 5D.
- Modify: `docs/superpowers/plans/2026-06-23-small-model-train-full-roadmap.md`
  - Replace the future plan pointer with this plan file after the manifest/report scaffolding exists.

---

## Task 1: Stage 5E Experiment Manifest

**Files:**
- Create: `src/small_model_train/evaluation/__init__.py`
- Create: `src/small_model_train/evaluation/experiment_manifest.py`
- Create: `scripts/build_stage5e_experiment_manifest.py`
- Create: `tests/test_stage5e_experiment_manifest.py`

- [ ] **Step 1: Add failing manifest tests**

Create `tests/test_stage5e_experiment_manifest.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _gate(path: Path, *, passed: bool = True) -> Path:
    _write_json(
        path,
        {
            "passed": passed,
            "errors": [] if passed else ["blocked"],
            "entry": "stage5e_controlled_experimentation",
        },
    )
    return path


def test_build_experiment_manifest_records_hashes_and_one_primary_variable(tmp_path: Path):
    from small_model_train.evaluation.experiment_manifest import build_experiment_manifest

    gate = _gate(tmp_path / "reports" / "stage5e_entry_check.json")
    dataset = tmp_path / "data_sft" / "stage5d_rejection_sampling_sft.jsonl"
    cards = tmp_path / "data_cards" / "approved.jsonl"
    config = tmp_path / "configs" / "sft.yaml"
    for path, text in (
        (dataset, "{\"id\":\"row-1\"}\n"),
        (cards, "{\"card_id\":\"card-1\"}\n"),
        (config, "learning_rate: 0.0002\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    manifest = build_experiment_manifest(
        experiment_id="stage5e-lr-0002-vs-0001",
        baseline_run_id="stage5d-current-sft",
        candidate_run_id="stage5e-lr-0002",
        primary_variable={
            "name": "learning_rate",
            "baseline_value": "0.0001",
            "candidate_value": "0.0002",
        },
        stage5e_entry_check=gate,
        artifact_paths={
            "sft_dataset": dataset,
            "eval_cards": cards,
            "config": config,
        },
        paired_eval={
            "cards": "data_cards/eval_execution_cards_50.jsonl",
            "baseline_generated": "outputs/stage5e/baseline/generated.jsonl",
            "candidate_generated": "outputs/stage5e/candidate/generated.jsonl",
        },
    )

    assert manifest["schema_version"] == 1
    assert manifest["stage5e_entry"]["passed"] is True
    assert manifest["primary_variable"]["name"] == "learning_rate"
    assert set(manifest["artifacts"]) == {"config", "eval_cards", "sft_dataset"}
    assert all(len(row["sha256"]) == 64 for row in manifest["artifacts"].values())


def test_validate_experiment_manifest_rejects_failed_gate(tmp_path: Path):
    from small_model_train.evaluation.experiment_manifest import build_experiment_manifest

    gate = _gate(tmp_path / "gate.json", passed=False)
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Stage 5E entry gate must pass"):
        build_experiment_manifest(
            experiment_id="blocked",
            baseline_run_id="base",
            candidate_run_id="candidate",
            primary_variable={
                "name": "learning_rate",
                "baseline_value": "0.0001",
                "candidate_value": "0.0002",
            },
            stage5e_entry_check=gate,
            artifact_paths={"sft_dataset": dataset},
            paired_eval={"cards": "cards.jsonl"},
        )


def test_validate_experiment_manifest_rejects_multi_variable_change(tmp_path: Path):
    from small_model_train.evaluation.experiment_manifest import validate_experiment_manifest

    manifest = {
        "schema_version": 1,
        "experiment_id": "bad",
        "baseline_run_id": "base",
        "candidate_run_id": "candidate",
        "primary_variable": {"name": "learning_rate", "baseline_value": "1", "candidate_value": "2"},
        "controlled_variables": [{"name": "rank", "baseline_value": "8", "candidate_value": "16"}],
        "stage5e_entry": {"path": "reports/stage5e_entry_check.json", "passed": True},
        "artifacts": {},
        "paired_eval": {"cards": "cards.jsonl"},
        "boundary": "controlled_experiment_one_primary_variable",
    }

    with pytest.raises(ValueError, match="controlled variable changed"):
        validate_experiment_manifest(manifest)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
python -m pytest tests/test_stage5e_experiment_manifest.py -q
```

Expected: fails because `small_model_train.evaluation.experiment_manifest` does not exist.

- [ ] **Step 3: Create evaluation package marker**

Create `src/small_model_train/evaluation/__init__.py`:

```python
"""Evaluation helpers for controlled Stage 5E experiments."""
```

- [ ] **Step 4: Implement experiment manifest helpers**

Create `src/small_model_train/evaluation/experiment_manifest.py`:

```python
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
BOUNDARY = "controlled_experiment_one_primary_variable"


def build_experiment_manifest(
    *,
    experiment_id: str,
    baseline_run_id: str,
    candidate_run_id: str,
    primary_variable: dict[str, Any],
    stage5e_entry_check: str | Path,
    artifact_paths: dict[str, str | Path],
    paired_eval: dict[str, Any],
    controlled_variables: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    gate_path = Path(stage5e_entry_check)
    gate = _read_json(gate_path)
    if gate.get("passed") is not True:
        raise ValueError("Stage 5E entry gate must pass")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now_iso(),
        "experiment_id": _required_string(experiment_id, "experiment_id"),
        "baseline_run_id": _required_string(baseline_run_id, "baseline_run_id"),
        "candidate_run_id": _required_string(candidate_run_id, "candidate_run_id"),
        "primary_variable": _validated_variable(primary_variable, require_change=True),
        "controlled_variables": [
            _validated_variable(variable, require_change=False)
            for variable in (controlled_variables or [])
        ],
        "stage5e_entry": {
            "path": str(gate_path),
            "passed": True,
            "entry": gate.get("entry", ""),
        },
        "artifacts": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in sorted(artifact_paths.items())
        },
        "paired_eval": dict(paired_eval),
        "boundary": BOUNDARY,
    }
    return validate_experiment_manifest(manifest)


def validate_experiment_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("experiment manifest must be a JSON object")
    for field in (
        "schema_version",
        "experiment_id",
        "baseline_run_id",
        "candidate_run_id",
        "primary_variable",
        "controlled_variables",
        "stage5e_entry",
        "artifacts",
        "paired_eval",
        "boundary",
    ):
        if field not in manifest:
            raise ValueError(f"{field} is required")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if manifest["boundary"] != BOUNDARY:
        raise ValueError(f"boundary must be {BOUNDARY}")
    _validated_variable(manifest["primary_variable"], require_change=True)
    for variable in manifest["controlled_variables"]:
        _validated_variable(variable, require_change=False)
        if variable["baseline_value"] != variable["candidate_value"]:
            raise ValueError(f"controlled variable changed: {variable['name']}")
    if manifest["stage5e_entry"].get("passed") is not True:
        raise ValueError("Stage 5E entry gate must pass")
    if not isinstance(manifest["artifacts"], dict):
        raise ValueError("artifacts must be an object")
    for name, artifact in manifest["artifacts"].items():
        if not isinstance(name, str) or not name:
            raise ValueError("artifact name must be a non-empty string")
        if not isinstance(artifact, dict):
            raise ValueError(f"artifact must be an object: {name}")
        _required_string(artifact.get("path"), f"artifact path {name}")
        _lower_hex_sha256(artifact.get("sha256"), f"artifact sha256 {name}")
    if not isinstance(manifest["paired_eval"], dict):
        raise ValueError("paired_eval must be an object")
    return manifest


def write_experiment_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    validated = validate_experiment_manifest(manifest)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(validated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_sha256(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.is_file():
        raise ValueError(f"artifact file not found: {file_path}")
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON file is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def _validated_variable(variable: dict[str, Any], *, require_change: bool) -> dict[str, str]:
    if not isinstance(variable, dict):
        raise ValueError("variable must be an object")
    name = _required_string(variable.get("name"), "variable name")
    baseline_value = _required_string(variable.get("baseline_value"), f"{name} baseline_value")
    candidate_value = _required_string(variable.get("candidate_value"), f"{name} candidate_value")
    if require_change and baseline_value == candidate_value:
        raise ValueError(f"primary variable did not change: {name}")
    return {
        "name": name,
        "baseline_value": baseline_value,
        "candidate_value": candidate_value,
    }


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _lower_hex_sha256(value: Any, field: str) -> str:
    text = _required_string(value, field)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{field} must be a 64-character lowercase hex string")
    return text


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
```

- [ ] **Step 5: Add the manifest CLI**

Create `scripts/build_stage5e_experiment_manifest.py`:

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

from small_model_train.evaluation.experiment_manifest import (
    build_experiment_manifest,
    write_experiment_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Stage 5E controlled experiment manifest.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--baseline-run-id", required=True)
    parser.add_argument("--candidate-run-id", required=True)
    parser.add_argument("--primary-variable-name", required=True)
    parser.add_argument("--primary-baseline-value", required=True)
    parser.add_argument("--primary-candidate-value", required=True)
    parser.add_argument("--stage5e-entry-check", required=True)
    parser.add_argument("--artifact", action="append", default=[], help="name=path")
    parser.add_argument("--paired-eval-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        manifest = build_experiment_manifest(
            experiment_id=args.experiment_id,
            baseline_run_id=args.baseline_run_id,
            candidate_run_id=args.candidate_run_id,
            primary_variable={
                "name": args.primary_variable_name,
                "baseline_value": args.primary_baseline_value,
                "candidate_value": args.primary_candidate_value,
            },
            stage5e_entry_check=args.stage5e_entry_check,
            artifact_paths=_artifact_paths(args.artifact),
            paired_eval=json.loads(args.paired_eval_json),
        )
        write_experiment_manifest(args.output, manifest)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote Stage 5E experiment manifest to {args.output}")
    return 0


def _artifact_paths(items: list[str]) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"artifact must be name=path: {item}")
        name, path = item.split("=", 1)
        if not name.strip() or not path.strip():
            raise ValueError(f"artifact must be name=path: {item}")
        if name in artifacts:
            raise ValueError(f"duplicate artifact name: {name}")
        artifacts[name] = path
    return artifacts


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Run manifest tests**

Run:

```powershell
python -m pytest tests/test_stage5e_experiment_manifest.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 1**

```powershell
git add src/small_model_train/evaluation/__init__.py src/small_model_train/evaluation/experiment_manifest.py scripts/build_stage5e_experiment_manifest.py tests/test_stage5e_experiment_manifest.py
git commit -m "feat: add stage5e experiment manifests"
```

---

## Task 2: Paired Evaluation Core And Report

**Files:**
- Create: `src/small_model_train/evaluation/paired_eval.py`
- Create: `scripts/build_paired_eval_report.py`
- Create: `tests/test_paired_eval.py`

- [ ] **Step 1: Add failing paired-eval tests**

Create `tests/test_paired_eval.py`:

```python
from __future__ import annotations


def test_summarize_paired_eval_uses_review_judgment_over_metric_tie():
    from small_model_train.evaluation.paired_eval import summarize_paired_eval

    summary = summarize_paired_eval(
        baseline_metrics=[{"id": "card-1", "hard_gate_pass": True, "failure_types": []}],
        candidate_metrics=[{"id": "card-1", "hard_gate_pass": True, "failure_types": []}],
        judgments=[
            {
                "id": "card-1",
                "winner": "candidate",
                "reviewer": "author",
                "reason": "同剧情下候选文本节奏更贴近作者。",
            }
        ],
    )

    assert summary["paired_rows"] == 1
    assert summary["wins"] == 1
    assert summary["losses"] == 0
    assert summary["ties"] == 0
    assert summary["comparisons"][0]["winner"] == "candidate"
    assert summary["comparisons"][0]["winner_source"] == "review"


def test_summarize_paired_eval_flags_candidate_regression_without_review():
    from small_model_train.evaluation.paired_eval import summarize_paired_eval

    summary = summarize_paired_eval(
        baseline_metrics=[{"id": "card-1", "hard_gate_pass": True, "failure_types": []}],
        candidate_metrics=[
            {"id": "card-1", "hard_gate_pass": False, "failure_types": ["outline_leak"]}
        ],
        judgments=[],
    )

    assert summary["wins"] == 0
    assert summary["losses"] == 1
    assert summary["regression_ids"] == ["card-1"]
    assert summary["comparisons"][0]["winner"] == "baseline"


def test_render_paired_eval_report_keeps_counts_and_boundary():
    from small_model_train.evaluation.paired_eval import render_paired_eval_report

    report = render_paired_eval_report(
        {
            "paired_rows": 1,
            "wins": 1,
            "losses": 0,
            "ties": 0,
            "regression_ids": [],
            "comparisons": [{"id": "card-1", "winner": "candidate", "winner_source": "review"}],
            "boundary": "paired_eval_no_training",
        }
    )

    assert "# Stage 5E Paired Eval Report" in report
    assert "- Candidate wins: 1" in report
    assert "paired_eval_no_training" in report
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
python -m pytest tests/test_paired_eval.py -q
```

Expected: fails because `small_model_train.evaluation.paired_eval` does not exist.

- [ ] **Step 3: Implement paired-eval helpers**

Create `src/small_model_train/evaluation/paired_eval.py`:

```python
from __future__ import annotations

import json
from typing import Any


BOUNDARY = "paired_eval_no_training"
WINNERS = {"baseline", "candidate", "tie"}


def summarize_paired_eval(
    *,
    baseline_metrics: list[dict[str, Any]],
    candidate_metrics: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_by_id = _rows_by_id(baseline_metrics, "baseline metrics")
    candidate_by_id = _rows_by_id(candidate_metrics, "candidate metrics")
    judgments_by_id = _judgments_by_id(judgments)
    paired_ids = sorted(set(baseline_by_id) & set(candidate_by_id))
    comparisons = [
        _comparison(sample_id, baseline_by_id[sample_id], candidate_by_id[sample_id], judgments_by_id)
        for sample_id in paired_ids
    ]
    return {
        "paired_rows": len(comparisons),
        "wins": sum(1 for row in comparisons if row["winner"] == "candidate"),
        "losses": sum(1 for row in comparisons if row["winner"] == "baseline"),
        "ties": sum(1 for row in comparisons if row["winner"] == "tie"),
        "missing_baseline_ids": sorted(set(candidate_by_id) - set(baseline_by_id)),
        "missing_candidate_ids": sorted(set(baseline_by_id) - set(candidate_by_id)),
        "regression_ids": [
            row["id"]
            for row in comparisons
            if row["winner"] == "baseline" and row["candidate_score"] < row["baseline_score"]
        ],
        "comparisons": comparisons,
        "boundary": BOUNDARY,
    }


def render_paired_eval_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Stage 5E Paired Eval Report",
        "",
        "## Summary",
        "",
        f"- Paired rows: {summary.get('paired_rows', 0)}",
        f"- Candidate wins: {summary.get('wins', 0)}",
        f"- Candidate losses: {summary.get('losses', 0)}",
        f"- Ties: {summary.get('ties', 0)}",
        f"- Boundary: {summary.get('boundary', BOUNDARY)}",
        "",
        "## Regressions",
        "",
    ]
    regression_ids = summary.get("regression_ids", [])
    if regression_ids:
        lines.extend(f"- {sample_id}" for sample_id in regression_ids)
    else:
        lines.append("- None")
    lines.extend(["", "## Comparisons", ""])
    for row in summary.get("comparisons", []):
        lines.append(
            f"- {row['id']}: {row['winner']} ({row['winner_source']}) "
            f"baseline={row['baseline_score']} candidate={row['candidate_score']}"
        )
    if not summary.get("comparisons"):
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def write_paired_eval_summary(path: str, summary: dict[str, Any]) -> None:
    from pathlib import Path

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _comparison(
    sample_id: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    judgments_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    baseline_score = _metric_score(baseline)
    candidate_score = _metric_score(candidate)
    judgment = judgments_by_id.get(sample_id)
    if judgment is not None:
        winner = judgment["winner"]
        winner_source = "review"
    elif candidate_score > baseline_score:
        winner = "candidate"
        winner_source = "metrics"
    elif candidate_score < baseline_score:
        winner = "baseline"
        winner_source = "metrics"
    else:
        winner = "tie"
        winner_source = "metrics"
    return {
        "id": sample_id,
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "baseline_failures": list(baseline.get("failure_types", [])),
        "candidate_failures": list(candidate.get("failure_types", [])),
        "winner": winner,
        "winner_source": winner_source,
    }


def _metric_score(row: dict[str, Any]) -> int:
    hard_gate = 1 if row.get("hard_gate_pass") is True else 0
    failures = row.get("failure_types", [])
    failure_count = len(failures) if isinstance(failures, list) else 1
    return hard_gate * 10 - failure_count


def _rows_by_id(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{label} row {index} must be an object")
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"{label} row {index} missing id")
        if sample_id in by_id:
            raise ValueError(f"{label} duplicate id: {sample_id}")
        by_id[sample_id] = row
    return by_id


def _judgments_by_id(judgments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(judgments, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"judgment row {index} must be an object")
        sample_id = row.get("id")
        winner = row.get("winner")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"judgment row {index} missing id")
        if winner not in WINNERS:
            raise ValueError(f"judgment row {index} winner must be baseline, candidate, or tie")
        if sample_id in by_id:
            raise ValueError(f"duplicate judgment id: {sample_id}")
        by_id[sample_id] = row
    return by_id
```

- [ ] **Step 4: Add paired-eval CLI**

Create `scripts/build_paired_eval_report.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.evaluation.paired_eval import (
    render_paired_eval_report,
    summarize_paired_eval,
    write_paired_eval_summary,
)
from small_model_train.io_utils import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Stage 5E paired-eval report.")
    parser.add_argument("--baseline-metrics", required=True)
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--judgments", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--report-output", required=True)
    args = parser.parse_args()

    try:
        summary = summarize_paired_eval(
            baseline_metrics=read_jsonl(args.baseline_metrics),
            candidate_metrics=read_jsonl(args.candidate_metrics),
            judgments=read_jsonl(args.judgments),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    write_paired_eval_summary(args.summary_output, summary)
    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_paired_eval_report(summary), encoding="utf-8")
    print(f"wrote Stage 5E paired eval report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run paired-eval tests**

Run:

```powershell
python -m pytest tests/test_paired_eval.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 2**

```powershell
git add src/small_model_train/evaluation/paired_eval.py scripts/build_paired_eval_report.py tests/test_paired_eval.py
git commit -m "feat: add stage5e paired eval report"
```

---

## Task 3: Experiment Matrix Dry-Run Runner

**Files:**
- Create: `scripts/run_experiment_matrix.py`
- Create: `tests/test_run_experiment_matrix.py`

- [ ] **Step 1: Add failing matrix tests**

Create `tests/test_run_experiment_matrix.py`:

```python
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_experiment_matrix.py"


def test_run_experiment_matrix_writes_dry_run_commands(tmp_path: Path):
    manifest = {
        "schema_version": 1,
        "experiment_id": "stage5e-lr",
        "baseline_run_id": "baseline",
        "candidate_run_id": "candidate",
        "primary_variable": {
            "name": "learning_rate",
            "baseline_value": "0.0001",
            "candidate_value": "0.0002",
        },
        "controlled_variables": [],
        "stage5e_entry": {"path": "reports/stage5e_entry_check.json", "passed": True},
        "artifacts": {"config": {"path": "configs/sft.yaml", "sha256": "a" * 64}},
        "paired_eval": {"cards": "data_cards/eval_execution_cards_50.jsonl"},
        "boundary": "controlled_experiment_one_primary_variable",
    }
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "commands.jsonl"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "experiment_id": "stage5e-lr",
            "run_id": "candidate",
            "dry_run": True,
            "primary_variable": "learning_rate",
            "command": [
                "python",
                "scripts/run_sft_train.py",
                "--config",
                "configs/sft.yaml",
                "--run-name",
                "candidate",
            ],
        }
    ]
```

- [ ] **Step 2: Run test and verify failure**

Run:

```powershell
python -m pytest tests/test_run_experiment_matrix.py -q
```

Expected: fails because `scripts/run_experiment_matrix.py` does not exist.

- [ ] **Step 3: Implement dry-run matrix script**

Create `scripts/run_experiment_matrix.py`:

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

from small_model_train.evaluation.experiment_manifest import validate_experiment_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Stage 5E experiment matrix commands.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        manifest = validate_experiment_manifest(
            json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        )
        rows = [_command_row(manifest, dry_run=args.dry_run)]
        _write_jsonl(args.output, rows)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {len(rows)} Stage 5E experiment commands to {args.output}")
    return 0


def _command_row(manifest: dict, *, dry_run: bool) -> dict:
    config_path = manifest["artifacts"].get("config", {}).get("path", "")
    if not config_path:
        raise ValueError("manifest artifact config is required")
    return {
        "experiment_id": manifest["experiment_id"],
        "run_id": manifest["candidate_run_id"],
        "dry_run": bool(dry_run),
        "primary_variable": manifest["primary_variable"]["name"],
        "command": [
            "python",
            "scripts/run_sft_train.py",
            "--config",
            config_path,
            "--run-name",
            manifest["candidate_run_id"],
        ],
    }


def _write_jsonl(path: str, rows: list[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run matrix tests**

Run:

```powershell
python -m pytest tests/test_run_experiment_matrix.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 3**

```powershell
git add scripts/run_experiment_matrix.py tests/test_run_experiment_matrix.py
git commit -m "feat: add stage5e experiment matrix dry run"
```

---

## Task 4: Stage 5E Documentation And Roadmap Handoff

**Files:**
- Create: `docs/stage5e-controlled-experimentation-efficiency.zh.md`
- Modify: `README.md`
- Modify: `docs/index.zh.md`
- Modify: `docs/pipeline-flow.zh.md`
- Modify: `docs/superpowers/plans/2026-06-23-small-model-train-full-roadmap.md`

- [ ] **Step 1: Create the Stage 5E runbook**

Create `docs/stage5e-controlled-experimentation-efficiency.zh.md`:

```markdown
# Stage 5E 受控实验与效率指南

Stage 5E 只在 Stage 5D 入场检查通过且完整 `python -m pytest` 通过后开始。它的第一批产物是实验 manifest、paired eval 报告和 dry-run 实验矩阵，不自动运行 DPO、SimPO、ORPO、KTO 或 reward model training。

## 入场检查

```powershell
python scripts/check_stage5e_entry.py --summary reports/stage5d_review_summary.json --review-records data_review/stage5d_review_records.jsonl --revisions data_review/stage5d_revisions.jsonl --rejection-sampling-rows data_sft/stage5d_rejection_sampling_sft.jsonl --preference-rows data_pref/stage5d_same_plot_preference.jsonl --generation-records outputs/stage5d_generation_records.jsonl --output reports/stage5e_entry_check.json
python -m pytest -q
```

只有 `reports/stage5e_entry_check.json` 中 `"passed": true`，并且完整测试通过，才可以生成 Stage 5E manifest。

## 实验 manifest

```powershell
python scripts/build_stage5e_experiment_manifest.py --experiment-id stage5e-lr-0002-vs-0001 --baseline-run-id stage5d-current-sft --candidate-run-id stage5e-lr-0002 --primary-variable-name learning_rate --primary-baseline-value 0.0001 --primary-candidate-value 0.0002 --stage5e-entry-check reports/stage5e_entry_check.json --artifact config=configs/sft_qlora_qwen3_4b_smoke_6144.yaml --artifact sft_dataset=data_sft/stage5d_rejection_sampling_sft.jsonl --artifact eval_cards=data_cards/eval_execution_cards_50.jsonl --paired-eval-json "{\"cards\":\"data_cards/eval_execution_cards_50.jsonl\"}" --output reports/stage5e_experiment_manifest.json
```

Manifest 必须只改变一个 primary variable。其他变量如果列入 controlled variables，baseline 和 candidate 值必须一致。

## Dry-Run 实验矩阵

```powershell
python scripts/run_experiment_matrix.py --manifest reports/stage5e_experiment_manifest.json --output reports/stage5e_experiment_commands.jsonl --dry-run
```

Dry-run 输出命令清单，不启动训练。真正训练必须由人工确认 manifest、预算和评测卡后再执行。

## Paired Eval 报告

```powershell
python scripts/build_paired_eval_report.py --baseline-metrics outputs/stage5e/baseline/metrics.jsonl --candidate-metrics outputs/stage5e/candidate/metrics.jsonl --judgments data_review/stage5e_paired_judgments.jsonl --summary-output reports/stage5e_paired_eval_summary.json --report-output reports/stage5e_paired_eval_report.md
```

报告只说明 paired comparison 的 win/loss/tie 和 regression samples；它不替代作者最终审阅。

## 边界

- 每次实验只改变一个 primary variable。
- 不用 sanitized-only artifact 推进实验。
- 不把 rule projection 当成人审。
- 不把 Stage 5D preference rows 当作已经训练过的偏好优化结果。
- 不因效率提升忽略剧情执行、作者接受和回归样本。
```

- [ ] **Step 2: Update README and docs index**

In `README.md`, add the Stage 5E guide next to Stage 5D:

```markdown
- [Stage 5E 受控实验与效率指南](docs/stage5e-controlled-experimentation-efficiency.zh.md)
```

In `docs/index.zh.md`, add:

```markdown
- [Stage 5E 受控实验与效率指南](stage5e-controlled-experimentation-efficiency.zh.md)：解释 Stage 5E manifest、paired eval、dry-run 实验矩阵和单变量实验边界。
```

- [ ] **Step 3: Update pipeline flow**

In `docs/pipeline-flow.zh.md`, add after the Stage 5D candidate section:

```markdown
### Stage 5E 控制面：受控实验与效率

Stage 5E 在 `scripts/check_stage5e_entry.py` 通过且完整 `python -m pytest` 通过后开始。第一步生成 `reports/stage5e_experiment_manifest.json`，记录 primary variable、controlled variables、dataset/config/eval artifact hashes 和 Stage 5D gate 证据；第二步用 `scripts/run_experiment_matrix.py --dry-run` 写出候选实验命令；第三步用 `scripts/build_paired_eval_report.py` 对 baseline/candidate 指标和人审判断做 paired comparison。

注意：Stage 5E 控制面不自动运行 preference optimization，不把 efficiency win 当作 prose win，也不允许一次改变多个 primary variables。
```

- [ ] **Step 4: Update roadmap pointer**

In `docs/superpowers/plans/2026-06-23-small-model-train-full-roadmap.md`, replace:

```markdown
**Future plan file:** `docs/superpowers/plans/2026-06-23-stage5e-controlled-experimentation-efficiency.md`
```

with:

```markdown
**Plan file:** `docs/superpowers/plans/2026-06-28-stage5e-controlled-experimentation-efficiency.md`
```

- [ ] **Step 5: Run docs scan**

Run:

```powershell
rg -n "Stage 5E|stage5e|paired eval|experiment manifest|run_experiment_matrix|build_paired_eval_report" README.md docs --glob "!docs/superpowers/specs/**"
```

Expected: docs point to Stage 5E manifest, paired eval, dry-run matrix, and single-variable boundaries.

- [ ] **Step 6: Commit Task 4**

```powershell
git add docs/stage5e-controlled-experimentation-efficiency.zh.md README.md docs/index.zh.md docs/pipeline-flow.zh.md docs/superpowers/plans/2026-06-23-small-model-train-full-roadmap.md
git commit -m "docs: add stage5e controlled experiment workflow"
```

---

## Task 5: Final Verification

**Files:**
- All files changed by Tasks 1-4.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
python -m pytest tests/test_stage5e_experiment_manifest.py tests/test_paired_eval.py tests/test_run_experiment_matrix.py tests/test_stage5e_entry.py -q
```

Expected: all focused tests pass.

- [ ] **Step 2: Run full suite**

Run:

```powershell
python -m pytest -q
```

Expected: full suite passes.

- [ ] **Step 3: Verify Stage 5E gate still passes**

Run:

```powershell
python scripts/check_stage5e_entry.py --summary reports/stage5d_review_summary.json --review-records data_review/stage5d_review_records.jsonl --revisions data_review/stage5d_revisions.jsonl --rejection-sampling-rows data_sft/stage5d_rejection_sampling_sft.jsonl --preference-rows data_pref/stage5d_same_plot_preference.jsonl --generation-records outputs/stage5d_generation_records.jsonl --output reports/stage5e_entry_check.json
```

Expected: exit code 0 and stdout includes `Stage 5E entry gate passed`.

- [ ] **Step 4: Run whitespace check**

Run:

```powershell
git diff --check
```

Expected: no output.

- [ ] **Step 5: Run red-flag scan**

Run:

```powershell
$patterns = @("to" + "do", "tb" + "d", "place" + "holder", "fa" + "ke", "st" + "ub", "not" + " implemented", "pass$")
rg -n -i ($patterns -join "|") src scripts tests docs README.md --glob "!docs/superpowers/plans/**" --glob "!docs/superpowers/specs/**"
```

Expected: no production-code matches. Test-double matches are acceptable only in tests.

- [ ] **Step 6: Commit final verification docs if needed**

If Task 5 changes files, commit them:

```powershell
git status --short
git add src scripts tests docs README.md
git commit -m "chore: verify stage5e control plane"
```

If no files changed, do not create an empty commit.

---

## Stage 5E Entry-To-Implementation Exit Criteria

Stage 5E scaffolding is ready only when all of these are true:

- Stage 5D gate still exits 0 and `reports/stage5e_entry_check.json` contains `"passed": true`.
- Full pytest passes.
- Experiment manifests record Stage 5D gate evidence and artifact hashes.
- Manifest validation rejects multi-variable changes.
- Dry-run experiment matrix writes commands without starting training.
- Paired eval report records candidate wins/losses/ties and regression ids.
- Docs state that Stage 5E control-plane work does not prove model-quality improvement.

## Self-Review

- Spec coverage: Task 1 covers manifest and entry-gate evidence; Task 2 covers paired eval; Task 3 covers dry-run experiment commands; Task 4 covers operator docs and roadmap; Task 5 covers verification.
- Scope control: No task runs DPO, SimPO, ORPO, KTO, reward-model training, or large experiment batches.
- Type consistency: `experiment_id`, `baseline_run_id`, `candidate_run_id`, `primary_variable`, `controlled_variables`, `stage5e_entry`, `artifacts`, `paired_eval`, and `boundary` are used consistently across tests, modules, scripts, and docs.
