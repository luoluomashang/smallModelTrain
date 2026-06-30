from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


BOUNDARY = "paired_eval_no_training"
WINNERS = {"baseline", "candidate", "tie"}


def summarize_paired_eval(
    *,
    baseline_metrics: list[dict[str, Any]],
    candidate_metrics: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_by_id = _index_metric_rows(baseline_metrics, "baseline")
    candidate_by_id = _index_metric_rows(candidate_metrics, "candidate")
    judgment_by_id = _index_judgments(judgments)

    baseline_ids = set(baseline_by_id)
    candidate_ids = set(candidate_by_id)
    missing_baseline_ids = sorted(candidate_ids - baseline_ids)
    missing_candidate_ids = sorted(baseline_ids - candidate_ids)
    missing_messages = []
    if missing_baseline_ids:
        missing_messages.append(f"missing_baseline_ids={missing_baseline_ids}")
    if missing_candidate_ids:
        missing_messages.append(f"missing_candidate_ids={missing_candidate_ids}")
    if missing_messages:
        raise ValueError("; ".join(missing_messages))

    paired_ids = sorted(baseline_ids & candidate_ids)
    if not paired_ids:
        raise ValueError("paired eval requires at least one paired row")

    stale_judgment_ids = sorted(set(judgment_by_id) - set(paired_ids))
    if stale_judgment_ids:
        raise ValueError(f"judgment id not found in paired rows: {stale_judgment_ids[0]}")

    comparisons = []
    wins = 0
    losses = 0
    ties = 0
    regression_ids = []

    for row_id in paired_ids:
        baseline_row = baseline_by_id[row_id]
        candidate_row = candidate_by_id[row_id]
        baseline_score = _metric_score(baseline_row)
        candidate_score = _metric_score(candidate_row)
        if candidate_score < baseline_score:
            regression_ids.append(row_id)

        if row_id in judgment_by_id:
            winner = judgment_by_id[row_id]["winner"]
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

        if winner == "candidate":
            wins += 1
        elif winner == "baseline":
            losses += 1
        else:
            ties += 1

        comparisons.append(
            {
                "id": row_id,
                "baseline_score": baseline_score,
                "candidate_score": candidate_score,
                "baseline_failures": _failure_list(baseline_row),
                "candidate_failures": _failure_list(candidate_row),
                "winner": winner,
                "winner_source": winner_source,
            }
        )

    return {
        "paired_rows": len(paired_ids),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "missing_baseline_ids": sorted(candidate_ids - baseline_ids),
        "missing_candidate_ids": sorted(baseline_ids - candidate_ids),
        "regression_ids": regression_ids,
        "comparisons": comparisons,
        "boundary": BOUNDARY,
    }


def render_paired_eval_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Stage 5E Paired Eval Report",
        "",
        "## Summary",
        f"- Boundary: {summary.get('boundary')}",
        f"- Paired rows: {summary.get('paired_rows', 0)}",
        f"- Candidate wins: {summary.get('wins', 0)}",
        f"- Candidate losses: {summary.get('losses', 0)}",
        f"- Ties: {summary.get('ties', 0)}",
        "",
        "## Regressions",
    ]

    regression_ids = summary.get("regression_ids") or []
    if regression_ids:
        lines.extend(f"- {row_id}" for row_id in regression_ids)
    else:
        lines.append("- None")

    lines.extend(["", "## Comparisons"])
    comparisons = summary.get("comparisons") or []
    if comparisons:
        for comparison in comparisons:
            lines.append(
                "- {id}: {winner} ({winner_source}); baseline={baseline_score}, "
                "candidate={candidate_score}".format(**comparison)
            )
    else:
        lines.append("- None")

    return "\n".join(lines) + "\n"


def write_paired_eval_summary(path: str | Path, summary: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _metric_score(row: dict[str, Any]) -> int:
    score = 10 if row.get("hard_gate_pass") is True else 0
    failures = row.get("failure_types")
    if isinstance(failures, list):
        return score - len(failures)
    return score - 1


def _index_metric_rows(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"{label} metrics must be a list")

    indexed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{label} metric row {index} must be an object")
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id.strip():
            raise ValueError(f"{label} metric row {index} id must be a non-empty string")
        if row_id in indexed:
            raise ValueError(f"duplicate {label} metric id: {row_id}")
        indexed[row_id] = copy.deepcopy(row)
    return indexed


def _index_judgments(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError("judgments must be a list")

    indexed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"judgment row {index} must be an object")
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id.strip():
            raise ValueError(f"judgment row {index} id must be a non-empty string")
        winner = row.get("winner")
        if winner not in WINNERS:
            raise ValueError(f"judgment row {index} winner must be one of {sorted(WINNERS)}")
        if row_id in indexed:
            raise ValueError(f"duplicate judgment id: {row_id}")
        indexed[row_id] = copy.deepcopy(row)
    return indexed


def _failure_list(row: dict[str, Any]) -> list[Any]:
    failures = row.get("failure_types")
    if isinstance(failures, list):
        return copy.deepcopy(failures)
    return []
