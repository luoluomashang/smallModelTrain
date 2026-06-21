"""Stage 4.1 quality-eval helpers.

These helpers summarize artifacts from real eval runs without copying generated
chapter text into reports. The generated JSONL remains the source of truth.
"""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any


OUTLINE_MARKERS = ("【", "】", "章节结构", "以下是正文")


def select_quality_subset(
    cards: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []

    metrics_by_id = {str(row.get("id", "")): row for row in metrics}

    def sort_key(indexed_card: tuple[int, dict[str, Any]]) -> tuple[int, int]:
        index, card = indexed_card
        metric = metrics_by_id.get(str(card.get("id", "")), {})
        failures = set(metric.get("failure_types", []))
        if "outline_leak" in failures:
            priority = 0
        elif failures:
            priority = 1
        else:
            priority = 2
        return (priority, index)

    selected = sorted(enumerate(cards), key=sort_key)[:count]
    return [card for _, card in selected]


def detect_outline_markers(text: str) -> list[str]:
    return [marker for marker in OUTLINE_MARKERS if marker in text]


def summarize_quality_budget(
    cards: list[dict[str, Any]],
    generated_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_rows = len(cards)
    generated_by_id = {str(row.get("id", "")): row for row in generated_rows}
    metric_by_id = {str(row.get("id", "")): row for row in metric_rows}
    expected_ids = [str(card.get("id", "")) for card in cards]

    char_counts = [
        int(row.get("char_count_zh", 0))
        for row in metric_rows
        if isinstance(row.get("char_count_zh", 0), (int, float))
    ]
    failure_counts = Counter(
        failure for row in metric_rows for failure in row.get("failure_types", [])
    )
    hard_gate_passes = sum(1 for row in metric_rows if row.get("hard_gate_pass"))
    max_new_tokens = sorted(
        {
            int(params["max_new_tokens"])
            for row in generated_rows
            if isinstance((params := row.get("params")), dict)
            and isinstance(params.get("max_new_tokens"), int)
        }
    )
    outline_leaks = []
    for row in metric_rows:
        failures = set(row.get("failure_types", []))
        if "outline_leak" not in failures:
            continue
        sample_id = str(row.get("id", ""))
        output = str(generated_by_id.get(sample_id, {}).get("output", ""))
        outline_leaks.append(
            {
                "id": sample_id,
                "char_count_zh": int(row.get("char_count_zh", 0)),
                "markers": detect_outline_markers(output),
                "failure_types": list(row.get("failure_types", [])),
            }
        )

    summary = {
        "expected_rows": expected_rows,
        "generated_rows": len(generated_rows),
        "metrics_rows": len(metric_rows),
        "missing_generated_ids": [
            sample_id for sample_id in expected_ids if sample_id not in generated_by_id
        ],
        "missing_metric_ids": [
            sample_id for sample_id in expected_ids if sample_id not in metric_by_id
        ],
        "max_new_tokens": max_new_tokens,
        "char_count_min": min(char_counts) if char_counts else 0,
        "char_count_max": max(char_counts) if char_counts else 0,
        "char_count_avg": round(mean(char_counts), 2) if char_counts else 0,
        "hard_gate_pass": hard_gate_passes,
        "hard_gate_total": len(metric_rows),
        "hard_gate_pass_rate": round(hard_gate_passes / len(metric_rows), 4)
        if metric_rows
        else 0,
        "failure_counts": dict(sorted(failure_counts.items())),
        "outline_leaks": outline_leaks,
    }
    summary["decision"] = _quality_decision(summary)
    summary["recommendation"] = _recommendation(summary["decision"])
    return summary


def render_quality_budget_report(title: str, summary: dict[str, Any]) -> str:
    lines = [
        f"# {title}",
        "",
        "## Decision",
        f"- {summary['decision']}",
        f"- {summary['recommendation']}",
        "",
        "## Rows",
        f"- expected: {summary['expected_rows']}",
        f"- generated: {summary['generated_rows']}",
        f"- metrics: {summary['metrics_rows']}",
        "",
        "## Inference Budget",
        f"- max_new_tokens: {', '.join(map(str, summary['max_new_tokens'])) or 'unknown'}",
        "",
        "## Character Counts",
        f"- min: {summary['char_count_min']}",
        f"- max: {summary['char_count_max']}",
        f"- avg: {summary['char_count_avg']}",
        "",
        "## Hard Gate",
        f"- pass: {summary['hard_gate_pass']}/{summary['hard_gate_total']}",
        f"- pass_rate: {summary['hard_gate_pass_rate']}",
        "",
        "## Failure Counts",
    ]
    if summary["failure_counts"]:
        for failure, count in summary["failure_counts"].items():
            lines.append(f"- {failure}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "## Outline Leak Triage"])
    outline_leaks = summary.get("outline_leaks", [])
    if outline_leaks:
        for row in outline_leaks:
            markers = ", ".join(row.get("markers", [])) or "marker not found"
            lines.append(
                f"- {row['id']}: char_count={row['char_count_zh']}; markers={markers}"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Missing Rows"])
    if summary["missing_generated_ids"]:
        lines.append(
            "- missing generated: " + ", ".join(summary["missing_generated_ids"])
        )
    else:
        lines.append("- missing generated: none")
    if summary["missing_metric_ids"]:
        lines.append("- missing metrics: " + ", ".join(summary["missing_metric_ids"]))
    else:
        lines.append("- missing metrics: none")

    return "\n".join(lines) + "\n"


def _quality_decision(summary: dict[str, Any]) -> str:
    if summary["generated_rows"] < summary["expected_rows"]:
        return "blocked_incomplete_generation"
    if summary["metrics_rows"] < summary["expected_rows"]:
        return "blocked_incomplete_metrics"
    failures = summary["failure_counts"]
    if failures.get("length_short", 0):
        return "blocked_length_short"
    if failures.get("length_long", 0):
        return "blocked_length_long"
    if failures.get("outline_leak", 0):
        return "blocked_outline_leak"
    if summary["expected_rows"] and summary["hard_gate_pass"] == summary["expected_rows"]:
        return "ready_for_full_50_long_eval"
    return "needs_quality_review"


def _recommendation(decision: str) -> str:
    recommendations = {
        "blocked_incomplete_generation": "先修复推理完成率或降低 subset/budget。",
        "blocked_incomplete_metrics": "先补齐 scoring，再判断质量。",
        "blocked_length_short": "提高 long-generation budget，直到接近 2000-2500 中文汉字目标。",
        "blocked_length_long": "降低 budget 或增加停止策略，避免超过长度门槛。",
        "blocked_outline_leak": "抽查 outline leak 样本，优先修 prompt/output 格式。",
        "ready_for_full_50_long_eval": "可以进入 full 50 long eval，不要直接扩到 100。",
        "needs_quality_review": "需要人工复核失败类型后再决定下一步。",
    }
    return recommendations[decision]
