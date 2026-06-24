"""Stage 4.1 quality-eval helpers.

These helpers summarize artifacts from real eval runs without copying generated
chapter text into reports. The generated JSONL remains the source of truth.
"""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

from small_model_train.agent_review import REVIEWERS, SAFE_ISSUE_LABEL_RE
from small_model_train.execution_cards import VALID_TARGET_PLATFORMS


OUTLINE_MARKERS = ("【", "】", "章节结构", "以下是正文")
AGENT_REVIEW_DECISIONS = {
    "ready_for_human_spot_check",
    "ready_for_next_expansion",
    "blocked_by_agent_review",
    "blocked_by_human_arbitration",
    "blocked_incomplete_agent_review",
    "rules_pass_agent_pending",
}
REQUIRED_AGENT_SUMMARY_FIELDS = (
    "target_platform",
    "rubric_version",
    "expected_rows",
    "reviewed_rows",
    "reviewed_card_ids",
    "missing_review_ids",
    "agent_gate_pass",
    "blocked_ids",
    "arbitration_ids",
    "issue_counts",
    "decision",
    "malformed_review_rows",
)


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


def validate_agent_summary(
    agent_summary: dict[str, Any],
    expected_card_ids: list[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(agent_summary, dict):
        raise ValueError("agent summary must be a dict")

    missing = [
        field
        for field in REQUIRED_AGENT_SUMMARY_FIELDS
        if field not in agent_summary or agent_summary.get(field) in (None, "")
    ]
    if missing:
        raise ValueError("agent summary missing fields: " + ", ".join(sorted(missing)))

    decision = agent_summary["decision"]
    if not isinstance(decision, str):
        raise ValueError("agent summary decision must be a string")
    if decision not in AGENT_REVIEW_DECISIONS:
        raise ValueError(f"unknown agent summary decision: {decision}")

    for field in ("target_platform", "rubric_version"):
        if not isinstance(agent_summary[field], str):
            raise ValueError(f"agent summary {field} must be a string")
    if agent_summary["target_platform"] not in VALID_TARGET_PLATFORMS:
        raise ValueError(
            f"unknown agent summary target_platform: {agent_summary['target_platform']}"
        )

    for field in ("expected_rows", "reviewed_rows"):
        value = agent_summary[field]
        if type(value) is not int or value < 0:
            raise ValueError(f"agent summary {field} must be a non-negative integer")

    if type(agent_summary["agent_gate_pass"]) is not bool:
        raise ValueError("agent summary agent_gate_pass must be a boolean")

    if "review_backend" in agent_summary:
        review_backend = agent_summary["review_backend"]
        if (
            not isinstance(review_backend, str)
            or not review_backend.strip()
            or not SAFE_ISSUE_LABEL_RE.fullmatch(review_backend)
        ):
            raise ValueError(
                "agent summary review_backend must be a non-empty safe value"
            )

    if "projection_only" in agent_summary:
        if type(agent_summary["projection_only"]) is not bool:
            raise ValueError("agent summary projection_only must be a boolean")
        if agent_summary["projection_only"] and agent_summary["agent_gate_pass"]:
            raise ValueError(
                "agent summary projection_only conflicts with agent_gate_pass"
            )

    if agent_summary.get("review_backend") == "rule_projection":
        if agent_summary.get("projection_only") is not True:
            raise ValueError(
                "agent summary rule_projection requires projection_only true"
            )
        if agent_summary["agent_gate_pass"] is not False:
            raise ValueError(
                "agent summary rule_projection requires agent_gate_pass false"
            )
        if decision != "rules_pass_agent_pending":
            raise ValueError(
                "agent summary rule_projection requires rules_pass_agent_pending decision"
            )

    for field in (
        "missing_review_ids",
        "blocked_ids",
        "arbitration_ids",
        "malformed_review_rows",
        "reviewed_card_ids",
    ):
        if not isinstance(agent_summary[field], list):
            raise ValueError(f"agent summary {field} must be a list")

    if not isinstance(agent_summary["issue_counts"], dict):
        raise ValueError("agent summary issue_counts must be a dict")

    if not all(isinstance(sample_id, str) for sample_id in agent_summary["reviewed_card_ids"]):
        raise ValueError("agent summary reviewed_card_ids must contain strings")

    if expected_card_ids is not None:
        if agent_summary["reviewed_card_ids"] != expected_card_ids:
            raise ValueError("agent summary card ids mismatch")
        expected_review_rows = len(expected_card_ids) * len(REVIEWERS)
        if agent_summary["expected_rows"] != expected_review_rows:
            raise ValueError("agent summary expected_rows mismatch")

    ready_decisions = {"ready_for_human_spot_check", "ready_for_next_expansion"}
    has_blocking_state = (
        not agent_summary["agent_gate_pass"]
        or bool(agent_summary["missing_review_ids"])
        or bool(agent_summary["blocked_ids"])
        or bool(agent_summary["arbitration_ids"])
        or bool(agent_summary["malformed_review_rows"])
        or bool(agent_summary.get("projection_only"))
        or agent_summary["reviewed_rows"] != agent_summary["expected_rows"]
    )
    if decision in ready_decisions and has_blocking_state:
        raise ValueError("agent summary ready decision conflicts with blocking fields")

    return agent_summary



def _duplicate_ids(rows: list[dict[str, Any]]) -> list[str]:
    counts: Counter[str] = Counter(str(row.get("id", "")) for row in rows)
    return sorted(sample_id for sample_id, count in counts.items() if sample_id and count > 1)

def _generated_outline_text(row: dict[str, Any]) -> str:
    for field in ("raw_output", "output", "text"):
        value = row.get(field)
        if value not in (None, ""):
            return str(value)
    return ""


def summarize_quality_budget(
    cards: list[dict[str, Any]],
    generated_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    agent_summary: dict[str, Any] | None = None,
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
        output = _generated_outline_text(generated_by_id.get(sample_id, {}))
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
        "duplicate_generated_ids": _duplicate_ids(generated_rows),
        "duplicate_metric_ids": _duplicate_ids(metric_rows),
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
    rule_decision = _quality_decision(summary)
    if agent_summary is not None:
        agent_summary = validate_agent_summary(agent_summary, expected_ids)
        summary["agent_review"] = agent_summary
    summary["decision"] = _combined_decision(rule_decision, agent_summary)
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
        "## Agent Review",
        *_agent_review_lines(summary.get("agent_review")),
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


def _agent_review_lines(agent_review: dict[str, Any] | None) -> list[str]:
    if agent_review is None:
        return [
            "- decision: rules_pass_agent_pending",
            "- agent_gate_pass: pending",
            "- blocked_ids: none",
            "- arbitration_ids: none",
        ]

    blocked_ids = [str(sample_id) for sample_id in agent_review.get("blocked_ids", [])]
    arbitration_ids = [
        str(sample_id) for sample_id in agent_review.get("arbitration_ids", [])
    ]
    return [
        f"- decision: {agent_review.get('decision', 'rules_pass_agent_pending')}",
        f"- agent_gate_pass: {agent_review.get('agent_gate_pass', 'unknown')}",
        "- blocked_ids: " + (", ".join(blocked_ids) if blocked_ids else "none"),
        "- arbitration_ids: "
        + (", ".join(arbitration_ids) if arbitration_ids else "none"),
    ]


def _combined_decision(
    rule_decision: str,
    agent_summary: dict[str, Any] | None,
) -> str:
    if rule_decision != "ready_for_full_50_long_eval":
        return rule_decision
    if agent_summary is None:
        return "rules_pass_agent_pending"
    return str(agent_summary.get("decision", "rules_pass_agent_pending"))


def _quality_decision(summary: dict[str, Any]) -> str:
    if summary["generated_rows"] != summary["expected_rows"]:
        return "blocked_incomplete_generation"
    if summary.get("missing_generated_ids") or summary.get("duplicate_generated_ids"):
        return "blocked_incomplete_generation"
    if summary["metrics_rows"] != summary["expected_rows"]:
        return "blocked_incomplete_metrics"
    if summary.get("missing_metric_ids") or summary.get("duplicate_metric_ids"):
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
        "ready_for_human_spot_check": "规则门和 agent gate 已通过，可进入人工抽查。",
        "ready_for_next_expansion": "规则门和 agent gate 已通过，可进入下一轮扩量。",
        "blocked_by_agent_review": "先处理 agent review 标记的质量问题，再进入下一步。",
        "blocked_by_human_arbitration": "等待人工仲裁样本处理后再继续。",
        "blocked_incomplete_agent_review": "补齐 agent review 结果后再判断扩量。",
        "rules_pass_agent_pending": "规则门已通过，等待 agent review 后再继续。",
    }
    return recommendations[decision]
