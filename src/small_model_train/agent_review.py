from __future__ import annotations

import re
from typing import Any

from small_model_train.execution_cards import VALID_TARGET_PLATFORMS


REVIEWERS = {
    "readthrough_structure",
    "male_genre_payoff",
    "platform_style_compliance",
}
SEVERITIES = {"none", "minor", "major", "blocker"}
SAFE_ISSUE_LABEL_RE = re.compile(r"^[a-z0-9_]{1,80}$")

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
    if not isinstance(row, dict):
        raise ValueError("review row must be a dict")

    missing = [
        field
        for field in REQUIRED_REVIEW_FIELDS
        if field not in row or row.get(field) in (None, "")
    ]
    if missing:
        raise ValueError("missing review fields: " + ", ".join(sorted(missing)))

    reviewer = row.get("reviewer")
    if reviewer not in REVIEWERS:
        raise ValueError(f"unknown reviewer: {reviewer}")

    severity = row.get("severity")
    if severity not in SEVERITIES:
        raise ValueError(f"unknown severity: {severity}")

    target_platform = row.get("target_platform")
    if target_platform not in VALID_TARGET_PLATFORMS:
        raise ValueError(f"unknown target_platform: {target_platform}")

    if type(row["pass"]) is not bool:
        raise ValueError("pass must be a boolean")

    for field in ("genre_tags", "issues", "evidence"):
        if not isinstance(row.get(field), list):
            raise ValueError(f"{field} must be a list")

    if not all(isinstance(issue, str) and issue.strip() for issue in row["issues"]):
        raise ValueError("issues must contain non-empty strings")
    if not all(SAFE_ISSUE_LABEL_RE.fullmatch(issue) for issue in row["issues"]):
        raise ValueError("issues must be safe code-like labels")

    return row


def aggregate_agent_reviews(
    expected_ids: list[str],
    review_rows: list[Any],
    target_platform: str,
    rubric_version: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected_id_set = set(expected_ids)
    votes_by_id = {
        sample_id: {
            "id": sample_id,
            "target_platform": target_platform,
            "review_count": 0,
            "pass_votes": 0,
            "fail_votes": 0,
            "blocker_votes": 0,
            "agent_gate_pass": False,
            "requires_human_arbitration": False,
            "issues": [],
            "reviewers": [],
        }
        for sample_id in expected_ids
    }
    seen_reviewers = {sample_id: set() for sample_id in expected_ids}
    malformed_review_rows: list[dict[str, Any]] = []
    issue_counts: dict[str, int] = {}

    for row_number, row in enumerate(review_rows, start=1):
        try:
            validate_review_row(row)
            if row["id"] not in expected_id_set:
                raise ValueError(f"unexpected review id: {row['id']}")
            if row["target_platform"] != target_platform:
                raise ValueError(f"target_platform mismatch: {row['target_platform']}")
            if row["rubric_version"] != rubric_version:
                raise ValueError(f"rubric_version mismatch: {row['rubric_version']}")
            if row["reviewer"] in seen_reviewers[row["id"]]:
                raise ValueError(f"duplicate reviewer for sample: {row['reviewer']}")
        except ValueError as exc:
            malformed_review_rows.append({"row_number": row_number, "error": str(exc)})
            continue

        vote = votes_by_id[row["id"]]
        seen_reviewers[row["id"]].add(row["reviewer"])
        vote["reviewers"].append(row["reviewer"])
        vote["review_count"] += 1
        if row["pass"]:
            vote["pass_votes"] += 1
        else:
            vote["fail_votes"] += 1
        if row["severity"] == "blocker":
            vote["blocker_votes"] += 1
            vote["requires_human_arbitration"] = True
        vote["issues"].extend(row["issues"])
        for issue in row["issues"]:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

    missing_review_ids = []
    blocked_ids = []
    arbitration_ids = []
    for sample_id in expected_ids:
        vote = votes_by_id[sample_id]
        vote["reviewers"] = sorted(vote["reviewers"])
        if seen_reviewers[sample_id] != REVIEWERS:
            missing_review_ids.append(sample_id)
        elif vote["pass_votes"] >= 2:
            vote["agent_gate_pass"] = True
        else:
            blocked_ids.append(sample_id)

        if vote["requires_human_arbitration"]:
            arbitration_ids.append(sample_id)

    if malformed_review_rows or missing_review_ids:
        decision = "blocked_incomplete_agent_review"
    elif arbitration_ids:
        decision = "blocked_by_human_arbitration"
    elif blocked_ids:
        decision = "blocked_by_agent_review"
    elif any(vote["pass_votes"] == 2 for vote in votes_by_id.values()):
        decision = "ready_for_human_spot_check"
    else:
        decision = "ready_for_next_expansion"

    votes = [votes_by_id[sample_id] for sample_id in expected_ids]
    summary = {
        "target_platform": target_platform,
        "rubric_version": rubric_version,
        "expected_rows": len(expected_ids) * len(REVIEWERS),
        "reviewed_rows": sum(vote["review_count"] for vote in votes),
        "missing_review_ids": missing_review_ids,
        "agent_gate_pass": (
            not malformed_review_rows
            and not missing_review_ids
            and not blocked_ids
            and not arbitration_ids
            and all(vote["agent_gate_pass"] for vote in votes)
        ),
        "blocked_ids": blocked_ids,
        "arbitration_ids": arbitration_ids,
        "issue_counts": issue_counts,
        "decision": decision,
        "malformed_review_rows": malformed_review_rows,
    }
    return summary, votes


def render_agent_review_report(
    title: str,
    summary: dict[str, Any],
    votes: list[dict[str, Any]],
) -> str:
    lines = [
        f"# {title}",
        "",
        f"- decision: {summary['decision']}",
        f"- target_platform: {summary['target_platform']}",
        f"- rubric_version: {summary['rubric_version']}",
        f"- expected_rows: {summary['expected_rows']}",
        f"- reviewed_rows: {summary['reviewed_rows']}",
        f"- agent_gate_pass: {summary['agent_gate_pass']}",
        f"- malformed_review_rows: {len(summary.get('malformed_review_rows', []))}",
        "",
        "## Issue Counts",
    ]
    for key, value in sorted(summary["issue_counts"].items()):
        lines.append(f"- {key}: {value}")
    if not summary["issue_counts"]:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Missing Review IDs",
            _format_id_list(summary["missing_review_ids"]),
            "",
            "## Blocked Samples",
            _format_id_list(summary["blocked_ids"]),
            "",
            "## Arbitration Samples",
            _format_id_list(summary["arbitration_ids"]),
            "",
            "## Sample Votes",
        ]
    )
    for vote in sorted(votes, key=lambda item: item["id"]):
        lines.append(
            "- "
            f"{vote['id']}: pass={vote['pass_votes']}, "
            f"fail={vote['fail_votes']}, blockers={vote['blocker_votes']}, "
            f"agent_gate_pass={vote['agent_gate_pass']}, "
            f"requires_human_arbitration={vote['requires_human_arbitration']}"
        )

    return "\n".join(lines) + "\n"


def _format_id_list(values: list[str]) -> str:
    if not values:
        return "- none"
    return "\n".join(f"- {value}" for value in values)
