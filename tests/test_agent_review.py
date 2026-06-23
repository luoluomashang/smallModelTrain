from __future__ import annotations

import pytest

from small_model_train.agent_review import (
    aggregate_agent_reviews,
    render_agent_review_report,
    validate_review_row,
)
from small_model_train.execution_cards import DEFAULT_TARGET_PLATFORM, RUBRIC_VERSION


SUMMARY_KEYS = {
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
}
VOTE_KEYS = {
    "id",
    "target_platform",
    "review_count",
    "pass_votes",
    "fail_votes",
    "blocker_votes",
    "agent_gate_pass",
    "requires_human_arbitration",
    "issues",
    "reviewers",
}


def _review(
    sample_id: str,
    reviewer: str,
    passed: bool,
    severity: str = "none",
    issues: list[str] | None = None,
) -> dict:
    return {
        "id": sample_id,
        "target_platform": DEFAULT_TARGET_PLATFORM,
        "genre_tags": ["xuanhuan", "system"],
        "rubric_version": RUBRIC_VERSION,
        "reviewer": reviewer,
        "pass": passed,
        "severity": severity,
        "issues": [] if passed else (issues or ["payoff_weak"]),
        "evidence": ["generated evidence note"],
        "recommendation": "accept" if passed else "revise",
        "confidence": 0.9,
    }


def _aggregate(rows: list[dict]) -> tuple[dict, list[dict]]:
    return aggregate_agent_reviews(
        ["case1"], rows, DEFAULT_TARGET_PLATFORM, RUBRIC_VERSION
    )


def test_validate_review_row_accepts_known_reviewer():
    row = _review("case1", "readthrough_structure", True)

    assert validate_review_row(row) == row


def test_validate_review_row_blocks_unknown_reviewer():
    row = _review("case1", "unknown_agent", True)

    with pytest.raises(ValueError, match="unknown reviewer"):
        validate_review_row(row)


def test_validate_review_row_blocks_unknown_severity():
    row = _review("case1", "readthrough_structure", True, severity="critical")

    with pytest.raises(ValueError, match="unknown severity"):
        validate_review_row(row)


def test_validate_review_row_blocks_unknown_target_platform():
    row = _review("case1", "readthrough_structure", True)
    row["target_platform"] = "unknown_platform"

    with pytest.raises(ValueError, match="unknown target_platform"):
        validate_review_row(row)


def test_validate_review_row_blocks_non_bool_pass():
    row = _review("case1", "readthrough_structure", True)
    row["pass"] = 1

    with pytest.raises(ValueError, match="pass must be a boolean"):
        validate_review_row(row)


@pytest.mark.parametrize("field", ["genre_tags", "issues", "evidence"])
def test_validate_review_row_blocks_non_list_fields(field: str):
    row = _review("case1", "readthrough_structure", True)
    row[field] = "not a list"

    with pytest.raises(ValueError, match=f"{field} must be a list"):
        validate_review_row(row)


def test_aggregate_agent_reviews_passes_two_of_three_with_spec_schema():
    rows = [
        _review("case1", "readthrough_structure", True),
        _review("case1", "male_genre_payoff", True),
        _review(
            "case1",
            "platform_style_compliance",
            False,
            severity="major",
            issues=["semantic_repetition"],
        ),
    ]

    summary, votes = _aggregate(rows)
    vote = votes[0]

    assert set(summary) == SUMMARY_KEYS
    assert summary["expected_rows"] == 3
    assert summary["reviewed_rows"] == 3
    assert summary["reviewed_card_ids"] == ["case1"]
    assert summary["agent_gate_pass"] is True
    assert summary["blocked_ids"] == []
    assert summary["arbitration_ids"] == []
    assert summary["issue_counts"] == {"semantic_repetition": 1}
    assert summary["decision"] == "ready_for_human_spot_check"
    assert isinstance(votes, list)
    assert len(votes) == 1
    assert set(vote) == VOTE_KEYS
    assert vote["id"] == "case1"
    assert vote["target_platform"] == DEFAULT_TARGET_PLATFORM
    assert vote["review_count"] == 3
    assert vote["pass_votes"] == 2
    assert vote["fail_votes"] == 1
    assert vote["blocker_votes"] == 0
    assert vote["agent_gate_pass"] is True
    assert vote["requires_human_arbitration"] is False
    assert vote["issues"] == ["semantic_repetition"]
    assert vote["reviewers"] == [
        "male_genre_payoff",
        "platform_style_compliance",
        "readthrough_structure",
    ]
    assert "pass_count" not in vote
    assert "fail_count" not in vote
    assert "missing_reviewers" not in vote
    assert "expected_review_rows" not in summary
    assert "valid_review_rows" not in summary
    assert "blocked_sample_ids" not in summary
    assert "arbitration_sample_ids" not in summary


def test_aggregate_agent_reviews_passes_three_of_three():
    rows = [
        _review("case1", "readthrough_structure", True),
        _review("case1", "male_genre_payoff", True),
        _review("case1", "platform_style_compliance", True),
    ]

    summary, votes = _aggregate(rows)
    vote = votes[0]

    assert vote["agent_gate_pass"] is True
    assert vote["pass_votes"] == 3
    assert vote["fail_votes"] == 0
    assert summary["agent_gate_pass"] is True
    assert summary["decision"] == "ready_for_next_expansion"


def test_aggregate_agent_reviews_blocks_one_of_three():
    rows = [
        _review("case1", "readthrough_structure", True),
        _review("case1", "male_genre_payoff", False, severity="major"),
        _review("case1", "platform_style_compliance", False, severity="major"),
    ]

    summary, votes = _aggregate(rows)
    vote = votes[0]

    assert vote["agent_gate_pass"] is False
    assert vote["pass_votes"] == 1
    assert vote["fail_votes"] == 2
    assert summary["agent_gate_pass"] is False
    assert summary["blocked_ids"] == ["case1"]
    assert summary["decision"] == "blocked_by_agent_review"


def test_aggregate_agent_reviews_blocks_zero_of_three():
    rows = [
        _review("case1", "readthrough_structure", False, severity="major"),
        _review("case1", "male_genre_payoff", False, severity="major"),
        _review("case1", "platform_style_compliance", False, severity="major"),
    ]

    summary, votes = _aggregate(rows)
    vote = votes[0]

    assert vote["agent_gate_pass"] is False
    assert vote["pass_votes"] == 0
    assert vote["fail_votes"] == 3
    assert summary["blocked_ids"] == ["case1"]
    assert summary["decision"] == "blocked_by_agent_review"


def test_aggregate_agent_reviews_prioritizes_blocker_arbitration_over_majority_block():
    rows = [
        _review("case1", "readthrough_structure", True),
        _review("case1", "male_genre_payoff", False, severity="blocker"),
        _review("case1", "platform_style_compliance", False, severity="major"),
    ]

    summary, votes = _aggregate(rows)
    vote = votes[0]

    assert vote["agent_gate_pass"] is False
    assert vote["blocker_votes"] == 1
    assert summary["blocked_ids"] == ["case1"]
    assert summary["arbitration_ids"] == ["case1"]
    assert summary["decision"] == "blocked_by_human_arbitration"


def test_aggregate_agent_reviews_blocker_prevents_batch_gate_pass():
    rows = [
        _review("case1", "readthrough_structure", True),
        _review("case1", "male_genre_payoff", True),
        _review("case1", "platform_style_compliance", False, severity="blocker"),
    ]

    summary, votes = _aggregate(rows)

    assert votes[0]["agent_gate_pass"] is True
    assert summary["arbitration_ids"] == ["case1"]
    assert summary["decision"] == "blocked_by_human_arbitration"
    assert summary["agent_gate_pass"] is False


def test_aggregate_agent_reviews_sends_blocker_to_arbitration():
    rows = [
        _review("case1", "readthrough_structure", True),
        _review("case1", "male_genre_payoff", True),
        _review("case1", "platform_style_compliance", False, severity="blocker"),
    ]

    summary, votes = _aggregate(rows)
    vote = votes[0]

    assert vote["agent_gate_pass"] is True
    assert vote["requires_human_arbitration"] is True
    assert vote["blocker_votes"] == 1
    assert summary["arbitration_ids"] == ["case1"]
    assert summary["decision"] == "blocked_by_human_arbitration"


def test_aggregate_agent_reviews_blocks_incomplete_reviews():
    rows = [_review("case1", "readthrough_structure", True)]

    summary, votes = _aggregate(rows)
    vote = votes[0]

    assert vote["agent_gate_pass"] is False
    assert vote["review_count"] == 1
    assert vote["pass_votes"] == 1
    assert summary["missing_review_ids"] == ["case1"]
    assert summary["agent_gate_pass"] is False
    assert summary["decision"] == "blocked_incomplete_agent_review"


def test_aggregate_agent_reviews_preserves_malformed_non_dict_row():
    rows = [
        None,
        _review("case1", "readthrough_structure", True),
        _review("case1", "male_genre_payoff", True),
        _review("case1", "platform_style_compliance", True),
    ]

    summary, votes = aggregate_agent_reviews(
        ["case1"], rows, DEFAULT_TARGET_PLATFORM, RUBRIC_VERSION
    )

    assert votes[0]["agent_gate_pass"] is True
    assert summary["malformed_review_rows"] == [
        {"row_number": 1, "error": "review row must be a dict"}
    ]
    assert summary["decision"] == "blocked_incomplete_agent_review"


@pytest.mark.parametrize("issues", [[{"label": "x"}], [""]])
def test_aggregate_agent_reviews_preserves_malformed_issue_elements(issues: list):
    rows = [
        _review("case1", "readthrough_structure", True),
        _review("case1", "male_genre_payoff", True),
        _review(
            "case1",
            "platform_style_compliance",
            False,
            severity="major",
            issues=issues,
        ),
    ]

    summary, votes = _aggregate(rows)

    assert votes[0]["review_count"] == 2
    assert summary["malformed_review_rows"] == [
        {"row_number": 3, "error": "issues must contain non-empty strings"}
    ]
    assert summary["decision"] == "blocked_incomplete_agent_review"


def test_aggregate_agent_reviews_counts_issue_labels_not_severities():
    rows = [
        _review(
            "case1",
            "readthrough_structure",
            False,
            severity="major",
            issues=["semantic_repetition"],
        ),
        _review(
            "case1",
            "male_genre_payoff",
            False,
            severity="blocker",
            issues=["semantic_repetition", "style_drift"],
        ),
        _review("case1", "platform_style_compliance", True),
    ]

    summary, votes = _aggregate(rows)

    assert votes[0]["issues"] == ["semantic_repetition", "semantic_repetition", "style_drift"]
    assert summary["issue_counts"] == {"semantic_repetition": 2, "style_drift": 1}
    assert "major" not in summary["issue_counts"]
    assert "blocker" not in summary["issue_counts"]


@pytest.mark.parametrize(
    "unsafe_issue",
    ["这是一段很长的生成正文，不应该进入报告", "# heading"],
)
def test_render_agent_review_report_omits_unsafe_issue_text(unsafe_issue: str):
    rows = [
        _review("case1", "readthrough_structure", True),
        _review("case1", "male_genre_payoff", True),
        _review(
            "case1",
            "platform_style_compliance",
            False,
            severity="major",
            issues=[unsafe_issue],
        ),
    ]
    summary, votes = _aggregate(rows)

    report = render_agent_review_report("Agent Review", summary, votes)

    assert summary["malformed_review_rows"] == [
        {"row_number": 3, "error": "issues must be safe code-like labels"}
    ]
    assert summary["decision"] == "blocked_incomplete_agent_review"
    assert unsafe_issue not in report
    assert "generated evidence note" not in report


def test_render_agent_review_report_preserves_malformed_row_errors():
    rows = [
        None,
        _review("case1", "readthrough_structure", True),
    ]
    summary, votes = _aggregate(rows)

    report = render_agent_review_report("Agent Review", summary, votes)

    assert "Malformed Review Rows" in report
    assert "row 1: review row must be a dict" in report

def test_render_agent_review_report_includes_gate_and_missing_review_ids():
    rows = [_review("case1", "readthrough_structure", True)]
    summary, votes = _aggregate(rows)

    report = render_agent_review_report("Agent Review", summary, votes)

    assert "agent_gate_pass: False" in report
    assert "Missing Review IDs" in report
    assert "case1" in report
    assert "generated evidence note" not in report


def test_render_agent_review_report_uses_spec_fields_and_omits_generated_text():
    rows = [
        _review("case1", "readthrough_structure", True),
        _review("case1", "male_genre_payoff", True),
        _review(
            "case1",
            "platform_style_compliance",
            False,
            severity="major",
            issues=["semantic_repetition"],
        ),
    ]
    summary, votes = _aggregate(rows)

    report = render_agent_review_report("Agent Review", summary, votes)

    assert "# Agent Review" in report
    assert "ready_for_human_spot_check" in report
    assert "expected_rows: 3" in report
    assert "reviewed_rows: 3" in report
    assert "semantic_repetition: 1" in report
    assert "case1: pass=2, fail=1, blockers=0" in report
    assert "generated evidence note" not in report
