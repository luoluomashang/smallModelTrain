from __future__ import annotations

import argparse
from collections import Counter
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.agent_review import (  # noqa: E402
    REVIEWERS,
    aggregate_agent_reviews,
    render_agent_review_report,
)
from small_model_train.execution_cards import (  # noqa: E402
    RUBRIC_VERSION,
    validate_execution_cards,
)
from small_model_train.io_utils import read_jsonl, write_jsonl  # noqa: E402


SAFE_ISSUE_LABEL_RE = re.compile(r"^[a-z0-9_]{1,80}$")


def _read_required_jsonl(path: str | Path, label: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"{label} file is missing or empty: {path}")
    return rows


def _row_ids(label: str, rows: list[dict[str, Any]]) -> list[str]:
    row_ids: list[str] = []
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{label} row {row_number} must be an object")
        raw_id = row.get("id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise ValueError(f"{label} row {row_number} missing id")
        row_ids.append(raw_id.strip())
    return row_ids


def _missing_ids(expected_ids: list[str], row_ids: list[str]) -> list[str]:
    present = set(row_ids)
    return [expected_id for expected_id in expected_ids if expected_id not in present]


def _duplicate_ids(row_ids: list[str]) -> list[str]:
    counts = Counter(row_id for row_id in row_ids if row_id)
    return sorted(row_id for row_id, count in counts.items() if count > 1)


def _unexpected_ids(expected_ids: list[str], row_ids: list[str]) -> list[str]:
    expected = set(expected_ids)
    return sorted({row_id for row_id in row_ids if row_id and row_id not in expected})


def _validate_exact_ids(
    label: str,
    expected_ids: list[str],
    rows: list[dict[str, Any]],
) -> None:
    row_ids = _row_ids(label, rows)
    missing = _missing_ids(expected_ids, row_ids)
    if missing:
        raise ValueError(f"{label} missing rows for card ids: {', '.join(missing)}")
    duplicates = _duplicate_ids(row_ids)
    if duplicates:
        raise ValueError(f"{label} duplicate rows for card ids: {', '.join(duplicates)}")
    unexpected = _unexpected_ids(expected_ids, row_ids)
    if unexpected:
        raise ValueError(f"{label} unexpected rows for card ids: {', '.join(unexpected)}")


def _safe_issue_labels(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []

    labels: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        normalized = re.sub(r"[^a-z0-9_]+", "_", value.strip().lower()).strip("_")
        normalized = re.sub(r"_+", "_", normalized)[:80]
        if normalized and SAFE_ISSUE_LABEL_RE.fullmatch(normalized):
            labels.append(normalized)
    return labels


def _mock_reviews(
    cards: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    target_platform: str,
    rubric_version: str,
) -> list[dict[str, Any]]:
    metrics_by_id = {
        row.get("id"): row for row in metrics if isinstance(row, dict) and row.get("id")
    }
    rows: list[dict[str, Any]] = []
    for card in cards:
        metric = metrics_by_id.get(card["id"], {})
        passed = metric.get("hard_gate_pass") is True
        issues = [] if passed else _safe_issue_labels(metric.get("failure_types"))
        if not passed and not issues:
            issues = ["rule_gate_failed"]

        for reviewer in sorted(REVIEWERS):
            rows.append(
                {
                    "id": card["id"],
                    "target_platform": target_platform,
                    "genre_tags": card["genre_tags"],
                    "rubric_version": rubric_version,
                    "reviewer": reviewer,
                    "pass": passed,
                    "severity": "none" if passed else "major",
                    "issues": issues,
                    "evidence": [
                        "mock review derived from Stage 4 hard gate metrics"
                    ],
                    "recommendation": "accept" if passed else "revise",
                    "confidence": 1.0,
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", required=True)
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--target-platform", required=True)
    parser.add_argument("--rubric-version", default=RUBRIC_VERSION)
    parser.add_argument("--backend", choices=["mock"], default="mock")
    parser.add_argument("--reviews-import")
    parser.add_argument("--output", required=True)
    parser.add_argument("--votes-output", required=True)
    parser.add_argument("--summary-output")
    parser.add_argument("--report", required=True)
    parser.add_argument("--title", default="Stage 4 Agent Review Report")
    args = parser.parse_args()

    try:
        cards = validate_execution_cards(_read_required_jsonl(args.cards, "cards"))
        expected_ids = [card["id"] for card in cards]
        outputs = _read_required_jsonl(args.outputs, "outputs")
        metrics = _read_required_jsonl(args.metrics, "metrics")
        _validate_exact_ids("outputs", expected_ids, outputs)
        _validate_exact_ids("metrics", expected_ids, metrics)

        if args.reviews_import:
            review_rows = read_jsonl(args.reviews_import)
        else:
            review_rows = _mock_reviews(
                cards, metrics, args.target_platform, args.rubric_version
            )

        summary, votes = aggregate_agent_reviews(
            expected_ids, review_rows, args.target_platform, args.rubric_version
        )
        report = render_agent_review_report(args.title, summary, votes)

        write_jsonl(args.output, review_rows)
        write_jsonl(args.votes_output, votes)
        if args.summary_output:
            write_jsonl(args.summary_output, [summary])
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"wrote {len(review_rows)} agent review rows to {args.output}")
    print(f"wrote {len(votes)} agent review votes to {args.votes_output}")
    if args.summary_output:
        print(f"wrote agent review summary to {args.summary_output}")
    print(f"wrote agent review report to {args.report}")
    return 0 if summary["decision"] in {
        "ready_for_human_spot_check",
        "ready_for_next_expansion",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
