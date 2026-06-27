from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.io_utils import read_jsonl
from small_model_train.review.stage5d_report import (
    build_stage5d_summary,
    render_stage5d_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-records", required=True)
    parser.add_argument("--revisions", required=True)
    parser.add_argument("--rejection-sampling-rows", required=True)
    parser.add_argument("--preference-rows", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--report-output", required=True)
    args = parser.parse_args()

    try:
        review_records = _read_required_jsonl(args.review_records)
        revision_records = _read_required_jsonl(args.revisions)
        rejection_sampling_rows = _read_required_jsonl(args.rejection_sampling_rows)
        preference_rows = _read_required_jsonl(args.preference_rows)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    summary = build_stage5d_summary(
        review_records,
        revision_records,
        rejection_sampling_rows,
        preference_rows,
    )
    report = render_stage5d_report(summary)

    summary_path = Path(args.summary_output)
    report_path = Path(args.report_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(report, encoding="utf-8")
    print(f"wrote Stage 5D summary to {summary_path}")
    return 0


def _read_required_jsonl(path: str) -> list[dict]:
    file_path = Path(path)
    if not file_path.exists():
        raise ValueError(f"required input path does not exist: {file_path}")
    try:
        return read_jsonl(file_path)
    except ValueError as exc:
        raise ValueError(f"required input path is invalid: {file_path}: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
