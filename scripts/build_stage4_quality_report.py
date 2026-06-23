from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.io_utils import read_jsonl
from small_model_train.stage4_quality import (
    render_quality_budget_report,
    summarize_quality_budget,
    validate_agent_summary,
)


def _read_agent_summary(path: str) -> dict:
    if not path.strip():
        raise ValueError("agent summary path is empty")
    file_path = Path(path)
    if not file_path.exists():
        raise ValueError(f"agent summary file does not exist: {file_path}")
    try:
        rows = read_jsonl(file_path)
    except ValueError as exc:
        raise ValueError(f"agent summary file is not valid: {exc}") from exc
    if not rows:
        raise ValueError(f"agent summary file is empty: {file_path}")
    return validate_agent_summary(rows[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", required=True)
    parser.add_argument("--generated", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--agent-summary")
    parser.add_argument("--title", default="Stage 4.1 Quality Eval Budget Report")
    args = parser.parse_args()

    try:
        agent_summary = (
            _read_agent_summary(args.agent_summary)
            if args.agent_summary is not None
            else None
        )
    except ValueError as exc:
        parser.error(str(exc))

    summary = summarize_quality_budget(
        read_jsonl(args.cards),
        read_jsonl(args.generated),
        read_jsonl(args.metrics),
        agent_summary=agent_summary,
    )
    report = render_quality_budget_report(args.title, summary)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"wrote Stage 4.1 quality report to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
