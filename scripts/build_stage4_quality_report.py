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
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", required=True)
    parser.add_argument("--generated", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--title", default="Stage 4.1 Quality Eval Budget Report")
    args = parser.parse_args()

    summary = summarize_quality_budget(
        read_jsonl(args.cards),
        read_jsonl(args.generated),
        read_jsonl(args.metrics),
    )
    report = render_quality_budget_report(args.title, summary)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"wrote Stage 4.1 quality report to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
