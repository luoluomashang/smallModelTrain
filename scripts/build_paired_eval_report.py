from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.evaluation.paired_eval import (  # noqa: E402
    render_paired_eval_report,
    summarize_paired_eval,
    write_paired_eval_summary,
)
from small_model_train.io_utils import read_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-metrics", required=True)
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--judgments", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--report-output", required=True)
    args = parser.parse_args()

    try:
        summary = summarize_paired_eval(
            baseline_metrics=_read_required_jsonl(args.baseline_metrics),
            candidate_metrics=_read_required_jsonl(args.candidate_metrics),
            judgments=_read_required_jsonl(args.judgments),
        )
        write_paired_eval_summary(args.summary_output, summary)
        report_path = Path(args.report_output)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_paired_eval_report(summary), encoding="utf-8")
    except (OSError, ValueError) as exc:
        _remove_output_file(args.summary_output)
        _remove_output_file(args.report_output)
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote Stage 5E paired eval report to {report_path}")
    return 0


def _read_required_jsonl(path: str) -> list[dict]:
    file_path = Path(path)
    if not file_path.exists():
        raise ValueError(f"required input path does not exist: {file_path}")
    return read_jsonl(file_path)


def _remove_output_file(output: str) -> None:
    output_path = Path(output)
    try:
        if output_path.is_file():
            output_path.unlink()
    except OSError:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
