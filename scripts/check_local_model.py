from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.stage2_model_check import (
    check_model_files,
    render_model_check_report,
    run_transformers_load_checks,
)

DEFAULT_MODEL_DIR = r"E:\models\Qwen3-4B-Instruct-2507"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report", default="reports/model_check_report.md")
    parser.add_argument("--skip-transformers-load", action="store_true")
    args = parser.parse_args()

    result = check_model_files(args.model_dir)
    if result["passed"] and not args.skip_transformers_load:
        run_transformers_load_checks(result)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_model_check_report(result), encoding="utf-8")
    print(f"wrote report to {report_path}")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
