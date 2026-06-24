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
from small_model_train.preflight_reports import (
    build_preflight_report,
    write_preflight_report,
)

DEFAULT_MODEL_DIR = r"E:\models\Qwen3-4B-Instruct-2507"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report", default="reports/model_check_report.md")
    parser.add_argument("--json-output", default="reports/model_check_report.json")
    parser.add_argument("--skip-transformers-load", action="store_true")
    args = parser.parse_args()

    result = check_model_files(args.model_dir)
    if result["passed"] and not args.skip_transformers_load:
        run_transformers_load_checks(result)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_model_check_report(result), encoding="utf-8")
    print(f"wrote report to {report_path}")

    write_preflight_report(
        args.json_output,
        build_preflight_report(
            kind="model",
            passed=bool(result.get("passed", False)),
            payload=result,
            errors=_model_preflight_errors(result),
        ),
    )

    return 0 if result["passed"] else 1


def _model_preflight_errors(result: dict[str, object]) -> list[str]:
    errors: list[str] = []
    _extend_unique(errors, result.get("errors", []))
    _extend_unique(
        errors,
        (
            f"missing required file: {name}"
            for name in result.get("missing_files", [])
        ),
    )
    _extend_unique(
        errors,
        (
            f"zero-size model shard: {name}"
            for name in result.get("zero_size_files", [])
        ),
    )
    return errors


def _extend_unique(errors: list[str], candidates: object) -> None:
    if isinstance(candidates, str):
        values = [candidates]
    elif hasattr(candidates, "__iter__"):
        values = candidates
    else:
        return

    for candidate in values:
        text = str(candidate)
        if text not in errors:
            errors.append(text)


if __name__ == "__main__":
    raise SystemExit(main())
