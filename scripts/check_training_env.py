from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.preflight_reports import (
    build_preflight_report,
    write_preflight_report,
)
from small_model_train.stage2_env_check import collect_training_env, render_env_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="reports/training_env_report.md")
    parser.add_argument("--json-output", default="reports/training_env_report.json")
    args = parser.parse_args()

    snapshot = collect_training_env()
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_env_report(snapshot), encoding="utf-8")
    print(f"wrote report to {report_path}")

    recommendation = snapshot.get("recommendation", {})
    write_preflight_report(
        args.json_output,
        build_preflight_report(
            kind="environment",
            passed=bool(recommendation.get("allow_training", False)),
            payload=snapshot,
            errors=_environment_preflight_errors(recommendation),
        ),
    )

    return 0 if snapshot["recommendation"]["allow_training"] else 1


def _environment_preflight_errors(recommendation: dict[str, object]) -> list[str]:
    blocking_reasons = [
        str(reason)
        for reason in recommendation.get("blocking_reasons", [])
    ]
    if blocking_reasons:
        return blocking_reasons
    if recommendation.get("allow_training", False):
        return []

    message = str(recommendation.get("message", "")).strip()
    if message:
        return [message]
    return ["training environment recommendation disallows training"]


if __name__ == "__main__":
    raise SystemExit(main())
