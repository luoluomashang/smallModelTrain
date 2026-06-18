from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.stage2_env_check import collect_training_env, render_env_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="reports/training_env_report.md")
    args = parser.parse_args()

    snapshot = collect_training_env()
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_env_report(snapshot), encoding="utf-8")
    print(f"wrote report to {report_path}")

    return 0 if snapshot["recommendation"]["allow_training"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
