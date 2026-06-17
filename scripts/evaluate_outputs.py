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
from small_model_train.reporting import build_markdown_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--title", default="Evaluation Report")
    parser.add_argument("--config-json", default="{}")
    args = parser.parse_args()

    config_snapshot = json.loads(args.config_json)
    report = build_markdown_report(args.title, read_jsonl(args.scores), config_snapshot)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report, encoding="utf-8")
    print(f"wrote report to {args.report}")


if __name__ == "__main__":
    main()
