from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.dataset_split import split_rows
from small_model_train.io_utils import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--eval-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260617)
    args = parser.parse_args()

    rows = split_rows(read_jsonl(args.input), eval_count=args.eval_count, seed=args.seed)
    write_jsonl(args.output, rows)
    write_jsonl(args.eval_output, [row for row in rows if row["split"] == "eval"])
    print(f"wrote {len(rows)} split rows to {args.output}")


if __name__ == "__main__":
    main()
