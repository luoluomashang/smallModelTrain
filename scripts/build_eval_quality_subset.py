from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.io_utils import read_jsonl, write_jsonl
from small_model_train.stage4_quality import select_quality_subset


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", required=True)
    parser.add_argument("--metrics")
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=positive_int, default=8)
    args = parser.parse_args()

    cards = read_jsonl(args.cards)
    metrics = read_jsonl(args.metrics) if args.metrics else []
    subset = select_quality_subset(cards, metrics, args.count)
    write_jsonl(args.output, subset)
    print(f"wrote {len(subset)} quality eval cards to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
