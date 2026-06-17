from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.io_utils import read_jsonl, write_jsonl
from small_model_train.preference_builder import build_preference_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", required=True)
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = build_preference_candidates(
        read_jsonl(args.cards),
        read_jsonl(args.outputs),
        read_jsonl(args.scores),
    )
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} preference candidates to {args.output}")


if __name__ == "__main__":
    main()
