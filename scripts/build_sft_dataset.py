from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.io_utils import read_jsonl, write_jsonl
from small_model_train.sft_builder import build_sft_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", required=True)
    parser.add_argument("--chapters", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = build_sft_rows(read_jsonl(args.cards), read_jsonl(args.chapters))
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} SFT rows to {args.output}")


if __name__ == "__main__":
    main()
