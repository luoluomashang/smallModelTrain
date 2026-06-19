from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from small_model_train.chapter_cards import build_draft_chapter_cards
from small_model_train.io_utils import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build draft chapter cards from split train chapters.")
    parser.add_argument("--chapters", default="data_clean/chapters_split.jsonl")
    parser.add_argument("--output", default="data_cards/chapter_cards.jsonl")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--min-chars", type=int, default=2000)
    parser.add_argument("--max-chars", type=int, default=3000)
    args = parser.parse_args()

    chapters = read_jsonl(args.chapters)
    cards = build_draft_chapter_cards(
        chapters,
        count=args.count,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )
    write_jsonl(args.output, cards)
    print(f"wrote {len(cards)} chapter cards to {args.output}")


if __name__ == "__main__":
    main()
