from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.io_utils import read_jsonl, write_jsonl
from small_model_train.text_utils import count_chinese_chars, normalize_newlines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-chars", type=int, default=500)
    parser.add_argument("--max-chars", type=int, default=5000)
    args = parser.parse_args()

    seen_texts: set[str] = set()
    cleaned_rows: list[dict] = []
    for row in read_jsonl(args.input):
        text = normalize_newlines(row.get("text", ""))
        char_count = count_chinese_chars(text)
        if char_count < args.min_chars or char_count > args.max_chars:
            continue
        if text in seen_texts:
            continue
        seen_texts.add(text)
        updated = dict(row)
        updated["text"] = text
        updated["char_count_zh"] = char_count
        cleaned_rows.append(updated)
    write_jsonl(args.output, cleaned_rows)
    print(f"wrote {len(cleaned_rows)} cleaned chapters to {args.output}")


if __name__ == "__main__":
    main()
