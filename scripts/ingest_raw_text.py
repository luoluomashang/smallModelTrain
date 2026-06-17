from __future__ import annotations

import argparse
from pathlib import Path

from small_model_train.chapter_splitter import split_chapters
from small_model_train.io_utils import read_text_auto, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quality-tag", default="A")
    args = parser.parse_args()

    rows: list[dict] = []
    input_dir = Path(args.input_dir)
    for path in sorted(input_dir.rglob("*")):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        work_id = path.stem.replace(" ", "_")
        rows.extend(
            split_chapters(
                read_text_auto(path),
                work_id=work_id,
                quality_tag=args.quality_tag,
            )
        )
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} chapters to {args.output}")


if __name__ == "__main__":
    main()
