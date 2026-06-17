from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.chapter_splitter import split_chapters
from small_model_train.io_utils import read_text_auto, write_jsonl


def work_id_from_path(path: Path, input_dir: Path) -> str:
    relative_path = path.relative_to(input_dir).with_suffix("")
    work_id_parts = []
    for part in relative_path.parts:
        sanitized = re.sub(r"\s+", "_", part)
        sanitized = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", sanitized)
        work_id_parts.append(sanitized.strip("_") or "part")
    work_id = "__".join(work_id_parts)
    return work_id.strip("_") or "work"


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
        work_id = work_id_from_path(path, input_dir)
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
