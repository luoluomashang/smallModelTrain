from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.io_utils import read_jsonl, write_jsonl
from small_model_train.sft_builder import build_sft_rows


def _write_dataset_info(path: str | Path, output_path: str | Path) -> None:
    path = Path(path)
    output_path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset_name = output_path.stem
    info = {
        dataset_name: {
            "file_name": output_path.name,
            "formatting": "alpaca",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        }
    }
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", required=True)
    parser.add_argument("--chapters", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-info-output")
    parser.add_argument("--allow-draft-cards", action="store_true")
    args = parser.parse_args()

    if args.dataset_info_output and Path(args.output).resolve() == Path(args.dataset_info_output).resolve():
        parser.error("--dataset-info-output must not be the same path as --output")

    rows = build_sft_rows(
        read_jsonl(args.cards),
        read_jsonl(args.chapters),
        require_approved_cards=not args.allow_draft_cards,
    )
    write_jsonl(args.output, rows)
    if args.dataset_info_output:
        _write_dataset_info(args.dataset_info_output, args.output)
    print(f"wrote {len(rows)} SFT rows to {args.output}")


if __name__ == "__main__":
    main()
