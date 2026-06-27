from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.io_utils import read_jsonl, write_jsonl
from small_model_train.review.rejection_sampling import build_rejection_sampling_sft_rows
from small_model_train.style_contract import read_style_contract_asset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revisions", required=True)
    parser.add_argument("--cards", required=True)
    parser.add_argument("--style-contract-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        revisions = read_jsonl(args.revisions)
        cards = read_jsonl(args.cards)
        style_contract = read_style_contract_asset(args.style_contract_json)
        rows = build_rejection_sampling_sft_rows(revisions, cards, style_contract)
        write_jsonl(args.output, rows)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"wrote {len(rows)} rejection-sampling SFT rows to {args.output}")


if __name__ == "__main__":
    main()
