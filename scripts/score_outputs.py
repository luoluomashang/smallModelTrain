from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.execution_cards import validate_execution_cards
from small_model_train.io_utils import read_jsonl, write_jsonl
from small_model_train.scoring import score_output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", required=True)
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        card_rows = validate_execution_cards(read_jsonl(args.cards))
        cards = {str(row["id"]): row for row in card_rows}
        scores = []
        seen_output_ids: set[str] = set()
        for row in read_jsonl(args.outputs):
            sample_id = str(row["id"])
            if sample_id in seen_output_ids:
                raise ValueError(f"duplicate output id: {sample_id}")
            seen_output_ids.add(sample_id)
            if sample_id not in cards:
                raise ValueError(f"output id not found in cards: {sample_id}")
            text = row.get("output", row.get("text", ""))
            scores.append(score_output(sample_id, cards[sample_id], text))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    write_jsonl(args.output, scores)
    print(f"wrote {len(scores)} scores to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
