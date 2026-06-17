from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.io_utils import read_jsonl, write_jsonl
from small_model_train.scoring import score_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", required=True)
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cards = {row["id"]: row for row in read_jsonl(args.cards)}
    scores = []
    for row in read_jsonl(args.outputs):
        sample_id = row["id"]
        text = row.get("output", row.get("text", ""))
        scores.append(score_output(sample_id, cards.get(sample_id, {}), text))
    write_jsonl(args.output, scores)
    print(f"wrote {len(scores)} scores to {args.output}")


if __name__ == "__main__":
    main()
