from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.io_utils import read_jsonl, write_jsonl
from small_model_train.preference_builder import build_same_plot_preference_candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revisions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        revisions_path = Path(args.revisions)
        if not revisions_path.exists():
            raise ValueError(f"revisions JSONL not found: {revisions_path}")

        rows = build_same_plot_preference_candidates(read_jsonl(revisions_path))
        write_jsonl(args.output, rows)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"wrote {len(rows)} same-plot preference rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
