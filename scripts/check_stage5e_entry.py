from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.io_utils import read_jsonl
from small_model_train.review.stage5e_entry import check_stage5e_entry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--review-records", required=True)
    parser.add_argument("--revisions", required=True)
    parser.add_argument("--rejection-sampling-rows", required=True)
    parser.add_argument("--preference-rows", required=True)
    parser.add_argument("--generation-records", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        result = check_stage5e_entry(
            summary=_read_summary(args.summary),
            review_records=_read_required_jsonl(args.review_records, "review records"),
            revision_records=_read_required_jsonl(args.revisions, "revisions"),
            rejection_sampling_rows=_read_required_jsonl(
                args.rejection_sampling_rows,
                "rejection-sampling rows",
            ),
            preference_rows=_read_required_jsonl(args.preference_rows, "preference rows"),
            generation_records=_read_required_jsonl(
                args.generation_records,
                "generation records",
            ),
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if not result["passed"]:
        for error in result["errors"]:
            print(error, file=sys.stderr)
        return 1

    print(f"Stage 5E entry gate passed; wrote {output_path}")
    return 0


def _read_summary(path: str) -> dict[str, Any]:
    summary_path = Path(path)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"summary JSON must be an object: {summary_path}")
    return payload


def _read_required_jsonl(path: str, label: str) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        raise ValueError(f"{label} JSONL not found: {input_path}")
    return read_jsonl(input_path)


if __name__ == "__main__":
    raise SystemExit(main())
