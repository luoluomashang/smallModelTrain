from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.artifact_manifest import file_sha256
from small_model_train.io_utils import read_jsonl
from small_model_train.style_contract import (
    build_style_contract_asset,
    render_style_contract_markdown,
    write_style_contract_asset,
)
from small_model_train.style_profile import build_style_profile


def _distinct_paths(*paths: str | None) -> bool:
    resolved_paths = [Path(path).resolve() for path in paths if path]
    return len(resolved_paths) == len(set(resolved_paths))


def _split_summary(rows: list[dict]) -> dict[str, int]:
    split_counts = Counter(str(row.get("split") or "unknown") for row in rows)
    return {split: split_counts[split] for split in sorted(split_counts)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapters", required=True)
    parser.add_argument(
        "--contract-json-output",
        default="data_style/style_contract_author_main_v1.json",
    )
    parser.add_argument("--contract-output", required=True)
    parser.add_argument(
        "--metrics-output",
        default="data_style/style_metrics_author_main_v1.json",
    )
    parser.add_argument("--profile-output")
    parser.add_argument("--style-contract-id", default="author_main_v1")
    parser.add_argument(
        "--approval-status",
        default="pending_review",
        choices=("draft", "pending_review", "approved", "frozen", "rejected"),
    )
    parser.add_argument("--author-notes", default="")
    args = parser.parse_args()

    if not _distinct_paths(
        args.contract_json_output,
        args.contract_output,
        args.metrics_output,
        args.profile_output,
    ):
        parser.error("output paths must be distinct")

    all_rows = read_jsonl(args.chapters)
    selected_rows = [row for row in all_rows if row.get("quality_tag") == "A"]
    profile = build_style_profile(selected_rows)
    profile["source_filter"] = {
        "total_rows": len(all_rows),
        "selected_rows": len(selected_rows),
        "skipped_rows": len(all_rows) - len(selected_rows),
        "quality_filter": "quality_tag=A",
    }
    source_corpus = {
        "path": str(Path(args.chapters)),
        "sha256": file_sha256(args.chapters),
        "quality_filter": "quality_tag=A",
        "row_count": len(all_rows),
        "selected_rows": len(selected_rows),
        "split_summary": _split_summary(all_rows),
    }
    asset = build_style_contract_asset(
        style_contract_id=args.style_contract_id,
        approval_status=args.approval_status,
        source_corpus=source_corpus,
        profile_metrics=profile,
        author_notes=args.author_notes,
    )

    write_style_contract_asset(args.contract_json_output, asset)
    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(args.contract_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.contract_output).write_text(
        render_style_contract_markdown(asset),
        encoding="utf-8",
    )
    if args.profile_output:
        profile_path = Path(args.profile_output)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(
            json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(f"wrote style contract JSON to {args.contract_json_output}")
    print(f"wrote style contract markdown to {args.contract_output}")
    print(f"wrote style metrics to {args.metrics_output}")
    if args.profile_output:
        print(f"wrote legacy style profile to {args.profile_output}")


if __name__ == "__main__":
    main()
