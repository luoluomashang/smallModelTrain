from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.io_utils import read_jsonl
from small_model_train.style_profile import build_style_profile, render_style_contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapters", required=True)
    parser.add_argument("--contract-output", required=True)
    parser.add_argument("--profile-output", required=True)
    args = parser.parse_args()

    rows = [row for row in read_jsonl(args.chapters) if row.get("quality_tag") == "A"]
    profile = build_style_profile(rows)
    Path(args.profile_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.profile_output).write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    Path(args.contract_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.contract_output).write_text(render_style_contract(profile), encoding="utf-8")
    print(f"wrote style profile to {args.profile_output}")
    print(f"wrote style contract to {args.contract_output}")


if __name__ == "__main__":
    main()
