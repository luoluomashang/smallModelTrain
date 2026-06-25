from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.cards.card_compiler import compile_chapter_execution_card
from small_model_train.io_utils import read_jsonl
from small_model_train.schemas.chapter_execution_card import write_chapter_execution_cards
from small_model_train.style_contract import read_style_contract_asset


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile draft cards into formal chapter execution cards.")
    parser.add_argument("--cards", required=True)
    parser.add_argument("--chapters", required=True)
    parser.add_argument("--style-contract-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        _require_jsonl_file(args.cards, "cards")
        _require_jsonl_file(args.chapters, "chapters")
        draft_cards = read_jsonl(args.cards)
        chapters = {chapter.get("id"): chapter for chapter in read_jsonl(args.chapters)}
        style_contract = read_style_contract_asset(args.style_contract_json)
        formal_cards = []
        for draft_card in draft_cards:
            chapter = _chapter_for(chapters, draft_card)
            formal_cards.append(
                compile_chapter_execution_card(
                    draft_card=draft_card,
                    chapter=chapter,
                    style_contract=style_contract,
                    group_id=str(draft_card.get("group_id") or f"group-{draft_card.get('id')}"),
                    split=str(chapter.get("split") or ""),
                )
            )
        write_chapter_execution_cards(args.output, formal_cards)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {len(formal_cards)} chapter execution cards to {args.output}")
    return 0


def _chapter_for(chapters: dict[Any, dict], draft_card: dict) -> dict:
    chapter_id = draft_card.get("id")
    chapter = chapters.get(chapter_id)
    if chapter is None:
        raise ValueError(f"chapter not found for draft card id: {chapter_id}")
    return chapter


def _require_jsonl_file(path: str, label: str) -> None:
    if not Path(path).is_file():
        raise ValueError(f"{label} JSONL not found: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
