from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.artifact_manifest import file_sha256
from small_model_train.cards.card_validator import validate_formal_card_batch
from small_model_train.data.dataset_manifest import build_dataset_manifest, write_dataset_manifest
from small_model_train.data.dedup import find_near_duplicate_pairs
from small_model_train.io_utils import read_jsonl, write_jsonl
from small_model_train.schemas.chapter_execution_card import text_sha256
from small_model_train.sft_builder import build_formal_sft_rows, build_sft_rows
from small_model_train.style_contract import read_style_contract_asset


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


def _split_manifest(chapters: list[dict]) -> dict[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for chapter in chapters:
        split = str(chapter.get("split") or "")
        if not split:
            continue
        counts[split] = counts.get(split, 0) + 1
    return {"counts": counts}


def _card_hashes(cards: list[dict]) -> dict[str, str]:
    return {
        str(card["card_id"]): str(card["card_sha256"])
        for card in cards
        if "card_id" in card and "card_sha256" in card
    }


def _chapter_hashes(chapters: list[dict]) -> dict[str, str]:
    return {
        str(chapter["id"]): text_sha256(str(chapter.get("text") or ""))
        for chapter in chapters
        if "id" in chapter
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", required=True)
    parser.add_argument("--chapters", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-info-output")
    parser.add_argument("--dataset-manifest-output")
    parser.add_argument("--style-contract-json")
    parser.add_argument("--allow-draft-cards", action="store_true")
    args = parser.parse_args()

    if args.dataset_info_output and Path(args.output).resolve() == Path(args.dataset_info_output).resolve():
        parser.error("--dataset-info-output must not be the same path as --output")

    try:
        cards = read_jsonl(args.cards)
        chapters = read_jsonl(args.chapters)
        style_contract = None
        if args.style_contract_json:
            style_contract = read_style_contract_asset(args.style_contract_json)
            expected_chapters_sha256 = style_contract["source_corpus"]["sha256"]
            actual_chapters_sha256 = file_sha256(args.chapters)
            if actual_chapters_sha256 != expected_chapters_sha256:
                raise ValueError(
                    "chapters sha256 does not match style contract "
                    "source_corpus.sha256"
                )
            rows = build_formal_sft_rows(cards, chapters, style_contract)
        elif args.allow_draft_cards:
            rows = build_sft_rows(
                cards,
                chapters,
                require_approved_cards=False,
                style_contract=None,
            )
        elif not args.allow_draft_cards:
            raise ValueError("style contract JSON is required for formal SFT")

        write_jsonl(args.output, rows)
        if args.dataset_manifest_output:
            if style_contract is None or not args.style_contract_json:
                raise ValueError("style contract JSON is required for formal dataset manifest")
            leakage_report = validate_formal_card_batch(cards, chapters, style_contract)
            if not leakage_report["passed"]:
                raise ValueError("\n".join(leakage_report["errors"]))
            manifest = build_dataset_manifest(
                sft_dataset_path=args.output,
                chapters_path=args.chapters,
                cards_path=args.cards,
                style_contract_path=args.style_contract_json,
                style_contract=style_contract,
                split_manifest=_split_manifest(chapters),
                card_hashes=_card_hashes(cards),
                chapter_hashes=_chapter_hashes(chapters),
                leakage_report={
                    "passed": leakage_report["passed"],
                    "errors": leakage_report["errors"],
                },
                near_duplicate_report=find_near_duplicate_pairs(chapters),
                formal_dataset=True,
            )
            write_dataset_manifest(args.dataset_manifest_output, manifest)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    if args.dataset_info_output:
        _write_dataset_info(args.dataset_info_output, args.output)
    print(f"wrote {len(rows)} SFT rows to {args.output}")


if __name__ == "__main__":
    main()
