from __future__ import annotations

from pathlib import Path
from typing import Any

from small_model_train.execution_cards import validate_execution_cards
from small_model_train.io_utils import read_jsonl
from small_model_train.sft_builder import render_sft_input


def default_inference_params() -> dict[str, Any]:
    return {
        "max_new_tokens": 5120,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.05,
    }


def render_eval_prompt(card: dict) -> str:
    return render_sft_input(card)


def build_generation_row(
    sample_id: str,
    output: str,
    model: str,
    params: dict,
) -> dict[str, Any]:
    return {
        "id": sample_id,
        "output": output,
        "model": model,
        "params": dict(params),
    }


def load_eval_cards(path: str | Path) -> list[dict]:
    cards_path = Path(path)
    if not cards_path.exists():
        raise ValueError(f"cards file is missing: {cards_path}")

    rows = read_jsonl(cards_path)
    if not rows:
        raise ValueError(f"cards file has no rows: {cards_path}")

    return validate_execution_cards(rows)
