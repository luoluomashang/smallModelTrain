from __future__ import annotations

from small_model_train.sft_builder import render_sft_input


def build_preference_candidates(
    cards: list[dict],
    outputs: list[dict],
    scores: list[dict],
) -> list[dict]:
    cards_by_id = {row["id"]: row for row in cards}
    outputs_by_id = {row["id"]: row for row in outputs}
    rows: list[dict] = []
    for score in scores:
        failure_types = score.get("failure_types", [])
        if not failure_types:
            continue
        sample_id = score["id"]
        card = cards_by_id.get(sample_id, {})
        output = outputs_by_id.get(sample_id, {})
        reject_type = failure_types[0] if failure_types else "unknown"
        rows.append(
            {
                "id": sample_id,
                "prompt": card["prompt"] if "prompt" in card else render_sft_input(card),
                "rejected": output.get("output", output.get("text", "")),
                "reject_type": reject_type,
                "chosen": "",
                "source": "failed_eval",
            }
        )
    return rows
