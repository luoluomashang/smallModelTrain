from __future__ import annotations

import random


def split_rows(rows: list[dict], eval_count: int, seed: int = 20260617) -> list[dict]:
    if eval_count < 0:
        raise ValueError("eval_count must be >= 0")
    ids = [row["id"] for row in rows]
    rng = random.Random(seed)
    eval_ids = set(rng.sample(ids, k=min(eval_count, len(ids))))
    output: list[dict] = []
    for row in rows:
        updated = dict(row)
        updated["split"] = "eval" if row["id"] in eval_ids else "train"
        output.append(updated)
    return output
