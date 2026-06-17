from __future__ import annotations

import random


def split_rows(rows: list[dict], eval_count: int, seed: int = 20260617) -> list[dict]:
    if eval_count < 0:
        raise ValueError("eval_count must be >= 0")
    rng = random.Random(seed)
    eval_indexes = set(rng.sample(range(len(rows)), k=min(eval_count, len(rows))))
    output: list[dict] = []
    for index, row in enumerate(rows):
        updated = dict(row)
        updated["split"] = "eval" if index in eval_indexes else "train"
        output.append(updated)
    return output
