"""Deterministic train/eval splitting for cleaned chapter rows.

The split is seed-based so repeated runs produce the same fixed evaluation set.
That fixed set is what makes Stage 2 adapter comparisons meaningful.
"""

from __future__ import annotations

import hashlib
import random


def split_rows(rows: list[dict], eval_count: int, seed: int = 20260617) -> list[dict]:
    if eval_count < 0:
        raise ValueError("eval_count must be >= 0")
    # The eval subset is fixed by seed and index assignment; it is not re-sampled per training run.
    rng = random.Random(seed)
    eval_indexes = set(rng.sample(range(len(rows)), k=min(eval_count, len(rows))))
    output: list[dict] = []
    for index, row in enumerate(rows):
        updated = dict(row)
        updated["split"] = "eval" if index in eval_indexes else "train"
        output.append(updated)
    return output


def split_grouped_rows(
    rows: list[dict],
    validation_count: int,
    sealed_count: int,
    seed: int = 20260625,
) -> list[dict]:
    if validation_count < 0:
        raise ValueError("validation_count must be >= 0")
    if sealed_count < 0:
        raise ValueError("sealed_count must be >= 0")

    ranked = sorted(
        enumerate(rows),
        key=lambda item: _split_rank(item[1], seed, item[0]),
    )
    validation_indexes = {index for index, _row in ranked[:validation_count]}
    sealed_indexes = {
        index
        for index, _row in ranked[validation_count : validation_count + sealed_count]
    }

    output: list[dict] = []
    for index, row in enumerate(rows):
        updated = dict(row)
        group_sha = _group_sha256(row, seed, index)
        if index in validation_indexes:
            split = "validation"
        elif index in sealed_indexes:
            split = "sealed"
        else:
            split = "train"
        updated["split"] = split
        updated["group_id"] = f"group-{group_sha[:16]}"
        updated["group_sha256"] = group_sha
        output.append(updated)
    return output


def _split_rank(row: dict, seed: int, index: int) -> str:
    return _group_sha256(row, seed, index)


def _group_sha256(row: dict, seed: int, index: int) -> str:
    row_id = row.get("id", index)
    if row_id is None:
        row_id = index
    text = row.get("text", "")
    if text is None:
        text = ""
    payload = f"{seed}\n{index}\n{row_id}\n{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
