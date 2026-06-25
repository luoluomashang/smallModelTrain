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

    group_source_hashes: dict[str, list[str]] = {}
    for row in rows:
        group_key = _group_key(row)
        group_source_hashes.setdefault(group_key, []).append(_source_text_sha256(row))

    group_shas = {
        group_key: _group_sha256(seed, group_key, source_hashes)
        for group_key, source_hashes in group_source_hashes.items()
    }
    ranked_group_keys = sorted(group_shas, key=lambda group_key: group_shas[group_key])
    validation_group_keys = set(ranked_group_keys[:validation_count])
    sealed_group_keys = set(
        ranked_group_keys[validation_count : validation_count + sealed_count]
    )

    output: list[dict] = []
    for row in rows:
        updated = dict(row)
        group_key = _group_key(row)
        group_sha = group_shas[group_key]
        if group_key in validation_group_keys:
            split = "validation"
        elif group_key in sealed_group_keys:
            split = "sealed"
        else:
            split = "train"
        updated["split"] = split
        updated["group_id"] = f"group-{group_sha[:16]}"
        updated["group_sha256"] = group_sha
        output.append(updated)
    return output


def _group_key(row: dict) -> str:
    row_id = row.get("id")
    if row_id is not None and str(row_id).strip():
        return f"id:{row_id}"
    return f"text:{_source_text_sha256(row)}"


def _source_text_sha256(row: dict) -> str:
    text = row.get("text", "")
    if text is None:
        text = ""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _group_sha256(seed: int, group_key: str, source_hashes: list[str]) -> str:
    payload = "\n".join([str(seed), group_key, *sorted(source_hashes)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
