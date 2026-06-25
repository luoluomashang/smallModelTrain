from __future__ import annotations

import re
from typing import Any

CHINESE_RE = re.compile(r"[\u4e00-\u9fff]+")


def normalized_chinese_text(text: str) -> str:
    return "".join(CHINESE_RE.findall(text))


def chinese_shingles(text: str, shingle_size: int = 12) -> set[str]:
    normalized = normalized_chinese_text(text)
    if not normalized:
        return set()
    if len(normalized) <= shingle_size:
        return {normalized}
    return {
        normalized[index : index + shingle_size]
        for index in range(0, len(normalized) - shingle_size + 1)
    }


def jaccard_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return round(len(left & right) / len(left | right), 4)


def find_near_duplicate_pairs(
    rows: list[dict[str, Any]],
    threshold: float = 0.82,
    shingle_size: int = 12,
) -> list[dict[str, Any]]:
    fingerprints = [
        (
            str(row.get("id") or ""),
            str(row.get("split") or ""),
            chinese_shingles(str(row.get("text") or ""), shingle_size),
        )
        for row in rows
    ]

    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(fingerprints):
        for right in fingerprints[left_index + 1 :]:
            left_id, left_split, left_shingles = left
            right_id, right_split, right_shingles = right
            if left_split == right_split:
                continue
            overlap = jaccard_overlap(left_shingles, right_shingles)
            if overlap >= threshold:
                pairs.append(
                    {
                        "left_id": left_id,
                        "left_split": left_split,
                        "right_id": right_id,
                        "right_split": right_split,
                        "overlap": overlap,
                    }
                )
    return pairs
