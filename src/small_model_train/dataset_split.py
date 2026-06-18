"""确定性 train/eval 切分。

第一阶段需要固定评测集：同一份输入、同一个 seed，每次都切出同样的 eval
样本，这样后续 baseline、SFT v1、SFT v2 的报告才可比较。
"""

from __future__ import annotations

import random


def split_rows(rows: list[dict], eval_count: int, seed: int = 20260617) -> list[dict]:
    """按位置抽取 eval 样本，并在每行写入 split 字段。

    使用行号而不是 id 抽样，是为了防止上游原稿意外产生重复 id 时，一个 id
    同时影响多行。输出保持原顺序，只改每行的 split 标记。
    """

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
