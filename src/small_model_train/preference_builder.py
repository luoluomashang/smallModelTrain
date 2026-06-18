"""从失败样本中挖掘偏好训练候选。

偏好数据的 rejected 来自固定评测里的坏输出；chosen 先留空，等待人工轻修
或后续更强模型生成对照答案。这样第一阶段能先把“哪些样本值得修”收集好。
"""

from __future__ import annotations

from small_model_train.sft_builder import render_sft_input


def build_preference_candidates(
    cards: list[dict],
    outputs: list[dict],
    scores: list[dict],
) -> list[dict]:
    """把带 failure_types 的评分记录转成偏好候选行。"""

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
                # 有些 eval card 已经保存了完整 prompt；没有时按 SFT 模板现场渲染。
                "prompt": card["prompt"] if "prompt" in card else render_sft_input(card),
                "rejected": output.get("output", output.get("text", "")),
                "reject_type": reject_type,
                "chosen": "",
                "source": "failed_eval",
            }
        )
    return rows
