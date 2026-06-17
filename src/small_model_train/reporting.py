from __future__ import annotations

from collections import Counter
from statistics import mean


def summarize_scores(scores: list[dict]) -> dict:
    sample_count = len(scores)
    hard_gate_passes = sum(1 for score in scores if score.get("hard_gate_pass"))
    failure_counts = Counter(
        failure
        for score in scores
        for failure in score.get("failure_types", [])
    )
    avg_chars = mean([score.get("char_count_zh", 0) for score in scores]) if scores else 0
    return {
        "sample_count": sample_count,
        "hard_gate_pass_rate": round(hard_gate_passes / sample_count, 4)
        if sample_count
        else 0,
        "avg_chinese_chars": round(avg_chars, 2),
        "failure_counts": dict(failure_counts),
    }


def build_markdown_report(
    title: str,
    scores: list[dict],
    config_snapshot: dict | None = None,
) -> str:
    summary = summarize_scores(scores)
    worst = sorted(
        scores,
        key=lambda score: (score.get("hard_gate_pass", False), score.get("char_count_zh", 0)),
    )[:10]
    lines = [
        f"# {title}",
        "",
        "## 配置快照",
        "```json",
        str(config_snapshot or {}).replace("'", '"'),
        "```",
        "",
        "## 汇总",
        f"- 样本数：{summary['sample_count']}",
        f"- 硬门槛通过率：{summary['hard_gate_pass_rate']}",
        f"- 平均中文汉字数：{summary['avg_chinese_chars']}",
        "",
        "## 失败类型分布",
    ]
    for failure, count in sorted(summary["failure_counts"].items()):
        lines.append(f"- {failure}: {count}")
    lines.extend(["", "## 最差样本", ""])
    for score in worst:
        lines.append(f"- {score.get('id')}: {', '.join(score.get('failure_types', [])) or 'pass'}")
    decision = "可以进入下一阶段" if summary["hard_gate_pass_rate"] >= 0.65 else "继续修数据和配置"
    lines.extend(["", "## 是否进入下一阶段", decision])
    return "\n".join(lines) + "\n"
