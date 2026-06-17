from __future__ import annotations

import json
from collections import Counter, defaultdict
from statistics import mean


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _format_number(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def summarize_scores(scores: list[dict]) -> dict:
    sample_count = len(scores)
    hard_gate_passes = sum(1 for score in scores if score.get("hard_gate_pass"))
    failure_counts = Counter(
        failure
        for score in scores
        for failure in score.get("failure_types", [])
    )
    avg_chars = mean([score.get("char_count_zh", 0) for score in scores]) if scores else 0
    score_totals = [
        float(score["score_total"])
        for score in scores
        if _is_number(score.get("score_total"))
    ]
    sub_score_values: defaultdict[str, list[float]] = defaultdict(list)
    for score in scores:
        sub_scores = score.get("sub_scores", {})
        if not isinstance(sub_scores, dict):
            continue
        for name, value in sub_scores.items():
            if _is_number(value):
                sub_score_values[name].append(float(value))
    return {
        "sample_count": sample_count,
        "hard_gate_pass_rate": round(hard_gate_passes / sample_count, 4)
        if sample_count
        else 0,
        "avg_chinese_chars": round(avg_chars, 2),
        "failure_counts": dict(failure_counts),
        "score_total_avg": round(mean(score_totals), 2) if score_totals else None,
        "sub_score_avgs": {
            name: round(mean(values), 2)
            for name, values in sorted(sub_score_values.items())
        },
    }


def build_markdown_report(
    title: str,
    scores: list[dict],
    config_snapshot: dict | None = None,
) -> str:
    config_snapshot = config_snapshot or {}
    summary = summarize_scores(scores)
    worst = sorted(
        scores,
        key=lambda score: (score.get("hard_gate_pass", False), score.get("char_count_zh", 0)),
    )[:10]
    top_preference_failures = sorted(
        [score for score in scores if score.get("failure_types", [])],
        key=lambda score: (
            score.get("hard_gate_pass", True),
            -len(score.get("failure_types", [])),
            str(score.get("id", "")),
        ),
    )[:20]
    lines = [
        f"# {title}",
        "",
        "## 配置快照",
        "```json",
        json.dumps(config_snapshot, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 汇总",
        f"- 样本数：{summary['sample_count']}",
        f"- 硬门槛通过率：{summary['hard_gate_pass_rate']}",
        f"- 平均中文汉字数：{summary['avg_chinese_chars']}",
        "",
        "## 数据与推理",
        "### Dataset",
        "```json",
        json.dumps(config_snapshot.get("dataset", {}), ensure_ascii=False, indent=2),
        "```",
        "### Inference",
        "```json",
        json.dumps(config_snapshot.get("inference", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "## 100 分制评分",
    ]
    if summary["score_total_avg"] is None:
        lines.append("- 100 分制均分：未提供")
    else:
        lines.append(f"- 100 分制均分：{_format_number(summary['score_total_avg'])}")
    if summary["sub_score_avgs"]:
        lines.append("- 分项均分：")
        for name, value in summary["sub_score_avgs"].items():
            lines.append(f"- {name}：{_format_number(value)}")
    else:
        lines.append("- 分项均分：未提供")
    lines.extend(
        [
            "",
            "## 偏好候选 Top 20",
        ]
    )
    if top_preference_failures:
        for score in top_preference_failures:
            lines.append(f"- {score.get('id')}: {', '.join(score.get('failure_types', []))}")
    else:
        lines.append("- 无")
    lines.extend(
        [
            "",
            "## 失败类型分布",
        ]
    )
    for failure, count in sorted(summary["failure_counts"].items()):
        lines.append(f"- {failure}: {count}")
    lines.extend(["", "## 最差样本", ""])
    for score in worst:
        lines.append(f"- {score.get('id')}: {', '.join(score.get('failure_types', [])) or 'pass'}")
    decision = "可以进入下一阶段" if summary["hard_gate_pass_rate"] >= 0.65 else "继续修数据和配置"
    lines.extend(["", "## 是否进入下一阶段", decision])
    return "\n".join(lines) + "\n"
