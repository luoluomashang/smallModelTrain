"""Rule-based output scoring and AI-trace detection.

These scores are not a replacement for human literary review. They provide
stable gates for length, required plot coverage, repetition, and generic AI
phrasing so failed samples can be triaged consistently.
"""

from __future__ import annotations

from small_model_train.text_utils import count_chinese_chars, repeated_ngram_ratio


AI_TRACE_PHRASES = [
    "空气仿佛凝固了",
    "难以言喻的情绪",
    "心中涌起一股复杂的情绪",
    "命运的齿轮开始转动",
    "眼神逐渐坚定起来",
    "嘴角勾起一抹弧度",
    "这一刻，他终于明白",
    "所有人都意识到",
    "一种前所未有的感觉",
]


def detect_ai_trace(text: str) -> dict:
    # The phrase list is a deterministic triage rule, not proof that a model generated the text.
    matches = [phrase for phrase in AI_TRACE_PHRASES if phrase in text]
    return {"count": len(matches), "matches": matches}


def _coverage(required: list[str], text: str) -> float:
    if not required:
        return 1.0
    hits = sum(1 for item in required if item and item in text)
    return hits / len(required)


def score_output(sample_id: str, card: dict, output: str) -> dict:
    char_count = count_chinese_chars(output)
    ai_trace = detect_ai_trace(output)
    repetition = repeated_ngram_ratio(output, n=4)
    must_include = card.get("must_include", [])
    must_not_include = card.get("must_not_include", [])
    include_coverage = _coverage(must_include, output)
    forbidden_hits = [item for item in must_not_include if item and item in output]

    failure_types: list[str] = []
    if char_count < 2000:
        failure_types.append("length_short")
    if char_count > 2500:
        failure_types.append("length_long")
    if any(marker in output for marker in ["【", "】", "章节结构", "以下是正文"]):
        failure_types.append("outline_leak")
    if include_coverage < 1.0:
        failure_types.append("must_include_missing")
    if forbidden_hits:
        failure_types.append("forbidden_violation")
    if repetition > 0.1:
        failure_types.append("repetition")
    if ai_trace["count"] > 0:
        failure_types.append("ai_trace")

    hard_gate_failures = {
        "length_short",
        "length_long",
        "outline_leak",
        "forbidden_violation",
        "repetition",
    }
    hard_gate_pass = not any(item in hard_gate_failures for item in failure_types)

    return {
        "id": sample_id,
        "char_count_zh": char_count,
        "hard_gate_pass": hard_gate_pass,
        "must_include_coverage": round(include_coverage, 4),
        "forbidden_hits": forbidden_hits,
        "ai_trace_count": ai_trace["count"],
        "ai_trace_matches": ai_trace["matches"],
        "repeated_ngram_ratio": round(repetition, 4),
        "failure_types": failure_types,
    }
