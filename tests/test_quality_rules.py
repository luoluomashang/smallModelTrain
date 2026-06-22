from __future__ import annotations

from small_model_train.quality_rules import detect_quality_issues


def test_detect_quality_issues_flags_markdown_and_disclaimer():
    result = detect_quality_issues(
        {},
        "> 林默走进大厅。\n作为AI，我无法保证以上内容完全准确。",
    )

    assert "markdown_residue" in result["issues"]
    assert "disclaimer_residue" in result["issues"]


def test_detect_quality_issues_flags_semantic_repetition():
    text = (
        "林默终于明白自己不能退。他知道自己必须向前。他清楚自己不能退缩。"
        "林默终于明白自己不能退。他知道自己必须向前。他清楚自己不能退缩。"
        "林默终于明白自己不能退。他知道自己必须向前。他清楚自己不能退缩。"
    )

    result = detect_quality_issues({}, text)

    assert "semantic_repetition" in result["issues"]


def test_detect_quality_issues_flags_weak_ending():
    result = detect_quality_issues({}, "林默看向窗外，心里有了某种决定")

    assert "unnatural_ending" in result["issues"]


def test_detect_quality_issues_flags_missing_external_payoff_terms():
    card = {"payoff_beat": "合同证据让岳家闭嘴", "ending_hook": "真正买家出现"}
    result = detect_quality_issues(card, "林默沉默了很久，最后只是握紧拳头。")

    assert "no_visible_payoff" in result["issues"]
    assert "weak_ending_hook" in result["issues"]


def test_detect_quality_issues_flags_meta_evaluation_residue():
    result = detect_quality_issues({}, "最终确认：本章完成，符合要求。")

    assert "meta_evaluation_residue" in result["issues"]


def test_detect_quality_issues_flags_generic_ai_phrase():
    result = detect_quality_issues({}, "林默深吸一口气。岳家人也深吸一口气。")

    assert "generic_ai_phrase" in result["issues"]
    assert result["details"]["generic_phrase_hits"] == ["深吸一口气"]


def test_detect_quality_issues_flags_padding_to_length():
    text = "林默终于明白自己不能退" * 223 + "。"

    result = detect_quality_issues({}, text)

    assert "semantic_repetition" in result["issues"]
    assert "padding_to_length" in result["issues"]
