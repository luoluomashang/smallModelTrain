from __future__ import annotations

import pytest


def test_validate_style_defect_accepts_known_label_and_severity():
    from small_model_train.review.style_defects import validate_style_defect

    defect = validate_style_defect(
        {
            "label": "generic_phrase",
            "severity": "major",
            "evidence_text": "他知道，这一刻已经没有退路。",
            "evidence_start": 2,
            "evidence_end": 12,
            "suggested_fix": "改成具体动作承压。",
        }
    )

    assert defect["label"] == "generic_phrase"
    assert defect["severity"] == "major"


@pytest.mark.parametrize("label", ["unknown", "", 123])
def test_validate_style_defect_rejects_unknown_label(label):
    from small_model_train.review.style_defects import validate_style_defect

    with pytest.raises(ValueError, match="defects\\[0\\].label"):
        validate_style_defect(
            {
                "label": label,
                "severity": "minor",
                "evidence_text": "文本",
                "evidence_start": 0,
                "evidence_end": 2,
                "suggested_fix": "",
            }
        )


def test_validate_style_defect_rejects_empty_evidence_text():
    from small_model_train.review.style_defects import validate_style_defect

    with pytest.raises(ValueError, match="evidence_text"):
        validate_style_defect(
            {
                "label": "generic_phrase",
                "severity": "minor",
                "evidence_text": "   ",
                "evidence_start": 0,
                "evidence_end": 2,
                "suggested_fix": "",
            }
        )


def test_validate_style_defect_rejects_zero_width_evidence_span():
    from small_model_train.review.style_defects import validate_style_defect

    with pytest.raises(ValueError, match="evidence_end"):
        validate_style_defect(
            {
                "label": "generic_phrase",
                "severity": "minor",
                "evidence_text": "文本",
                "evidence_start": 1,
                "evidence_end": 1,
                "suggested_fix": "",
            }
        )


def test_summarize_style_defects_counts_labels_and_severity():
    from small_model_train.review.style_defects import summarize_style_defects

    summary = summarize_style_defects(
        [
            {"label": "generic_phrase", "severity": "minor"},
            {"label": "generic_phrase", "severity": "major"},
            {"label": "hook_blur", "severity": "blocker"},
        ]
    )

    assert summary["total_defects"] == 3
    assert summary["by_label"] == {"generic_phrase": 2, "hook_blur": 1}
    assert summary["by_severity"] == {"blocker": 1, "major": 1, "minor": 1}
