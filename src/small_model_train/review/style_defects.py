from __future__ import annotations

from typing import Any


DEFECT_LABELS = {
    "generic_phrase",
    "explanation_voice",
    "summary_narration",
    "empty_intensity",
    "repeated_psychology",
    "dialogue_flatness",
    "payoff_blur",
    "hook_blur",
    "style_contract_drift",
    "plan_execution_regression",
}
DEFECT_SEVERITIES = {"minor", "major", "blocker"}


def validate_style_defect(defect: dict[str, Any], *, index: int = 0) -> dict[str, Any]:
    if not isinstance(defect, dict):
        raise ValueError(f"defects[{index}] must be a JSON object")
    label = defect.get("label")
    if label not in DEFECT_LABELS:
        raise ValueError(f"defects[{index}].label must be one of: {', '.join(sorted(DEFECT_LABELS))}")
    severity = defect.get("severity")
    if severity not in DEFECT_SEVERITIES:
        raise ValueError(
            f"defects[{index}].severity must be one of: {', '.join(sorted(DEFECT_SEVERITIES))}"
        )
    evidence_text = defect.get("evidence_text")
    if not isinstance(evidence_text, str) or not evidence_text.strip():
        raise ValueError(f"defects[{index}].evidence_text must be a non-empty string")
    if not isinstance(defect.get("suggested_fix"), str):
        raise ValueError(f"defects[{index}].suggested_fix must be a string")
    for field in ("evidence_start", "evidence_end"):
        value = defect.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"defects[{index}].{field} must be an int >= 0")
    if defect["evidence_end"] <= defect["evidence_start"]:
        raise ValueError(f"defects[{index}].evidence_end must be > evidence_start")
    return defect


def validate_style_defects(defects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(defects, list):
        raise ValueError("defects must be a list")
    return [validate_style_defect(defect, index=index) for index, defect in enumerate(defects)]


def summarize_style_defects(defects: list[dict[str, Any]]) -> dict[str, Any]:
    by_label: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    for defect in defects:
        label = str(defect.get("label"))
        severity = str(defect.get("severity"))
        by_label[label] = by_label.get(label, 0) + 1
        by_severity[severity] = by_severity.get(severity, 0) + 1
    return {
        "total_defects": len(defects),
        "by_label": dict(sorted(by_label.items())),
        "by_severity": dict(sorted(by_severity.items())),
    }
