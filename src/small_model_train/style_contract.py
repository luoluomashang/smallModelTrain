"""Structured StyleContract assets for Stage 5B formal SFT gates."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from small_model_train.style_profile import render_style_contract


SCHEMA_VERSION = 1
APPROVAL_STATUSES = {"draft", "pending_review", "approved", "frozen", "rejected"}
APPROVED_FORMAL_STATUSES = {"approved", "frozen"}

OUTPUT_RULE = "只输出正文。不要输出提纲、小标题、解释、分析或提示语。"
LOWER_HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TOP_LEVEL_FIELD_ORDER = (
    "schema_version",
    "style_contract_id",
    "approval_status",
    "contract_sha256",
    "created_at",
    "source_corpus",
    "profile_metrics",
    "prompt_rules",
    "ai_taste_guardrails",
    "author_notes",
    "review",
)
NESTED_FIELD_ORDER = {
    "source_corpus": (
        "path",
        "sha256",
        "quality_filter",
        "row_count",
        "selected_rows",
        "split_summary",
    ),
    "prompt_rules": ("system_role", "style_contract_text", "output"),
    "ai_taste_guardrails": ("banned_phrases", "policy"),
    "review": ("reviewer", "reviewed_at", "review_notes"),
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def canonical_style_contract_sha256(asset: dict[str, Any]) -> str:
    canonical_asset = copy.deepcopy(asset)
    canonical_asset.pop("contract_sha256", None)
    payload = json.dumps(
        canonical_asset,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_style_contract_asset(
    *,
    style_contract_id: str,
    approval_status: str,
    source_corpus: dict[str, Any],
    profile_metrics: dict[str, Any],
    author_notes: str = "",
    created_at: str | None = None,
) -> dict[str, Any]:
    source_corpus_copy = copy.deepcopy(source_corpus)
    profile_metrics_copy = copy.deepcopy(profile_metrics)
    contract_text = render_style_contract(profile_metrics_copy)
    ai_taste = profile_metrics_copy.get("ai_taste", {})
    phrase_hits = ai_taste.get("phrase_hits", {}) if isinstance(ai_taste, dict) else {}
    banned_phrases = list(phrase_hits.keys()) if isinstance(phrase_hits, dict) else []
    asset: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "style_contract_id": style_contract_id,
        "approval_status": approval_status,
        "contract_sha256": "",
        "created_at": created_at or utc_now_iso(),
        "source_corpus": source_corpus_copy,
        "profile_metrics": profile_metrics_copy,
        "prompt_rules": {
            "system_role": "你是作者的正文执行器，只负责根据章节执行卡写正文。",
            "style_contract_text": contract_text,
            "output": OUTPUT_RULE,
        },
        "ai_taste_guardrails": {
            "banned_phrases": banned_phrases,
            "policy": "避免使用语料中标记的 AI 味短语；如原文统计出现命中，生成时应继续压低。",
        },
        "author_notes": author_notes,
        "review": {
            "reviewer": "",
            "reviewed_at": "",
            "review_notes": "",
        },
    }
    asset["contract_sha256"] = canonical_style_contract_sha256(asset)
    return validate_style_contract_asset(asset)


def validate_style_contract_asset(asset: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(asset, dict):
        raise ValueError("style contract asset must be a JSON object")

    required_fields = {
        "schema_version",
        "style_contract_id",
        "approval_status",
        "contract_sha256",
        "created_at",
        "source_corpus",
        "profile_metrics",
        "prompt_rules",
        "ai_taste_guardrails",
        "author_notes",
        "review",
    }
    missing = sorted(required_fields - set(asset))
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")

    if asset["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if not isinstance(asset["style_contract_id"], str) or not asset["style_contract_id"].strip():
        raise ValueError("style_contract_id must be a non-empty string")
    if not isinstance(asset["created_at"], str) or not asset["created_at"].strip():
        raise ValueError("created_at must be a non-empty string")
    if not isinstance(asset["author_notes"], str):
        raise ValueError("author_notes must be a string")
    if asset["approval_status"] not in APPROVAL_STATUSES:
        raise ValueError(
            "approval_status must be one of: "
            + ", ".join(sorted(APPROVAL_STATUSES))
        )
    if not _is_lower_hex_sha256(asset["contract_sha256"]):
        raise ValueError("contract_sha256 must be a 64-character lowercase hex string")

    for field in (
        "source_corpus",
        "profile_metrics",
        "prompt_rules",
        "ai_taste_guardrails",
        "review",
    ):
        if not isinstance(asset[field], dict):
            raise ValueError(f"{field} must be a JSON object")

    _validate_source_corpus(asset["source_corpus"])
    _validate_prompt_rules(asset["prompt_rules"])
    _validate_ai_taste_guardrails(asset["ai_taste_guardrails"])
    _validate_review(asset["review"])

    expected_sha256 = canonical_style_contract_sha256(asset)
    if asset["contract_sha256"] != expected_sha256:
        raise ValueError("contract_sha256 mismatch")

    return asset


def is_contract_approved_for_formal_sft(asset: dict[str, Any]) -> bool:
    validated = validate_style_contract_asset(asset)
    return validated["approval_status"] in APPROVED_FORMAL_STATUSES


def write_style_contract_asset(path: str | Path, asset: dict[str, Any]) -> None:
    validated = validate_style_contract_asset(asset)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_ordered_style_contract_asset(validated), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_style_contract_asset(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    try:
        asset = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{input_path} is not valid JSON") from exc
    return validate_style_contract_asset(asset)


def render_style_contract_markdown(asset: dict[str, Any]) -> str:
    validated = validate_style_contract_asset(asset)
    source = validated["source_corpus"]
    prompt_rules = validated["prompt_rules"]
    return "\n".join(
        [
            f"# Style Contract {validated['style_contract_id']}",
            "",
            f"approval_status: {validated['approval_status']}",
            f"contract_sha256: {validated['contract_sha256']}",
            "",
            "## Source Corpus",
            "",
            f"- path: {source.get('path', '')}",
            f"- sha256: {source.get('sha256', '')}",
            f"- selected_rows: {source.get('selected_rows', '')}",
            "",
            "## Prompt Rules",
            "",
            f"system_role: {prompt_rules.get('system_role', '')}",
            "",
            str(prompt_rules.get("style_contract_text", "")),
            "",
            f"output: {prompt_rules.get('output', '')}",
            "",
            "## Author Notes",
            "",
            str(validated.get("author_notes", "")),
            "",
        ]
    )


def _is_lower_hex_sha256(value: object) -> bool:
    return isinstance(value, str) and LOWER_HEX_SHA256_RE.fullmatch(value) is not None


def _require_fields(section_name: str, section: dict[str, Any], fields: tuple[str, ...]) -> None:
    missing = [field for field in fields if field not in section]
    if missing:
        raise ValueError(f"{section_name}.{missing[0]} is required")


def _require_non_empty_string(section_name: str, section: dict[str, Any], field: str) -> None:
    value = section.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{section_name}.{field} must be a non-empty string")


def _validate_source_corpus(source_corpus: dict[str, Any]) -> None:
    _require_fields("source_corpus", source_corpus, ("path", "sha256", "selected_rows"))
    if not isinstance(source_corpus["path"], str):
        raise ValueError("source_corpus.path must be a string")
    if not _is_lower_hex_sha256(source_corpus["sha256"]):
        raise ValueError("source_corpus.sha256 must be a 64-character lowercase hex string")
    if not isinstance(source_corpus["selected_rows"], int) or isinstance(
        source_corpus["selected_rows"], bool
    ):
        raise ValueError("source_corpus.selected_rows must be an int")


def _validate_prompt_rules(prompt_rules: dict[str, Any]) -> None:
    _require_fields(
        "prompt_rules",
        prompt_rules,
        ("system_role", "style_contract_text", "output"),
    )
    _require_non_empty_string("prompt_rules", prompt_rules, "system_role")
    _require_non_empty_string("prompt_rules", prompt_rules, "style_contract_text")
    _require_non_empty_string("prompt_rules", prompt_rules, "output")


def _validate_ai_taste_guardrails(guardrails: dict[str, Any]) -> None:
    _require_fields("ai_taste_guardrails", guardrails, ("banned_phrases", "policy"))
    banned_phrases = guardrails["banned_phrases"]
    if not isinstance(banned_phrases, list) or not all(
        isinstance(phrase, str) for phrase in banned_phrases
    ):
        raise ValueError("ai_taste_guardrails.banned_phrases must be a list of strings")
    _require_non_empty_string("ai_taste_guardrails", guardrails, "policy")


def _validate_review(review: dict[str, Any]) -> None:
    _require_fields("review", review, ("reviewer", "reviewed_at", "review_notes"))
    for field in ("reviewer", "reviewed_at", "review_notes"):
        if not isinstance(review[field], str):
            raise ValueError(f"review.{field} must be a string")


def _ordered_style_contract_asset(asset: dict[str, Any]) -> dict[str, Any]:
    return _ordered_dict(asset, TOP_LEVEL_FIELD_ORDER)


def _ordered_dict(values: dict[str, Any], field_order: tuple[str, ...]) -> dict[str, Any]:
    ordered: dict[str, Any] = {}
    for field in field_order:
        if field in values:
            nested_order = NESTED_FIELD_ORDER.get(field, ())
            ordered[field] = _ordered_value(values[field], nested_order)
    for field in sorted(key for key in values if key not in ordered):
        ordered[field] = _ordered_value(values[field], ())
    return ordered


def _ordered_value(value: Any, field_order: tuple[str, ...]) -> Any:
    if isinstance(value, dict):
        return _ordered_dict(value, field_order or tuple(sorted(value)))
    if isinstance(value, list):
        return [_ordered_value(item, ()) for item in value]
    return value
