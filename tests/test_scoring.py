from __future__ import annotations

import subprocess
import sys

from small_model_train.execution_cards import DEFAULT_TARGET_PLATFORM
from small_model_train.io_utils import read_jsonl, write_jsonl
from small_model_train.scoring import detect_ai_trace, score_output

def _execution_card(sample_id: str = "case1") -> dict:
    return {
        "id": sample_id,
        "target_platform": DEFAULT_TARGET_PLATFORM,
        "genre_tags": ["xuanhuan", "system"],
        "style_contract": "男频爽文，节奏紧，强钩子。",
        "chapter_goal": "主角发现系统任务并反击压迫者。",
        "chapter_structure": [
            {
                "step": 1,
                "name": "开局压迫",
                "goal": "建立困境和目标",
                "estimated_chars": "800",
            }
        ],
        "conflict_beat": "旧势力当众羞辱主角。",
        "payoff_beat": "主角用系统奖励完成反杀。",
        "must_include": ["系统面板", "当众反击"],
        "must_not_include": ["女频误会流"],
        "ending_hook": "新的悬赏任务出现。",
        "target_word_count": 1800,
    }

def test_detect_ai_trace_counts_known_phrases():
    result = detect_ai_trace("空气仿佛凝固了，他心中涌起一股复杂的情绪。")
    assert result["count"] == 2
    assert "空气仿佛凝固了" in result["matches"]


def test_score_output_applies_hard_gates_and_failure_types():
    card = {
        "must_include": ["加钱"],
        "must_not_include": ["真相是他父亲"],
    }
    output = "林默说加钱。真相是他父亲。"
    score = score_output("case1", card, output)
    assert score["hard_gate_pass"] is False
    assert "forbidden_violation" in score["failure_types"]
    assert score["must_include_coverage"] == 1.0


def test_score_output_marks_length_short():
    score = score_output("case2", {"must_include": [], "must_not_include": []}, "太短。")
    assert "length_short" in score["failure_types"]


def test_score_output_marks_outline_leak():
    output = "以下是正文：" + "林默加钱。" * 500
    score = score_output("case3", {"must_include": [], "must_not_include": []}, output)
    assert "outline_leak" in score["failure_types"]
    assert score["hard_gate_pass"] is False


def test_score_output_marks_missing_must_include():
    card = {"must_include": ["加钱", "雨夜"], "must_not_include": []}
    score = score_output("case4", card, "林默走进房间。" * 400)
    assert "must_include_missing" in score["failure_types"]
    assert score["must_include_coverage"] == 0.0


def test_score_outputs_cli_reads_cards_and_outputs_jsonl(tmp_path):
    cards_path = tmp_path / "cards.jsonl"
    outputs_path = tmp_path / "outputs.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    write_jsonl(
        cards_path,
        [
            dict(
                _execution_card("case1"),
                must_include=["加钱"],
                must_not_include=["真相是他父亲"],
                payoff_beat="加钱",
                ending_hook="真相是他父亲",
            )
        ],
    )
    write_jsonl(outputs_path, [{"id": "case1", "output": "林默说加钱。真相是他父亲。"}])

    result = subprocess.run(
        [
            sys.executable,
            "scripts/score_outputs.py",
            "--cards",
            str(cards_path),
            "--outputs",
            str(outputs_path),
            "--output",
            str(scores_path),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    rows = read_jsonl(scores_path)
    assert len(rows) == 1
    assert rows[0]["id"] == "case1"
    assert rows[0]["hard_gate_pass"] is False
    assert "forbidden_violation" in rows[0]["failure_types"]


def test_score_outputs_cli_rejects_raw_cards(tmp_path):
    cards_path = tmp_path / "cards.jsonl"
    outputs_path = tmp_path / "outputs.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    write_jsonl(cards_path, [{"id": "case1", "text": "原文"}])
    write_jsonl(outputs_path, [{"id": "case1", "output": "正文"}])

    result = subprocess.run(
        [
            sys.executable,
            "scripts/score_outputs.py",
            "--cards",
            str(cards_path),
            "--outputs",
            str(outputs_path),
            "--output",
            str(scores_path),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    assert "missing execution-card fields" in result.stderr
    assert not scores_path.exists()


def test_score_outputs_cli_rejects_unknown_output_id(tmp_path):
    cards_path = tmp_path / "cards.jsonl"
    outputs_path = tmp_path / "outputs.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    write_jsonl(cards_path, [_execution_card("case1")])
    write_jsonl(outputs_path, [{"id": "case-missing", "output": "正文"}])

    result = subprocess.run(
        [
            sys.executable,
            "scripts/score_outputs.py",
            "--cards",
            str(cards_path),
            "--outputs",
            str(outputs_path),
            "--output",
            str(scores_path),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    assert "output id not found in cards: case-missing" in result.stderr
    assert not scores_path.exists()


def test_score_output_merges_quality_rule_failures():
    card = {
        "must_include": [],
        "must_not_include": [],
        "payoff_beat": "合同证据让岳家闭嘴",
        "ending_hook": "真正买家出现",
    }
    output = "> 林默沉默了很久，最后只是握紧拳头。"

    score = score_output("case-quality", card, output)

    assert "markdown_residue" in score["failure_types"]
    assert "no_visible_payoff" in score["failure_types"]
    assert score["hard_gate_pass"] is False


def _assert_quality_hard_gate(flag: str, output: str, card: dict | None = None):
    score = score_output(
        f"case-{flag}",
        card or {"must_include": [], "must_not_include": []},
        output,
    )

    assert flag in score["failure_types"]
    assert score["hard_gate_pass"] is False
    return score


def test_score_output_hard_gates_disclaimer_residue():
    _assert_quality_hard_gate("disclaimer_residue", "作为AI，我无法保证这段内容完全准确。")


def test_score_output_hard_gates_meta_evaluation_residue():
    _assert_quality_hard_gate("meta_evaluation_residue", "最终确认：本章完成，符合要求。")


def test_score_output_hard_gates_semantic_repetition_and_returns_details():
    output = (
        "林默终于明白自己不能退。他知道自己必须向前。他清楚自己不能退缩。"
        "林默终于明白自己不能退。他知道自己必须向前。他清楚自己不能退缩。"
        "林默终于明白自己不能退。他知道自己必须向前。他清楚自己不能退缩。"
    )

    score = _assert_quality_hard_gate("semantic_repetition", output)

    assert score["quality_rule_details"]["repeated_runs"]


def test_score_output_hard_gates_padding_to_length():
    output = "林默终于明白自己不能退" * 223 + "。"

    _assert_quality_hard_gate("padding_to_length", output)


def test_score_output_hard_gates_unnatural_ending():
    _assert_quality_hard_gate("unnatural_ending", "林默看向窗外，心里有了某种决定")


def test_score_output_hard_gates_weak_ending_hook():
    card = {
        "must_include": [],
        "must_not_include": [],
        "ending_hook": "真正买家出现",
    }

    _assert_quality_hard_gate("weak_ending_hook", "林默握紧拳头。", card)
