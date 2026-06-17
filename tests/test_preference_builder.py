from __future__ import annotations

import subprocess
import sys

from small_model_train.io_utils import read_jsonl, write_jsonl
from small_model_train.preference_builder import build_preference_candidates


def test_build_preference_candidates_uses_failed_scores():
    cards = [{"id": "c1", "prompt": "卡", "style_contract": "契约"}]
    outputs = [{"id": "c1", "output": "坏正文"}]
    scores = [{"id": "c1", "hard_gate_pass": False, "failure_types": ["ai_trace"]}]

    rows = build_preference_candidates(cards, outputs, scores)

    assert rows == [
        {
            "id": "c1",
            "prompt": "卡",
            "rejected": "坏正文",
            "reject_type": "ai_trace",
            "chosen": "",
            "source": "failed_eval",
        }
    ]


def test_build_preference_candidates_skips_passing_scores():
    cards = [{"id": "c1", "prompt": "卡", "style_contract": "契约"}]
    outputs = [{"id": "c1", "output": "正文"}]
    scores = [{"id": "c1", "hard_gate_pass": True, "failure_types": []}]

    assert build_preference_candidates(cards, outputs, scores) == []


def test_build_preference_candidates_keeps_soft_failures_even_when_hard_gate_passes():
    cards = [{"id": "c1", "prompt": "卡", "style_contract": "契约"}]
    outputs = [{"id": "c1", "output": "像AI总结一样的正文"}]
    scores = [{"id": "c1", "hard_gate_pass": True, "failure_types": ["ai_trace"]}]

    rows = build_preference_candidates(cards, outputs, scores)

    assert len(rows) == 1
    assert rows[0]["reject_type"] == "ai_trace"
    assert rows[0]["rejected"] == "像AI总结一样的正文"


def test_build_preference_candidates_renders_prompt_when_missing():
    cards = [
        {
            "id": "c1",
            "style_contract": "契约",
            "previous_summary": "前情",
            "chapter_goal": "目标",
            "target_word_count": "2000-2500中文汉字",
            "chapter_structure": [],
            "character_states": [],
            "must_include": [],
            "must_not_include": [],
            "ending_hook": "",
        }
    ]
    outputs = [{"id": "c1", "output": "坏正文"}]
    scores = [{"id": "c1", "hard_gate_pass": False, "failure_types": ["format"]}]

    rows = build_preference_candidates(cards, outputs, scores)

    assert "【输出要求】" in rows[0]["prompt"]
    assert "只输出正文" in rows[0]["prompt"]


def test_build_preference_candidates_preserves_explicit_empty_prompt():
    cards = [{"id": "c1", "prompt": ""}]
    outputs = [{"id": "c1", "output": "坏正文"}]
    scores = [{"id": "c1", "hard_gate_pass": False, "failure_types": ["format"]}]

    rows = build_preference_candidates(cards, outputs, scores)

    assert rows[0]["prompt"] == ""


def test_build_preference_candidates_uses_text_when_output_missing():
    cards = [{"id": "c1", "prompt": "卡", "style_contract": "契约"}]
    outputs = [{"id": "c1", "text": "文本正文"}]
    scores = [{"id": "c1", "hard_gate_pass": False, "failure_types": ["ai_trace"]}]

    rows = build_preference_candidates(cards, outputs, scores)

    assert rows[0]["rejected"] == "文本正文"


def test_build_preference_dataset_cli_writes_jsonl(tmp_path):
    cards_path = tmp_path / "cards.jsonl"
    outputs_path = tmp_path / "outputs.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    output_path = tmp_path / "preference.jsonl"
    write_jsonl(cards_path, [{"id": "c1", "prompt": "卡", "style_contract": "契约"}])
    write_jsonl(outputs_path, [{"id": "c1", "output": "坏正文"}])
    write_jsonl(
        scores_path,
        [{"id": "c1", "hard_gate_pass": False, "failure_types": ["ai_trace"]}],
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/build_preference_dataset.py",
            "--cards",
            str(cards_path),
            "--outputs",
            str(outputs_path),
            "--scores",
            str(scores_path),
            "--output",
            str(output_path),
        ],
        check=True,
    )

    rows = read_jsonl(output_path)
    assert rows == [
        {
            "id": "c1",
            "prompt": "卡",
            "rejected": "坏正文",
            "reject_type": "ai_trace",
            "chosen": "",
            "source": "failed_eval",
        }
    ]
