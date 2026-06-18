import json
import subprocess
from pathlib import Path

import pytest

from small_model_train.io_utils import write_jsonl
from small_model_train.stage2_inference import (
    build_generation_row,
    default_inference_params,
    render_eval_prompt,
)


def _card(sample_id: str = "eval-1") -> dict:
    return {
        "id": sample_id,
        "style_contract": "短句推进，动作细节清楚。",
        "previous_summary": "主角刚抵达旧城。",
        "chapter_goal": "让主角发现密室钥匙。",
        "chapter_structure": [
            {
                "step": 1,
                "name": "搜寻",
                "goal": "在书房找到线索",
                "estimated_chars": "800",
            }
        ],
        "character_states": [
            {
                "name": "林照",
                "state": "谨慎但兴奋",
                "speech_style": "简短直接",
            }
        ],
        "must_include": ["铜钥匙", "雨声"],
        "must_not_include": ["解释设定"],
        "ending_hook": "门后传来第二个人的呼吸声。",
        "target_word_count": "2000-2500中文汉字",
        "source_text": "这是一段不应该泄漏到提示词里的原文内容而且足够长",
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_render_eval_prompt_contains_card_fields():
    prompt = render_eval_prompt(_card())

    assert "让主角发现密室钥匙" in prompt
    assert "短句推进，动作细节清楚。" in prompt
    assert "铜钥匙" in prompt
    assert "雨声" in prompt
    assert "不应该泄漏到提示词" not in prompt


def test_build_generation_row_uses_fixed_schema():
    params = {"temperature": 0.7}

    row = build_generation_row("eval-1", "正文", "sft_v1", params)

    assert row == {
        "id": "eval-1",
        "output": "正文",
        "model": "sft_v1",
        "params": params,
    }


def test_default_inference_params_match_stage2_eval_defaults():
    assert default_inference_params() == {
        "max_new_tokens": 5120,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.05,
    }


def test_dry_run_cli_writes_generation_rows_without_subprocess(
    tmp_path: Path,
    monkeypatch,
):
    from scripts import run_eval_inference

    cards = [_card("eval-1"), _card("eval-2")]
    cards_path = tmp_path / "cards.jsonl"
    output_path = tmp_path / "generated.jsonl"
    write_jsonl(cards_path, cards)

    def fail_run(*args, **kwargs):
        raise AssertionError("dry-run must not call subprocess.run")

    monkeypatch.setattr(run_eval_inference.subprocess, "run", fail_run)

    exit_code = run_eval_inference.main(
        [
            "--cards",
            str(cards_path),
            "--output",
            str(output_path),
            "--model-name",
            "dry_eval",
            "--dry-run",
        ]
    )

    rows = _read_jsonl(output_path)
    assert exit_code == 0
    assert len(rows) == 2
    assert rows[0]["id"] == "eval-1"
    assert rows[0]["model"] == "dry_eval"
    assert rows[0]["output"].startswith("[DRY RUN] ")
    assert rows[0]["params"] == default_inference_params()
    assert set(rows[0]) == {"id", "output", "model", "params"}


def test_non_dry_run_failure_writes_logs_events_and_classification(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    from scripts import run_eval_inference

    output_path = tmp_path / "generated.jsonl"
    event_log = tmp_path / "events.jsonl"
    stderr_log = tmp_path / "stderr.log"
    seen_commands = []

    completed = subprocess.CompletedProcess(
        args=["python", "worker"],
        returncode=23,
        stdout="",
        stderr="RuntimeError: CUDA out of memory",
    )

    def fake_run(command, **kwargs):
        seen_commands.append((command, kwargs))
        return completed

    monkeypatch.setattr(run_eval_inference.subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as exc_info:
        run_eval_inference.main(
            [
                "--cards",
                str(tmp_path / "cards.jsonl"),
                "--model-dir",
                "model",
                "--adapter-dir",
                "adapter",
                "--output",
                str(output_path),
                "--event-log",
                str(event_log),
                "--stderr-log",
                str(stderr_log),
            ]
        )

    assert exc_info.value.code == 23
    assert seen_commands == [
        (
            [
                run_eval_inference.sys.executable,
                "scripts/stage2_eval_worker.py",
                "--cards",
                str(tmp_path / "cards.jsonl"),
                "--model-dir",
                "model",
                "--adapter-dir",
                "adapter",
                "--output",
                str(output_path),
                "--model-name",
                "sft_v1",
            ],
            {"capture_output": True, "check": False, "text": True},
        )
    ]
    assert stderr_log.read_text(encoding="utf-8") == "RuntimeError: CUDA out of memory"
    events = _read_jsonl(event_log)
    assert [event["status"] for event in events] == ["start", "failed"]
    assert events[0]["detail"]["command"] == seen_commands[0][0]
    assert events[-1]["detail"]["exit_code"] == 23
    assert "cuda_oom" in capsys.readouterr().err


def test_non_dry_run_launcher_exception_exits_127_and_writes_failed_event(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    from scripts import run_eval_inference

    event_log = tmp_path / "events.jsonl"
    stderr_log = tmp_path / "stderr.log"

    def fake_run(command, **kwargs):
        raise OSError("launcher unavailable")

    monkeypatch.setattr(run_eval_inference.subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as exc_info:
        run_eval_inference.main(
            [
                "--cards",
                str(tmp_path / "cards.jsonl"),
                "--model-dir",
                "model",
                "--adapter-dir",
                "adapter",
                "--output",
                str(tmp_path / "generated.jsonl"),
                "--event-log",
                str(event_log),
                "--stderr-log",
                str(stderr_log),
            ]
        )

    assert exc_info.value.code == 127
    assert "OSError: launcher unavailable" in stderr_log.read_text(encoding="utf-8")
    events = _read_jsonl(event_log)
    assert [event["status"] for event in events] == ["start", "failed"]
    assert events[-1]["detail"]["exit_code"] == 127
    assert "process_killed" in capsys.readouterr().err
