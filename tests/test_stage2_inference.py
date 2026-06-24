import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from small_model_train.execution_cards import DEFAULT_TARGET_PLATFORM
from small_model_train.io_utils import write_jsonl
from small_model_train.prompt_renderer import prompt_sha256
from small_model_train.stage2_inference import (
    build_generation_row,
    default_inference_params,
    load_eval_cards,
    render_eval_model_input,
    render_eval_prompt,
    sanitize_generated_output,
    sanitize_generated_output_with_events,
)
from small_model_train.text_utils import count_chinese_chars

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _card(sample_id: str = "eval-1") -> dict:
    return {
        "id": sample_id,
        "target_platform": DEFAULT_TARGET_PLATFORM,
        "genre_tags": ["悬疑", "男频"],
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
        "conflict_beat": "主角必须在管家返回前打开书房暗格。",
        "payoff_beat": "暗格弹开，铜钥匙和旧城密道图一起出现。",
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


class _FakeWorkerProcess:
    def __init__(
        self,
        *,
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout.splitlines(keepends=True)
        self.stderr = _FakePipe(stderr)
        self.wait_called = False

    def wait(self) -> int:
        self.wait_called = True
        return self.returncode


class _FakePipe:
    def __init__(self, text: str) -> None:
        self.text = text

    def read(self) -> str:
        return self.text

    def close(self) -> None:
        pass


class _RecordingTokenizer:
    def __init__(self) -> None:
        self.tokenize = None
        self.add_generation_prompt = None

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.tokenize = tokenize
        self.add_generation_prompt = add_generation_prompt
        return "templated-prefix"


def test_load_eval_cards_requires_execution_card_schema(tmp_path):
    cards_path = tmp_path / "raw_eval_cards.jsonl"
    cards_path.write_text(
        '{"id":"case1","text":"原文","quality_tag":"A","split":"eval"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        load_eval_cards(cards_path)

    assert "missing execution-card fields" in str(excinfo.value)

def test_render_eval_prompt_contains_card_fields():
    prompt = render_eval_prompt(_card())

    assert "让主角发现密室钥匙" in prompt
    assert "短句推进，动作细节清楚。" in prompt
    assert "铜钥匙" in prompt
    assert "雨声" in prompt
    assert "不应该泄漏到提示词" not in prompt


def test_render_eval_prompt_rejects_source_text_fragment_leakage():
    card = _card()
    card["previous_summary"] = "这是一段不应该泄漏到提示词"

    with pytest.raises(ValueError, match="SFT input contains source_text fragment"):
        render_eval_prompt(card)


def test_render_eval_model_input_uses_tokenizer_chat_template():
    tokenizer = _RecordingTokenizer()

    rendered = render_eval_model_input(_card(), tokenizer)

    assert rendered == "templated-prefix"
    assert tokenizer.tokenize is False
    assert tokenizer.add_generation_prompt is True


def test_render_eval_model_input_rejects_source_text_fragment_leakage():
    card = _card()
    card["previous_summary"] = "这是一段不应该泄漏到提示词"

    with pytest.raises(ValueError, match="SFT input contains source_text fragment"):
        render_eval_model_input(card, _RecordingTokenizer())


def test_build_generation_row_uses_fixed_schema():
    params = {"temperature": 0.7}

    row = build_generation_row("eval-1", "正文", "sft_v1", params)

    assert row == {
        "id": "eval-1",
        "output": "正文",
        "raw_output": "正文",
        "sanitized_output": "正文",
        "model": "sft_v1",
        "adapter_dir": "",
        "params": params,
        "finish_reason": "unknown",
        "generated_tokens": 0,
        "prompt_sha256": "",
        "sanitizer_events": [],
    }


def test_build_generation_row_copies_params():
    params = {"temperature": 0.7}

    row = build_generation_row("eval-1", "正文", "sft_v1", params)
    params["temperature"] = 0.1

    assert row["params"] == {"temperature": 0.7}
    assert row["params"] is not params


def test_build_generation_row_preserves_raw_and_sanitized_evidence():
    params = {"temperature": 0.7}
    events = [{"type": "drop_meta_line", "reason": "outline_heading", "line_number": 2}]

    row = build_generation_row(
        "eval-1",
        "【章节结构】\n正文",
        "sft_v1",
        params,
        sanitized_output="正文",
        adapter_dir="outputs/sft_v1",
        finish_reason="length",
        generated_tokens=128,
        prompt_sha256="abc123",
        sanitizer_events=events,
    )

    assert row == {
        "id": "eval-1",
        "output": "【章节结构】\n正文",
        "raw_output": "【章节结构】\n正文",
        "sanitized_output": "正文",
        "model": "sft_v1",
        "adapter_dir": "outputs/sft_v1",
        "params": {"temperature": 0.7},
        "finish_reason": "length",
        "generated_tokens": 128,
        "prompt_sha256": "abc123",
        "sanitizer_events": events,
    }


def test_default_inference_params_match_stage2_eval_defaults():
    assert default_inference_params() == {
        "max_new_tokens": 5120,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.05,
    }


def test_sanitize_generated_output_removes_outline_blocks_and_keeps_prose():
    raw = "\n".join(
        [
            "我推开门，雨声从楼道深处涌了进来。",
            "",
            "【章节结构】",
            "- 承接：交代开场状态",
            "- 加压：制造阻碍",
            "---",
            "她没有回头，只把铜钥匙压在掌心。",
        ]
    )

    cleaned = sanitize_generated_output(raw)

    assert cleaned == "我推开门，雨声从楼道深处涌了进来。\n\n她没有回头，只把铜钥匙压在掌心。"
    assert "【" not in cleaned
    assert "章节结构" not in cleaned
    assert "承接" not in cleaned


def test_sanitize_generated_output_with_events_keeps_audit_trail():
    raw = "\n".join(
        [
            "我推开门，雨声从楼道深处涌了进来。她没有回头，只把铜钥匙压在掌心。",
            "【章节结构】",
            "承接旧线索",
            "- 加压：制造阻碍",
            "---",
            "第二个人的呼吸声从门后停住。",
        ]
    )

    sanitized, events = sanitize_generated_output_with_events(raw, max_chinese_chars=12)

    assert "【章节结构】" in raw
    assert "【章节结构】" not in sanitized
    assert count_chinese_chars(sanitized) == 12
    assert {event["type"] for event in events} == {
        "drop_meta_line",
        "drop_meta_continuation",
        "drop_list_line",
        "drop_separator",
        "cap_chinese_chars",
    }
    assert all("reason" in event for event in events)
    line_events = [event for event in events if event["type"].startswith("drop_")]
    cap_event = next(event for event in events if event["type"] == "cap_chinese_chars")
    assert [event["line_number"] for event in line_events] == [2, 3, 4, 5]
    assert all(event["preview"] and len(event["preview"]) <= 80 for event in line_events)
    assert cap_event["preview"]
    assert len(cap_event["preview"]) <= 80
    assert cap_event["preview"] != raw


def test_sanitize_generated_output_drops_inline_meta_directives():
    raw = "\n".join(
        [
            "请严格遵循【风格契约】，只输出正文。",
            "（请根据以上所有信息，生成完整章节正文。）",
            "门后的呼吸声停了一下，像有人把手按在了木板另一侧。",
        ]
    )

    cleaned = sanitize_generated_output(raw)

    assert cleaned == "门后的呼吸声停了一下，像有人把手按在了木板另一侧。"


def test_sanitize_generated_output_drops_writing_meta_language():
    raw = "\n".join(
        [
            "第一人称视角，现实主义语言风格，禁止抒情。",
            "雨水顺着门缝流进来，我听见里面有人轻轻咳了一声。",
        ]
    )

    cleaned = sanitize_generated_output(raw)

    assert cleaned == "雨水顺着门缝流进来，我听见里面有人轻轻咳了一声。"


def test_sanitize_generated_output_drops_body_marker_variants():
    raw = "\n".join(
        [
            "以下是正文：",
            "【正文开始",
            "门轴响了一下，我把手里的钥匙攥紧。",
        ]
    )

    cleaned = sanitize_generated_output(raw)

    assert cleaned == "门轴响了一下，我把手里的钥匙攥紧。"


def test_sanitize_generated_output_caps_chinese_chars_at_sentence_boundary():
    raw = ("我" * 2400) + "。" + ("他" * 200) + "。"

    cleaned = sanitize_generated_output(raw)

    assert count_chinese_chars(cleaned) == 2400
    assert cleaned.endswith("。")


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
    assert rows[0]["raw_output"] == rows[0]["output"]
    assert rows[0]["sanitized_output"] == rows[0]["output"]
    assert rows[0]["sanitizer_events"] == []
    for row, card in zip(rows, cards):
        expected_hash = prompt_sha256(render_eval_model_input(card))
        assert row["prompt_sha256"] == expected_hash
        assert len(row["prompt_sha256"]) == 64
        int(row["prompt_sha256"], 16)
    params = default_inference_params()
    params["seed"] = 20260623
    assert rows[0]["params"] == params
    assert set(rows[0]) == {
        "id",
        "output",
        "raw_output",
        "sanitized_output",
        "model",
        "adapter_dir",
        "params",
        "finish_reason",
        "generated_tokens",
        "prompt_sha256",
        "sanitizer_events",
    }


def test_dry_run_cli_records_max_new_tokens_override(
    tmp_path: Path,
    monkeypatch,
):
    from scripts import run_eval_inference

    cards_path = tmp_path / "cards.jsonl"
    output_path = tmp_path / "generated.jsonl"
    write_jsonl(cards_path, [_card("eval-1")])

    def fail_run(*args, **kwargs):
        raise AssertionError("dry-run must not call subprocess.run")

    monkeypatch.setattr(run_eval_inference.subprocess, "run", fail_run)

    exit_code = run_eval_inference.main(
        [
            "--cards",
            str(cards_path),
            "--output",
            str(output_path),
            "--dry-run",
            "--max-new-tokens",
            "128",
        ]
    )

    params = default_inference_params()
    params["max_new_tokens"] = 128
    params["seed"] = 20260623
    assert exit_code == 0
    assert _read_jsonl(output_path)[0]["params"] == params


def test_dry_run_cli_records_repetition_penalty_override(
    tmp_path: Path,
    monkeypatch,
):
    from scripts import run_eval_inference

    cards_path = tmp_path / "cards.jsonl"
    output_path = tmp_path / "generated.jsonl"
    write_jsonl(cards_path, [_card("eval-1")])

    def fail_run(*args, **kwargs):
        raise AssertionError("dry-run must not call subprocess.run")

    monkeypatch.setattr(run_eval_inference.subprocess, "run", fail_run)

    exit_code = run_eval_inference.main(
        [
            "--cards",
            str(cards_path),
            "--output",
            str(output_path),
            "--dry-run",
            "--repetition-penalty",
            "1.12",
        ]
    )

    params = default_inference_params()
    params["repetition_penalty"] = 1.12
    params["seed"] = 20260623
    assert exit_code == 0
    assert _read_jsonl(output_path)[0]["params"] == params


def test_dry_run_cli_records_no_repeat_ngram_size_override(
    tmp_path: Path,
    monkeypatch,
):
    from scripts import run_eval_inference

    cards_path = tmp_path / "cards.jsonl"
    output_path = tmp_path / "generated.jsonl"
    write_jsonl(cards_path, [_card("eval-1")])

    def fail_run(*args, **kwargs):
        raise AssertionError("dry-run must not call subprocess.run")

    monkeypatch.setattr(run_eval_inference.subprocess, "run", fail_run)

    exit_code = run_eval_inference.main(
        [
            "--cards",
            str(cards_path),
            "--output",
            str(output_path),
            "--dry-run",
            "--no-repeat-ngram-size",
            "8",
        ]
    )

    params = default_inference_params()
    params["no_repeat_ngram_size"] = 8
    params["seed"] = 20260623
    assert exit_code == 0
    assert _read_jsonl(output_path)[0]["params"] == params


def test_dry_run_cli_records_seed_override(
    tmp_path: Path,
    monkeypatch,
):
    from scripts import run_eval_inference

    cards_path = tmp_path / "cards.jsonl"
    output_path = tmp_path / "generated.jsonl"
    write_jsonl(cards_path, [_card("eval-1")])

    def fail_run(*args, **kwargs):
        raise AssertionError("dry-run must not call subprocess.run")

    monkeypatch.setattr(run_eval_inference.subprocess, "run", fail_run)

    exit_code = run_eval_inference.main(
        [
            "--cards",
            str(cards_path),
            "--output",
            str(output_path),
            "--dry-run",
            "--seed",
            "12345",
        ]
    )

    params = default_inference_params()
    params["seed"] = 12345
    assert exit_code == 0
    assert _read_jsonl(output_path)[0]["params"] == params


def test_dry_run_cli_rejects_non_positive_max_new_tokens(
    tmp_path: Path,
    capsys,
):
    from scripts import run_eval_inference

    with pytest.raises(SystemExit) as exc_info:
        run_eval_inference.main(
            [
                "--cards",
                str(tmp_path / "cards.jsonl"),
                "--output",
                str(tmp_path / "generated.jsonl"),
                "--dry-run",
                "--max-new-tokens",
                "0",
            ]
        )

    assert exc_info.value.code == 2
    assert "must be a positive integer" in capsys.readouterr().err


def test_dry_run_cli_rejects_non_positive_repetition_penalty(
    tmp_path: Path,
    capsys,
):
    from scripts import run_eval_inference

    with pytest.raises(SystemExit) as exc_info:
        run_eval_inference.main(
            [
                "--cards",
                str(tmp_path / "cards.jsonl"),
                "--output",
                str(tmp_path / "generated.jsonl"),
                "--dry-run",
                "--repetition-penalty",
                "0",
            ]
        )

    assert exc_info.value.code == 2
    assert "must be a positive float" in capsys.readouterr().err


def test_dry_run_cli_fails_for_missing_cards_without_subprocess(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    from scripts import run_eval_inference

    def fail_run(*args, **kwargs):
        raise AssertionError("invalid dry-run inputs must not call subprocess.run")

    monkeypatch.setattr(run_eval_inference.subprocess, "run", fail_run)

    with pytest.raises(SystemExit) as exc_info:
        run_eval_inference.main(
            [
                "--cards",
                str(tmp_path / "missing.jsonl"),
                "--output",
                str(tmp_path / "generated.jsonl"),
                "--dry-run",
            ]
        )

    assert exc_info.value.code == 1
    assert "cards file is missing" in capsys.readouterr().err


def test_dry_run_cli_fails_for_empty_cards_without_subprocess(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    from scripts import run_eval_inference

    cards_path = tmp_path / "empty.jsonl"
    cards_path.write_text("\n", encoding="utf-8")

    def fail_run(*args, **kwargs):
        raise AssertionError("invalid dry-run inputs must not call subprocess.run")

    monkeypatch.setattr(run_eval_inference.subprocess, "run", fail_run)

    with pytest.raises(SystemExit) as exc_info:
        run_eval_inference.main(
            [
                "--cards",
                str(cards_path),
                "--output",
                str(tmp_path / "generated.jsonl"),
                "--dry-run",
            ]
        )

    assert exc_info.value.code == 1
    assert "cards file has no rows" in capsys.readouterr().err


def test_worker_card_loader_fails_before_gpu_imports_for_missing_or_empty_cards(
    tmp_path: Path,
):
    from scripts.stage2_eval_worker import load_eval_cards

    missing = tmp_path / "missing.jsonl"
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="cards file is missing"):
        load_eval_cards(missing)
    with pytest.raises(ValueError, match="cards file has no rows"):
        load_eval_cards(empty)


def test_worker_resets_stale_output_and_appends_rows_incrementally(
    tmp_path: Path,
):
    from scripts.stage2_eval_worker import append_generation_row, reset_generation_output

    output_path = tmp_path / "nested" / "generated.jsonl"
    output_path.parent.mkdir()
    output_path.write_text('{"id":"stale"}\n', encoding="utf-8")

    reset_generation_output(output_path)

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == ""

    append_generation_row(output_path, {"id": "eval-1", "output": "一"})

    assert _read_jsonl(output_path) == [{"id": "eval-1", "output": "一"}]

    append_generation_row(output_path, {"id": "eval-2", "output": "二"})

    assert _read_jsonl(output_path) == [
        {"id": "eval-1", "output": "一"},
        {"id": "eval-2", "output": "二"},
    ]


def test_worker_progress_message_includes_count_total_and_id(capsys):
    from scripts.stage2_eval_worker import print_generation_progress

    print_generation_progress(completed=3, total=50, sample_id="eval-003")

    assert capsys.readouterr().out == "generated 3/50 eval-003\n"


def test_worker_inference_params_default_and_override():
    from scripts.stage2_eval_worker import build_inference_params

    assert build_inference_params(None, None, None) == default_inference_params()

    params = build_inference_params(64, 1.12, 8)
    expected = default_inference_params()
    expected["max_new_tokens"] = 64
    expected["repetition_penalty"] = 1.12
    expected["no_repeat_ngram_size"] = 8
    assert params == expected
    assert default_inference_params()["max_new_tokens"] == 5120


def test_worker_cli_rejects_non_positive_max_new_tokens(tmp_path: Path, capsys):
    from scripts import stage2_eval_worker

    with pytest.raises(SystemExit) as exc_info:
        stage2_eval_worker.main(
            [
                "--cards",
                str(tmp_path / "cards.jsonl"),
                "--model-dir",
                "model",
                "--adapter-dir",
                "adapter",
                "--output",
                str(tmp_path / "generated.jsonl"),
                "--max-new-tokens",
                "0",
            ]
        )

    assert exc_info.value.code == 2
    assert "must be a positive integer" in capsys.readouterr().err


def test_worker_cli_rejects_non_positive_repetition_penalty(tmp_path: Path, capsys):
    from scripts import stage2_eval_worker

    with pytest.raises(SystemExit) as exc_info:
        stage2_eval_worker.main(
            [
                "--cards",
                str(tmp_path / "cards.jsonl"),
                "--model-dir",
                "model",
                "--adapter-dir",
                "adapter",
                "--output",
                str(tmp_path / "generated.jsonl"),
                "--repetition-penalty",
                "0",
            ]
        )

    assert exc_info.value.code == 2
    assert "must be a positive float" in capsys.readouterr().err


def test_worker_cli_rejects_non_positive_no_repeat_ngram_size(
    tmp_path: Path,
    capsys,
):
    from scripts import stage2_eval_worker

    with pytest.raises(SystemExit) as exc_info:
        stage2_eval_worker.main(
            [
                "--cards",
                str(tmp_path / "cards.jsonl"),
                "--model-dir",
                "model",
                "--adapter-dir",
                "adapter",
                "--output",
                str(tmp_path / "generated.jsonl"),
                "--no-repeat-ngram-size",
                "0",
            ]
        )

    assert exc_info.value.code == 2
    assert "must be a positive integer" in capsys.readouterr().err


def test_worker_main_uses_model_input_and_preserves_raw_generation(
    tmp_path: Path,
    monkeypatch,
):
    from scripts import stage2_eval_worker

    cards_path = tmp_path / "cards.jsonl"
    output_path = tmp_path / "generated.jsonl"
    write_jsonl(cards_path, [_card("eval-1")])

    seed_calls: list[int] = []
    cuda_seed_calls: list[int] = []

    class FakeInputs(dict):
        def __init__(self) -> None:
            super().__init__(input_ids=SimpleNamespace(shape=(1, 2)))

        def to(self, device):
            self["device"] = device
            return self

    class FakeTokenizer:
        eos_token = "<eos>"
        eos_token_id = 99
        pad_token_id = None

        def __init__(self) -> None:
            self.prompts: list[str] = []

        def __call__(self, prompt, *, return_tensors, add_special_tokens):
            self.prompts.append(prompt)
            assert return_tensors == "pt"
            assert add_special_tokens is False
            return FakeInputs()

        def decode(self, tokens, *, skip_special_tokens):
            assert list(tokens) == [31, 32, 33]
            assert skip_special_tokens is True
            return "【章节结构】\n- 加压：制造阻碍\n正文来了。"

    class FakeModel:
        device = "cpu"

        def eval(self):
            return None

        def generate(self, **kwargs):
            assert kwargs["max_new_tokens"] == 5120
            return [[10, 11, 31, 32, 33]]

    fake_tokenizer = FakeTokenizer()
    fake_model = FakeModel()

    monkeypatch.setattr(
        stage2_eval_worker,
        "render_eval_model_input",
        lambda card, tokenizer: "MODEL PREFIX",
        raising=False,
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            manual_seed=lambda seed: seed_calls.append(seed),
            cuda=SimpleNamespace(
                is_available=lambda: True,
                manual_seed_all=lambda seed: cuda_seed_calls.append(seed),
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: fake_tokenizer
            ),
            AutoModelForCausalLM=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: object()
            ),
            BitsAndBytesConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(
            PeftModel=SimpleNamespace(
                from_pretrained=lambda base_model, adapter_dir: fake_model
            )
        ),
    )

    exit_code = stage2_eval_worker.main(
        [
            "--cards",
            str(cards_path),
            "--model-dir",
            "model",
            "--adapter-dir",
            "adapter",
            "--output",
            str(output_path),
            "--seed",
            "12345",
        ]
    )

    row = _read_jsonl(output_path)[0]
    assert exit_code == 0
    assert seed_calls == [12345]
    assert cuda_seed_calls == [12345]
    assert fake_tokenizer.prompts == ["MODEL PREFIX"]
    assert row["output"] == "【章节结构】\n- 加压：制造阻碍\n正文来了。"
    assert row["raw_output"] == row["output"]
    assert row["sanitized_output"] == "正文来了。"
    assert row["adapter_dir"] == "adapter"
    assert row["generated_tokens"] == 3
    assert row["params"]["seed"] == 12345
    assert isinstance(row["prompt_sha256"], str)
    assert len(row["prompt_sha256"]) == 64
    assert {event["type"] for event in row["sanitizer_events"]} == {
        "drop_meta_line",
        "drop_list_line",
    }


def test_generated_token_count_uses_tensor_numel():
    from scripts.stage2_eval_worker import _generated_token_count

    class TensorLike:
        def numel(self):
            return "7"

    assert _generated_token_count(TensorLike()) == 7


def test_non_dry_run_success_writes_and_echoes_stdout(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    from scripts import run_eval_inference

    event_log = tmp_path / "events.jsonl"
    stderr_log = tmp_path / "stderr.log"
    stdout_log = tmp_path / "stdout.log"

    process = _FakeWorkerProcess(
        returncode=0,
        stdout="generated 1/2 eval-1\nwrote 2 generations to output\n",
        stderr="",
    )
    seen_commands = []

    def fake_popen(command, **kwargs):
        seen_commands.append((command, kwargs))
        return process

    monkeypatch.setattr(run_eval_inference.subprocess, "Popen", fake_popen)

    exit_code = run_eval_inference.main(
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
            "--stdout-log",
            str(stdout_log),
        ]
    )

    assert exit_code == 0
    assert stdout_log.read_text(encoding="utf-8") == (
        "generated 1/2 eval-1\nwrote 2 generations to output\n"
    )
    assert stderr_log.read_text(encoding="utf-8") == ""
    assert process.wait_called
    assert seen_commands[0][1] == {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "bufsize": 1,
    }
    assert "capture_output" not in seen_commands[0][1]
    captured = capsys.readouterr()
    assert captured.out.count("generated 1/2 eval-1\n") == 1
    assert captured.out.count("wrote 2 generations to output\n") == 1
    events = _read_jsonl(event_log)
    assert [event["status"] for event in events] == ["start", "ok"]


def test_build_worker_command_omits_max_new_tokens_by_default(tmp_path: Path):
    from scripts import run_eval_inference

    args = argparse.Namespace(
        cards=tmp_path / "cards.jsonl",
        model_dir="model",
        adapter_dir="adapter",
        output=tmp_path / "generated.jsonl",
        model_name="sft_v1",
        max_new_tokens=None,
        repetition_penalty=None,
        no_repeat_ngram_size=None,
        seed=20260623,
    )

    command = run_eval_inference._build_worker_command(args)

    assert "--max-new-tokens" not in command
    assert command[-2:] == ["--seed", "20260623"]


def test_build_worker_command_includes_max_new_tokens_when_supplied(tmp_path: Path):
    from scripts import run_eval_inference

    args = argparse.Namespace(
        cards=tmp_path / "cards.jsonl",
        model_dir="model",
        adapter_dir="adapter",
        output=tmp_path / "generated.jsonl",
        model_name="sft_v1",
        max_new_tokens=96,
        repetition_penalty=None,
        no_repeat_ngram_size=None,
        seed=20260623,
    )

    command = run_eval_inference._build_worker_command(args)

    assert command[-2:] == ["--max-new-tokens", "96"]


def test_build_worker_command_includes_repetition_penalty_when_supplied(
    tmp_path: Path,
):
    from scripts import run_eval_inference

    args = argparse.Namespace(
        cards=tmp_path / "cards.jsonl",
        model_dir="model",
        adapter_dir="adapter",
        output=tmp_path / "generated.jsonl",
        model_name="sft_v1",
        max_new_tokens=None,
        repetition_penalty=1.12,
        no_repeat_ngram_size=None,
        seed=20260623,
    )

    command = run_eval_inference._build_worker_command(args)

    assert command[-2:] == ["--repetition-penalty", "1.12"]


def test_build_worker_command_includes_no_repeat_ngram_size_when_supplied(
    tmp_path: Path,
):
    from scripts import run_eval_inference

    args = argparse.Namespace(
        cards=tmp_path / "cards.jsonl",
        model_dir="model",
        adapter_dir="adapter",
        output=tmp_path / "generated.jsonl",
        model_name="sft_v1",
        max_new_tokens=None,
        repetition_penalty=None,
        no_repeat_ngram_size=8,
        seed=20260623,
    )

    command = run_eval_inference._build_worker_command(args)

    assert command[-2:] == ["--no-repeat-ngram-size", "8"]


def test_build_worker_command_includes_seed_when_supplied(tmp_path: Path):
    from scripts import run_eval_inference

    args = argparse.Namespace(
        cards=tmp_path / "cards.jsonl",
        model_dir="model",
        adapter_dir="adapter",
        output=tmp_path / "generated.jsonl",
        model_name="sft_v1",
        max_new_tokens=None,
        repetition_penalty=None,
        no_repeat_ngram_size=None,
        seed=12345,
    )

    command = run_eval_inference._build_worker_command(args)

    assert command[-2:] == ["--seed", "12345"]


def test_non_dry_run_failure_writes_logs_events_and_classification(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    from scripts import run_eval_inference

    output_path = tmp_path / "generated.jsonl"
    event_log = tmp_path / "events.jsonl"
    stderr_log = tmp_path / "stderr.log"
    stdout_log = tmp_path / "stdout.log"
    seen_commands = []

    process = _FakeWorkerProcess(
        returncode=23,
        stdout="loaded tokenizer before failure\n",
        stderr="RuntimeError: CUDA out of memory",
    )

    def fake_popen(command, **kwargs):
        seen_commands.append((command, kwargs))
        return process

    monkeypatch.setattr(run_eval_inference.subprocess, "Popen", fake_popen)

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
                "--stdout-log",
                str(stdout_log),
            ]
        )

    worker_path = str(run_eval_inference.REPO_ROOT / "scripts" / "stage2_eval_worker.py")
    assert exc_info.value.code == 23
    assert seen_commands == [
        (
            [
                run_eval_inference.sys.executable,
                worker_path,
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
                "--seed",
                "20260623",
            ],
            {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "bufsize": 1,
            },
        )
    ]
    assert stderr_log.read_text(encoding="utf-8") == "RuntimeError: CUDA out of memory"
    assert stdout_log.read_text(encoding="utf-8") == "loaded tokenizer before failure\n"
    events = _read_jsonl(event_log)
    assert [event["status"] for event in events] == ["start", "failed"]
    assert events[0]["detail"]["command"] == seen_commands[0][0]
    assert events[-1]["detail"]["exit_code"] == 23
    captured = capsys.readouterr()
    assert captured.out.count("loaded tokenizer before failure\n") == 1
    assert "cuda_oom" in captured.err


def test_non_dry_run_failure_classifies_stdout_only_cuda_oom(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    from scripts import run_eval_inference

    process = _FakeWorkerProcess(
        returncode=23,
        stdout="RuntimeError: CUDA out of memory\n",
        stderr="",
    )

    monkeypatch.setattr(
        run_eval_inference.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )

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
                str(tmp_path / "events.jsonl"),
                "--stderr-log",
                str(tmp_path / "stderr.log"),
                "--stdout-log",
                str(tmp_path / "stdout.log"),
            ]
        )

    events = _read_jsonl(tmp_path / "events.jsonl")
    assert exc_info.value.code == 23
    assert events[-1]["detail"]["error"]["error_type"] == "cuda_oom"
    assert "cuda_oom" in capsys.readouterr().err


def test_non_dry_run_launcher_exception_exits_127_and_writes_failed_event(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    from scripts import run_eval_inference

    event_log = tmp_path / "events.jsonl"
    stderr_log = tmp_path / "stderr.log"
    stdout_log = tmp_path / "stdout.log"

    def fake_popen(command, **kwargs):
        raise OSError("launcher unavailable")

    monkeypatch.setattr(run_eval_inference.subprocess, "Popen", fake_popen)

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
                "--stdout-log",
                str(stdout_log),
            ]
        )

    assert exc_info.value.code == 127
    assert "OSError: launcher unavailable" in stderr_log.read_text(encoding="utf-8")
    assert stdout_log.read_text(encoding="utf-8") == ""
    events = _read_jsonl(event_log)
    assert [event["status"] for event in events] == ["start", "failed"]
    assert events[-1]["detail"]["exit_code"] == 127
    assert "process_killed" in capsys.readouterr().err
