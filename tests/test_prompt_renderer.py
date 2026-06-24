from __future__ import annotations

from small_model_train.prompt_renderer import (
    SYSTEM_PROMPT,
    build_chat_messages,
    prompt_sha256,
    render_execution_input,
    render_model_input_prefix,
)


def _card() -> dict:
    return {
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


class _RecordingTokenizer:
    def __init__(self) -> None:
        self.messages = None
        self.tokenize = None
        self.add_generation_prompt = None

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.messages = messages
        self.tokenize = tokenize
        self.add_generation_prompt = add_generation_prompt
        return "templated-prefix"


def test_render_execution_input_includes_card_fields_without_source_text():
    rendered = render_execution_input(_card())

    assert "【风格契约】\n短句推进，动作细节清楚。" in rendered
    assert "【前情摘要】\n主角刚抵达旧城。" in rendered
    assert "【本章目标】\n让主角发现密室钥匙。" in rendered
    assert "【冲突推进】\n主角必须在管家返回前打开书房暗格。" in rendered
    assert "【爽点兑现】\n暗格弹开，铜钥匙和旧城密道图一起出现。" in rendered
    assert "- 1. 搜寻：在书房找到线索（建议 800）" in rendered
    assert "- 林照：谨慎但兴奋；说话方式：简短直接" in rendered
    assert "【必须出现】\n- 铜钥匙\n- 雨声" in rendered
    assert "【禁止事项】\n- 解释设定" in rendered
    assert "【章末钩子】\n门后传来第二个人的呼吸声。" in rendered
    assert "【目标字数】\n2000-2500中文汉字" in rendered
    assert "只输出正文，不输出提纲、小标题、解释、分析或提示语。" in rendered
    assert "不应该泄漏到提示词" not in rendered


def test_system_prompt_literal_is_pinned():
    assert SYSTEM_PROMPT == "你是作者的正文执行器。严格执行章节卡，并保持指定作者风格。"


def test_build_chat_messages_returns_system_and_user_messages():
    messages = build_chat_messages(_card())

    assert messages == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_execution_input(_card())},
    ]


def test_render_model_input_prefix_uses_tokenizer_chat_template():
    tokenizer = _RecordingTokenizer()

    rendered = render_model_input_prefix(_card(), tokenizer)

    assert rendered == "templated-prefix"
    assert tokenizer.messages == build_chat_messages(_card())
    assert tokenizer.tokenize is False
    assert tokenizer.add_generation_prompt is True


def test_render_model_input_prefix_falls_back_when_tokenizer_has_no_chat_template():
    rendered = render_model_input_prefix(_card(), object())

    assert rendered == f"{SYSTEM_PROMPT}\n\n{render_execution_input(_card())}\n\n"


def test_prompt_sha256_is_stable_hex_digest():
    prompt = render_execution_input(_card())

    first = prompt_sha256(prompt)
    second = prompt_sha256(prompt)

    assert first == second
    assert len(first) == 64
