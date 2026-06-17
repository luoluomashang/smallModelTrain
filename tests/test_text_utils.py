from pathlib import Path

from small_model_train.io_utils import read_jsonl, read_text_auto, write_jsonl
from small_model_train.text_utils import (
    count_chinese_chars,
    dialogue_ratio,
    normalize_newlines,
    paragraph_lengths,
    repeated_ngram_ratio,
)


def test_count_chinese_chars_ignores_ascii_and_punctuation():
    assert count_chinese_chars("第1章 Hello，林默说：加钱。") == 7


def test_normalize_newlines_and_paragraph_lengths():
    text = "第一段\r\n\r\n\r\n第二段\n\n第三段"
    normalized = normalize_newlines(text)
    assert normalized == "第一段\n\n第二段\n\n第三段"
    assert paragraph_lengths(normalized) == [3, 3, 3]


def test_dialogue_ratio_counts_dialogue_paragraphs():
    text = "林默看了她一眼。\n\n“这单加钱。”\n\n苏小满愣住：“多少？”"
    assert dialogue_ratio(text) == 2 / 3


def test_repeated_ngram_ratio_detects_repetition():
    text = "加钱加钱加钱加钱"
    assert repeated_ngram_ratio(text, n=2) > 0.5


def test_jsonl_round_trip(tmp_path: Path):
    path = tmp_path / "items.jsonl"
    rows = [{"id": "a", "text": "正文"}, {"id": "b", "text": "另一章"}]
    write_jsonl(path, rows)
    assert read_jsonl(path) == rows


def test_read_text_auto_handles_utf8_sig(tmp_path: Path):
    path = tmp_path / "novel.txt"
    path.write_text("正文", encoding="utf-8-sig")
    assert read_text_auto(path) == "正文"
