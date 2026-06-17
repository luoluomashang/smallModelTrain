from small_model_train.chapter_splitter import clean_raw_text, split_chapters


def test_clean_raw_text_removes_author_notes_and_separators():
    text = "作者有话说：今天两更\n\n---\n\n第1章 开始\n\n正文"
    assert clean_raw_text(text) == "第1章 开始\n\n正文"


def test_split_chapters_extracts_numbered_chapters():
    text = "第1章 开始\n\n林默回来了。\n\n第2章 加钱\n\n苏小满愣住。"
    chapters = split_chapters(text, work_id="work_001")
    assert [chapter["id"] for chapter in chapters] == [
        "work_001_chapter_0001",
        "work_001_chapter_0002",
    ]
    assert chapters[0]["chapter_title"] == "第1章 开始"
    assert chapters[0]["char_count_zh"] == 5
    assert chapters[0]["quality_tag"] == "A"
    assert chapters[0]["split"] == "train"


def test_split_chapters_uses_single_chapter_for_untitled_text():
    chapters = split_chapters("林默回来了。", work_id="solo")
    assert len(chapters) == 1
    assert chapters[0]["chapter_title"] == "未命名章节"
