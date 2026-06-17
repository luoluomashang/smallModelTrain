import json
import subprocess
import sys

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


def test_split_chapters_supports_spaced_and_indented_heading():
    text = "　 第 1 章 开始\n\n林默回来了。"
    chapters = split_chapters(text, work_id="work_001")
    assert len(chapters) == 1
    assert chapters[0]["chapter_title"] == "第 1 章 开始"
    assert chapters[0]["text"] == "林默回来了。"


def test_split_chapters_preserves_preface_before_first_heading():
    text = "楔子\n\n林默站在雨里。\n\n第1章 开始\n\n苏小满来了。"
    chapters = split_chapters(text, work_id="work_001")
    assert [chapter["id"] for chapter in chapters] == [
        "work_001_chapter_0001",
        "work_001_chapter_0002",
    ]
    assert chapters[0]["chapter_title"] == "前置正文"
    assert chapters[0]["text"] == "楔子\n\n林默站在雨里。"
    assert chapters[1]["chapter_title"] == "第1章 开始"


def test_clean_raw_text_removes_multiline_author_note_blocks():
    text = "　题外话：今天两更\n这一段也不是正文\n\n第1章 开始\n\n正文"
    assert clean_raw_text(text) == "第1章 开始\n\n正文"


def test_split_chapters_preserves_next_heading_after_author_note_without_blank_line():
    text = "第1章 开始\n\n正文一\n\n作者有话说：明天见\n第2章 继续\n\n正文二"
    chapters = split_chapters(text, work_id="work_001")
    assert [chapter["chapter_title"] for chapter in chapters] == ["第1章 开始", "第2章 继续"]
    assert chapters[0]["text"] == "正文一"
    assert chapters[1]["text"] == "正文二"


def test_ingest_raw_text_script_runs_from_repo_root_and_uses_unique_work_ids(tmp_path):
    input_dir = tmp_path / "raw"
    (input_dir / "alpha").mkdir(parents=True)
    (input_dir / "beta").mkdir(parents=True)
    (input_dir / "alpha" / "book.txt").write_text("第1章 开始\n\n林默回来了。", encoding="utf-8")
    (input_dir / "beta" / "book.txt").write_text("第1章 开始\n\n苏小满愣住。", encoding="utf-8")
    output_path = tmp_path / "chapters.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/ingest_raw_text.py",
            "--input-dir",
            str(input_dir),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    work_ids = [row["work_id"] for row in rows]
    assert work_ids[0].startswith("alpha__book__")
    assert work_ids[1].startswith("beta__book__")
    assert len(set(work_ids)) == 2
    assert [row["id"] for row in rows] == [f"{work_id}_chapter_0001" for work_id in work_ids]


def test_ingest_raw_text_script_preserves_path_boundaries_in_work_ids(tmp_path):
    input_dir = tmp_path / "raw"
    (input_dir / "a").mkdir(parents=True)
    (input_dir / "a" / "book.txt").write_text("第1章 开始\n\n林默回来了。", encoding="utf-8")
    (input_dir / "a_book.txt").write_text("第1章 开始\n\n苏小满愣住。", encoding="utf-8")
    (input_dir / "a b.txt").write_text("第1章 开始\n\n雨声很急。", encoding="utf-8")
    (input_dir / "a" / "b.txt").write_text("第1章 开始\n\n灯火亮起。", encoding="utf-8")
    output_path = tmp_path / "chapters.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/ingest_raw_text.py",
            "--input-dir",
            str(input_dir),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    work_ids = [row["work_id"] for row in rows]
    assert work_ids[0].startswith("a__b__")
    assert work_ids[1].startswith("a__book__")
    assert work_ids[2].startswith("a_b__")
    assert work_ids[3].startswith("a_book__")
    assert len({row["work_id"] for row in rows}) == 4
    assert [row["id"] for row in rows] == [f"{work_id}_chapter_0001" for work_id in work_ids]


def test_ingest_raw_text_script_uses_hashes_to_avoid_sanitized_work_id_collisions(tmp_path):
    input_dir = tmp_path / "raw"
    (input_dir / "a").mkdir(parents=True)
    (input_dir / "a" / "b.txt").write_text("第1章 开始\n\n林默回来了。", encoding="utf-8")
    (input_dir / "a__b.txt").write_text("第1章 开始\n\n苏小满愣住。", encoding="utf-8")
    (input_dir / "a b.txt").write_text("第1章 开始\n\n雨声很急。", encoding="utf-8")
    (input_dir / "a_b.txt").write_text("第1章 开始\n\n灯火亮起。", encoding="utf-8")
    output_path = tmp_path / "chapters.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/ingest_raw_text.py",
            "--input-dir",
            str(input_dir),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    work_ids = [row["work_id"] for row in rows]
    id_prefixes = [row["id"].removesuffix("_chapter_0001") for row in rows]
    assert len(work_ids) == 4
    assert len(set(work_ids)) == 4
    assert id_prefixes == work_ids


def test_clean_chapters_script_runs_from_repo_root(tmp_path):
    input_path = tmp_path / "chapters.jsonl"
    output_path = tmp_path / "cleaned.jsonl"
    row = {
        "id": "work_001_chapter_0001",
        "work_id": "work_001",
        "chapter_title": "第1章 开始",
        "text": "林默回来了。",
        "char_count_zh": 5,
        "quality_tag": "A",
        "split": "train",
    }
    input_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/clean_chapters.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--min-chars",
            "1",
            "--max-chars",
            "20",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [row]
