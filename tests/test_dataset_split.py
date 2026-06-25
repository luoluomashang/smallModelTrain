import subprocess
import sys

import pytest

from small_model_train.dataset_split import split_rows
from small_model_train.io_utils import read_jsonl, write_jsonl


def test_split_rows_is_deterministic_and_marks_split():
    rows = [{"id": f"chapter_{index:04d}", "text": "正文"} for index in range(10)]
    first = split_rows(rows, eval_count=3, seed=7)
    second = split_rows(rows, eval_count=3, seed=7)
    assert first == second
    assert sum(1 for row in first if row["split"] == "eval") == 3
    assert sum(1 for row in first if row["split"] == "train") == 7


def test_split_rows_keeps_source_fields():
    rows = [{"id": "a", "work_id": "w", "text": "正文", "quality_tag": "A"}]
    split = split_rows(rows, eval_count=1, seed=1)
    assert split[0]["work_id"] == "w"
    assert split[0]["quality_tag"] == "A"


def test_split_rows_samples_duplicate_ids_by_position():
    rows = [{"id": "duplicate", "text": f"正文{index}"} for index in range(3)]
    split = split_rows(rows, eval_count=1, seed=1)
    assert sum(1 for row in split if row["split"] == "eval") == 1


def test_split_rows_rejects_negative_eval_count():
    rows = [{"id": "a", "text": "正文"}]
    with pytest.raises(ValueError):
        split_rows(rows, eval_count=-1)


def test_split_train_eval_script_runs_from_repo_root(tmp_path):
    input_path = tmp_path / "cleaned.jsonl"
    output_path = tmp_path / "split.jsonl"
    eval_output_path = tmp_path / "eval.jsonl"
    rows = [{"id": f"chapter_{index:04d}", "text": "正文"} for index in range(5)]
    write_jsonl(input_path, rows)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/split_train_eval.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--eval-output",
            str(eval_output_path),
            "--eval-count",
            "2",
            "--seed",
            "7",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    split_rows_output = read_jsonl(output_path)
    eval_rows = read_jsonl(eval_output_path)
    assert len(split_rows_output) == 5
    assert len(eval_rows) == 2
    assert eval_rows == [row for row in split_rows_output if row["split"] == "eval"]


def test_split_grouped_rows_is_deterministic_and_non_overlapping():
    from small_model_train.dataset_split import split_grouped_rows

    rows = [{"id": f"chapter_{index:04d}", "text": f"正文{index}"} for index in range(12)]
    first = split_grouped_rows(rows, validation_count=2, sealed_count=3, seed=9)
    second = split_grouped_rows(rows, validation_count=2, sealed_count=3, seed=9)

    assert first == second
    validation_ids = {row["id"] for row in first if row["split"] == "validation"}
    sealed_ids = {row["id"] for row in first if row["split"] == "sealed"}
    train_ids = {row["id"] for row in first if row["split"] == "train"}
    assert len(validation_ids) == 2
    assert len(sealed_ids) == 3
    assert len(train_ids) == 7
    assert validation_ids.isdisjoint(sealed_ids)
    assert validation_ids.isdisjoint(train_ids)
    assert sealed_ids.isdisjoint(train_ids)
    assert all(row["group_id"].startswith("group-") for row in first)
    assert all(len(row["group_sha256"]) == 64 for row in first)


def test_split_grouped_rows_rejects_negative_counts():
    from small_model_train.dataset_split import split_grouped_rows

    with pytest.raises(ValueError, match="validation_count"):
        split_grouped_rows([{"id": "a", "text": "正文"}], validation_count=-1, sealed_count=0)
    with pytest.raises(ValueError, match="sealed_count"):
        split_grouped_rows([{"id": "a", "text": "正文"}], validation_count=0, sealed_count=-1)


def test_find_near_duplicate_pairs_flags_high_overlap():
    from small_model_train.data.dedup import find_near_duplicate_pairs

    rows = [
        {"id": "train_1", "split": "train", "text": "林默走进旧仓库发现箱子正在响动"},
        {"id": "sealed_1", "split": "sealed", "text": "林默走进旧仓库发现箱子正在响动"},
        {"id": "sealed_2", "split": "sealed", "text": "另一条完全不同的章节内容"},
    ]

    pairs = find_near_duplicate_pairs(rows, threshold=0.8, shingle_size=4)

    assert pairs == [
        {
            "left_id": "train_1",
            "left_split": "train",
            "right_id": "sealed_1",
            "right_split": "sealed",
            "overlap": 1.0,
        }
    ]
