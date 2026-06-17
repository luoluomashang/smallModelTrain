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
