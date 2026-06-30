from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from small_model_train.evaluation.experiment_manifest import file_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_experiment_matrix_writes_candidate_dry_run_command(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "commands.jsonl"
    manifest = _manifest(tmp_path)
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_experiment_matrix.py",
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ] == [
        {
            "experiment_id": "stage5e-lr",
            "run_id": "candidate",
            "dry_run": True,
            "primary_variable": "learning_rate",
            "command": [
                "python",
                "scripts/run_sft_train.py",
                "--config",
                manifest["artifacts"]["config"]["path"],
                "--output-dir",
                "outputs/stage5e/candidate",
                "--sft-dataset",
                manifest["artifacts"]["sft_dataset"]["path"],
                "--eval-cards",
                manifest["artifacts"]["eval_cards"]["path"],
                "--dry-run",
            ],
        }
    ]


def test_run_experiment_matrix_generated_command_never_uses_run_name(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "commands.jsonl"
    manifest_path.write_text(json.dumps(_manifest(tmp_path)) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_experiment_matrix.py",
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    unsupported_arg = "--run" + "-name"
    assert all(unsupported_arg not in row["command"] for row in rows)


def test_run_experiment_matrix_requires_config_artifact(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["artifacts"] = {}
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "commands.jsonl"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_experiment_matrix.py",
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "error: manifest artifact config is required" in result.stderr
    assert not output_path.exists()


def test_run_experiment_matrix_removes_stale_output_on_failure(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["artifacts"] = {}
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "commands.jsonl"
    stale_row = {
        "experiment_id": "old",
        "run_id": "old-candidate",
        "dry_run": True,
        "primary_variable": "learning_rate",
        "command": ["python", "old.py"],
    }
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    output_path.write_text(json.dumps(stale_row) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_experiment_matrix.py",
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "error: manifest artifact config is required" in result.stderr
    assert not output_path.exists()


def test_run_experiment_matrix_rejects_artifact_sha256_mismatch_and_removes_output(
    tmp_path,
):
    manifest = _manifest(tmp_path)
    manifest["artifacts"]["config"]["sha256"] = "0" * 64
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "commands.jsonl"
    output_path.write_text('{"stale": true}\n', encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_experiment_matrix.py",
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "error: artifact sha256 mismatch: config" in result.stderr
    assert not output_path.exists()


def test_remove_output_file_ignores_unlink_errors(tmp_path, monkeypatch):
    output_path = tmp_path / "commands.jsonl"
    output_path.write_text('{"stale": true}\n', encoding="utf-8")
    module = _load_run_experiment_matrix()

    def raise_oserror(self):
        raise OSError("locked")

    monkeypatch.setattr(type(output_path), "unlink", raise_oserror)

    module._remove_output_file(str(output_path))


def _manifest(tmp_path: Path) -> dict:
    artifact_paths = _write_manifest_artifacts(tmp_path)
    return {
        "schema_version": 1,
        "created_at": "2026-06-30T00:00:00Z",
        "experiment_id": "stage5e-lr",
        "baseline_run_id": "baseline",
        "candidate_run_id": "candidate",
        "primary_variable": {
            "name": "learning_rate",
            "baseline_value": "0.0001",
            "candidate_value": "0.0002",
        },
        "controlled_variables": [],
        "stage5e_entry": {
            "path": "reports/stage5e_entry.json",
            "passed": True,
            "entry": "stage5e_controlled_experimentation",
        },
        "artifacts": {
            "config": {
                "path": str(artifact_paths["config"]),
                "sha256": file_sha256(artifact_paths["config"]),
            },
            "sft_dataset": {
                "path": str(artifact_paths["sft_dataset"]),
                "sha256": file_sha256(artifact_paths["sft_dataset"]),
            },
            "eval_cards": {
                "path": str(artifact_paths["eval_cards"]),
                "sha256": file_sha256(artifact_paths["eval_cards"]),
            },
        },
        "paired_eval": {"cards": ["baseline", "candidate"]},
        "boundary": "controlled_experiment_one_primary_variable",
    }


def _write_manifest_artifacts(tmp_path: Path) -> dict[str, Path]:
    config = tmp_path / "sft.yaml"
    sft_dataset = tmp_path / "sft.jsonl"
    eval_cards = tmp_path / "eval_cards.jsonl"
    config.write_text("learning_rate: 0.0002\n", encoding="utf-8")
    sft_dataset.write_text('{"text": "train"}\n', encoding="utf-8")
    eval_cards.write_text('{"id": "eval-1"}\n', encoding="utf-8")
    return {
        "config": config,
        "sft_dataset": sft_dataset,
        "eval_cards": eval_cards,
    }


def _load_run_experiment_matrix():
    script_path = REPO_ROOT / "scripts" / "run_experiment_matrix.py"
    spec = importlib.util.spec_from_file_location("run_experiment_matrix", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
