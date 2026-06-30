from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_experiment_matrix_writes_candidate_dry_run_command(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "commands.jsonl"
    manifest_path.write_text(json.dumps(_manifest()) + "\n", encoding="utf-8")

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
                "configs/sft.yaml",
                "--run-name",
                "candidate",
            ],
        }
    ]


def test_run_experiment_matrix_requires_config_artifact(tmp_path):
    manifest = _manifest()
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
    manifest = _manifest()
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


def test_remove_output_file_ignores_unlink_errors(tmp_path, monkeypatch):
    output_path = tmp_path / "commands.jsonl"
    output_path.write_text('{"stale": true}\n', encoding="utf-8")
    module = _load_run_experiment_matrix()

    def raise_oserror(self):
        raise OSError("locked")

    monkeypatch.setattr(type(output_path), "unlink", raise_oserror)

    module._remove_output_file(str(output_path))


def _manifest() -> dict:
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
                "path": "configs/sft.yaml",
                "sha256": "a" * 64,
            }
        },
        "paired_eval": {"cards": ["baseline", "candidate"]},
        "boundary": "controlled_experiment_one_primary_variable",
    }


def _load_run_experiment_matrix():
    script_path = REPO_ROOT / "scripts" / "run_experiment_matrix.py"
    spec = importlib.util.spec_from_file_location("run_experiment_matrix", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
