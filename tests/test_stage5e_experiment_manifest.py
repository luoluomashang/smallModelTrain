from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from small_model_train.evaluation.experiment_manifest import (
    build_experiment_manifest,
    file_sha256,
    validate_experiment_manifest,
    write_experiment_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
STAGE5E_ENTRY_MARKER = "stage5e_controlled_experimentation"


def test_build_experiment_manifest_records_gate_primary_variable_and_artifact_hashes(
    tmp_path,
):
    stage5e_entry = tmp_path / "stage5e_entry.json"
    stage5e_entry.write_text(
        json.dumps({"passed": True, "entry": STAGE5E_ENTRY_MARKER}) + "\n",
        encoding="utf-8",
    )
    sft_dataset = tmp_path / "sft.jsonl"
    eval_cards = tmp_path / "eval_cards.jsonl"
    config = tmp_path / "config.json"
    sft_dataset.write_text('{"text": "train"}\n', encoding="utf-8")
    eval_cards.write_text('{"id": "eval-1"}\n', encoding="utf-8")
    config.write_text('{"learning_rate": 0.0002}\n', encoding="utf-8")

    paired_eval = {"metric": "win_rate", "baseline": 0.42, "candidate": 0.57}
    manifest = build_experiment_manifest(
        experiment_id="exp-stage5e-001",
        baseline_run_id="baseline-run",
        candidate_run_id="candidate-run",
        primary_variable={
            "name": "learning_rate",
            "baseline_value": "0.0001",
            "candidate_value": "0.0002",
        },
        stage5e_entry_check=stage5e_entry,
        artifact_paths={
            "sft_dataset": sft_dataset,
            "eval_cards": eval_cards,
            "config": config,
        },
        paired_eval=paired_eval,
    )

    assert manifest["schema_version"] == 1
    assert manifest["stage5e_entry"]["passed"] is True
    assert manifest["stage5e_entry"]["entry"] == STAGE5E_ENTRY_MARKER
    assert manifest["primary_variable"]["name"] == "learning_rate"
    assert set(manifest["artifacts"]) == {"sft_dataset", "eval_cards", "config"}
    for artifact_name, artifact in manifest["artifacts"].items():
        assert artifact["path"] == str(
            {
                "sft_dataset": sft_dataset,
                "eval_cards": eval_cards,
                "config": config,
            }[artifact_name]
        )
        assert SHA256_RE.match(artifact["sha256"])
    assert manifest["paired_eval"] == paired_eval

    output_path = tmp_path / "manifest.json"
    write_experiment_manifest(output_path, manifest)
    written_manifest = json.loads(output_path.read_text(encoding="utf-8"))
    assert written_manifest["stage5e_entry"]["entry"] == STAGE5E_ENTRY_MARKER


def test_build_experiment_manifest_rejects_failed_stage5e_gate(tmp_path):
    stage5e_entry = tmp_path / "stage5e_entry.json"
    stage5e_entry.write_text(
        json.dumps({"passed": False, "errors": ["stage 5d incomplete"]}) + "\n",
        encoding="utf-8",
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Stage 5E entry gate must pass"):
        build_experiment_manifest(
            experiment_id="exp-stage5e-001",
            baseline_run_id="baseline-run",
            candidate_run_id="candidate-run",
            primary_variable={
                "name": "learning_rate",
                "baseline_value": "0.0001",
                "candidate_value": "0.0002",
            },
            stage5e_entry_check=stage5e_entry,
            artifact_paths={"config": artifact},
            paired_eval={},
        )


def test_build_experiment_manifest_rejects_unchanged_primary_variable(tmp_path):
    stage5e_entry = _write_stage5e_entry(tmp_path)
    artifact = _write_artifact(tmp_path)

    with pytest.raises(ValueError, match="primary variable must change"):
        build_experiment_manifest(
            experiment_id="exp-stage5e-001",
            baseline_run_id="baseline-run",
            candidate_run_id="candidate-run",
            primary_variable={
                "name": "learning_rate",
                "baseline_value": "0.0002",
                "candidate_value": "0.0002",
            },
            stage5e_entry_check=stage5e_entry,
            artifact_paths={"config": artifact},
            paired_eval={},
        )


def test_build_experiment_manifest_rejects_wrong_stage5e_entry_marker(tmp_path):
    stage5e_entry = tmp_path / "stage5e_entry.json"
    stage5e_entry.write_text(
        json.dumps({"passed": True, "entry": "not_stage5e"}) + "\n",
        encoding="utf-8",
    )
    artifact = _write_artifact(tmp_path)

    with pytest.raises(ValueError, match="Stage 5E entry gate marker"):
        build_experiment_manifest(
            experiment_id="exp-stage5e-001",
            baseline_run_id="baseline-run",
            candidate_run_id="candidate-run",
            primary_variable={
                "name": "learning_rate",
                "baseline_value": "0.0001",
                "candidate_value": "0.0002",
            },
            stage5e_entry_check=stage5e_entry,
            artifact_paths={"config": artifact},
            paired_eval={},
        )


def test_file_sha256_rejects_directory(tmp_path):
    with pytest.raises(ValueError, match="artifact file not found or not a file"):
        file_sha256(tmp_path)


def test_build_stage5e_experiment_manifest_cli_writes_manifest(tmp_path):
    stage5e_entry = _write_stage5e_entry(tmp_path)
    artifact = _write_artifact(tmp_path)
    output_path = tmp_path / "manifest.json"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_stage5e_experiment_manifest.py"),
            "--experiment-id",
            "exp-stage5e-001",
            "--baseline-run-id",
            "baseline-run",
            "--candidate-run-id",
            "candidate-run",
            "--primary-variable-name",
            "learning_rate",
            "--primary-baseline-value",
            "0.0001",
            "--primary-candidate-value",
            "0.0002",
            "--stage5e-entry-check",
            str(stage5e_entry),
            "--artifact",
            f"config={artifact}",
            "--paired-eval-json",
            '{"metric": "win_rate"}',
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == f"wrote Stage 5E experiment manifest to {output_path}"
    written_manifest = json.loads(output_path.read_text(encoding="utf-8"))
    assert written_manifest["stage5e_entry"]["entry"] == STAGE5E_ENTRY_MARKER


def test_build_stage5e_experiment_manifest_cli_records_controlled_variable(tmp_path):
    stage5e_entry = _write_stage5e_entry(tmp_path)
    artifact = _write_artifact(tmp_path)
    output_path = tmp_path / "manifest.json"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_stage5e_experiment_manifest.py"),
            "--experiment-id",
            "exp-stage5e-001",
            "--baseline-run-id",
            "baseline-run",
            "--candidate-run-id",
            "candidate-run",
            "--primary-variable-name",
            "learning_rate",
            "--primary-baseline-value",
            "0.0001",
            "--primary-candidate-value",
            "0.0002",
            "--controlled-variable",
            "seed=20260628=20260628",
            "--stage5e-entry-check",
            str(stage5e_entry),
            "--artifact",
            f"config={artifact}",
            "--paired-eval-json",
            '{"metric": "win_rate"}',
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    written_manifest = json.loads(output_path.read_text(encoding="utf-8"))
    assert written_manifest["controlled_variables"] == [
        {
            "name": "seed",
            "baseline_value": "20260628",
            "candidate_value": "20260628",
        }
    ]


def test_build_stage5e_experiment_manifest_cli_rejects_changed_controlled_variable(
    tmp_path,
):
    stage5e_entry = _write_stage5e_entry(tmp_path)
    artifact = _write_artifact(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_stage5e_experiment_manifest.py"),
            "--experiment-id",
            "exp-stage5e-001",
            "--baseline-run-id",
            "baseline-run",
            "--candidate-run-id",
            "candidate-run",
            "--primary-variable-name",
            "learning_rate",
            "--primary-baseline-value",
            "0.0001",
            "--primary-candidate-value",
            "0.0002",
            "--controlled-variable",
            "rank=8=16",
            "--stage5e-entry-check",
            str(stage5e_entry),
            "--artifact",
            f"config={artifact}",
            "--paired-eval-json",
            '{"metric": "win_rate"}',
            "--output",
            str(tmp_path / "manifest.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "controlled variable changed" in result.stderr


def test_build_stage5e_experiment_manifest_cli_rejects_duplicate_controlled_variable(
    tmp_path,
):
    stage5e_entry = _write_stage5e_entry(tmp_path)
    artifact = _write_artifact(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_stage5e_experiment_manifest.py"),
            "--experiment-id",
            "exp-stage5e-001",
            "--baseline-run-id",
            "baseline-run",
            "--candidate-run-id",
            "candidate-run",
            "--primary-variable-name",
            "learning_rate",
            "--primary-baseline-value",
            "0.0001",
            "--primary-candidate-value",
            "0.0002",
            "--controlled-variable",
            "seed=13=13",
            "--controlled-variable",
            "seed=21=21",
            "--stage5e-entry-check",
            str(stage5e_entry),
            "--artifact",
            f"config={artifact}",
            "--paired-eval-json",
            '{"metric": "win_rate"}',
            "--output",
            str(tmp_path / "manifest.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "duplicate controlled variable name: seed" in result.stderr


def test_build_stage5e_experiment_manifest_cli_rejects_malformed_controlled_variable(
    tmp_path,
):
    stage5e_entry = _write_stage5e_entry(tmp_path)
    artifact = _write_artifact(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_stage5e_experiment_manifest.py"),
            "--experiment-id",
            "exp-stage5e-001",
            "--baseline-run-id",
            "baseline-run",
            "--candidate-run-id",
            "candidate-run",
            "--primary-variable-name",
            "learning_rate",
            "--primary-baseline-value",
            "0.0001",
            "--primary-candidate-value",
            "0.0002",
            "--controlled-variable",
            "seed=20260628",
            "--stage5e-entry-check",
            str(stage5e_entry),
            "--artifact",
            f"config={artifact}",
            "--paired-eval-json",
            '{"metric": "win_rate"}',
            "--output",
            str(tmp_path / "manifest.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert (
        "controlled variable must be name=baseline_value=candidate_value"
        in result.stderr
    )


def test_build_stage5e_experiment_manifest_cli_rejects_malformed_artifact(tmp_path):
    stage5e_entry = _write_stage5e_entry(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_stage5e_experiment_manifest.py"),
            "--experiment-id",
            "exp-stage5e-001",
            "--baseline-run-id",
            "baseline-run",
            "--candidate-run-id",
            "candidate-run",
            "--primary-variable-name",
            "learning_rate",
            "--primary-baseline-value",
            "0.0001",
            "--primary-candidate-value",
            "0.0002",
            "--stage5e-entry-check",
            str(stage5e_entry),
            "--artifact",
            "config",
            "--paired-eval-json",
            '{"metric": "win_rate"}',
            "--output",
            str(tmp_path / "manifest.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "error: malformed artifact: config" in result.stderr


def test_validate_experiment_manifest_rejects_changed_controlled_variable(tmp_path):
    stage5e_entry = _write_stage5e_entry(tmp_path)
    artifact = _write_artifact(tmp_path)
    manifest = build_experiment_manifest(
        experiment_id="exp-stage5e-001",
        baseline_run_id="baseline-run",
        candidate_run_id="candidate-run",
        primary_variable={
            "name": "learning_rate",
            "baseline_value": "0.0001",
            "candidate_value": "0.0002",
        },
        controlled_variables=[
            {
                "name": "seed",
                "baseline_value": "13",
                "candidate_value": "13",
            }
        ],
        stage5e_entry_check=stage5e_entry,
        artifact_paths={"config": artifact},
        paired_eval={},
    )
    manifest["controlled_variables"][0]["candidate_value"] = "21"

    with pytest.raises(ValueError, match="controlled variable changed"):
        validate_experiment_manifest(manifest)


def _write_stage5e_entry(tmp_path: Path) -> Path:
    stage5e_entry = tmp_path / "stage5e_entry.json"
    stage5e_entry.write_text(
        json.dumps({"passed": True, "entry": STAGE5E_ENTRY_MARKER}) + "\n",
        encoding="utf-8",
    )
    return stage5e_entry


def _write_artifact(tmp_path: Path) -> Path:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")
    return artifact
