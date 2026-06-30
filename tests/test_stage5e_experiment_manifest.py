from __future__ import annotations

import json
import re

import pytest

from small_model_train.evaluation.experiment_manifest import (
    build_experiment_manifest,
    validate_experiment_manifest,
    write_experiment_manifest,
)


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def test_build_experiment_manifest_records_gate_primary_variable_and_artifact_hashes(
    tmp_path,
):
    stage5e_entry = tmp_path / "stage5e_entry.json"
    stage5e_entry.write_text(
        json.dumps({"passed": True, "entry": "stage5e_controlled_experimentation"})
        + "\n",
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
    assert manifest["stage5e_entry"]["entry"] == "stage5e_controlled_experimentation"
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
    assert written_manifest["stage5e_entry"]["entry"] == "stage5e_controlled_experimentation"


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


def test_validate_experiment_manifest_rejects_changed_controlled_variable(tmp_path):
    stage5e_entry = tmp_path / "stage5e_entry.json"
    stage5e_entry.write_text(
        json.dumps({"passed": True}) + "\n",
        encoding="utf-8",
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")
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
                "candidate_value": "21",
            }
        ],
        stage5e_entry_check=stage5e_entry,
        artifact_paths={"config": artifact},
        paired_eval={},
    )

    with pytest.raises(ValueError, match="controlled variable changed"):
        validate_experiment_manifest(manifest)
