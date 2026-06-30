from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
BOUNDARY = "controlled_experiment_one_primary_variable"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def file_sha256(path: str | Path) -> str:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise ValueError(f"artifact missing: {artifact_path}")

    digest = hashlib.sha256()
    with artifact_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_experiment_manifest(
    *,
    experiment_id: str,
    baseline_run_id: str,
    candidate_run_id: str,
    primary_variable: dict[str, Any],
    stage5e_entry_check: str | Path,
    artifact_paths: dict[str, str | Path],
    paired_eval: dict[str, Any],
    controlled_variables: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    entry_path = Path(stage5e_entry_check)
    gate = json.loads(entry_path.read_text(encoding="utf-8"))
    if not isinstance(gate, dict):
        raise ValueError("Stage 5E entry gate must be a JSON object")
    if gate.get("passed") is not True:
        raise ValueError("Stage 5E entry gate must pass")

    artifacts = {
        name: {
            "path": str(path),
            "sha256": file_sha256(path),
        }
        for name, path in artifact_paths.items()
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now_iso(),
        "experiment_id": experiment_id,
        "baseline_run_id": baseline_run_id,
        "candidate_run_id": candidate_run_id,
        "primary_variable": copy.deepcopy(primary_variable),
        "controlled_variables": copy.deepcopy(controlled_variables or []),
        "stage5e_entry": {
            "path": str(entry_path),
            "passed": True,
            "entry": copy.deepcopy(gate.get("entry", gate)),
        },
        "artifacts": artifacts,
        "paired_eval": copy.deepcopy(paired_eval),
        "boundary": BOUNDARY,
    }


def validate_experiment_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("experiment manifest must be an object")

    required_fields = (
        "schema_version",
        "created_at",
        "experiment_id",
        "baseline_run_id",
        "candidate_run_id",
        "primary_variable",
        "controlled_variables",
        "stage5e_entry",
        "artifacts",
        "paired_eval",
        "boundary",
    )
    for field in required_fields:
        if field not in manifest:
            raise ValueError(f"missing required field: {field}")

    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    if manifest["boundary"] != BOUNDARY:
        raise ValueError("unsupported experiment boundary")

    _require_non_empty_string(manifest["created_at"], "created_at")
    if not manifest["created_at"].endswith("Z"):
        raise ValueError("created_at must be a UTC ISO string ending Z")
    for field in ("experiment_id", "baseline_run_id", "candidate_run_id"):
        _require_non_empty_string(manifest[field], field)

    _validate_primary_variable(manifest["primary_variable"])
    _validate_controlled_variables(manifest["controlled_variables"])
    _validate_stage5e_entry(manifest["stage5e_entry"])
    _validate_artifacts(manifest["artifacts"])

    if not isinstance(manifest["paired_eval"], dict):
        raise ValueError("paired_eval must be an object")
    return manifest


def write_experiment_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    validated = validate_experiment_manifest(manifest)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(validated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _validate_primary_variable(primary_variable: Any) -> None:
    if not isinstance(primary_variable, dict):
        raise ValueError("primary_variable must be an object")
    _require_non_empty_string(primary_variable.get("name"), "primary_variable.name")
    if "baseline_value" not in primary_variable:
        raise ValueError("primary_variable.baseline_value is required")
    if "candidate_value" not in primary_variable:
        raise ValueError("primary_variable.candidate_value is required")
    if primary_variable["baseline_value"] == primary_variable["candidate_value"]:
        raise ValueError("primary variable must change")


def _validate_controlled_variables(controlled_variables: Any) -> None:
    if not isinstance(controlled_variables, list):
        raise ValueError("controlled_variables must be a list")
    for index, variable in enumerate(controlled_variables):
        if not isinstance(variable, dict):
            raise ValueError(f"controlled_variables[{index}] must be an object")
        _require_non_empty_string(variable.get("name"), f"controlled_variables[{index}].name")
        if "baseline_value" not in variable:
            raise ValueError(f"controlled_variables[{index}].baseline_value is required")
        if "candidate_value" not in variable:
            raise ValueError(f"controlled_variables[{index}].candidate_value is required")
        if variable["baseline_value"] != variable["candidate_value"]:
            raise ValueError("controlled variable changed")


def _validate_stage5e_entry(stage5e_entry: Any) -> None:
    if not isinstance(stage5e_entry, dict):
        raise ValueError("stage5e_entry must be an object")
    _require_non_empty_string(stage5e_entry.get("path"), "stage5e_entry.path")
    if stage5e_entry.get("passed") is not True:
        raise ValueError("Stage 5E entry gate must pass")
    if not isinstance(stage5e_entry.get("entry"), dict):
        raise ValueError("stage5e_entry.entry must be an object")


def _validate_artifacts(artifacts: Any) -> None:
    if not isinstance(artifacts, dict):
        raise ValueError("artifacts must be an object")
    for name, artifact in artifacts.items():
        _require_non_empty_string(name, "artifact name")
        if not isinstance(artifact, dict):
            raise ValueError(f"artifact {name} must be an object")
        _require_non_empty_string(artifact.get("path"), f"artifacts.{name}.path")
        if not _is_sha256(artifact.get("sha256")):
            raise ValueError(f"artifacts.{name}.sha256 must be 64-character lowercase hex")


def _require_non_empty_string(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.match(value) is not None
