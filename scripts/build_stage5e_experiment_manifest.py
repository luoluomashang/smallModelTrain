from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.evaluation.experiment_manifest import (  # noqa: E402
    build_experiment_manifest,
    write_experiment_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--baseline-run-id", required=True)
    parser.add_argument("--candidate-run-id", required=True)
    parser.add_argument("--primary-variable-name", required=True)
    parser.add_argument("--primary-baseline-value", required=True)
    parser.add_argument("--primary-candidate-value", required=True)
    parser.add_argument("--stage5e-entry-check", required=True)
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument(
        "--controlled-variable",
        action="append",
        default=[],
        help="name=baseline_value=candidate_value",
    )
    parser.add_argument("--paired-eval-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        artifacts = _parse_artifacts(args.artifact)
        controlled_variables = _parse_controlled_variables(args.controlled_variable)
        paired_eval = json.loads(args.paired_eval_json)
        if not isinstance(paired_eval, dict):
            raise ValueError("paired eval JSON must be an object")
        manifest = build_experiment_manifest(
            experiment_id=args.experiment_id,
            baseline_run_id=args.baseline_run_id,
            candidate_run_id=args.candidate_run_id,
            primary_variable={
                "name": args.primary_variable_name,
                "baseline_value": args.primary_baseline_value,
                "candidate_value": args.primary_candidate_value,
            },
            controlled_variables=controlled_variables,
            stage5e_entry_check=args.stage5e_entry_check,
            artifact_paths=artifacts,
            paired_eval=paired_eval,
        )
        write_experiment_manifest(args.output, manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote Stage 5E experiment manifest to {args.output}")
    return 0


def _parse_artifacts(values: list[str]) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"malformed artifact: {value}")
        name, path = value.split("=", 1)
        if not name.strip() or not path.strip():
            raise ValueError(f"malformed artifact: {value}")
        if name in artifacts:
            raise ValueError(f"duplicate artifact name: {name}")
        artifacts[name] = path
    return artifacts


def _parse_controlled_variables(values: list[str]) -> list[dict[str, str]]:
    controlled_variables: list[dict[str, str]] = []
    names: set[str] = set()
    for value in values:
        parts = value.split("=", 2)
        if len(parts) != 3:
            raise ValueError(
                f"controlled variable must be name=baseline_value=candidate_value: {value}"
            )
        name, baseline_value, candidate_value = (part.strip() for part in parts)
        if not name or not baseline_value or not candidate_value:
            raise ValueError(
                f"controlled variable must be name=baseline_value=candidate_value: {value}"
            )
        if name in names:
            raise ValueError(f"duplicate controlled variable name: {name}")
        names.add(name)
        controlled_variables.append(
            {
                "name": name,
                "baseline_value": baseline_value,
                "candidate_value": candidate_value,
            }
        )
    return controlled_variables


if __name__ == "__main__":
    raise SystemExit(main())
