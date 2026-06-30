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
    validate_experiment_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        validate_experiment_manifest(manifest)
        rows = [_build_candidate_row(manifest, dry_run=args.dry_run)]
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        _remove_output_file(args.output)
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {len(rows)} Stage 5E experiment commands to {args.output}")
    return 0


def _build_candidate_row(manifest: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    config = manifest["artifacts"].get("config")
    if not isinstance(config, dict) or not config.get("path"):
        raise ValueError("manifest artifact config is required")

    run_id = manifest["candidate_run_id"]
    command = [
        "python",
        "scripts/run_sft_train.py",
        "--config",
        config["path"],
        "--output-dir",
        f"outputs/stage5e/{run_id}",
    ]
    for artifact_name, argument_name in (
        ("sft_dataset", "--sft-dataset"),
        ("eval_cards", "--eval-cards"),
    ):
        artifact = manifest["artifacts"].get(artifact_name)
        if isinstance(artifact, dict) and artifact.get("path"):
            command.extend([argument_name, artifact["path"]])
    if dry_run:
        command.append("--dry-run")

    return {
        "experiment_id": manifest["experiment_id"],
        "run_id": run_id,
        "dry_run": dry_run,
        "primary_variable": manifest["primary_variable"]["name"],
        "command": command,
    }


def _remove_output_file(output: str) -> None:
    output_path = Path(output)
    try:
        if output_path.is_file():
            output_path.unlink()
    except OSError:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
