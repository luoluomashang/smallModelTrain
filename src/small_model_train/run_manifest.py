from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

SCHEMA_VERSION = 1


def current_git_commit(cwd: str | Path | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            cwd=cwd,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def build_run_manifest(
    *,
    run_name: str,
    command: Sequence[str],
    training_exit_code: int,
    model_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    preflight_reports: dict[str, Any],
    adapter_check: dict[str, Any],
    passed: bool,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now_iso(),
        "git_commit": current_git_commit(repo_root),
        "run_name": run_name,
        "command": [str(part) for part in command],
        "training_exit_code": int(training_exit_code),
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "config_path": str(config_path),
        "preflight_reports": preflight_reports,
        "adapter_check": adapter_check,
        "passed": bool(passed),
    }


def write_run_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
