from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.artifact_manifest import file_sha256, summarize_jsonl_artifact
from small_model_train.preflight_reports import read_preflight_report
from small_model_train.run_manifest import build_run_manifest, write_run_manifest
from small_model_train.stage2_adapter import check_adapter_dir
from small_model_train.stage2_training import (
    build_train_run,
    run_training_dry,
    run_training_subprocess,
    validate_training_inputs,
)
from small_model_train.style_contract import read_style_contract_asset

DEFAULT_MODEL_DIR = r"E:\models\Qwen3-4B-Instruct-2507"


def validate_full_training_prerequisites(
    model_report: str | Path,
    env_report: str | Path,
    smoke_adapter_dir: str | Path,
) -> dict[str, object]:
    errors = []
    for label, expected_kind, raw_path in (
        ("model", "model", model_report),
        ("environment", "environment", env_report),
    ):
        try:
            report = read_preflight_report(raw_path, expected_kind=expected_kind)
        except ValueError as exc:
            errors.append(f"{label} preflight report invalid: {exc}")
            continue
        if report["passed"] is not True:
            errors.append(f"{label} preflight report did not pass: {raw_path}")
            errors.extend(
                f"{label} preflight error: {error}"
                for error in report.get("errors", [])
            )

    adapter_result = check_adapter_dir(smoke_adapter_dir)
    if not adapter_result["passed"]:
        errors.append(f"smoke adapter check failed: {smoke_adapter_dir}")
        errors.extend(
            f"missing smoke adapter file: {name}"
            for name in adapter_result["missing_files"]
        )
        errors.extend(
            f"zero-size smoke adapter file: {name}"
            for name in adapter_result["zero_size_files"]
        )
        errors.extend(str(error) for error in adapter_result["errors"])

    return {"passed": not errors, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", default="configs/sft_qlora_qwen3_4b.yaml")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", default="outputs/sft_v1")
    parser.add_argument("--log-dir", default="logs/training")
    parser.add_argument("--sft-dataset", default="data_sft/sft_chapter_v1.jsonl")
    parser.add_argument("--eval-cards", default="data_cards/eval_execution_cards_50.jsonl")
    parser.add_argument("--model-report-json", default="reports/model_check_report.json")
    parser.add_argument("--env-report-json", default="reports/training_env_report.json")
    parser.add_argument("--smoke-adapter-dir", default="outputs/sft_smoke")
    parser.add_argument("--style-contract-json")
    parser.add_argument("--skip-prereq-checks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    validation = validate_training_inputs(args.sft_dataset, args.eval_cards)
    if not validation["passed"]:
        for error in validation["errors"]:
            print(error, file=sys.stderr)
        return 1
    input_artifacts = validation.get("artifacts", {})

    if not args.skip_prereq_checks:
        prerequisites = validate_full_training_prerequisites(
            model_report=args.model_report_json,
            env_report=args.env_report_json,
            smoke_adapter_dir=args.smoke_adapter_dir,
        )
        if not prerequisites["passed"]:
            for error in prerequisites["errors"]:
                print(error, file=sys.stderr)
            return 1

    preflight_reports = _preflight_reports_for_manifest(
        model_report=args.model_report_json,
        env_report=args.env_report_json,
        skipped=args.skip_prereq_checks,
    )
    style_contract_summary = _style_contract_for_manifest(args.style_contract_json)
    if (
        style_contract_summary is not None
        and not style_contract_summary["schema"]["valid"]
    ):
        for error in style_contract_summary["schema"]["errors"]:
            print(error, file=sys.stderr)
        return 1

    run = build_train_run(
        name="sft_v1",
        source_config=args.config,
        model_dir=args.model_dir,
        output_dir=Path(args.output_dir),
        log_dir=Path(args.log_dir),
        smoke=False,
    )
    # full-run dry-run validates prerequisites and the launch command without consuming GPU memory.
    result = run_training_dry(run) if args.dry_run else run_training_subprocess(run)
    print(result["command_text"])

    training_exit_code = int(result["exit_code"])
    adapter_check = _adapter_check_for_run(args.output_dir, args.dry_run, training_exit_code)
    manifest_passed = _manifest_passed(args.dry_run, training_exit_code, adapter_check)
    sft_summary = summarize_jsonl_artifact(
        args.sft_dataset,
        label="sft_dataset",
        validate_execution_card_schema=False,
    )
    eval_summary = input_artifacts.get("eval_cards")
    formal_evidence = (
        not args.dry_run
        and training_exit_code == 0
        and all(report.get("passed") is True for report in preflight_reports.values())
        and adapter_check.get("passed") is True
        and sft_summary.get("schema", {}).get("valid") is True
        and isinstance(eval_summary, dict)
        and eval_summary.get("schema", {}).get("valid") is True
        and style_contract_summary is not None
        and style_contract_summary.get("schema", {}).get("valid") is True
        and style_contract_summary.get("approval_status") in {"approved", "frozen"}
    )
    write_run_manifest(
        Path(args.output_dir) / "run_manifest.json",
        build_run_manifest(
            run_name=run["name"],
            command=run["command"],
            training_exit_code=training_exit_code,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            config_path=run["config_path"],
            preflight_reports=preflight_reports,
            adapter_check=adapter_check,
            passed=manifest_passed,
            sft_dataset=sft_summary,
            eval_cards=eval_summary,
            style_contract=style_contract_summary,
            formal_evidence=formal_evidence,
            repo_root=REPO_ROOT,
        ),
    )

    if training_exit_code == 0 and not args.dry_run and not adapter_check["passed"]:
        _print_adapter_check_errors(adapter_check)
        return 1
    return training_exit_code


def _style_contract_for_manifest(path: str | Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    try:
        asset = read_style_contract_asset(path)
    except ValueError as exc:
        return {
            "path": str(path),
            "schema": {
                "name": "style_contract",
                "valid": False,
                "errors": [str(exc)],
            },
        }
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "style_contract_id": asset["style_contract_id"],
        "contract_sha256": asset["contract_sha256"],
        "approval_status": asset["approval_status"],
        "schema": {"name": "style_contract", "valid": True, "errors": []},
    }


def _preflight_reports_for_manifest(
    *,
    model_report: str | Path,
    env_report: str | Path,
    skipped: bool,
) -> dict[str, object]:
    reports = {}
    for label, expected_kind, raw_path in (
        ("model", "model", model_report),
        ("environment", "environment", env_report),
    ):
        if skipped:
            reports[label] = {
                "path": str(raw_path),
                "kind": expected_kind,
                "status": "skipped",
                "passed": False,
                "errors": [],
                "warnings": [],
            }
            continue
        report = read_preflight_report(raw_path, expected_kind=expected_kind)
        reports[label] = {
            "path": str(raw_path),
            "kind": report["kind"],
            "passed": report["passed"],
            "checked_at": report["checked_at"],
            "errors": report["errors"],
            "warnings": report["warnings"],
        }
    return reports


def _adapter_check_for_run(
    output_dir: str | Path,
    dry_run: bool,
    training_exit_code: int,
) -> dict[str, object]:
    if dry_run:
        return {
            "status": "skipped",
            "passed": False,
            "reason": "dry-run does not produce an adapter",
        }
    if training_exit_code != 0:
        return {
            "status": "not_run",
            "passed": False,
            "reason": "training command exited nonzero",
        }
    result = check_adapter_dir(output_dir)
    result["status"] = "checked"
    return result


def _manifest_passed(
    dry_run: bool,
    training_exit_code: int,
    adapter_check: dict[str, object],
) -> bool:
    if dry_run:
        return training_exit_code == 0
    return training_exit_code == 0 and adapter_check.get("passed") is True


def _print_adapter_check_errors(adapter_check: dict[str, object]) -> None:
    print("trained adapter check failed", file=sys.stderr)
    for name in adapter_check.get("missing_files", []):
        print(f"missing adapter file: {name}", file=sys.stderr)
    for name in adapter_check.get("zero_size_files", []):
        print(f"zero-size adapter file: {name}", file=sys.stderr)
    for error in adapter_check.get("errors", []):
        print(str(error), file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
