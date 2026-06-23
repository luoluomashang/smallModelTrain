from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.stage2_adapter import check_adapter_dir
from small_model_train.stage2_training import (
    build_train_run,
    run_training_dry,
    run_training_subprocess,
    validate_training_inputs,
)

DEFAULT_MODEL_DIR = r"E:\models\Qwen3-4B-Instruct-2507"


def validate_full_training_prerequisites(
    model_report: str | Path,
    env_report: str | Path,
    smoke_adapter_dir: str | Path,
) -> dict[str, object]:
    errors = []
    for label, raw_path in (
        ("model report", model_report),
        ("training env report", env_report),
    ):
        path = Path(raw_path)
        if not path.is_file():
            errors.append(f"{label} is missing: {path}")
        elif path.stat().st_size == 0:
            errors.append(f"{label} is empty: {path}")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft_qlora_qwen3_4b.yaml")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", default="outputs/sft_v1")
    parser.add_argument("--log-dir", default="logs/training")
    parser.add_argument("--sft-dataset", default="data_sft/sft_chapter_v1.jsonl")
    parser.add_argument("--eval-cards", default="data_cards/eval_execution_cards_50.jsonl")
    parser.add_argument("--model-report", default="reports/model_check_report.md")
    parser.add_argument("--env-report", default="reports/training_env_report.md")
    parser.add_argument("--smoke-adapter-dir", default="outputs/sft_smoke")
    parser.add_argument("--skip-prereq-checks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    validation = validate_training_inputs(args.sft_dataset, args.eval_cards)
    if not validation["passed"]:
        for error in validation["errors"]:
            print(error, file=sys.stderr)
        return 1

    if not args.skip_prereq_checks:
        prerequisites = validate_full_training_prerequisites(
            model_report=args.model_report,
            env_report=args.env_report,
            smoke_adapter_dir=args.smoke_adapter_dir,
        )
        if not prerequisites["passed"]:
            for error in prerequisites["errors"]:
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
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
