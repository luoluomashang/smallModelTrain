from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.stage2_training import (
    build_train_run,
    run_training_dry,
    run_training_subprocess,
    validate_training_inputs,
)

DEFAULT_MODEL_DIR = r"E:\models\Qwen3-4B-Instruct-2507"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft_qlora_qwen3_4b.yaml")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", default="outputs/sft_smoke")
    parser.add_argument("--log-dir", default="logs/training")
    parser.add_argument("--sft-dataset", default="data_sft/sft_chapter_v1.jsonl")
    parser.add_argument("--eval-cards", default="data_cards/eval_execution_cards_50.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    validation = validate_training_inputs(args.sft_dataset, args.eval_cards)
    if not validation["passed"]:
        for error in validation["errors"]:
            print(error, file=sys.stderr)
        return 1

    run = build_train_run(
        name="sft_smoke",
        source_config=args.config,
        model_dir=args.model_dir,
        output_dir=Path(args.output_dir),
        log_dir=Path(args.log_dir),
        smoke=True,
    )
    # dry-run is a preflight for command construction and should not be described as a trained adapter.
    result = run_training_dry(run) if args.dry_run else run_training_subprocess(run)
    print(result["command_text"])
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
