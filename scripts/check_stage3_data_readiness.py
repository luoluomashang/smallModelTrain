from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.stage3_data_readiness import (
    build_stage3_summary,
    render_stage3_readiness_report,
)

DEFAULT_MODEL_DIR = r"E:\models\Qwen3-4B-Instruct-2507"
READY_DECISION = "ready_for_stage4_smoke_training"


def _run_smoke_dry_run(args: argparse.Namespace) -> dict:
    command = [
        sys.executable,
        "scripts/run_sft_smoke.py",
        "--dry-run",
        "--config",
        args.config,
        "--model-dir",
        args.model_dir,
        "--sft-dataset",
        args.sft_dataset,
        "--eval-cards",
        args.eval_cards,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return {
        "exit_code": result.returncode,
        "command": subprocess.list2cmdline(command),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data_raw/novels")
    parser.add_argument("--chapters-raw", default="data_clean/chapters_raw.jsonl")
    parser.add_argument("--chapters", default="data_clean/chapters.jsonl")
    parser.add_argument("--chapters-split", default="data_clean/chapters_split.jsonl")
    parser.add_argument("--chapter-cards", default="data_cards/chapter_cards.jsonl")
    parser.add_argument("--eval-cards", default="data_cards/eval_cards_20.jsonl")
    parser.add_argument("--sft-dataset", default="data_sft/sft_chapter_v1.jsonl")
    parser.add_argument("--report", default="reports/stage3_data_readiness_report.md")
    parser.add_argument("--config", default="configs/sft_qlora_qwen3_4b.yaml")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--min-trainable-sft", type=int, default=20)
    parser.add_argument("--min-eval-cards", type=int, default=10)
    parser.add_argument("--preferred-eval-cards", type=int, default=50)
    parser.add_argument("--run-smoke-dry-run", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    smoke_dry_run = (
        _run_smoke_dry_run(args)
        if args.run_smoke_dry_run
        else {"exit_code": None, "command": "", "stderr": "smoke dry-run has not been run"}
    )

    summary = build_stage3_summary(
        args.raw_dir,
        args.chapters_raw,
        args.chapters,
        args.chapters_split,
        args.chapter_cards,
        args.eval_cards,
        args.sft_dataset,
        smoke_dry_run=smoke_dry_run,
        min_trainable_sft=args.min_trainable_sft,
        min_eval_cards=args.min_eval_cards,
        preferred_eval_cards=args.preferred_eval_cards,
    )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_stage3_readiness_report(summary), encoding="utf-8")

    print(f"report: {report_path}")
    print(f"decision: {summary['decision']}")
    return 0 if summary["decision"] == READY_DECISION else 1


if __name__ == "__main__":
    raise SystemExit(main())
