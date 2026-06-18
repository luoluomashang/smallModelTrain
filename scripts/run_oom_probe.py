from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.stage2_oom_probe import (
    PROBES,
    render_probe_report,
    run_oom_probes,
)

DEFAULT_MODEL_DIR = r"E:\models\Qwen3-4B-Instruct-2507"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--cards", default="data_cards/eval_cards_50.jsonl")
    parser.add_argument("--sft-dataset", default="data_sft/sft_chapter_v1.jsonl")
    parser.add_argument("--config", default="configs/sft_qlora_qwen3_4b.yaml")
    parser.add_argument("--log-dir", default="logs/training/oom_probe")
    parser.add_argument("--report", default="reports/oom_probe_report.md")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.dry_run:
        results = None
    else:
        results = run_oom_probes(
            model_dir=args.model_dir,
            cards=args.cards,
            sft_dataset=args.sft_dataset,
            config=args.config,
            log_dir=args.log_dir,
        )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_probe_report(results), encoding="utf-8")
    print(f"wrote OOM probe report to {report_path}")

    if results is not None and any(result["status"] == "failed" for result in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
