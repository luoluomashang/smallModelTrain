from __future__ import annotations

import argparse
import subprocess
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.io_utils import write_jsonl
from small_model_train.stage2_inference import (
    build_generation_row,
    default_inference_params,
    load_eval_cards,
    render_eval_prompt,
)
from small_model_train.stage2_monitoring import (
    append_event,
    classify_training_error,
)

DEFAULT_MODEL_DIR = r"E:\models\Qwen3-4B-Instruct-2507"
PHASE = "eval_first_generation"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", default="data_cards/eval_cards_50.jsonl")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--adapter-dir", default="outputs/sft_v1")
    parser.add_argument("--output", default="outputs/sft_v1/generated.jsonl")
    parser.add_argument("--model-name", default="sft_v1")
    parser.add_argument(
        "--event-log",
        default="logs/training/sft_v1_eval_events.jsonl",
    )
    parser.add_argument(
        "--stderr-log",
        default="logs/training/sft_v1_eval_stderr.log",
    )
    parser.add_argument(
        "--stdout-log",
        default="logs/training/sft_v1_eval_stdout.log",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-new-tokens", type=positive_int)
    args = parser.parse_args(argv)

    if args.dry_run:
        try:
            _run_dry(args.cards, args.output, args.model_name, args.max_new_tokens)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from exc
        return 0

    command = _build_worker_command(args)
    append_event(args.event_log, PHASE, "start", {"command": command})

    try:
        returncode, stdout, stderr = _run_worker_streaming(command)
    except (FileNotFoundError, OSError) as exc:
        returncode = 127
        stdout = ""
        stderr = f"{type(exc).__name__}: {exc}"

    _write_text_log(args.stdout_log, stdout)
    _write_text_log(args.stderr_log, stderr)

    if returncode == 0:
        append_event(args.event_log, PHASE, "ok", {"exit_code": returncode})
        return 0

    error = classify_training_error(stderr + "\n" + stdout, returncode)
    append_event(
        args.event_log,
        PHASE,
        "failed",
        {"exit_code": returncode, "error": error},
    )
    print(f"{error['error_type']}: {error['suggestion']}", file=sys.stderr)
    raise SystemExit(returncode)


def _run_worker_streaming(command: list[str]) -> tuple[int, str, str]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []

    def read_stderr() -> None:
        if process.stderr is None:
            return
        stderr_parts.append(process.stderr.read())

    stderr_thread = threading.Thread(target=read_stderr)
    stderr_thread.start()

    if process.stdout is not None:
        for line in process.stdout:
            stdout_parts.append(line)
            print(line, end="", flush=True)

    returncode = int(process.wait())
    stderr_thread.join()
    return returncode, "".join(stdout_parts), "".join(stderr_parts)


def _run_dry(
    cards_path: str | Path,
    output_path: str | Path,
    model_name: str,
    max_new_tokens: int | None = None,
) -> None:
    params = default_inference_params()
    if max_new_tokens is not None:
        params = dict(params)
        params["max_new_tokens"] = max_new_tokens
    rows = []
    for card in load_eval_cards(cards_path):
        sample_id = str(card.get("id", ""))
        output = "[DRY RUN] " + render_eval_prompt(card)[:80]
        rows.append(build_generation_row(sample_id, output, model_name, params))
    write_jsonl(output_path, rows)


def _build_worker_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "stage2_eval_worker.py"),
        "--cards",
        str(args.cards),
        "--model-dir",
        str(args.model_dir),
        "--adapter-dir",
        str(args.adapter_dir),
        "--output",
        str(args.output),
        "--model-name",
        str(args.model_name),
    ]
    if args.max_new_tokens is not None:
        command.extend(["--max-new-tokens", str(args.max_new_tokens)])
    return command


def _write_text_log(path: str | Path, text: str) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
