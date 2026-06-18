from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.stage2_config import (
    build_llamafactory_command,
    make_training_snapshot,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=int, choices=range(1, 8), required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--cards", required=True)
    parser.add_argument("--sft-dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--log-dir", required=True)
    args = parser.parse_args(argv)

    if args.probe == 1:
        return _probe_tokenizer_and_config(args.model_dir)
    if args.probe == 2:
        return _probe_base_model_4bit(args.model_dir)
    if args.probe == 3:
        return _probe_lora(args.model_dir)
    if args.probe == 4:
        return _probe_tokenize_one_sample(args.model_dir, args.cards)
    return _probe_one_step_training(args)


def _probe_tokenizer_and_config(model_dir: str) -> int:
    from transformers import AutoConfig, AutoTokenizer

    AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    print("Loading tokenizer and config succeeded")
    return 0


def _probe_base_model_4bit(model_dir: str) -> int:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    print("load 4-bit base model succeeded")
    return 0


def _probe_lora(model_dir: str) -> int:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    get_peft_model(model, lora_config)
    print("Preparing LoRA adapter succeeded")
    return 0


def _probe_tokenize_one_sample(model_dir: str, cards: str) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    card = _load_first_jsonl(cards)
    prompt = json.dumps(card, ensure_ascii=False)
    tokenizer(prompt, truncation=True, max_length=8192)
    print("Tokenize dataset sample succeeded")
    return 0


def _probe_one_step_training(args: argparse.Namespace) -> int:
    sft_dataset = Path(args.sft_dataset)
    probe_overrides = {
        5: {"cutoff_len": 8192, "max_steps": 1},
        6: {"cutoff_len": 6144, "max_steps": 1},
        7: {"cutoff_len": 6144, "lora_rank": 8, "max_steps": 1},
    }
    overrides = {
        **probe_overrides[args.probe],
        "dataset": str(sft_dataset),
        "dataset_dir": str(sft_dataset.parent),
    }
    log_dir = Path(args.log_dir)
    probe_dir = log_dir / f"probe_{args.probe}"
    config_path = probe_dir / "training_config_snapshot.yaml"
    output_dir = probe_dir / "adapter"
    snapshot = make_training_snapshot(
        source_config=args.config,
        output_config=config_path,
        model_dir=args.model_dir,
        output_dir=output_dir,
        overrides=overrides,
    )
    print(
        "one-step training snapshot: "
        f"cutoff_len={snapshot.get('cutoff_len')}, "
        f"lora_rank={snapshot.get('lora_rank')}, "
        f"max_steps={snapshot.get('max_steps')}"
    )
    result = subprocess.run(
        build_llamafactory_command(config_path),
        capture_output=True,
        check=False,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
    return int(result.returncode)


def _load_first_jsonl(path: str | Path) -> dict[str, Any]:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            row = json.loads(stripped)
            if isinstance(row, dict):
                return row
            raise ValueError("first JSONL row must be an object")
    raise ValueError(f"no JSONL rows found in {path}")


if __name__ == "__main__":
    raise SystemExit(main())
