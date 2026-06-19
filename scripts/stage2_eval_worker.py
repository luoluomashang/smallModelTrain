"""Worker process for fixed eval inference.

The launcher keeps this GPU-heavy path in a child process so stdout, stderr, and
exit codes remain visible even when model loading or generation fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.stage2_inference import (
    build_generation_row,
    default_inference_params,
    load_eval_cards,
    render_eval_prompt,
)


def reset_generation_output(path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")


def append_generation_row(path: str | Path, row: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def print_generation_progress(completed: int, total: int, sample_id: str) -> None:
    print(f"generated {completed}/{total} {sample_id}", flush=True)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_inference_params(max_new_tokens: int | None = None) -> dict:
    params = default_inference_params()
    if max_new_tokens is not None:
        params = dict(params)
        params["max_new_tokens"] = max_new_tokens
    return params


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="sft_v1")
    parser.add_argument("--max-new-tokens", type=positive_int)
    args = parser.parse_args(argv)

    try:
        cards = load_eval_cards(args.cards)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    reset_generation_output(args.output)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    params = build_inference_params(args.max_new_tokens)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    for index, card in enumerate(cards, start=1):
        prompt = render_eval_prompt(card)
        inputs = tokenizer(prompt, return_tensors="pt")
        device = getattr(model, "device", None)
        if device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(device)

        generated = model.generate(
            **inputs,
            max_new_tokens=params["max_new_tokens"],
            temperature=params["temperature"],
            top_p=params["top_p"],
            top_k=params["top_k"],
            repetition_penalty=params["repetition_penalty"],
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        prompt_length = inputs["input_ids"].shape[-1]
        new_tokens = generated[0][prompt_length:]
        output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        sample_id = str(card.get("id", ""))
        append_generation_row(
            args.output,
            build_generation_row(
                sample_id,
                output,
                args.model_name,
                params,
            ),
        )
        print_generation_progress(index, len(cards), sample_id)

    print(f"wrote {len(cards)} generations to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
