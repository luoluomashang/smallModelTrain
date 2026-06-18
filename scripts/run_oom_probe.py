from __future__ import annotations

import argparse
from pathlib import Path


PROBES = [
    "Probe 1: load tokenizer and config",
    "Probe 2: load 4-bit base model",
    "Probe 3: inject LoRA",
    "Probe 4: tokenize one sample",
    "Probe 5: cutoff_len=8192, max_steps=1",
    "Probe 6: cutoff_len=6144, max_steps=1",
    "Probe 7: cutoff_len=6144, lora_rank=8, max_steps=1",
]


def render_probe_report() -> str:
    lines = ["# OOM Probe Report", "", "## Probe Plan"]
    lines.extend(f"- {probe}" for probe in PROBES)
    lines.extend(
        [
            "",
            "## Interpretation",
            "- Probe 2 失败：基座 4-bit 加载不稳，优先查 bitsandbytes / CUDA / 显存占用。",
            "- Probe 3 失败：LoRA target 或 PEFT 配置问题。",
            "- Probe 5 失败但 Probe 6 成功：8192 cutoff_len 过高。",
            "- Probe 6 失败但 Probe 7 成功：LoRA rank 或反向传播显存压力过高。",
            "- 所有 probe 通过但正式训练失败：检查数据长度分布、保存、日志或长时间运行稳定性。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="reports/oom_probe_report.md")
    args = parser.parse_args()

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_probe_report(), encoding="utf-8")
    print(f"wrote OOM probe report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
