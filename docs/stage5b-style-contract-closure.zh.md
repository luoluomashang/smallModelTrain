# Stage 5B StyleContract 闭环指南

## 目标

Stage 5B 把风格契约从根目录里的普通 Markdown 文本升级为可审阅、可哈希、可追踪的 StyleContract 资产。它不扩样、不批准章节卡，也不替代真人风格审阅。

这阶段的核心变化是：StyleContract JSON 成为 formal SFT 的机器门禁源，Markdown 继续服务人工审阅，metrics 用来复查风格统计。

## 生成 StyleContract

输入仍然来自 `data_clean/chapters_split.jsonl`。生成命令建议显式写出 JSON、Markdown、metrics 和 contract id：

```powershell
python scripts/build_style_contract.py --chapters data_clean/chapters_split.jsonl --contract-json-output data_style/style_contract_author_main_v1.json --contract-output style_contract.md --metrics-output data_style/style_metrics_author_main_v1.json --style-contract-id author_main_v1
```

输出三件套：

- `data_style/style_contract_author_main_v1.json`：机器读取的 StyleContract 资产。
- `style_contract.md`：给人工审阅使用的 Markdown 摘要。
- `data_style/style_metrics_author_main_v1.json`：用于复查篇幅、对白比例、段落长度等统计指标。

默认状态是 `pending_review`。这表示资产可以被审阅和 smoke/dev 引用，但不能进入 formal SFT。

## 审阅状态

- `pending_review`：默认状态，等待人工检查。
- `approved`：允许 formal SFT 使用。
- `frozen`：允许 formal SFT 使用，并表示同 ID 资产不应被覆盖。
- `draft` / `rejected`：不能进入 formal SFT。

## Formal SFT

formal SFT 必须显式传入 `approved` 或 `frozen` 的 StyleContract JSON：

```powershell
python scripts/build_sft_dataset.py --cards data_cards/chapter_cards_approved.jsonl --chapters data_clean/chapters_split.jsonl --output data_sft/sft_chapter_formal.jsonl --dataset-info-output data_sft/dataset_info_formal.json --style-contract-json data_style/style_contract_author_main_v1.json
```

如果 contract 仍是 `pending_review`，命令应失败。卡里的 `style_contract_id` 和 `style_contract_sha256` 必须与 JSON 完全一致。

## 训练 Manifest

`run_sft_train.py` 会在 manifest 中记录：

- SFT dataset 路径、sha256、行数。
- eval cards 路径、sha256、schema 校验结果。
- StyleContract 路径、id、hash、approval status。
- `formal_evidence` 是否为真实 formal 证据。

dry-run 可以验证命令构造，但不会产生 formal evidence。

## Stage 5B 不证明什么

- 不自动批准任何 StyleContract。
- 不把现有草稿章节卡变成正式执行卡。
- 不实现 Stage 5C 的 Card Compiler。
- 不做 sealed test、group split 或近重复检查。
- 不允许直接扩到 100/500 样本。
