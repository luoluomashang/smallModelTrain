# Stage 5C 正式章节执行卡与数据完整性指南

## 目标

Stage 5C 把 `data_cards/chapter_cards.jsonl` 里的草稿章节卡升级为可审阅、可批准、可哈希、可追踪的 `ChapterExecutionCard` 资产。正式训练不再把普通草稿卡当成 formal 证据，而是要求整批 formal SFT 数据能用正式卡、源章节、StyleContract 和 split 元数据做一致性复查。

Stage 5C 不扩样，不自动批准卡，不进入作者反馈，也不运行实验矩阵。它只负责把 formal SFT 的卡资产和数据完整性边界立起来。

## 草稿卡路径

`data_cards/chapter_cards.jsonl` 仍是 draft/smoke/dev 路径。只要用它构建 SFT 数据，就必须显式传入 `--allow-draft-cards`，避免把草稿卡误读为 formal 资产：

```powershell
python scripts/build_sft_dataset.py --cards data_cards/chapter_cards.jsonl --chapters data_clean/chapters_split.jsonl --output data_sft/sft_chapter_v1.jsonl --dataset-info-output data_sft/dataset_info.json --allow-draft-cards
```

这条路径可以检查小规模数据和训练链路，但不能作为正式扩量、正式质量结论或 formal SFT 证据。

## 编译正式卡候选

正式卡先由 Card Compiler 从草稿卡、章节正文和 StyleContract JSON 编译出来：

```powershell
python scripts/compile_chapter_execution_cards.py --cards data_cards/chapter_cards.jsonl --chapters data_clean/chapters_split.jsonl --style-contract-json data_style/style_contract_author_main_v1.json --output data_cards/chapter_execution_cards_reviewed.jsonl
```

输出的 `data_cards/chapter_execution_cards_reviewed.jsonl` 是 reviewed 候选，不是 approved/frozen 资产。它可以进入人工审阅，但不能直接进入 formal SFT。

人工审阅后，只有确认通过的正式卡才能把 `card_status` 字段改为 `approved` 或 `frozen`。修改 `card_status`、审阅记录或任何正式字段后，都要重新计算 `card_sha256`。formal SFT 读取的是审批冻结后的 `data_cards/chapter_execution_cards_approved.jsonl`。

## Formal SFT 命令

formal SFT 构建数据时使用 `card_status` 为 `approved` 或 `frozen` 的正式卡、approved/frozen 的 StyleContract JSON，并写出 dataset manifest：

```powershell
python scripts/build_sft_dataset.py --cards data_cards/chapter_execution_cards_approved.jsonl --chapters data_clean/chapters_split.jsonl --output data_sft/sft_chapter_formal.jsonl --dataset-info-output data_sft/dataset_info_formal.json --style-contract-json data_style/style_contract_author_main_v1.json --dataset-manifest-output data_sft/sft_chapter_formal_manifest.json
```

`--dataset-manifest-output` 写出的 manifest 是数据集证据，不替代训练后的 `run_manifest.json`。当前 manifest 记录 dataset/file hashes、汇总 split counts、`card_hashes`、`chapter_hashes`、leakage report 和 near-duplicate report；它证明批次级 provenance，并让审阅者交叉核对卡、章节和 StyleContract hash。SFT JSONL 行本身仍是 `instruction`、`input`、`output`，不包含逐行 card/chapter/split 映射。

## Formal 门禁检查

进入 formal SFT 前，至少要检查：

- 每个 train/A 章节恰好有一张 `card_status` 为 `approved` 或 `frozen` 的 `ChapterExecutionCard`。
- 卡里的 `style_contract_id` 与 StyleContract JSON 一致。
- 卡里的 `style_contract_sha256` 与 StyleContract JSON 的 `contract_sha256` 一致。
- 卡里的 `source_chapter_sha256` 与 `data_clean/chapters_split.jsonl` 中对应章节正文重新计算的 hash 一致。
- prompt 不泄漏目标正文、source_text、使用 grouped split 元数据标记的 validation/sealed 文本片段或禁止引用的长片段。
- dataset manifest 记录 dataset 文件、cards 文件、chapters 文件、StyleContract 文件、汇总 split counts、card hashes 和 chapter hashes，能支持批次级 hash provenance 复核。

任一项不通过，都应阻断 formal SFT，而不是降级为草稿路径。

## Sealed Split 边界

Stage 5C 的 validation/sealed split 是使用 grouped split 元数据时的正式数据边界。现有 `scripts/split_train_eval.py` 仍只创建 legacy train/eval smoke split；validation/sealed 分配来自 Stage 5C 的 `split_grouped_rows()` helper 和 formal card/data contract，而不是这个旧 CLI 的默认输出。

在使用 grouped split 元数据时，`sealed` 章节不能进入训练，不能用于调参，也不应用来挑 prompt、挑阈值或修卡。formal SFT 只能使用 train 侧符合门禁的章节；validation 可以服务开发期比较和预算调参；sealed 只留给后续最终证明。当前 dataset manifest 记录汇总 split counts；配合 formal cards 和 grouped split 元数据，可用于复查 formal 数据集没有把 validation/sealed 内容带入训练。

## Stage 5C 不证明什么

- 不证明模型质量已经提升。
- 不证明可以扩大到 100/500 样本。
- 不替代人工审批正式卡。
- 不证明 StyleContract 本身已经通过真人风格审阅；那属于 Stage 5B 的审批状态。
- 不处理作者反馈、same-plot 修订、rejection sampling、DPO 或实验矩阵。
- 不证明 sealed eval 已经通过；它只提供复查 sealed 边界是否被 formal SFT 训练数据消耗的证据。
