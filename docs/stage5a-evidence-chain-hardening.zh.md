# Stage 5A 证据链修正操作指南

## 目标

Stage 5A 的目标，是把训练、推理、评分和审阅结果变成可追溯的证据链：训练前检查同时产出人读 Markdown 和机器读 JSON，eval 生成保留原始输出，评分默认看 `raw_output`，规则投影不会伪装成独立审阅，正式 SFT 只吃 `card_status` 为 `approved` 或 `frozen` 的正式章节执行卡。

这阶段不承诺扩量、不发布 adapter，也不把 smoke/dev 草稿流当成 formal SFT 证据。它只证明链路不会被草稿卡、清洗器、mock review 或缺失 manifest 掩盖。

## 适用范围

- 适用于 Stage 4/4.1 之后的证据链加固复跑。
- 适用于 smoke/dev draft flow：可以用 `build_chapter_cards.py` 生成的 `draft_only: true` 草稿卡构建小规模 SFT 数据，但必须显式选择 `--allow-draft-cards`。
- 适用于 formal SFT 前的门禁复查：formal SFT 默认拒绝 draft cards。Stage 5C 起，只有 `ChapterExecutionCard` 的 `card_status` 为 `approved` 或 `frozen`，且带有 `style_contract_id`、64 位十六进制 `style_contract_sha256` 等 style contract metadata 的卡，才允许进入 formal。
- 不适用于把 `rule_projection` 当作真人或模型 agent review，也不适用于直接宣布 100/500 样本扩量。

## 入口命令

先生成训练前检查证据。Stage 5A 起，preflight 必须同时保留 Markdown 和 JSON；后续 formal training 默认读取 JSON 门禁。

```powershell
python scripts/check_local_model.py --model-dir E:\models\Qwen3-4B-Instruct-2507 --report reports/model_check_report.md --json-output reports/model_check_report.json
python scripts/check_training_env.py --report reports/training_env_report.md --json-output reports/training_env_report.json
```

smoke/dev draft flow 可以从草稿章节卡构建 SFT 数据，但必须显式声明这是草稿路径：

```powershell
python scripts/build_sft_dataset.py --cards data_cards/chapter_cards.jsonl --chapters data_clean/chapters_split.jsonl --output data_sft/sft_chapter_v1.jsonl --dataset-info-output data_sft/dataset_info.json --allow-draft-cards
```

运行 smoke training：

```powershell
python scripts/run_sft_smoke.py --config configs/sft_qlora_qwen3_4b_smoke_6144.yaml --eval-cards data_cards/eval_execution_cards_50.jsonl
```

eval generation 写 raw-first JSONL。输出文件名建议显式带 `_raw`，seed 固定为 `20260623`，便于复跑比较：

```powershell
python scripts/run_eval_inference.py --cards data_cards/eval_execution_cards_50.jsonl --adapter-dir outputs/sft_smoke --output outputs/sft_smoke/generated_raw.jsonl --model-name sft_smoke --max-new-tokens 1024 --seed 20260623
```

评分默认使用生成 JSONL 里的 `raw_output`；只有缺少 `raw_output` 时才回退到旧字段：

```powershell
python scripts/score_outputs.py --cards data_cards/eval_execution_cards_50.jsonl --outputs outputs/sft_smoke/generated_raw.jsonl --output outputs/sft_smoke/metrics_raw.jsonl
```

`rule_projection` 只把 rules metrics 投影成 review-shaped 产物，用于保持下游报告链路完整：

```powershell
python scripts/run_agent_review.py --cards data_cards/eval_execution_cards_50.jsonl --outputs outputs/sft_smoke/generated_raw.jsonl --metrics outputs/sft_smoke/metrics_raw.jsonl --target-platform hybrid_fanqie_qidian --backend rule_projection --output outputs/sft_smoke/reviews_projection.jsonl --votes-output outputs/sft_smoke/votes_projection.jsonl --summary-output outputs/sft_smoke/review_projection_summary.jsonl --report reports/stage5a_rule_projection_report.md
```

formal SFT 构建数据时不要加 `--allow-draft-cards`：

```powershell
python scripts/build_sft_dataset.py --cards data_cards/chapter_execution_cards_approved.jsonl --chapters data_clean/chapters_split.jsonl --output data_sft/sft_chapter_formal.jsonl --dataset-info-output data_sft/dataset_info_formal.json --style-contract-json data_style/style_contract_author_main_v1.json --dataset-manifest-output data_sft/sft_chapter_formal_manifest.json
```

formal training 会读取 JSON preflight，并在输出目录写 run manifest：

```powershell
python scripts/run_sft_train.py --config configs/sft_qlora_qwen3_4b.yaml --sft-dataset data_sft/sft_chapter_formal.jsonl --eval-cards data_cards/eval_execution_cards_50.jsonl --model-report-json reports/model_check_report.json --env-report-json reports/training_env_report.json --output-dir outputs/sft_v1 --style-contract-json data_style/style_contract_author_main_v1.json
```

上面两条 formal 命令保留 Stage 5A 的证据链语境，但按 Stage 5B/5C 的资产闭环写法补上了 StyleContract JSON、正式 `ChapterExecutionCard` 和 dataset manifest。Stage 5B 起，缺少 `--style-contract-json` 不应被当作 formal 证据；Stage 5C 起，缺少 `card_status` 为 `approved` 或 `frozen` 的正式卡，或缺少 `--dataset-manifest-output`，也不应被当作完整 formal 数据证据。即使命令能 dry-run，`formal_evidence` 也应保持 `false`。

## 成功标志

- `reports/model_check_report.md` 与 `reports/model_check_report.json` 都存在，JSON 的 `kind` 为 `model` 且 `passed` 为 `true`。
- `reports/training_env_report.md` 与 `reports/training_env_report.json` 都存在，JSON 的 `kind` 为 `environment` 且 `passed` 为 `true`。
- smoke/dev SFT 若使用草稿卡，命令中明确出现 `--allow-draft-cards`；formal SFT 命令中不出现该参数。
- formal SFT 构建数据时使用 `data_cards/chapter_execution_cards_approved.jsonl`，并写出 `data_sft/sft_chapter_formal_manifest.json`。
- `outputs/sft_smoke/generated_raw.jsonl` 每行保留 `raw_output`、`sanitized_output`、`output`、`params.seed`、`prompt_sha256`、`generated_tokens` 和 `sanitizer_events`。
- `outputs/sft_smoke/metrics_raw.jsonl` 来自 `generated_raw.jsonl`，并按 `raw_output` 评分。
- `review_projection_summary.jsonl` 中 `review_backend` 为 `rule_projection`、`projection_only` 为 `true`、`agent_gate_pass` 为 `false`、`decision` 为 `rules_pass_agent_pending`。
- formal training 成功后，`outputs/<run>/run_manifest.json` 存在；成功标志是 manifest 顶层 `passed: true`。该 manifest 应记录 `training_exit_code: 0`、preflight JSON 摘要、config path、model dir、output dir 和 adapter check。

## 不能误读的结果

- `rule_projection` 不是独立审阅。它没有真人/模型 reviewer 的外部判断，只是从 Stage 4 hard gate metrics 生成流程投影。
- `rule_projection` 正常结果应是 `rules_pass_agent_pending`，并且命令退出码通常是非 0。这个非 0 表示“仍待真正 agent review”，不是产物缺失。
- `rule_projection` 不能产生 `ready_for_next_expansion`、发布结论或扩量结论。扩量必须等真实 agent review、人工仲裁和质量报告共同通过。
- `sanitized_output` 只用于排查清洗器行为；质量评分默认看 `raw_output`，避免 outline leak 或格式污染被清洗后误判为通过。
- `run_sft_smoke.py` 证明 smoke 训练链路；formal SFT 证据以 `run_sft_train.py` 的 JSON preflight 门禁和 `run_manifest.json` 为准。
- Stage 5B 起，formal SFT 还必须显式绑定 `--style-contract-json data_style/style_contract_author_main_v1.json`；缺少它或只跑 dry-run 时，`formal_evidence` 不应为 `true`。
- `--allow-draft-cards` 是 smoke/dev 草稿流的显式豁免，不是 formal SFT 的开关。formal 卡缺少 `card_status: approved/frozen` 或 style contract metadata 时，应失败而不是绕过。

## 下一阶段入口

Stage 5A 通过后，下一步不是直接扩大样本，而是进入 Stage 5B/5C 的正式资产闭环：

- Stage 5B：收敛 StyleContract，补齐可批准、可哈希、可绑定到 adapter manifest 的风格契约资产。
- Stage 5C：编译正式 ChapterExecutionCard，完成 approval lifecycle、leakage check、group split、sealed test 和近重复检查。
- Stage 5D：在真实审阅证据基础上整理 AI-taste 缺陷分类、same-plot 作者修订、rejection-sampling SFT 候选和 same-plot preference 候选数据；不运行偏好优化训练。

只有 formal cards、raw-first eval、raw scoring、真实审阅和 manifest 证据都闭环后，才讨论下一次样本扩量。
