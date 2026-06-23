# 完整数据流说明

这页按‘文件怎么变成另一个文件’的顺序解释项目。你不需要先懂训练原理，只要记住：每一步都有输入、命令、输出和检查方式。

## 总览

```text
data_raw/novels/
  -> data_clean/chapters_raw.jsonl
  -> data_clean/chapters.jsonl
  -> data_clean/chapters_split.jsonl + data_cards/eval_cards_50.jsonl
  -> style_contract.md + style_profile.json
  -> data_cards/chapter_cards.jsonl
  -> data_sft/sft_chapter_v1.jsonl
  -> reports/stage3_data_readiness_report.md
  -> outputs/sft_smoke/
  -> outputs/sft_smoke/generated.jsonl
  -> outputs/sft_smoke/metrics.jsonl
  -> reports/sft_smoke_eval_report.md
  -> reports/stage4_1_quality_eval_budget_report.md
```

## 1. 原始小说到原始章节

输入：`data_raw/novels/`

命令：

```powershell
python scripts/ingest_raw_text.py --input-dir data_raw/novels --output data_clean/chapters_raw.jsonl
```

输出：`data_clean/chapters_raw.jsonl`

成功标志：文件存在，每一行是一章或一段可处理文本。

## 2. 原始章节到清洗章节

输入：`data_clean/chapters_raw.jsonl`

命令：

```powershell
python scripts/clean_chapters.py --input data_clean/chapters_raw.jsonl --output data_clean/chapters.jsonl --min-chars 500 --max-chars 5000
```

输出：`data_clean/chapters.jsonl`

成功标志：过短、过长或格式不合适的章节被过滤。

## 3. 划分训练集和评测集

输入：`data_clean/chapters.jsonl`

命令：

```powershell
python scripts/split_train_eval.py --input data_clean/chapters.jsonl --output data_clean/chapters_split.jsonl --eval-output data_cards/eval_cards_50.jsonl --eval-count 50
```

输出：

- `data_clean/chapters_split.jsonl`
- `data_cards/eval_cards_50.jsonl`

成功标志：评测集数量符合 `--eval-count`，训练数据仍然保留在 split 文件里。

## 4. 生成风格契约

输入：`data_clean/chapters_split.jsonl`

命令：

```powershell
python scripts/build_style_contract.py --chapters data_clean/chapters_split.jsonl --contract-output style_contract.md --profile-output style_profile.json
```

输出：

- `style_contract.md`
- `style_profile.json`

成功标志：Markdown 里有风格摘要，JSON 里有统计字段。

## 5. 生成章节卡

输入：`data_clean/chapters_split.jsonl`

命令：

```powershell
python scripts/build_chapter_cards.py --chapters data_clean/chapters_split.jsonl --output data_cards/chapter_cards.jsonl --count 50 --min-chars 2000 --max-chars 3000
```

输出：`data_cards/chapter_cards.jsonl`

成功标志：章节卡字段完整，没有直接塞入完整原文。

## 6. 构建 SFT 数据

输入：

- `data_cards/chapter_cards.jsonl`
- `data_clean/chapters_split.jsonl`

命令：

```powershell
python scripts/build_sft_dataset.py --cards data_cards/chapter_cards.jsonl --chapters data_clean/chapters_split.jsonl --output data_sft/sft_chapter_v1.jsonl --dataset-info-output data_sft/dataset_info.json
```

输出：

- `data_sft/sft_chapter_v1.jsonl`
- `data_sft/dataset_info.json`

成功标志：SFT 数据可以被训练脚本引用。

## 7. 训练前 readiness

命令：

```powershell
python scripts/check_stage3_data_readiness.py --eval-cards data_cards/eval_cards_50.jsonl --run-smoke-dry-run
```

输出：`reports/stage3_data_readiness_report.md`

成功标志：报告给出可以进入 smoke training 的状态。

## 8. 模型和环境检查

命令：

```powershell
python scripts/check_local_model.py --model-dir E:\models\Qwen3-4B-Instruct-2507 --report reports/model_check_report.md
python scripts/check_training_env.py --report reports/training_env_report.md
```

输出：

- `reports/model_check_report.md`
- `reports/training_env_report.md`

成功标志：模型文件存在，训练环境报告没有阻断项。

## 9. Smoke training

命令：

```powershell
python scripts/run_sft_smoke.py --config configs/sft_qlora_qwen3_4b_smoke_6144.yaml --eval-cards data_cards/eval_execution_cards_50.jsonl
```

输出：`outputs/sft_smoke/`

成功标志：adapter 文件生成，训练日志写入 `logs/training/`。

## 10. Adapter 检查

命令：

```powershell
python scripts/check_adapter.py --adapter-dir outputs/sft_smoke --report reports/sft_smoke_report.md --title "SFT Smoke Adapter Check"
```

输出：`reports/sft_smoke_report.md`

成功标志：报告确认 adapter 结构可读。

## 11. Eval 推理和评分

命令：

```powershell
python scripts/run_eval_inference.py --cards data_cards/eval_execution_cards_50.jsonl --adapter-dir outputs/sft_smoke --output outputs/sft_smoke/generated.jsonl --model-name sft_smoke --max-new-tokens 256
python scripts/score_outputs.py --cards data_cards/eval_execution_cards_50.jsonl --outputs outputs/sft_smoke/generated.jsonl --output outputs/sft_smoke/metrics.jsonl
python scripts/evaluate_outputs.py --scores outputs/sft_smoke/metrics.jsonl --report reports/sft_smoke_eval_report.md --title "SFT Smoke Eval Report"
```

输出：

- `outputs/sft_smoke/generated.jsonl`
- `outputs/sft_smoke/metrics.jsonl`
- `reports/sft_smoke_eval_report.md`

成功标志：生成结果、metrics 和 Markdown 报告三者数量能对上。

## 12. Stage 4.1 质量评测

命令：

```powershell
python scripts/build_eval_quality_subset.py --cards data_cards/eval_execution_cards_50.jsonl --metrics outputs/sft_smoke/metrics.jsonl --output data_cards/eval_cards_quality_subset.jsonl --count 8
python scripts/run_eval_inference.py --cards data_cards/eval_cards_quality_subset.jsonl --adapter-dir outputs/sft_smoke --output outputs/sft_smoke/generated_subset_1024.jsonl --model-name sft_smoke_subset_1024 --max-new-tokens 1024
python scripts/score_outputs.py --cards data_cards/eval_cards_quality_subset.jsonl --outputs outputs/sft_smoke/generated_subset_1024.jsonl --output outputs/sft_smoke/metrics_subset_1024.jsonl
python scripts/build_stage4_quality_report.py --cards data_cards/eval_cards_quality_subset.jsonl --generated outputs/sft_smoke/generated_subset_1024.jsonl --metrics outputs/sft_smoke/metrics_subset_1024.jsonl --report reports/stage4_1_quality_eval_budget_report.md --title "Stage 4.1 Quality Eval Budget Report"
```

输出：`reports/stage4_1_quality_eval_budget_report.md`

成功标志：报告说明是否通过质量门槛，而不是只看命令有没有跑完。

## 阶段边界

- Stage 1：数据管线和基础评分报告。
- Stage 2：训练执行封装、模型检查、环境检查、adapter 检查、OOM 探测。
- Stage 3：真实数据接入和 readiness 证据，不直接追求最终训练质量。
- Stage 4：固定 50 卡 smoke training 和评测决策。
- Stage 4.1：长生成质量、预算和审阅门槛。
