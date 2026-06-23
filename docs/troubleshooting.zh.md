# 常见问题排查

这页按症状查问题。命令失败时先看报错和产物，不要连续盲试。

## 找不到文件或目录

症状：PowerShell 报 `No such file`、`FileNotFoundError`、路径不存在。

通常原因：

- 没在项目根目录运行命令。
- 上一步输出没有生成。
- 参数里的路径拼错。

先检查：

```powershell
Get-Location
Test-Path data_raw/novels
Test-Path data_clean/chapters.jsonl
```

安全处理：

- 回到 `E:\codex\smallModelTrain`。
- 按 [完整数据流说明](pipeline-flow.zh.md) 补跑缺失的上游步骤。
- 不要手写伪造中间 JSONL 来骗过后续脚本。

## JSONL 格式错误

症状：报 JSON 解析错误，或提示某一行无法读取。

通常原因：

- 文件不是一行一个 JSON。
- 手动编辑时删了引号、逗号或括号。
- 文件混入了普通文本。

先检查具体文件，例如：

```powershell
Get-Content data_cards/chapter_cards.jsonl -TotalCount 3
```

安全处理：

- 优先重新运行生成该文件的脚本。
- 如果必须手工修，只修报错行，并保持一行一个完整 JSON 对象。

## 章节卡或评测卡校验失败

症状：提示字段缺失、结构不合法、id 重复、样本数量不对。

通常原因：

- `data_cards/chapter_cards.jsonl` 或 eval cards 不是当前脚本期望的格式。
- 章节卡和章节 split 文件没有对齐。
- 生成结果、metrics、review 三类文件的 sample id 对不上。
- `data_cards/eval_cards_50.jsonl` 和 `data_cards/eval_execution_cards_50.jsonl` 被混用。它们不是同一种卡，不能互相替代。

Stage 4 推理、评分和 Agent review 要求严格的 execution-card schema。`data_cards/eval_execution_cards_50.jsonl` 至少要包含这些执行字段：

- `id`
- `target_platform`
- `genre_tags`
- `style_contract`
- `chapter_goal`
- `chapter_structure`
- `conflict_beat`
- `payoff_beat`
- `must_include`
- `must_not_include`
- `ending_hook`
- `target_word_count`

先用 dry-run 校验执行卡：

```powershell
python scripts/run_eval_inference.py --cards data_cards/eval_execution_cards_50.jsonl --output outputs/sft_smoke/generated_dry_run.jsonl --model-name sft_smoke --dry-run
```

先看：

- `reports/stage3_data_readiness_report.md`
- `data_cards/chapter_cards.jsonl`
- `data_cards/eval_execution_cards_50.jsonl`

安全处理：

- 重新生成章节卡。
- 重新构建 SFT 数据。
- 不要跳过 readiness 报告直接训练。
- 如果 dry-run 失败，修复或重新创建 `data_cards/eval_execution_cards_50.jsonl`。只重新生成章节卡和 SFT 数据，不能修复 execution-card schema 错误。

## 本地模型路径错误

症状：模型检查失败，提示 `config.json`、tokenizer、safetensors 或 shard 文件缺失。

先运行：

```powershell
python scripts/check_local_model.py --model-dir E:\models\Qwen3-4B-Instruct-2507 --report reports/model_check_report.md
```

然后读：

```text
reports/model_check_report.md
```

安全处理：

- 确认 `--model-dir` 指向真实模型目录。
- 不要把 adapter 目录当成基座模型目录。

## CUDA 不可用或显存不够

症状：`CUDA unavailable`、`CUDA out of memory`、进程被系统杀掉、训练中断。

先运行：

```powershell
python scripts/check_training_env.py --report reports/training_env_report.md
python scripts/run_oom_probe.py --dry-run
```

如果需要定位显存问题，再运行：

```powershell
python scripts/run_oom_probe.py
```

重点看：

- `reports/training_env_report.md`
- `reports/oom_probe_report.md`
- `logs/training/`

安全处理：

- 先关掉占用显卡的其他进程。
- 先跑 smoke 配置，不要直接跑 full training。
- OOM 后先读报告，不要反复重试同一个大配置。

## LLaMA-Factory 启动失败

症状：训练命令启动后很快退出，stderr 有 import、命令不存在或配置字段错误。

重点看：

- `logs/training/*stderr*`
- `logs/training/*stdout*`
- `logs/training/*events*`
- `reports/training_env_report.md`

安全处理：

- 先确认环境检查报告。
- 再确认 `configs/sft_qlora_qwen3_4b.yaml` 或 `configs/sft_qlora_qwen3_4b_smoke_6144.yaml`。
- 不要改完多个配置再试。一次只改一个变量，方便回溯。

## 生成结果为空或太短

症状：`generated.jsonl` 存在，但正文为空、很短或明显被截断。

通常原因：

- `--max-new-tokens` 太小。
- adapter 质量不足。
- prompt 或 eval card 异常。
- 推理过程被中断。

重点看：

- `outputs/sft_smoke/generated.jsonl`
- `outputs/sft_smoke/metrics.jsonl`
- `reports/sft_smoke_eval_report.md`
- `reports/stage4_1_quality_eval_budget_report.md`

安全处理：

- 先用固定质量子集做长生成评测。
- 不要只因为命令成功就扩大训练规模。

## Scoring 或报告数量对不上

症状：评分脚本提示 sample id 缺失、重复或生成结果和评测卡不匹配。

先确认三类文件使用同一批卡：

```powershell
python scripts/score_outputs.py --cards data_cards/eval_execution_cards_50.jsonl --outputs outputs/sft_smoke/generated.jsonl --output outputs/sft_smoke/metrics.jsonl
python scripts/evaluate_outputs.py --scores outputs/sft_smoke/metrics.jsonl --report reports/sft_smoke_eval_report.md --title "SFT Smoke Eval Report"
python scripts/build_stage4_quality_report.py --cards data_cards/eval_execution_cards_50.jsonl --generated outputs/sft_smoke/generated.jsonl --metrics outputs/sft_smoke/metrics.jsonl --report reports/stage4_1_quality_eval_budget_report.md --title "Stage 4.1 Quality Eval Budget Report"
```

`build_stage4_quality_report.py` 会检查 expected、generated、metrics 的行数和 missing ids，更适合发现生成覆盖不完整、generated 或 metrics 行缺失的问题。

安全处理：

- 不要混用 20 卡、50 卡、quality subset 的生成结果和 metrics。
- 用文件名明确区分 full50、subset、1024 token 等运行。

## Agent review 产物对不上

症状：智能体审阅提示 reviews、votes、summary 或 metrics id 不一致。

重点看：

- `scripts/run_agent_review.py` 的参数。
- 输入的 cards、outputs、metrics 是否来自同一批样本。
- 输出的 review、votes、summary 是否被上一次运行覆盖。

安全处理：

- 每次审阅使用同一套 sample id。
- 出现重复或缺失 id 时，先修输入文件，不要改审阅聚合结果。
