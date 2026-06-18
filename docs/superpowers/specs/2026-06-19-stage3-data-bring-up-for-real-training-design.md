# Stage 3: Data Bring-Up for Real Training Design

> 日期：2026-06-19
> 中文名称：面向真实训练的数据准备阶段
> 适用前提：Stage 1 与 Stage 2 的代码能力已经完成，但项目当前还没有真实 `data_*` 训练/评测产物
> 设计选择：先从 0 准备可训练数据，再把真实 GPU 训练后移到 Stage 4

## 1. 阶段定位

Stage 3 的目标不是证明 QLoRA 已经训练成功，而是让项目第一次拥有真实、可复查、可交给训练脚本读取的数据资产。

当前项目已经具备两类能力：

```text
Stage 1: 清洗、切章、固定 train/eval、风格画像、SFT 构造、规则评分
Stage 2: 模型/环境检查、训练命令构造、dry-run、训练监督、adapter 检查、eval 推理入口
```

但这些能力还没有吃到真实数据。仓库中当前没有 `data_raw/`、`data_clean/`、`data_cards/`、`data_sft/`、`outputs/`、`reports/` 下的真实训练资产。因此 Stage 3 必须先补齐训练前数据，而不是直接要求 100 条 smoke 或 500 条正式训练。

Stage 3 完成时，项目应站在真实训练门口：数据存在、格式正确、泄漏检查通过、训练 dry-run 能识别这批数据。真实 smoke/full training 是 Stage 4 的目标。

## 2. 非目标

Stage 3 不做以下事情：

```text
不承诺完成真实 GPU 训练。
不要求生成 SFT v2 的 3000-5000 条大规模数据。
不做 ORPO / KTO / DPO / SimPO 偏好训练。
不做复杂 100 分制质量评测。
不把自动生成的大量章节卡直接视为高质量训练数据。
不修改 Stage 2 的训练监督框架，除非 dry-run 暴露出数据路径或数据注册问题。
```

这些任务在 Stage 4 或后续质量阶段处理。

## 3. 推荐路线与取舍

### 3.1 推荐路线：数据准备优先

主线如下：

```text
原始小说文本
-> ingest_raw_text.py
-> clean_chapters.py
-> split_train_eval.py
-> build_style_contract.py
-> 准备 chapter_cards.jsonl
-> build_sft_dataset.py
-> 数据验收报告
-> run_sft_smoke.py --dry-run
```

这条路线先解决最基础的问题：训练数据一条都没有。它的好处是每一步都有文件产物和脚本验收，失败时能知道是原稿、切章、章节卡、SFT 构造，还是 Stage 2 dry-run 对数据路径不满意。

### 3.2 备选路线：先补训练环境

也可以先跑模型和环境检查，但这不能替代 Stage 3。环境检查只证明机器可能能训练，不会产生训练样本。

这条路线适合穿插在 Stage 3 末尾：数据准备到最小可训练集后，再用 Stage 2 的 dry-run 和环境检查确认训练入口。

### 3.3 延后路线：质量评测与偏好优化

质量评测、坏例复盘、偏好数据清洗应等待真实 adapter 输出后再做。当前没有 baseline 输出、SFT v1 输出或人工轻修样本，过早设计偏好训练会变成空转。

## 4. 数据流设计

Stage 3 使用现有 Stage 1 脚本，形成以下产物链：

```text
data_raw/novels/*.txt
-> data_clean/chapters_raw.jsonl
-> data_clean/chapters.jsonl
-> data_clean/chapters_split.jsonl
-> style_contract.md
-> style_profile.json
-> data_cards/chapter_cards.jsonl
-> data_cards/eval_cards_*.jsonl
-> data_sft/sft_chapter_v1.jsonl
-> reports/stage3_data_readiness_report.md
```

`data_clean/chapters_split.jsonl` 是 Stage 3 的中间枢纽。它同时服务于风格画像、章节卡准备、SFT 构造和固定 eval。

`data_cards/chapter_cards.jsonl` 是 Stage 3 风险最高的文件。它不能简单复制原文，也不能把目标正文贴进 prompt。章节卡应描述目标、结构、人物状态、必须出现和禁止事项，而不是写出正文句子。

`data_sft/sft_chapter_v1.jsonl` 是 Stage 3 交给 Stage 4 的核心产物。它出现且非空，只表示训练数据准备好了，不表示模型已经训练。

## 5. 目录与产物

Stage 3 需要落地以下目录和文件：

```text
data_raw/novels/
data_clean/chapters_raw.jsonl
data_clean/chapters.jsonl
data_clean/chapters_split.jsonl
data_cards/chapter_cards.jsonl
data_cards/eval_cards_20.jsonl 或 data_cards/eval_cards_50.jsonl
data_sft/sft_chapter_v1.jsonl
style_contract.md
style_profile.json
reports/stage3_data_readiness_report.md
```

如果原始章节数量不足以固定 50 条 eval，先使用 `eval_cards_20.jsonl` 或更小的固定评测集。评测集数量可以小，但必须固定，且不能混入 SFT 训练样本。

## 6. 最小可训练集标准

Stage 3 第一版不追求大规模数据，先追求能真实喂给训练脚本。

最低验收线：

```text
至少 20-50 条可训练 SFT 样本。
至少 10-20 条固定 eval cards，原稿数量足够时优先 50 条。
所有 SFT 样本来自 split=train 且 quality_tag=A 的章节。
eval cards 对应章节不进入训练。
章节卡字段完整，包含 chapter_goal、chapter_structure、character_states、must_include、must_not_include、target_word_count。
source_text 不进入 prompt。
build_sft_dataset.py 的泄漏检查通过。
run_sft_smoke.py --dry-run 能读到配置和数据。
```

如果可用章节少于 20 条，Stage 3 仍可以产出数据报告，但结论应标记为“数据不足，暂不进入 Stage 4 smoke training”。

## 7. 章节卡设计原则

章节卡是 Stage 3 成败的关键。它的职责是给正文执行器提供可执行任务，而不是给模型答案。

每张章节卡应遵守：

```text
保留剧情结构，不保留原文句子。
保留人物状态，不复述正文细节。
must_include 写必须出现的信息点，不写完整桥段。
must_not_include 写禁止提前揭示的内容、禁止跑偏方向和禁用表达。
chapter_structure 使用 5-8 个步骤，每步写目标和字数预算。
target_word_count 第一版固定为 2000-2500中文汉字，必要时对短章单独标记。
source_text 只用于离线溯源和泄漏检查，不进入训练输入。
```

章节卡可以先人工准备小批量，也可以由外部模型辅助生成后人工复核。第一版不追求自动生成规模，而追求字段稳定和无泄漏。

## 8. 数据验收报告

Stage 3 需要一份面向训练入口的 Markdown 报告：

```text
reports/stage3_data_readiness_report.md
```

报告至少包含：

```text
原始文件数量。
清洗后章节数量。
train/eval 数量。
A/B/C 或 quality_tag 分布。
中文汉字长度分布。
章节卡数量。
SFT 样本数量。
eval cards 数量。
被跳过样本数量和原因。
source_text 泄漏检查结论。
是否允许进入 Stage 4。
如果不允许，列出阻断原因。
```

Stage 3 的最终结论只能是以下之一：

```text
ready_for_stage4_smoke_training
blocked_missing_raw_text
blocked_insufficient_chapters
blocked_missing_chapter_cards
blocked_sft_empty
blocked_source_leakage
blocked_eval_missing
blocked_stage2_dry_run_failed
```

## 9. 错误处理与阻断策略

Stage 3 失败时不应该静默跳过，也不应该用空数据进入训练。

阻断规则：

```text
没有 data_raw/novels 输入：阻断。
清洗后没有章节：阻断。
没有 train 样本：阻断。
没有 eval cards：阻断。
chapter_cards.jsonl 为空：阻断。
SFT JSONL 为空：阻断。
检测到 source_text 泄漏：阻断。
run_sft_smoke.py --dry-run 无法读取数据或配置：阻断。
```

非阻断但要报告的情况：

```text
eval cards 少于 50 条。
SFT 样本少于 100 条。
章节长度分布偏短或偏长。
must_include 或 must_not_include 平均数量过少。
章节卡由模型辅助生成但尚未全部人工复核。
```

## 10. 与 Stage 2 和 Stage 4 的边界

Stage 2 已经实现训练执行工具，但它默认假设数据存在。Stage 3 负责让这个假设第一次成立。

Stage 3 结束前必须完成以下训练入口验收动作：

```text
python scripts/run_sft_smoke.py --dry-run
```

Stage 4 承接真实训练动作：

```text
python scripts/check_local_model.py --model-dir E:\models\Qwen3-4B-Instruct-2507
python scripts/check_training_env.py
python scripts/run_sft_smoke.py
python scripts/check_adapter.py --adapter-dir outputs/sft_smoke --report reports/sft_smoke_report.md --title "SFT Smoke Adapter Check"
```

其中真实 `run_sft_smoke.py` 是 Stage 4 的第一批动作。Stage 4 可以重复执行 dry-run 做最终确认，但不能把重复 dry-run 视为真实训练已经开始。

## 11. 测试与验证

Stage 3 主要验证数据文件和脚本链路，不跑真实 GPU 训练。

必跑验证：

```powershell
python -m pytest -q
python scripts/ingest_raw_text.py --input-dir data_raw/novels --output data_clean/chapters_raw.jsonl
python scripts/clean_chapters.py --input data_clean/chapters_raw.jsonl --output data_clean/chapters.jsonl --min-chars 500 --max-chars 5000
python scripts/split_train_eval.py --input data_clean/chapters.jsonl --output data_clean/chapters_split.jsonl --eval-output data_cards/eval_cards_20.jsonl --eval-count 20
python scripts/build_style_contract.py --chapters data_clean/chapters_split.jsonl --contract-output style_contract.md --profile-output style_profile.json
python scripts/build_sft_dataset.py --cards data_cards/chapter_cards.jsonl --chapters data_clean/chapters_split.jsonl --output data_sft/sft_chapter_v1.jsonl
python scripts/run_sft_smoke.py --dry-run
```

如果原始章节数量足够，`eval-count` 提升到 50，并输出 `data_cards/eval_cards_50.jsonl`。

## 12. 完成标准

Stage 3 完成时必须满足：

```text
真实原稿已经进入 data_raw/novels。
data_clean/chapters_split.jsonl 存在且非空。
style_contract.md 和 style_profile.json 存在。
data_cards/chapter_cards.jsonl 存在且非空。
固定 eval cards 存在且不进入训练。
data_sft/sft_chapter_v1.jsonl 存在且非空。
泄漏检查通过。
reports/stage3_data_readiness_report.md 给出 ready_for_stage4_smoke_training 或明确阻断原因。
run_sft_smoke.py --dry-run 通过。
```

只有当报告结论为 `ready_for_stage4_smoke_training` 时，才进入 Stage 4 的真实 smoke training。

## 13. Stage 4 前瞻

Stage 4 建议命名为：

```text
Stage 4: Real Training Bring-Up and Acceptance
```

中文名称：

```text
真实训练拉起与验收闭环
```

Stage 4 使用 Stage 3 产出的数据，按以下顺序推进：

```text
模型检查
-> 环境检查
-> 真实 20-50 条 smoke training
-> smoke adapter 检查
-> 固定 eval 推理
-> 规则评分报告
-> 决定是否扩到 100 条、500 条或修数据
```

如果 Stage 4 训练失败，问题应回填到 Stage 3 的数据报告或 Stage 2 的训练诊断报告中，而不是直接进入 SFT v2 或偏好优化。
