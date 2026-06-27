# Stage 5D 作者反馈与 AI 味降低指南

Stage 5D 把“作者看过、改过、接受过”的同剧情证据变成候选训练数据。它先补齐 Stage 5C formal admission 的重复 id/hash 与 manifest 覆写门禁，再进入 AI 味缺陷审阅、作者 same-plot 修订、rejection-sampling SFT 候选和 same-plot preference 候选构建。

## 目标

- 让 formal SFT 入场先拒绝重复的 trainable chapter id、重复 card hash、重复 chapter hash，并阻止 dataset manifest 被静默覆写。
- 用可追溯的 review record 记录 AI 味缺陷、证据 span 和 raw output provenance。
- 用作者 same-plot 修订记录保留 card、StyleContract、prompt 和 raw output provenance。
- 从作者接受的修订中构建 rejection-sampling SFT 候选行。
- 从合法的作者接受修订中构建 same-plot preference 候选行。
- 输出 Stage 5D 报告 metrics，用于审阅数据健康度，而不是证明模型质量已经改善。

## 先修 formal admission

Stage 5D 继承 Stage 5C 的 formal SFT 边界，并先修复 formal admission 缺口。进入作者反馈数据构建前，应确认：

- `data_cards/chapter_execution_cards_approved.jsonl` 中没有重复 trainable chapter id。
- formal admission 会拒绝重复的 `card_sha256`。
- formal admission 会拒绝重复的 source chapter hash。
- formal SFT dataset manifest 不会在未显式允许时覆盖已有文件。
- 正式卡仍必须绑定 approved/frozen `ChapterExecutionCard`、StyleContract id/hash 和章节 hash。

这些门禁只说明数据入口更可信，不说明扩大到 100/500 条训练已经被允许。

## AI 味缺陷标签

AI 味缺陷标签用于记录“为什么这段输出不像作者本人会写”。每条 review record 应保留：

- 缺陷 taxonomy label。
- 缺陷严重度和是否阻断接受。
- 对应 generated/raw output 的 evidence spans。
- 评审对象的 card、prompt、raw output 和生成参数 provenance。

标签和证据 span 是审阅数据，不是自动文学裁判。人工判断仍然优先，尤其是风格、节奏和人物语气类问题。

## Same-Plot 作者修订

Same-plot 修订要求作者在同一张卡、同一个剧情目标和同一个 StyleContract 下改写小模型输出。修订记录应保留：

- 原始生成文本和 raw output provenance。
- 作者修订文本。
- card id、card hash、StyleContract id/hash。
- prompt hash 或 prompt 文本 provenance。
- 作者接受状态、修订原因和 major-edit 证据。

只有 accepted 且 provenance 完整的修订，才能进入后续候选数据构建。被拒绝、缺少 card/StyleContract/prompt/raw output provenance，或无法证明 same-plot 的记录，都不能被当作有效偏好样本。

## 构建候选数据

构建 rejection-sampling SFT 候选：

```powershell
python scripts/build_rejection_sampling_sft.py --revisions data_review/stage5d_revisions.jsonl --cards data_cards/chapter_execution_cards_approved.jsonl --style-contract-json data_style/style_contract_author_main_v1.json --output data_sft/stage5d_rejection_sampling_sft.jsonl
```

输出：`data_sft/stage5d_rejection_sampling_sft.jsonl`

成功标志：每行来自 accepted 作者修订，并绑定正式卡、StyleContract 和修订 provenance。它是 SFT 候选数据，不自动触发训练。

构建 same-plot preference 候选：

```powershell
python scripts/build_same_plot_preference_dataset.py --revisions data_review/stage5d_revisions.jsonl --output data_pref/stage5d_same_plot_preference.jsonl
```

输出：`data_pref/stage5d_same_plot_preference.jsonl`

成功标志：每行是同剧情、同卡、同 StyleContract 下的 chosen/rejected 候选对。Preference rows 只是 candidate data，不是 DPO、SimPO、ORPO、KTO，也不表示已经运行 reward model training 或 preference optimization。

## 报告

构建 Stage 5D 汇总和 Markdown 报告：

```powershell
python scripts/build_stage5d_review_report.py --review-records data_review/stage5d_review_records.jsonl --revisions data_review/stage5d_revisions.jsonl --rejection-sampling-rows data_sft/stage5d_rejection_sampling_sft.jsonl --preference-rows data_pref/stage5d_same_plot_preference.jsonl --summary-output reports/stage5d_review_summary.json --report-output reports/stage5d_review_report.md
```

输出：

- `reports/stage5d_review_summary.json`
- `reports/stage5d_review_report.md`

报告用于检查 review records、accepted revisions、rejection-sampling 候选和 same-plot preference 候选的数量与比例。它只能说明 Stage 5D 数据构建状态，不能替代 sealed evaluation、作者盲审或 paired eval。

## Stage 5D 不证明什么

- 不证明模型质量已经改善。
- 不证明可以扩大到 100/500 条正式训练。
- 不运行 DPO、SimPO、ORPO、KTO。
- 不运行 reward model training 或 preference optimization。
- 不把 preference rows 当成已训练过的偏好优化结果。
- 不让大模型接管最终写作；作者 same-plot 修订仍是核心监督来源。
