# Stage 4.1 Full50 Manual Review Findings

- 日期: 2026-06-21
- reviewed artifact: `outputs/sft_smoke/generated_full50_merged_2560_rp112_nr4_retry3072.jsonl`
- status: old hard-gate pass is insufficient; full50 promotion is blocked by manual review.

## 结论

本次 full50 merged 结果不能作为进入 100-sample expansion 的质量证据。

虽然旧评分器给出 `50/50` hard gate pass，但人工复核与补充诊断发现输出存在明显质量问题：

- 部分输出有补足字数式的语义循环：同一情绪、同一关系解释、同一结论反复换说法出现。
- 多条输出靠近 2500 字上限，表现为“写到封顶”而非自然完成。
- 输出仍残留非正文内容，如 markdown blockquote、免责声明、最终确认、内部结构说明。
- 套话密度偏高，跨样本重复出现“终于明白”“轻声说”“像是某种”“深吸一口气”等泛化表达。
- eval 输入 schema 错位：`data_cards/eval_cards_50.jsonl` 不是章节执行卡，缺少 `style_contract`、`chapter_goal`、`chapter_structure`、`must_include`、`ending_hook` 等字段，因此 full50 实际没有验证“按章节执行卡写正文”。

## 诊断摘要

针对 `generated_full50_merged_2560_rp112_nr4_retry3072.jsonl` 的补充扫描结果：

- near-cap rows: 39/50 rows >= 2450 Chinese chars。
- blockquote/markdown residue: 28 rows。
- final-confirmation/meta-evaluation residue: 10 rows。
- disclaimer residue: 8 rows。
- suspicious or incomplete endings: observed in multiple rows。
- traditional/simplified style mixing: 5 rows。
- exact repeated 5+ Chinese char grams with count >= 3: 12 rows。
- cross-sample generic phrase overuse:
  - `终于明白`: 47 hits in 34 rows。
  - `轻声说`: 22 hits in 19 rows。
  - `像是某种`: 18 hits in 15 rows。
  - `深吸一口气`: 13 hits in 13 rows。

当前 `repeated_ngram_ratio` 没有抓住这些问题，因为它主要衡量精确 4-gram 重复。`no_repeat_ngram_size=4` 也只能阻止精确 token n-gram 循环，不能阻止语义层面的“换句话重复”。

## 根因判断

1. **Eval schema 错位**

   `run_eval_inference.py` 使用章节执行卡渲染 prompt，但 full50 传入的是原文 eval chapters。缺失字段会让 prompt 变成弱约束或空目标，模型在没有明确章节任务时自由补写。

2. **长度门槛诱导 padding**

   2000-2500 字硬门槛、`2560/3072` token budget、以及输出层封顶共同造成“写够长度优先”的行为。模型倾向通过解释、回忆、情绪总结和重复关系信息补足字数。

3. **Scorer 覆盖不足**

   旧 scorer 只覆盖长度、精确重复、少量 outline markers、must_include 和 forbidden hits。由于 eval cards 缺少 must_include，coverage 默认通过；由于重复多为语义重复，精确 n-gram ratio 低估风险。

4. **Smoke adapter 训练强度不足**

   当前 adapter 来自 50-sample smoke training，训练目标是验证链路，不足以证明稳定文学生成质量。

## 修复门槛

恢复 Stage 4.1 晋级前，至少需要：

1. 构建真正的 eval execution cards，字段与 SFT chapter cards 一致。
2. 在 inference 入口增加 schema guard，缺少执行卡关键字段时直接失败。
3. 扩展 scorer：
   - meta/disclaimer/markdown residue hard gate。
   - semantic or chunk-level repetition diagnostics。
   - suspicious ending / truncation gate。
   - generic phrase overuse soft or hard gate。
   - non-empty `must_include` / `ending_hook` coverage gate。
4. 重新跑 quality subset，再跑 full50。
5. 只有新 full50 通过机器门禁与人工抽检后，才讨论 100-sample expansion。

## 当前决策

撤销此前 “Stage 4.1 full50 control evidence is sufficient” 的晋级判断。

当前状态应记录为：

`blocked_by_eval_schema_mismatch_and_semantic_repetition`
