# 项目文档索引

这页是 Small Model Train 的文档导航。你不需要一次读完所有文档，按自己的角色选择路线即可。

## 我完全没基础

按这个顺序读：

1. [零基础使用手册](zero-start.zh.md)
2. [术语表](glossary.zh.md)
3. [项目目录地图](project-map.zh.md)
4. [完整数据流说明](pipeline-flow.zh.md)
5. [常见问题排查](troubleshooting.zh.md)

读完这条路线，你应该能知道项目在做什么、文件放哪里、先跑哪些安全命令、输出结果去哪里看。

## 我只想跑流程

按这个顺序读：

1. [零基础使用手册](zero-start.zh.md) 的“第一次安全检查”和后续数据准备、训练前检查章节
2. [第三阶段真实数据准备指南](stage3-data-bring-up-guide.zh.md)
3. [第四阶段 Smoke Eval 指南](stage4-smoke-eval-guide.zh.md)
4. [Stage 4.1 Quality Eval Hardening 指南](stage4-1-quality-eval-guide.zh.md)
5. [Stage 4 决策日志](stage4-decision-log.zh.md)

这条路线适合已经会用 PowerShell、Python，并且只想按步骤复跑数据准备、训练检查、评测和报告的人。

## 我要改代码

按这个顺序读：

1. [代码设计说明](code-design.zh.md)
2. [项目目录地图](project-map.zh.md)
3. [完整数据流说明](pipeline-flow.zh.md)
4. [常见问题排查](troubleshooting.zh.md)
5. `tests/` 里的相关测试文件

改代码前先确认要改的是命令入口还是核心逻辑。一般来说，`scripts/` 负责接收参数和读写文件，`src/small_model_train/` 负责可测试的业务逻辑。

## 阶段文档

- [第一阶段数据管线中文说明](stage1-pipeline-guide.zh.md)：解释从原稿到基础数据、评分和报告的第一阶段流程。
- [第三阶段真实数据准备指南](stage3-data-bring-up-guide.zh.md)：解释真实数据接入、章节卡、SFT 数据和 readiness 报告。
- [第四阶段 Smoke Eval 指南](stage4-smoke-eval-guide.zh.md)：解释固定 50 卡 smoke training、adapter 检查、eval 推理和评分。
- [Stage 4.1 Quality Eval Hardening 指南](stage4-1-quality-eval-guide.zh.md)：解释长生成质量评测、质量子集、预算报告和审阅门槛。
- [Stage 4 决策日志](stage4-decision-log.zh.md)：记录当前是否允许扩大训练规模，以及原因。
- [Stage 4.1 Full50 归档](stage4-1-full50-archive.zh.md)：归档 full50 运行证据。
- [Stage 4.1 Full50 Manual Review Findings](stage4-1-full50-manual-review.zh.md)：记录人工复核发现。
- [第四阶段总结与下一阶段前瞻](stage4-summary-next-outlook.zh.md)：总结当前状态和下一阶段建议。
- [两阶段实现审计报告](two-stage-implementation-audit.zh.md)：审计早期实现是否符合计划。

## 产物去哪里看

- `data_raw/novels/`：你放进去的原始小说文本。
- `data_clean/`：清洗、切章、划分后的章节数据。
- `data_cards/`：章节卡、评测卡、质量子集卡。
- `data_sft/`：给训练用的 SFT 数据和数据集元信息。
- `outputs/`：训练产物、adapter、生成结果、metrics。
- `reports/`：模型检查、环境检查、readiness、评测和质量报告。
- `logs/`：训练、推理、OOM 探测等运行日志。
- `configs/`：训练和推理配置。
