# Two-Stage Implementation Audit And Docs Design

> 日期：2026-06-18
> 适用分支：`codex/two-stage-audit-docs`
> 依据文件：
> - `docs/superpowers/plans/2026-06-17-qwen3-qlora-stage1-pipeline.md`
> - `docs/superpowers/plans/2026-06-18-qwen3-qlora-stage2-training-execution.md`

## 1. 目标

本轮工作不新增训练能力，不重构已有管线，而是对已经完成的两阶段实现做一次可追溯整理：

```text
按计划核对实现
-> 补关键代码注释
-> 写第一阶段零基础中文说明文档
-> 输出两阶段实现审计报告
-> 检查是否存在降级、掩饰或虚假实现
```

最终交付应让两类读者都能看懂：

```text
零基础读者：知道系统为什么这样拆、每个文件负责什么、怎样从原稿走到训练数据和评测报告。
工程读者：能按计划文件逐项检查代码，知道哪些实现是真执行、哪些是显式 dry-run、哪些是阶段边界内的限制。
```

## 2. 非目标

本轮不做：

```text
不运行真实 QLoRA 训练。
不运行真实 GPU 推理。
不自动生成章节卡。
不修订 Stage 1 / Stage 2 的功能边界。
不大规模重构模块结构。
不新增外部依赖。
不把审计报告里的中长期建议直接实现成新功能。
```

如果发现确实存在代码问题，本轮优先写入审计报告。只有非常小、确定、不会改变行为边界的问题，才允许在实施计划中作为独立修复项列出，并必须有测试。

## 3. 审计对象

### 3.1 Stage 1

计划文件：

```text
docs/superpowers/plans/2026-06-17-qwen3-qlora-stage1-pipeline.md
```

需要核对的交付范围：

```text
项目脚手架与共享文本工具。
原稿清洗与切章。
确定性 train/eval 切分。
风格画像与风格契约。
SFT 数据构造。
规则评分与 AI 味检测。
偏好候选构造。
Markdown 评测报告。
QLoRA 与推理配置。
端到端 smoke test。
```

### 3.2 Stage 2

计划文件：

```text
docs/superpowers/plans/2026-06-18-qwen3-qlora-stage2-training-execution.md
```

需要核对的交付范围：

```text
本地模型文件与 transformers 加载检查。
训练环境、CUDA、依赖与 VRAM 检查。
配置快照与 LLaMA-Factory 命令构造。
训练事件日志、GPU 采样和错误分类。
smoke/full 训练启动器。
adapter 静态校验。
OOM probe 执行框架。
固定 eval 推理和 Stage 1 评分衔接。
README Stage 2 命令序列。
```

## 4. 审查口径

审计报告必须明确区分三类情况。

### 4.1 符合计划

代码实现了计划要求，并且自动测试覆盖了关键行为。例如：

```text
Stage 1 的 JSONL 读写、切章、评分、报告生成均有单元测试或 smoke test。
Stage 2 的训练 supervisor 会保留 stdout/stderr/event/gpu 日志，并能分类 stdout-only OOM。
```

### 4.2 合理阶段边界

计划本身不要求自动完成，或者出于本地资源限制显式留给人工执行。例如：

```text
Stage 1 不自动生成章节卡。
Stage 2 自动测试不跑真实 GPU 训练。
run_sft_smoke.py --dry-run 只预演命令和配置，不声明训练成功。
```

这类情况不能写成“已完成真实训练”，必须在审计报告里说明边界。

### 4.3 风险或问题

出现下列情况时，必须在审计报告中标为风险或问题：

```text
用静态文本冒充实际检查。
失败路径被吞掉，导致脚本返回成功。
dry-run 文案或报告让用户误以为真实训练已完成。
测试只覆盖 mock 路径，真实入口可能导入失败。
计划要求有可执行诊断，但代码只有空报告或占位输出。
报告路径、README 命令和代码默认值互相矛盾。
```

风险级别分为：

```text
Critical：会导致用户误判训练/评测成功，或可能覆盖/丢失数据。
Important：会影响定位问题、复现实验或按计划执行。
Minor：不会破坏流程，但影响理解、维护或文档一致性。
```

## 5. 文档交付

### 5.1 第一阶段中文说明文档

新增：

```text
docs/stage1-pipeline-guide.zh.md
```

目标读者：零基础读者。

文档结构：

```text
1. 第一阶段一句话说明。
2. 系统从原稿到评测报告的数据流。
3. 目录结构：scripts、src、小工具、tests、configs、reports。
4. 核心数据格式：JSONL、章节行、章节卡、SFT 行、评分行、偏好候选行。
5. 每个 Stage 1 脚本做什么、输入输出是什么。
6. 每个 Stage 1 模块做什么、为什么这样拆。
7. 怎样按 README 跑一遍 Stage 1。
8. 常见问题：章节卡为什么不自动生成、为什么固定 eval、为什么检查 source_text 泄漏。
9. Stage 1 和 Stage 2 的边界。
```

写作要求：

```text
用中文解释专业词。
每个概念先用一句话说明，再给文件名或小例子。
避免只堆命令。
明确“这一步做了什么”和“为什么需要这一步”。
```

### 5.2 两阶段实现审计报告

新增：

```text
docs/two-stage-implementation-audit.zh.md
```

报告结构：

```text
1. 审计结论摘要。
2. Stage 1 计划符合性矩阵。
3. Stage 2 计划符合性矩阵。
4. 降级、掩饰、虚假实现专项审查。
5. 已知合理边界。
6. 风险清单与建议后续动作。
7. 验证命令与结果。
```

矩阵每行必须包含：

```text
计划任务。
对应实现文件。
对应测试文件。
结论：符合 / 合理边界 / 风险。
简要说明。
```

## 6. 代码注释策略

本轮注释要解释“设计意图”和“边界”，避免写机械注释。

优先补注释的文件：

```text
src/small_model_train/io_utils.py
src/small_model_train/text_utils.py
src/small_model_train/chapter_splitter.py
src/small_model_train/dataset_split.py
src/small_model_train/style_profile.py
src/small_model_train/sft_builder.py
src/small_model_train/scoring.py
src/small_model_train/preference_builder.py
src/small_model_train/reporting.py
src/small_model_train/stage2_training.py
src/small_model_train/stage2_oom_probe.py
src/small_model_train/stage2_monitoring.py
scripts/stage2_oom_probe_worker.py
scripts/stage2_eval_worker.py
```

注释准则：

```text
模块 docstring：说明该文件在阶段中的责任。
函数 docstring：只给对外或关键内部函数补充，说明输入输出和边界。
行内注释：只用于容易误读的安全边界，例如 dry-run、stdout+stderr 分类、safetensors 只读 header。
不为简单赋值、明显循环、普通 argparse 参数写注释。
```

## 7. 验证方式

完成后必须运行：

```powershell
python -m pytest -q
```

如果只改文档和注释，也要运行测试，确认没有打坏导入、编码或脚本行为。

还要做静态检查：

```powershell
rg -n -i "todo|tbd|placeholder|fake|stub|notimplemented" docs src scripts
git status --short
git diff --check
```

审计报告的“验证命令与结果”必须写入实际运行结果，不允许写未验证的通过措辞。

## 8. 交付标准

本轮完成时应满足：

```text
第一阶段中文说明文档存在，且零基础读者可以按它理解系统。
两阶段实现审计报告存在，且逐项覆盖两个计划。
关键代码注释解释设计意图，不掩盖真实边界。
审计报告明确指出是否存在降级、掩饰、虚假实现。
测试通过。
工作树干净。
```

如果审计发现问题但本轮不修复，必须在报告里给出明确原因：

```text
为什么不在本轮修。
它是否阻断当前阶段使用。
后续应该用什么任务处理。
```

## 9. 自检

本规格没有要求真实 GPU 训练或推理，因为当前任务是审计与文档补强。真实训练仍属于 Stage 2 手动执行闭环。  
本规格没有要求重写现有模块结构，因为当前代码已经按 `src/` 核心逻辑和 `scripts/` 薄 CLI 分层。  
本规格要求审计报告如实写出 dry-run、mock 测试和真实执行边界，防止把“安全预演”描述成“真实完成”。
