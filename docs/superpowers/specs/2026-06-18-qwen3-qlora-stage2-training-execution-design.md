# Qwen3 QLoRA Stage 2 Training Execution Design

> 日期：2026-06-18
> 适用前提：Stage 1 数据与评测闭环已完成；模型已下载到 `E:\models\Qwen3-4B-Instruct-2507`
> 设计选择：方案 A，二阶段只做训练执行闭环
> 三阶段前瞻：数据扩容、质量评测、偏好优化准备

## 1. 二阶段定位

Stage 2 的目标是验证本机能否稳定完成 Qwen3-4B-Instruct-2507 的 QLoRA SFT 训练，并产出一个可被 Stage 1 评测闭环检查的 adapter。

核心闭环：

```text
本地模型检查
-> 训练环境检查
-> 100 条 SFT 冒烟训练
-> OOM 定位与降级记录
-> adapter 检查
-> 500-1000 条 SFT v1 正式训练
-> 固定 eval 推理
-> 复用 Stage 1 规则评分与报告
```

Stage 2 不负责生成章节卡，也不负责补齐训练数据。如果 SFT 数据或 eval cards 缺失，Stage 2 只输出明确阻断报告，不把数据生产纳入本阶段。

## 2. 非目标

Stage 2 第一版不做：

```text
不生成 chapter_cards.jsonl。
不扩充 SFT 数据。
不重做 Stage 1 清洗、切分、评分逻辑。
不做 SFT v2。
不做 DPO / ORPO / KTO / SimPO。
不做 LLM 盲评系统。
不做 Web UI。
不做长期训练服务。
```

这些任务进入 Stage 3 或更后续阶段。

## 3. 输入与输出

### 3.1 输入

模型目录：

```text
E:\models\Qwen3-4B-Instruct-2507
```

训练配置：

```text
configs/sft_qlora_qwen3_4b.yaml
configs/infer_eval_qwen3_4b.yaml
```

Stage 1 产物：

```text
data_sft/sft_chapter_v1.jsonl
data_cards/eval_cards_50.jsonl
```

评测脚本：

```text
scripts/score_outputs.py
scripts/evaluate_outputs.py
```

### 3.2 输出

训练与推理产物：

```text
outputs/sft_smoke/
outputs/sft_v1/
outputs/sft_v1/generated.jsonl
outputs/sft_v1/metrics.jsonl
```

报告产物：

```text
reports/model_check_report.md
reports/training_env_report.md
reports/sft_smoke_report.md
reports/sft_v1_training_report.md
reports/sft_v1_report.md
```

OOM 与崩溃定位产物：

```text
reports/oom_probe_report.md
logs/training/sft_smoke_events.jsonl
logs/training/sft_smoke_gpu.jsonl
logs/training/sft_smoke_stderr.log
logs/training/sft_v1_events.jsonl
logs/training/sft_v1_gpu.jsonl
logs/training/sft_v1_stderr.log
```

## 4. 脚本设计

### 4.1 `scripts/check_local_model.py`

职责：验证本地模型目录是否完整，并确认 `transformers` 能加载 config 和 tokenizer。

检查项：

```text
config.json 存在。
tokenizer.json 存在。
tokenizer_config.json 存在。
model.safetensors.index.json 存在。
safetensors 分片存在且大小非零。
generation_config.json 存在时记录。
AutoConfig.from_pretrained(local_path) 可执行。
AutoTokenizer.from_pretrained(local_path) 可执行。
```

输出：

```text
reports/model_check_report.md
```

失败策略：

```text
缺文件：报告列出缺失文件并停止。
config/tokenizer 加载失败：记录异常类型和错误摘要。
权重分片不完整：不进入训练。
```

### 4.2 `scripts/check_training_env.py`

职责：检查训练环境是否具备启动 QLoRA 的基本条件。

检查项：

```text
Python 版本。
torch 是否可导入。
torch 版本。
CUDA 是否可用。
GPU 名称。
总显存与当前空闲显存。
transformers 是否可导入及版本。
bitsandbytes 是否可导入。
peft 是否可导入。
LLaMA-Factory 命令是否可用。
HF_HOME / TRANSFORMERS_CACHE / HF_ENDPOINT。
```

显存建议：

```text
可用显存 >= 13GB：允许 8192 cutoff_len 冒烟训练。
可用显存 11-13GB：建议从 6144 cutoff_len 开始。
可用显存 < 11GB：不建议启动训练。
```

输出：

```text
reports/training_env_report.md
```

### 4.3 `scripts/run_sft_smoke.py`

职责：启动 100 条样本以内的 QLoRA 冒烟训练。

默认行为：

```text
读取 configs/sft_qlora_qwen3_4b.yaml。
将 model_name_or_path 覆盖为 E:\models\Qwen3-4B-Instruct-2507。
将 max_samples 覆盖为 100。
将 output_dir 覆盖为 outputs/sft_smoke。
保存实际训练配置快照。
通过子进程启动 LLaMA-Factory。
捕获 stdout、stderr、退出码。
采样 GPU 状态。
记录训练阶段心跳事件。
```

冒烟训练验收：

```text
训练命令启动成功。
模型能以 4-bit 方式加载。
LoRA adapter 初始化成功。
至少进入第一个训练 step。
adapter 文件保存成功。
```

### 4.4 `scripts/run_sft_train.py`

职责：启动正式 SFT v1 训练。

默认行为：

```text
读取 configs/sft_qlora_qwen3_4b.yaml。
将 model_name_or_path 覆盖为 E:\models\Qwen3-4B-Instruct-2507。
将 output_dir 设置为 outputs/sft_v1。
保存配置快照到 outputs/sft_v1/training_config_snapshot.yaml。
通过与 smoke 相同的子进程、日志、GPU 采样和心跳机制执行训练。
```

启动条件：

```text
本地模型检查通过。
训练环境检查通过。
100 条冒烟训练通过。
冒烟 adapter 检查通过。
Stage 1 测试通过。
用户已准备好 data_sft/sft_chapter_v1.jsonl。
```

### 4.5 `scripts/check_adapter.py`

职责：验证训练输出目录是否包含可加载 adapter。

检查项：

```text
adapter_config.json 存在。
adapter_model.safetensors 存在且大小非零。
trainer_state.json 或训练日志存在。
训练配置快照存在。
AutoPeftModel 或 LLaMA-Factory 可识别 adapter。
```

输出：

```text
reports/sft_smoke_report.md
reports/sft_v1_training_report.md
```

### 4.6 `scripts/run_eval_inference.py`

职责：用固定 eval cards 对 SFT v1 adapter 批量生成整章正文。

输入：

```text
data_cards/eval_cards_50.jsonl
outputs/sft_v1
configs/infer_eval_qwen3_4b.yaml
```

输出：

```text
outputs/sft_v1/generated.jsonl
```

输出格式固定为：

```json
{"id":"case_id","output":"模型生成正文","model":"sft_v1","params":{"temperature":0.7,"top_p":0.8,"top_k":20,"repetition_penalty":1.05}}
```

推理结束后复用 Stage 1：

```powershell
python scripts/score_outputs.py --cards data_cards/eval_cards_50.jsonl --outputs outputs/sft_v1/generated.jsonl --output outputs/sft_v1/metrics.jsonl
python scripts/evaluate_outputs.py --scores outputs/sft_v1/metrics.jsonl --report reports/sft_v1_report.md --title "SFT v1 Report"
```

## 5. OOM 与崩溃定位冗余机制

Stage 2 必须能回答两个问题：

```text
OOM 或崩溃发生在哪个阶段？
下一次应该降低哪个参数，而不是盲目重跑？
```

因此训练脚本不能只调用一次训练命令然后等待结果。必须引入子进程隔离、阶段心跳、GPU 采样、错误分类和阶梯降级。

### 5.1 子进程隔离

所有高风险 GPU 操作必须在子进程中执行：

```text
模型 4-bit 加载。
LoRA 初始化。
训练首个 step。
正式训练。
adapter 加载。
eval 推理。
```

主进程只负责：

```text
生成配置。
启动子进程。
捕获 stdout / stderr。
采样 GPU。
写入心跳事件。
解析退出码。
生成报告。
```

这样即使训练子进程因为 CUDA OOM、驱动重置、bitsandbytes 崩溃或 Python 进程被杀，主进程仍可保留最后日志并输出报告。

### 5.2 阶段心跳事件

训练启动器写入 JSONL 事件日志：

```text
logs/training/sft_smoke_events.jsonl
logs/training/sft_v1_events.jsonl
```

事件示例：

```json
{"time":"2026-06-18T12:00:00+08:00","phase":"prepare_config","status":"start"}
{"time":"2026-06-18T12:00:01+08:00","phase":"prepare_config","status":"ok"}
{"time":"2026-06-18T12:00:02+08:00","phase":"launch_train_subprocess","status":"start"}
{"time":"2026-06-18T12:00:20+08:00","phase":"load_base_model_4bit","status":"seen_in_log"}
{"time":"2026-06-18T12:00:45+08:00","phase":"first_train_step","status":"seen_in_log"}
```

标准阶段名：

```text
prepare_config
validate_input_files
launch_train_subprocess
load_tokenizer
load_base_model_4bit
prepare_lora
tokenize_dataset
first_forward
first_backward
first_optimizer_step
save_adapter
load_adapter_for_check
eval_first_generation
```

如果进程崩溃，报告使用最后一个已知阶段定位问题。例如最后事件停在 `load_base_model_4bit`，则判断崩溃点接近基座模型量化加载；如果停在 `first_backward`，则判断训练反向传播显存不足。

### 5.3 GPU 采样日志

训练期间每 2 秒采样一次 GPU 状态，写入：

```text
logs/training/sft_smoke_gpu.jsonl
logs/training/sft_v1_gpu.jsonl
```

优先使用 `nvidia-smi`，不可用时退化为 `torch.cuda.mem_get_info()`。

采样字段：

```json
{"time":"2026-06-18T12:00:10+08:00","gpu_name":"NVIDIA GPU","total_mb":16384,"free_mb":4212,"used_mb":12172,"processes":[{"pid":1234,"used_mb":11800,"name":"python.exe"}]}
```

报告中记录：

```text
训练前空闲显存。
峰值已用显存。
崩溃前最后一次空闲显存。
是否存在非训练进程占用 GPU。
```

### 5.4 错误分类

stderr 和日志按规则分类：

```text
cuda_oom：出现 CUDA out of memory、CUBLAS_STATUS_ALLOC_FAILED。
process_killed：进程无 Python traceback 且退出码异常。
driver_reset：出现 CUDA driver、device lost、unknown error。
bnb_load_error：bitsandbytes 导入或 4-bit 加载失败。
tokenizer_error：tokenizer/config 加载失败。
dataset_error：数据路径、格式、dataset_info 错误。
llamafactory_error：训练命令或参数不被 LLaMA-Factory 接受。
adapter_save_error：训练完成但 adapter 文件缺失或保存失败。
```

每个错误分类必须在报告里给出：

```text
错误类型。
最后已知阶段。
退出码。
stderr 摘要。
最后三条心跳事件。
最后三条 GPU 采样。
建议下一步动作。
```

### 5.5 阶梯探测而不是盲目重试

OOM 后不直接重跑完整训练。先执行最小探测：

```text
Probe 1: 只加载 tokenizer 和 config。
Probe 2: 加载 4-bit base model，不训练。
Probe 3: 注入 LoRA，不训练。
Probe 4: 取 1 条样本 tokenize。
Probe 5: cutoff_len=8192，max_steps=1。
Probe 6: cutoff_len=6144，max_steps=1。
Probe 7: cutoff_len=6144，lora_rank=8，max_steps=1。
```

探测结果写入：

```text
reports/oom_probe_report.md
```

探测目的：

```text
如果 Probe 2 失败：基座 4-bit 加载就不稳，优先查 bitsandbytes / CUDA / 显存占用。
如果 Probe 3 失败：LoRA target 或 PEFT 配置问题。
如果 Probe 5 失败但 Probe 6 成功：8192 cutoff_len 过高。
如果 Probe 6 失败但 Probe 7 成功：LoRA rank 或反向传播显存压力过高。
如果所有 probe 通过但正式训练失败：更可能是数据长度分布、保存、日志或长时间运行稳定性问题。
```

### 5.6 降级策略

降级顺序固定，不混乱组合：

```text
1. cutoff_len: 8192 -> 6144
2. lora_rank: 16 -> 8
3. gradient_accumulation_steps: 16 -> 32
4. max_samples: full -> 100 smoke
5. 暂停训练，提示关闭其他 GPU 进程后重试
```

不优先降到 `cutoff_len=4096`，因为整章任务需要学习后半章、反转和章末钩子。只有用户明确同意，才把 4096 作为临时诊断参数。

### 5.7 报告中的崩溃结论

训练失败报告必须包含一个明确结论：

```text
失败点：load_base_model_4bit / first_backward / save_adapter / eval_first_generation 等。
失败类型：cuda_oom / dataset_error / bnb_load_error 等。
最可能原因：显存不足、依赖不匹配、数据格式错误、adapter 保存失败等。
建议动作：降低 cutoff_len、降低 rank、关闭其他 GPU 程序、修数据路径、重装依赖等。
是否允许继续下一步：是 / 否。
```

## 6. 配置策略

现有配置文件保留为源配置：

```text
configs/sft_qlora_qwen3_4b.yaml
configs/infer_eval_qwen3_4b.yaml
```

Stage 2 运行时生成快照配置：

```text
outputs/sft_smoke/training_config_snapshot.yaml
outputs/sft_v1/training_config_snapshot.yaml
```

快照必须覆盖：

```yaml
model_name_or_path: E:\models\Qwen3-4B-Instruct-2507
output_dir: outputs/sft_smoke 或 outputs/sft_v1
```

冒烟训练额外覆盖：

```yaml
max_samples: 100
save_steps: 50
logging_steps: 5
num_train_epochs: 1
```

正式训练沿用：

```yaml
cutoff_len: 8192
lora_rank: 16
lora_alpha: 32
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 3.0e-5
num_train_epochs: 2
```

如果降级，快照中必须记录实际使用参数，并在报告中列出降级原因。

## 7. 数据边界

方案 A 不生产数据，但训练前会做最小存在性检查：

```text
data_sft/sft_chapter_v1.jsonl 存在。
data_cards/eval_cards_50.jsonl 存在。
SFT 文件非空。
eval cards 文件非空。
```

Stage 2 不检查章节卡质量，不检查 SFT 样本是否足够文学化，不修复泄漏问题。这些进入 Stage 3。

如果数据缺失，报告应写：

```text
Stage 2 blocked: missing SFT dataset or eval cards.
```

并列出需要用户或上游阶段补齐的文件。

## 8. 测试策略

Stage 2 新增脚本应采用“轻测试 + 可选集成测试”的方式。

必跑测试：

```text
check_local_model 的文件检查逻辑可用临时目录测试。
check_training_env 的报告渲染逻辑可用 mock 数据测试。
OOM 错误分类逻辑可用 stderr 样本文本测试。
训练事件日志写入逻辑可用临时 JSONL 测试。
adapter 检查逻辑可用临时文件测试。
```

不在自动化测试中强制跑真实训练，因为真实训练依赖 GPU、bitsandbytes、LLaMA-Factory 和本地数据。真实训练通过手动命令和报告验收。

每次实现后仍运行：

```powershell
python -m pytest -v
```

## 9. 二阶段完成标准

Stage 2 完成时应满足：

```text
本地模型检查通过。
训练环境检查报告生成。
100 条冒烟训练通过。
OOM/崩溃定位机制可用，并至少能产生日志与报告。
smoke adapter 检查通过。
SFT v1 adapter 生成。
SFT v1 adapter 检查通过。
固定 eval 推理完成。
outputs/sft_v1/generated.jsonl 生成。
outputs/sft_v1/metrics.jsonl 生成。
reports/sft_v1_report.md 生成。
```

如果训练失败，Stage 2 仍必须交付：

```text
失败报告。
最后已知阶段。
错误分类。
GPU 采样摘要。
降级建议。
是否阻断下一步。
```

## 10. 三阶段前瞻规划

Stage 3 建议命名为：

```text
Stage 3: Data Expansion, Quality Evaluation, Preference Preparation
```

中文定位：

```text
数据扩容、质量评测与偏好优化准备
```

Stage 3 解决 Stage 2 故意不处理的问题：模型是否真的更会写、数据是否够好、章节卡是否稳定、坏例如何回流。

### 10.1 Stage 3 主线

推荐主线：

```text
SFT v1 坏例复盘
-> 章节卡模板修订
-> 生产式 eval cards 引入
-> SFT v2 数据扩容到 3000-5000 条
-> 固定 eval 扩展到 200 张
-> 人工/LLM 100 分制盲评
-> 人工轻修可用率统计
-> 偏好候选数据清洗
-> ORPO/KTO 小规模实验
```

### 10.2 Stage 3 交付物

```text
docs/stage3-quality-loop-guide.zh.md
data_cards/eval_cards_200.jsonl
data_sft/sft_chapter_v2.jsonl
data_pref/preference_candidates_v1.jsonl
data_pref/kto_v1.jsonl
data_pref/orpo_v1.jsonl
reports/sft_v1_badcase_review.md
reports/sft_v2_report.md
reports/pref_v1_report.md
```

### 10.3 Stage 3 指标

Stage 3 不再只看训练是否成功，而看质量是否提升：

```text
硬门槛通过率。
100 分制均分。
人工轻修可用率。
AI 味下降。
擅自加戏下降。
must_include 覆盖率。
must_not_include 违反率。
章末钩子保留率。
后半章塌陷率。
风格相似度。
source_memorization 风险。
```

### 10.4 Stage 3 方法取舍

优先顺序：

```text
先做 SFT v2 数据扩容。
再做偏好数据 chosen/rejected 清洗。
优先尝试 ORPO 或 KTO。
成对偏好数据足够稳定后，再考虑 DPO 或 SimPO。
不直接用 GRPO/RLVR 优化小说正文质量。
```

理由：

```text
SFT v1 之后，首要问题通常不是偏好算法，而是章节卡分布、样本覆盖和坏例类型。
ORPO/KTO 对小显存和不完整偏好数据更友好。
DPO/SimPO 对 chosen/rejected 质量要求更高，适合在偏好数据成熟后对照。
小说质量难以被简单规则奖励完整覆盖，因此 RLVR 只适合字数、禁项等局部实验。
```

## 11. 推荐实施顺序

Stage 2 实现建议按以下顺序：

```text
1. 本地模型检查脚本。
2. 训练环境检查脚本。
3. 事件日志、GPU 采样、错误分类公共模块。
4. 100 条冒烟训练脚本。
5. OOM probe 报告。
6. adapter 检查脚本。
7. 正式 SFT v1 训练脚本。
8. 固定 eval 推理脚本。
9. 报告整合与 README 更新。
```

这个顺序先降低环境不确定性，再进入真正训练。训练相关脚本共享同一套日志和崩溃定位机制，避免 smoke 和正式训练各写一套。
