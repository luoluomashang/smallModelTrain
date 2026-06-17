# Qwen3 QLoRA Stage 2 Training Direction

> 日期：2026-06-17
> 适用前提：Stage 1 数据与评测闭环已完成
> 目标：模型下载、环境验证、冒烟训练、正式 QLoRA SFT、训练后评测

## 1. Stage 2 目标

Stage 2 负责把 Stage 1 产出的数据和配置接到真实训练流程中。它不重新设计数据清洗和评分逻辑，而是验证本机环境能否稳定完成 Qwen3-4B-Instruct-2507 的 QLoRA 微调，并产出可评测 adapter。

核心闭环：

```text
环境检查
-> 模型下载
-> 100 条样本冒烟训练
-> OOM 降级验证
-> 500-1000 条 SFT v1 正式训练
-> adapter 产物校验
-> 固定 eval 推理
-> 训练报告
```

## 2. Stage 2 非目标

第一版 Stage 2 不做：

- 大规模 SFT v2。
- DPO/ORPO/KTO/SimPO 正式偏好训练。
- 多 adapter 题材拆分。
- GRPO/RLVR。
- Web UI 或长期训练服务。

这些内容等 SFT v1 训练报告稳定后再进入后续阶段。

## 3. 环境验证

需要新增 `scripts/check_training_env.py`，输出训练环境报告：

```text
Python 版本
CUDA 是否可用
GPU 名称
总显存与空闲显存
torch 版本
transformers 版本
bitsandbytes 是否可导入
LLaMA-Factory 是否可执行
HF_HOME / TRANSFORMERS_CACHE / HF_ENDPOINT
Codex 运行期间建议保留的显存余量
```

验收建议：

```text
可用显存 >= 13GB：允许 8192 cutoff_len 冒烟训练。
可用显存 11-13GB：先使用 6144 cutoff_len。
可用显存 < 11GB：不启动训练，提示关闭占用 GPU 的进程。
```

## 4. 模型下载

需要新增 `scripts/download_model.py` 或 README 命令，负责下载：

```text
Qwen/Qwen3-4B-Instruct-2507
```

下载策略：

- 默认使用 Hugging Face cache。
- 支持通过环境变量设置 `HF_HOME` 或 `TRANSFORMERS_CACHE`。
- 支持中国大陆网络下配置镜像端点。
- 下载后运行 tokenizer 和 config 加载验证。

验收标准：

```text
模型 config 可加载。
tokenizer 可加载。
本地缓存路径可打印。
不会在训练目录复制完整模型权重。
```

## 5. LLaMA-Factory 集成

Stage 2 需要明确 LLaMA-Factory 的安装方式和命令封装。

推荐策略：

```text
外部安装 LLaMA-Factory
本项目只保存配置、数据、报告和启动脚本
```

需要新增：

```text
scripts/run_sft_smoke.py
scripts/run_sft_train.py
scripts/check_adapter.py
```

这些脚本可以优先生成并打印命令，后续再加入实际执行开关：

```powershell
python scripts/run_sft_smoke.py --config configs/sft_qlora_qwen3_4b.yaml --max-samples 100
python scripts/run_sft_train.py --config configs/sft_qlora_qwen3_4b.yaml
python scripts/check_adapter.py --adapter outputs/sft_v1
```

## 6. 冒烟训练

冒烟训练只使用 100 条以内 SFT 样本，目标是验证：

- 数据格式被训练框架接受。
- 4-bit 量化可加载。
- LoRA target 可解析。
- 8192 cutoff_len 或 6144 cutoff_len 不 OOM。
- loss 能正常下降或至少正常记录。
- adapter 能保存。

冒烟训练不以模型质量为目标。

推荐冒烟参数：

```yaml
max_samples: 100
save_steps: 50
logging_steps: 5
num_train_epochs: 1
```

如果 OOM，按顺序降级：

```text
cutoff_len 8192 -> 6144
lora_rank 16 -> 8
gradient_accumulation_steps 16 -> 32
关闭其他 GPU 进程后重试
```

## 7. 正式 SFT v1

正式 SFT v1 使用 500-1000 条 A 类整章样本。

启动条件：

```text
Stage 1 全部测试通过。
SFT JSONL 样本数量 >= 500。
冒烟训练通过。
固定 eval cards 已准备。
训练前显存检查通过。
```

成功标准：

```text
训练无 OOM。
adapter 文件完整。
训练日志保存。
配置快照保存。
数据版本保存。
```

质量判断不看训练 loss，而看训练后固定 eval 报告。

## 8. 训练后评测

Stage 2 需要加入训练后推理脚本方向：

```text
scripts/run_eval_inference.py
```

输出格式固定：

```json
{"id": "case_id", "output": "模型生成正文", "model": "sft_v1", "params": {"temperature": 0.7}}
```

评测仍复用 Stage 1：

```powershell
python scripts/score_outputs.py --cards data_cards/eval_cards_50.jsonl --outputs outputs/sft_v1/generated.jsonl --output outputs/sft_v1/metrics.jsonl
python scripts/evaluate_outputs.py --scores outputs/sft_v1/metrics.jsonl --report reports/sft_v1_report.md --title "SFT v1 Report"
```

## 9. 训练报告

需要新增训练报告或扩展现有报告：

```text
reports/training_env_report.md
reports/sft_smoke_report.md
reports/sft_v1_training_report.md
reports/sft_v1_report.md
```

报告包含：

- 环境快照。
- 模型缓存路径。
- 训练配置。
- 数据集版本和样本数量。
- 显存峰值。
- OOM 降级记录。
- adapter 文件检查结果。
- 固定 eval 评分。
- 是否进入 SFT v2 或偏好优化阶段。

## 10. Stage 2 完成标准

Stage 2 完成时应满足：

```text
模型已下载并通过加载验证。
训练环境报告存在。
100 条冒烟训练通过。
SFT v1 adapter 生成。
adapter 检查通过。
固定 eval 推理完成。
reports/sft_v1_report.md 生成。
AcceptanceAgent 给出下一阶段建议。
```

如果显存条件无法满足，Stage 2 应输出明确阻断报告，而不是反复重试训练。
