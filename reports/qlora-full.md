# QLoRA 正式训练与验证报告

## 1. 实验目的

在固定的 CMB-Exam v1 单项选择题子集上，完成一轮可复现的 QLoRA 训练，并用同一验证集、同一 prompt 和同一解码参数比较 Qwen3-1.7B 基础模型与 LoRA 适配器的 exact-match 表现。

本实验用于验证工程链路和结果对照，不代表医疗能力评估，也不构成诊断、治疗或其他医疗建议。

## 2. 数据与任务口径

- 基础模型：`Qwen/Qwen3-1.7B`
- 数据集：`FreedomIntelligence/CMB` 的 `CMB-Exam` 配置
- 固定数据：5,000 条 train、240 条有标签 val
- 任务：单项选择题，只保留 assistant target 的答案字母
- 评测指标：预测字母与参考答案完全匹配的 accuracy；无法解析为选项字母计为 invalid
- CMB test split 没有公开答案，因此本报告不报告 test accuracy
- assistant-only loss：只对 assistant 答案部分计算 loss，不对 system/user prompt 计算 loss

## 3. 训练配置

- epoch：1
- optimizer batch size：1
- gradient accumulation：8
- optimizer steps：625
- max sequence length：1,024；本次数据实际最大长度 363，平均长度 131.52
- learning rate：`2e-4`
- warmup ratio：`0.03`
- optimizer：`paged_adamw_8bit`
- gradient checkpointing：开启
- precision：FP16
- base quantization：4-bit NF4、double quantization、FP16 compute
- LoRA：`r=8`、`alpha=16`、`dropout=0.05`
- target modules：attention 的 q/k/v/o projection 与 MLP 的 gate/up/down projection
- seed：42
- 硬件：NVIDIA RTX 4060 Laptop GPU，8 GiB

可训练参数为 8,716,288 / 1,729,291,264，即 0.5040%；因此只保存约 33.3 MiB 的 adapter 参数，而不是保存完整基础模型。

## 4. 训练结果

| 指标 | 结果 |
| --- | ---: |
| train loss | 0.4069 |
| final eval loss | 0.4199 |
| train runtime | 2,337.3 s（约 38 分 57 秒） |
| train samples/s | 2.139 |
| peak CUDA allocated | 3,251.02 MB |
| peak CUDA reserved | 3,580.00 MB |

最终 adapter 输出目录为 `outputs/qlora-cmb-v1/`，其中包含 `adapter_model.safetensors`、`adapter_config.json`、tokenizer 和 `run_summary.json`。中间保留了 `checkpoint-600` 与 `checkpoint-625`，便于恢复或比较。

## 5. 基线与 QLoRA 对照

两组模型都使用验证集 240 条样本、相同 prompt、`max_new_tokens=4` 和 greedy decoding。

| 模型 | correct | total | accuracy | invalid rate |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-1.7B baseline | 107 | 240 | 0.4458（44.58%） | 0.0000 |
| QLoRA adapter | 121 | 240 | 0.5042（50.42%） | 0.0000 |

相对基础模型，QLoRA 在本验证集上净增加 14 道答对题，accuracy 提升 5.84 个百分点。样本级迁移如下：

- 基础模型答错、QLoRA 答对：23 条
- 基础模型答对、QLoRA 答错：9 条
- 两者都答对：98 条
- 两者都答错：110 条
- 预测字母发生变化：64 条

这说明本轮训练在固定验证集上出现了可观测的提升，但同时存在 9 条回退样本；在只有一个训练配置和一个验证切分的情况下，不能把该结果外推为稳定泛化能力。

## 6. 复现命令

```powershell
python scripts/train_qlora.py `
  --config configs/qlora.yaml `
  --output-dir outputs/qlora-cmb-v1

python scripts/run_baseline.py `
  --input data/processed/cmb-exam-v1/val.jsonl `
  --output outputs/qwen3-base-val-full.jsonl `
  --max-new-tokens 4 `
  --greedy

python scripts/run_baseline.py `
  --input data/processed/cmb-exam-v1/val.jsonl `
  --output outputs/qlora-cmb-v1-val-full.jsonl `
  --adapter outputs/qlora-cmb-v1 `
  --max-new-tokens 4 `
  --greedy

python scripts/evaluate_benchmark.py `
  --input outputs/qwen3-base-val-full.jsonl `
  --output reports/qwen3-base-val-full.json

python scripts/evaluate_benchmark.py `
  --input outputs/qlora-cmb-v1-val-full.jsonl `
  --output reports/qlora-cmb-v1-val-full.json
```

训练摘要：`outputs/qlora-cmb-v1/run_summary.json`。逐题评测结果：`reports/qwen3-base-val-full.json` 和 `reports/qlora-cmb-v1-val-full.json`。

## 7. 结论与局限

本阶段结论：先用 Qwen3-1.7B 做统一 prompt 的基线，再将 CMB-Exam 清洗为可控的单项选择任务；训练时采用 4-bit NF4 基座和 LoRA，只更新 0.5040% 的参数；最后用同一验证集做逐题 exact-match 对照，并记录错误迁移，而不是只汇报 loss。

主要局限：训练集只使用 5,000 条子集，训练仅 1 epoch，没有做超参搜索或多随机种子实验；验证任务是选择题字母匹配，不能替代开放式医疗问答质量、安全性和临床有效性评估。后续需要把 RAG、DPO、服务化和更严格的安全评测与当前结果分开记录。
